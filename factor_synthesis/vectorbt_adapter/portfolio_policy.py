"""Shared portfolio policy logic decoupled from equity/futures execution engines.

Centralizes dynamic pool lookup, causal signal alignment, factor scoring and
decay, directional tradability filtering, Top-K / overlay / MVO / partial
rebalance / liquidity-cap / long-short weighting, skip-month liquidations, and
alignment diagnostics so both the VectorBT equity pre-simulator and the
single-account futures execution engine execute the exact same strategy policy.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Protocol, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from utils.execution_schedule import resolve_calendar_signal_bar, resolve_runner_execution
from utils.output_paths import open_diagnostic
from utils.strategy_kernel import rank_scores

from ..core.calendar_skip import date_in_skip, skip_mask
from .slot_provider import (
    apply_magnitude_caps,
    compute_long_short_target_weights,
    sanitize_execution_prices,
)


class PolicyAccountView(Protocol):
    """Read-only view of the live portfolio account state at a single bar."""

    def held_instruments(self) -> Set[str]: ...

    def held_notional(self, instrument: str) -> float: ...

    def nav(self) -> float: ...

    def prev_weight_map(self) -> Dict[str, float]: ...

    def execution_price(self, instrument: str) -> Optional[float]: ...

    def mark_price(self, instrument: str) -> Optional[float]: ...

    def filter_tradeable(
        self,
        date_str: str,
        instruments: Sequence[str],
        *,
        side: str,
        require_non_st: bool,
    ) -> List[str]: ...

    def deployed_capital_for_liquidity(self, exposure: float) -> float: ...

    def base_capital(self, exposure: float) -> float: ...

    def branch3_buy_target_value(
        self,
        *,
        executed_sells: Set[str],
        remaining_held: Set[str],
        effective_top_k: int,
        buy_count: int,
        exposure: float,
    ) -> float: ...


@dataclass
class BarPolicyDecision:
    """Target instructions produced by the shared portfolio policy for one bar."""

    kind: Literal[
        "noop",
        "skip_liquidate",
        "liquidate_targetless",
        "no_signal",
        "branch_long_short",
        "branch_overlay",
        "branch_topk",
    ]
    date: Any
    date_normalized: pd.Timestamp
    date_str: str
    is_rebalance: bool
    is_skip: bool
    # Per-instrument target dollar value (0.0 for close, positive/negative for target,
    # omitted or NaN when holding unchanged or blocked).
    target_values: Dict[str, float] = field(default_factory=dict)
    # Per-instrument target weight fraction relative to account equity/NAV
    # (used by futures single-account sizing where leverage/margin_rate apply).
    target_weights: Dict[str, float] = field(default_factory=dict)
    # Instruments whose pending exit was cancelled because they were re-selected.
    cancelled_exits: Set[str] = field(default_factory=set)
    # Instruments whose exit was blocked by tradability or missing exec price.
    blocked_exits: Set[str] = field(default_factory=set)
    # Instruments to force-close (e.g. skip_liquidate / liquidate_targetless / branch_topk sells).
    sell_instruments: Set[str] = field(default_factory=set)
    # Instruments held unchanged (NaN in target matrix).
    hold_instruments: Set[str] = field(default_factory=set)
    # Instruments to keep/open with target_values (or NaN if forced in overlay).
    forced_instruments: Set[str] = field(default_factory=set)
    # Per-instrument fee overrides (used by nominal VectorBT asymmetric fees).
    order_fees: Dict[str, float] = field(default_factory=dict)
    # Optional alignment snapshot metadata for runner._alignment_writer.
    snapshot_payload: Optional[Dict[str, Any]] = None
    # Count of buys skipped due to strict_max_actual_holdings.
    strict_buys_skipped: int = 0
    # True when policy requires target_values[inst] to be treated as notional directly
    # (Branch 3 non-initial rebalance in futures mode or overlay reallocation).
    use_direct_notional_targets: bool = False


@dataclass
class PortfolioPolicyContext:
    """Prepared per-run state shared across bars by ``SharedPortfolioPolicy``."""

    all_dates: pd.Index
    all_dates_idx: pd.DatetimeIndex
    is_daily_naive: bool
    market_cal: Any
    all_stocks: List[str]
    all_stocks_set: Set[str]
    price_df: pd.DataFrame
    open_df: Optional[pd.DataFrame]
    exec_price_df: pd.DataFrame
    sorted_pool_dates: List[pd.Timestamp]
    normalized_pools: Dict[pd.Timestamp, Set[str]]
    rebalance_set: Set[pd.Timestamp]
    skip_liquidation_bars: Set[pd.Timestamp]
    entry_offset: int
    scheduler: Any
    buy_cost_multiplier: float
    sell_proceeds_multiplier: float
    buy_fee_rate: float
    sell_fee_rate: float
    capture_decisions: bool
    bar_key_fn: Callable[[Any], str]
    norm_bar_fn: Callable[[Any], pd.Timestamp]


class SharedPortfolioPolicy:
    """Stateless-per-bar portfolio policy parameterized by ``VectorBTBacktestRunner``."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner

    def prepare(
        self,
        *,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        dynamic_pools: Dict[str, List[str]],
        rebalance_dates: List[datetime],
        open_df: Optional[pd.DataFrame] = None,
    ) -> PortfolioPolicyContext:
        runner = self.runner
        all_dates = factor_df.index
        all_dates_idx = pd.DatetimeIndex(all_dates)
        is_daily_naive = runner._slot_provider.is_daily_naive_index(all_dates_idx)
        try:
            market_cal = runner._trading_calendar()
        except Exception:
            market_cal = None

        def _bar_key(ts: Any) -> str:
            return runner._slot_provider.bar_key(ts, daily_naive=is_daily_naive)

        def _norm_bar(ts: Any) -> pd.Timestamp:
            return runner._slot_provider.norm_bar_ts(ts, daily_naive=is_daily_naive)

        runner._score_decay_state = pd.Series(dtype=float)
        runner._score_decay_date = None
        runner._input_factor_df = factor_df
        runner._mvo_diagnostics = []

        all_stocks = factor_df.columns.tolist()
        if open_df is None and runner.slot is not None:
            open_df = runner._slot_provider.read_execution_price_frame(
                all_dates_idx, all_stocks
            )
        elif open_df is not None:
            open_df = sanitize_execution_prices(open_df)

        if runner._execution_scheduler is None:
            runner._execution_scheduler, runner.effective_execution_profile = (
                resolve_runner_execution(runner, "open" if open_df is not None else "close")
            )
        scheduler = runner._execution_scheduler
        entry_offset = scheduler.resolved_entry_offset_bars
        runner._backtest_previous_dates = scheduler.signal_timestamp_map(
            all_dates, key_fn=_bar_key
        )
        runner._factor_replay_dates = sorted(
            {
                runner._backtest_previous_dates[_bar_key(d)]
                for d in rebalance_dates
                if _bar_key(d) in runner._backtest_previous_dates
            }
        )

        all_stocks_set = set(all_stocks)
        sorted_pool_dates, normalized_pools = runner._slot_provider.resolve_dynamic_pools(
            all_dates_idx,
            dynamic_pools,
            runner.price_db_conn,
            runner.price_db_path,
        )
        rebalance_set = {_norm_bar(d) for d in rebalance_dates}

        skip_liquidation_bars: Set[pd.Timestamp] = set()
        if runner.skip_months:
            session_days = pd.DatetimeIndex(
                [
                    runner._slot_provider.session_date_for_bar(
                        d, daily_naive=is_daily_naive, cal=market_cal
                    )
                    for d in all_dates
                ]
            )
            in_skip = skip_mask(session_days, runner.skip_months)
            if in_skip.any():
                prev_in_skip = np.concatenate(([False], in_skip[:-1]))
                entry_bars = all_dates[in_skip & ~prev_in_skip]
                skip_liquidation_bars = {_norm_bar(d) for d in entry_bars}
                for d in entry_bars:
                    runner.logger.debug(f"[Skip] 识别清仓Bar: {_bar_key(d)}")

        exec_price_df = open_df if open_df is not None else price_df
        runner._active_exec_price_df = exec_price_df

        buy_fee_rate = max(float(runner.buy_cost_rate), 0.0) + max(float(runner.slippage), 0.0)
        sell_fee_rate = max(float(runner.sell_cost_rate), 0.0) + max(float(runner.slippage), 0.0)
        buy_cost_multiplier = (
            1.0 + max(float(runner.buy_cost_rate), 0.0) + max(float(runner.slippage), 0.0)
        )
        sell_proceeds_multiplier = max(
            0.0, 1.0 - max(float(runner.sell_cost_rate), 0.0) - max(float(runner.slippage), 0.0)
        )
        capture_decisions = bool(
            runner.diagnostics_output_dir and runner.return_series_trial_id
        )

        return PortfolioPolicyContext(
            all_dates=all_dates,
            all_dates_idx=all_dates_idx,
            is_daily_naive=is_daily_naive,
            market_cal=market_cal,
            all_stocks=all_stocks,
            all_stocks_set=all_stocks_set,
            price_df=price_df,
            open_df=open_df,
            exec_price_df=exec_price_df,
            sorted_pool_dates=sorted_pool_dates,
            normalized_pools=normalized_pools,
            rebalance_set=rebalance_set,
            skip_liquidation_bars=skip_liquidation_bars,
            entry_offset=entry_offset,
            scheduler=scheduler,
            buy_cost_multiplier=buy_cost_multiplier,
            sell_proceeds_multiplier=sell_proceeds_multiplier,
            buy_fee_rate=buy_fee_rate,
            sell_fee_rate=sell_fee_rate,
            capture_decisions=capture_decisions,
            bar_key_fn=_bar_key,
            norm_bar_fn=_norm_bar,
        )

    def evaluate_bar(
        self,
        *,
        ctx: PortfolioPolicyContext,
        date: Any,
        account: PolicyAccountView,
        exposure_by_date: Optional[Dict[Any, float]] = None,
    ) -> BarPolicyDecision:
        runner = self.runner
        date_normalized = ctx.norm_bar_fn(date)
        skip_lookup_day = runner._slot_provider.session_date_for_bar(
            date, daily_naive=ctx.is_daily_naive, cal=ctx.market_cal
        )
        date_key = ctx.bar_key_fn(date)
        is_rebalance = date_normalized in ctx.rebalance_set

        if date_normalized in ctx.skip_liquidation_bars:
            return BarPolicyDecision(
                kind="skip_liquidate",
                date=date,
                date_normalized=date_normalized,
                date_str=date_key,
                is_rebalance=is_rebalance,
                is_skip=True,
                sell_instruments=set(account.held_instruments()),
            )

        if not is_rebalance:
            return BarPolicyDecision(
                kind="noop",
                date=date,
                date_normalized=date_normalized,
                date_str=date_key,
                is_rebalance=False,
                is_skip=False,
            )

        if runner.skip_months and date_in_skip(skip_lookup_day, runner.skip_months):
            return BarPolicyDecision(
                kind="skip_liquidate",
                date=date,
                date_normalized=date_normalized,
                date_str=date_key,
                is_rebalance=True,
                is_skip=True,
                sell_instruments=set(account.held_instruments()),
            )

        if not ctx.sorted_pool_dates:
            valid_stocks = ctx.all_stocks_set
        else:
            pool_pos = bisect_right(ctx.sorted_pool_dates, date_normalized) - 1
            valid_stocks = (
                ctx.normalized_pools[ctx.sorted_pool_dates[pool_pos]] & ctx.all_stocks_set
                if pool_pos >= 0
                else set()
            )

        try:
            with open_diagnostic(
                "local_pools.txt",
                "a",
                directory=runner.diagnostics_output_dir,
                enabled=runner.verbose_alignment or bool(runner.diagnostics_output_dir),
            ) as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"DATE: {date_key}\n")
                f.write(f"POOL_PRE_FILTER: {len(valid_stocks)}\n")
                f.write(f"POOL_STOCKS: {','.join(sorted(valid_stocks))}\n")
        except Exception as exc:
            runner.logger.warning(f"Failed to dump pool: {exc}")

        if not valid_stocks:
            return BarPolicyDecision(
                kind="liquidate_targetless",
                date=date,
                date_normalized=date_normalized,
                date_str=date_key,
                is_rebalance=True,
                is_skip=False,
                sell_instruments=set(account.held_instruments()),
            )

        date_str = date_key
        if runner.score_source == "input_factor":
            position = ctx.all_dates.get_loc(date)
            if isinstance(position, slice):
                position = position.start
            sig_ts = ctx.scheduler.signal_timestamp_for_position(
                ctx.all_dates, int(position)
            )
            prev_date = ctx.bar_key_fn(sig_ts) if sig_ts is not None else None
        else:
            prev_date = runner._backtest_previous_dates.get(date_str)
            if prev_date is None and ctx.is_daily_naive:
                prev_date = (
                    runner._get_previous_trading_day(date_str)
                    if ctx.entry_offset == 1
                    else resolve_calendar_signal_bar(
                        ctx.market_cal
                        if ctx.market_cal is not None
                        else runner._trading_calendar(),
                        date_str,
                        frequency=runner.frequency,
                        entry_offset_bars=ctx.entry_offset,
                    )
                )

        if prev_date is None:
            runner.logger.debug(
                f"[VectorBT] No signal bar (offset={ctx.entry_offset}) for {date_str}, skipping"
            )
            return BarPolicyDecision(
                kind="no_signal",
                date=date,
                date_normalized=date_normalized,
                date_str=date_key,
                is_rebalance=True,
                is_skip=False,
            )
        prev_date_str = prev_date

        score_series, processed_factors_df, raw_factors_df = (
            runner._get_synthesis_score(prev_date_str, list(valid_stocks))
        )
        score_series = runner._apply_score_decay(score_series, prev_date_str)

        if len(score_series) > 0:
            runner.logger.debug(
                f"[DEBUG SCORE] Computed {len(score_series)} scores, "
                f"mean={score_series.mean():.4f}, std={score_series.std():.4f}, "
                f"top3={score_series.nlargest(3).to_dict()}"
            )
            if runner.verbose_alignment:
                ranked_scores = score_series.sort_values(ascending=False)
                top10 = ranked_scores.head(10)
                top10_str = "|".join([f"{s}:{v:.6f}" for s, v in top10.items()])
                runner.logger.info(
                    f"FACTOR_ALIGNMENT|{date_str}|TOP10|n={len(score_series)}|{top10_str}"
                )

        min_required = runner._get_effective_top_k(len(score_series))
        if len(score_series) < min_required:
            runner.logger.debug(
                f"[VectorBT] {date}: Only {len(score_series)} valid stocks (T-1={prev_date}), need ~{min_required}"
            )
            if len(score_series) == 0:
                return BarPolicyDecision(
                    kind="liquidate_targetless",
                    date=date,
                    date_normalized=date_normalized,
                    date_str=date_key,
                    is_rebalance=True,
                    is_skip=False,
                    sell_instruments=set(account.held_instruments()),
                )

        ranked = score_series.reindex(rank_scores(score_series))
        runner._log_rebalance_debug(
            date_str=date_str,
            prev_date_str=prev_date_str,
            ranked=ranked,
            raw_factors_df=raw_factors_df,
            score_series=score_series,
        )

        held_set = set(account.held_instruments())
        if runner.enable_tday_filter:
            ranked_names = ranked.index.tolist()
            buyable = set(
                account.filter_tradeable(
                    date_str, ranked_names, side="buy", require_non_st=True
                )
            )
            sellable_open = (
                set(
                    account.filter_tradeable(
                        date_str, ranked_names, side="sell", require_non_st=True
                    )
                )
                if runner.allow_short
                else set()
            )
            held_eligible = set(
                account.filter_tradeable(
                    date_str,
                    [s for s in ranked_names if s in held_set],
                    side="hold",
                    require_non_st=True,
                )
            )
            tradeable_stocks = [
                s
                for s in ranked_names
                if s in buyable or s in sellable_open or s in held_eligible
            ]
            stocks_removed_by_tday = set(ranked.index) - set(tradeable_stocks)
            if not runner.pretrade_fixed_targets:
                ranked = ranked.loc[[s for s in ranked.index if s in tradeable_stocks]]
        else:
            tradeable_stocks = ranked.index.tolist()
            stocks_removed_by_tday = set()

        stocks_removed_by_score = set(valid_stocks) - set(score_series.index)
        effective_top_k = runner._get_effective_top_k(len(ranked))
        if len(ranked) < effective_top_k:
            runner.logger.debug(
                f"[VectorBT] {date_str}: Only {len(ranked)} tradeable stocks after T-day filter, "
                f"effective_top_k={effective_top_k}"
            )

        strategy_weight_map: Optional[Dict[Any, float]] = None
        if runner._strategy_plugin is not None:
            strategy_weight_map = runner._strategy_plugin(ranked, effective_top_k)
            top_k_stocks = list(strategy_weight_map)
        elif runner.allow_short:
            strategy_weight_map = compute_long_short_target_weights(
                ranked, effective_top_k
            )
            top_k_stocks = list(strategy_weight_map)
        elif runner.all_equal_weight:
            top_k_stocks = ranked.index.tolist()
            effective_top_k = len(top_k_stocks)
        elif runner.overlay_active:
            top_k_stocks = runner._select_with_overlays(
                ranked, effective_top_k, held_set
            )
        else:
            top_k_stocks = ranked.head(effective_top_k).index.tolist()

        runner._record_selection_diagnostic(
            date=date_normalized,
            pool_count=len(valid_stocks),
            valid_score_count=len(score_series),
            tradeable_count=len(tradeable_stocks),
            ranked=ranked,
            selected=top_k_stocks,
            previous_holdings=held_set,
        )

        try:
            with open_diagnostic(
                "local_pools.txt",
                "a",
                directory=runner.diagnostics_output_dir,
                enabled=runner.verbose_alignment or bool(runner.diagnostics_output_dir),
            ) as f:
                f.write(
                    f"FILTER_SCORE_REMOVED: {len(stocks_removed_by_score)} ({','.join(sorted(stocks_removed_by_score)) if stocks_removed_by_score else 'none'})\n"
                )
                f.write(
                    f"FILTER_TDAY_REMOVED: {len(stocks_removed_by_tday)} ({','.join(sorted(stocks_removed_by_tday)) if stocks_removed_by_tday else 'none'})\n"
                )
                f.write(f"FINAL_POOL: {len(ranked)}\n")
                f.write(
                    f"EFFECTIVE_TOP_K: {effective_top_k} (top_k={runner.top_k}, top_pct={runner.top_pct})\n"
                )
                f.write(f"TOP_K_SELECTED: {','.join(top_k_stocks)}\n")
        except Exception:
            pass

        exposure = 1.0
        if exposure_by_date:
            exposure = float(exposure_by_date.get(date_normalized, 1.0))

        # Branch 1: allow_short and not overlay_active
        if runner.allow_short and not runner.overlay_active:
            target_map = strategy_weight_map or {}
            target_set = set(target_map)
            order_names = sorted(held_set | target_set)
            held_list = sorted(held_set)
            buyable_close = (
                set(
                    account.filter_tradeable(
                        date_str, held_list, side="buy", require_non_st=False
                    )
                )
                if held_list
                else set()
            )
            sellable_close = (
                set(
                    account.filter_tradeable(
                        date_str, held_list, side="sell", require_non_st=False
                    )
                )
                if held_list
                else set()
            )
            if runner.enable_tday_filter:
                buyable_open = set(
                    account.filter_tradeable(
                        date_str, order_names, side="buy", require_non_st=True
                    )
                )
                sellable_open = set(
                    account.filter_tradeable(
                        date_str, order_names, side="sell", require_non_st=True
                    )
                )
            else:
                buyable_open = set(order_names)
                sellable_open = set(order_names)

            base_cap = account.base_capital(exposure)
            target_values: Dict[str, float] = {}
            target_weights: Dict[str, float] = {}
            hold_instruments: Set[str] = set()
            blocked_exits: Set[str] = set()
            completed_exits: Set[str] = set()
            order_fees: Dict[str, float] = {}

            for stock in order_names:
                px = account.execution_price(stock)
                cur_val = account.held_notional(stock)
                w = float(target_map.get(stock, 0.0))
                tgt_val = w * base_cap
                if px is None:
                    hold_instruments.add(stock)
                    if stock not in target_set and stock in held_set:
                        blocked_exits.add(stock)
                    continue
                delta = tgt_val - cur_val
                if abs(delta) <= 1e-12:
                    if abs(tgt_val) > 1e-12:
                        target_values[stock] = tgt_val
                        target_weights[stock] = w * exposure
                    elif stock in held_set:
                        target_values[stock] = 0.0
                        target_weights[stock] = 0.0
                        completed_exits.add(stock)
                    continue
                if delta > 0:
                    closing_short = cur_val < 0 and tgt_val <= 1e-12
                    side_allowed = (
                        (stock in buyable_close)
                        if closing_short
                        else (stock in buyable_open)
                    )
                    order_fee = float(runner.buy_cost_rate)
                else:
                    closing_long = cur_val > 0 and tgt_val >= -1e-12
                    side_allowed = (
                        (stock in sellable_close)
                        if closing_long
                        else (stock in sellable_open)
                    )
                    order_fee = float(runner.sell_cost_rate)

                if not side_allowed:
                    hold_instruments.add(stock)
                    if stock not in target_set and stock in held_set:
                        blocked_exits.add(stock)
                    continue

                target_values[stock] = tgt_val
                target_weights[stock] = w * exposure
                order_fees[stock] = order_fee
                if abs(tgt_val) <= 1e-12 and stock in held_set:
                    completed_exits.add(stock)

            return BarPolicyDecision(
                kind="branch_long_short",
                date=date,
                date_normalized=date_normalized,
                date_str=date_key,
                is_rebalance=True,
                is_skip=False,
                target_values=target_values,
                target_weights=target_weights,
                cancelled_exits=set(target_set),
                blocked_exits=blocked_exits,
                sell_instruments=completed_exits,
                hold_instruments=hold_instruments,
                order_fees=order_fees,
            )

        # Branch 2: overlay_active
        if runner.overlay_active:
            target_set = set(top_k_stocks)
            nav = account.nav()
            prev_weight_map = account.prev_weight_map()
            weight_map = (
                strategy_weight_map
                if strategy_weight_map is not None
                else runner._compute_overlay_weights(
                    top_k_stocks,
                    score_series,
                    ctx.price_df,
                    prev_date_str,
                    prev_weight_map,
                )
            )
            if runner.overlay["partial_rebalance"]["enabled"]:
                weight_map = runner._blend_partial_rebalance(weight_map, prev_weight_map)
                target_set = set(weight_map)
            liq_as_of_str = prev_date_str
            if runner.overlay["liquidity_cap"]["enabled"]:
                liq_as_of_str = runner._slot_provider.liquidity_as_of_key(
                    prev_date_str, runner.price_db_conn, runner.price_db_path
                )
                weight_map = runner._apply_liquidity_cap(
                    weight_map,
                    liq_as_of_str,
                    account.deployed_capital_for_liquidity(exposure),
                )

            order_names = sorted(held_set | target_set)
            held_list = sorted(held_set)
            buyable = {
                stock
                for stock in account.filter_tradeable(
                    date_str, order_names, side="buy", require_non_st=True
                )
                if account.execution_price(stock) is not None
            }
            buyable_close = (
                {
                    stock
                    for stock in account.filter_tradeable(
                        date_str, held_list, side="buy", require_non_st=False
                    )
                    if account.execution_price(stock) is not None
                }
                if held_list
                else set()
            )
            sellable = {
                stock
                for stock in account.filter_tradeable(
                    date_str, order_names, side="sell", require_non_st=False
                )
                if account.execution_price(stock) is not None
            }
            sellable_open = (
                {
                    stock
                    for stock in account.filter_tradeable(
                        date_str, order_names, side="sell", require_non_st=True
                    )
                    if account.execution_price(stock) is not None
                }
                if runner.allow_short
                else sellable
            )

            strict_skipped = 0
            if runner.strict_max_actual_holdings:
                executable_exits = (held_set - target_set) & sellable
                surviving_holdings = held_set - executable_exits
                ranked_new_targets = [
                    stock
                    for stock in top_k_stocks
                    if stock in target_set and stock not in held_set
                ]
                available_slots = max(0, effective_top_k - len(surviving_holdings))
                allowed_new = set(ranked_new_targets[:available_slots])
                skipped_new = set(ranked_new_targets) - allowed_new
                strict_skipped = len(skipped_new)
                target_set.difference_update(skipped_new)
                weight_map = {
                    stock: weight
                    for stock, weight in weight_map.items()
                    if stock in target_set
                }
                order_names = sorted(held_set | target_set)

            current_values: Dict[str, float] = {}
            for stock in held_set:
                px = account.mark_price(stock)
                if px is not None:
                    current_values[stock] = account.held_notional(stock)

            base_capital = account.base_capital(exposure)
            dollar_caps: Dict[str, float] = {}
            if runner.overlay["liquidity_cap"]["enabled"]:
                liq_caps = runner._liquidity_weight_caps(
                    target_set,
                    liq_as_of_str,
                    account.deployed_capital_for_liquidity(exposure),
                )
                dollar_caps = {
                    stock: cap * base_capital for stock, cap in liq_caps.items()
                }
            desired = {
                stock: weight_map.get(stock, 0.0) * base_capital
                for stock in target_set
            }
            initial_gross = sum(abs(v) for v in desired.values())
            forced: Dict[str, float] = {}
            blocked_exits: Set[str] = set()

            for _ in range(len(order_names) + 1):
                newly_forced = False
                for stock in order_names:
                    if stock in forced:
                        continue
                    current = current_values.get(stock, 0.0)
                    target = desired.get(stock, 0.0)
                    if stock not in target_set:
                        exit_ok = (
                            (stock in buyable_close)
                            if current < 0
                            else (stock in sellable)
                        )
                        if not exit_ok:
                            forced[stock] = current
                            blocked_exits.add(stock)
                            newly_forced = True
                    elif target > current + 1e-12:
                        allowed = (
                            (stock in buyable_close)
                            if (current < 0 and target <= 1e-12)
                            else (stock in buyable)
                        )
                        if not allowed:
                            forced[stock] = current
                            newly_forced = True
                    elif target < current - 1e-12:
                        allowed = (
                            (stock in sellable)
                            if (current > 0 and target >= -1e-12)
                            else (stock in sellable_open)
                        )
                        if not allowed:
                            forced[stock] = current
                            newly_forced = True
                if runner.pretrade_fixed_targets or not forced:
                    break
                flexible_targets = [s for s in target_set if s not in forced]
                gross_budget = min(base_capital, initial_gross)
                available = max(
                    0.0, gross_budget - sum(abs(v) for v in forced.values())
                )
                flexible_weight = sum(
                    abs(weight_map.get(s, 0.0)) for s in flexible_targets
                )
                if not (flexible_targets and flexible_weight > 0):
                    break
                new_desired = {
                    stock: available * weight_map.get(stock, 0.0) / flexible_weight
                    for stock in flexible_targets
                }
                if dollar_caps:
                    new_desired = apply_magnitude_caps(
                        new_desired, dollar_caps, tol=1e-9
                    )
                changed = any(
                    abs(new_desired[stock] - desired.get(stock, 0.0)) > 1e-12
                    for stock in flexible_targets
                )
                desired.update(new_desired)
                if not newly_forced and not changed:
                    break

            target_values: Dict[str, float] = {}
            target_weights: Dict[str, float] = {}
            order_fees: Dict[str, float] = {}
            hold_instruments: Set[str] = set()
            sell_instruments: Set[str] = set()

            for stock in order_names:
                if stock in forced:
                    hold_instruments.add(stock)
                    continue
                px = account.mark_price(stock)
                if px is None:
                    if stock not in target_set:
                        blocked_exits.add(stock)
                    hold_instruments.add(stock)
                    continue
                current = current_values.get(stock, 0.0)
                target = desired.get(stock, 0.0)
                if target < current - 1e-12:
                    order_fees[stock] = float(runner.sell_cost_rate)
                elif target > current + 1e-12:
                    order_fees[stock] = float(runner.buy_cost_rate)
                target_values[stock] = target
                if abs(target) <= 1e-12:
                    target_weights[stock] = 0.0
                    sell_instruments.add(stock)
                elif base_capital > 0:
                    target_weights[stock] = (target / base_capital) * exposure
                else:
                    target_weights[stock] = 0.0

            snapshot_payload = None
            if runner._alignment_writer is not None:
                snapshot_payload = {
                    "rebalance_date": date_str,
                    "factor_date": prev_date_str,
                    "stock_pool": valid_stocks,
                    "raw_factors": raw_factors_df,
                    "processed_factors": processed_factors_df,
                    "composite_scores": score_series,
                    "target_holdings": [s for s in top_k_stocks if s in target_set],
                    "target_values": {
                        s: forced.get(s, desired.get(s, 0.0))
                        for s in sorted(target_set | set(forced))
                    },
                    "pool_size": len(valid_stocks),
                    "nav": nav / runner.init_cash,
                }

            return BarPolicyDecision(
                kind="branch_overlay",
                date=date,
                date_normalized=date_normalized,
                date_str=date_key,
                is_rebalance=True,
                is_skip=False,
                target_values=target_values,
                target_weights=target_weights,
                cancelled_exits=set(target_set),
                blocked_exits=blocked_exits,
                sell_instruments=sell_instruments,
                hold_instruments=hold_instruments,
                forced_instruments=set(forced),
                order_fees=order_fees,
                snapshot_payload=snapshot_payload,
                strict_buys_skipped=strict_skipped,
                use_direct_notional_targets=True,
            )

        # Branch 3: Standard long-only Top-K
        target_set = set(top_k_stocks)
        stocks_to_sell = sorted(held_set - target_set)
        stocks_to_keep = sorted(held_set & target_set)
        stocks_to_buy = sorted(target_set - held_set)

        sellable = set(
            account.filter_tradeable(
                date_str, stocks_to_sell, side="sell", require_non_st=False
            )
        )
        blocked_sells = set(stocks_to_sell) - sellable
        executed_sells: Set[str] = set()
        for stock in stocks_to_sell:
            if stock not in sellable:
                continue
            if account.held_notional(stock) > 0:
                if account.execution_price(stock) is None:
                    blocked_sells.add(stock)
                else:
                    executed_sells.add(stock)

        remaining_held = held_set - executed_sells
        strict_skipped = 0
        if runner.strict_max_actual_holdings and stocks_to_buy:
            ranked_new_targets = [
                stock for stock in top_k_stocks if stock in stocks_to_buy
            ]
            available_slots = max(0, effective_top_k - len(remaining_held))
            strict_skipped = max(0, len(ranked_new_targets) - available_slots)
            stocks_to_buy = ranked_new_targets[:available_slots]

        intended_buy_count = len(stocks_to_buy)
        if (
            runner.pretrade_fixed_targets
            and runner.enable_tday_filter
            and stocks_to_buy
        ):
            buyable = set(
                account.filter_tradeable(
                    date_str, stocks_to_buy, side="buy", require_non_st=True
                )
            )
            stocks_to_buy = [
                stock
                for stock in stocks_to_buy
                if stock in buyable and account.execution_price(stock) is not None
            ]
        elif ctx.exec_price_df is not ctx.price_df and stocks_to_buy:
            stocks_to_buy = [
                stock
                for stock in stocks_to_buy
                if account.execution_price(stock) is not None
            ]

        denom = (
            intended_buy_count
            if runner.pretrade_fixed_targets
            else len(stocks_to_buy)
        )
        value_per_new_stock = account.branch3_buy_target_value(
            executed_sells=executed_sells,
            remaining_held=remaining_held,
            effective_top_k=effective_top_k,
            buy_count=denom,
            exposure=exposure,
        )

        target_values = {}
        order_fees = {}
        hold_instruments = set(stocks_to_keep) | (set(stocks_to_sell) - executed_sells)
        for stock in executed_sells:
            target_values[stock] = 0.0
            order_fees[stock] = float(runner.sell_cost_rate)
        for stock in stocks_to_buy:
            target_values[stock] = value_per_new_stock
            order_fees[stock] = float(runner.buy_cost_rate)

        try:
            with open_diagnostic(
                "local_pools.txt",
                "a",
                directory=runner.diagnostics_output_dir,
                enabled=runner.verbose_alignment or bool(runner.diagnostics_output_dir),
            ) as f:
                f.write(
                    f"JQ_ALIGNED_V4_SELL: {','.join(sorted(stocks_to_sell)) if stocks_to_sell else 'none'}\n"
                )
                f.write(
                    f"JQ_ALIGNED_V4_KEEP: {','.join(sorted(stocks_to_keep)) if stocks_to_keep else 'none'} (NaN)\n"
                )
                f.write(
                    f"JQ_ALIGNED_V4_BUY: {','.join(sorted(stocks_to_buy)) if stocks_to_buy else 'none'} (每只={value_per_new_stock:,.0f}元)\n"
                )
        except Exception:
            pass

        snapshot_payload = None
        if runner._alignment_writer is not None:
            actions: Dict[str, str] = {}
            for s in stocks_to_sell:
                actions[s] = "SELL"
            for s in stocks_to_keep:
                actions[s] = "HOLD"
            for s in stocks_to_buy:
                actions[s] = "BUY"
            snapshot_payload = {
                "rebalance_date": date_str,
                "factor_date": prev_date_str,
                "stock_pool": valid_stocks,
                "raw_factors": raw_factors_df,
                "processed_factors": (
                    processed_factors_df.fillna(0.0)
                    if processed_factors_df is not None
                    else None
                ),
                "composite_scores": score_series,
                "target_holdings": top_k_stocks,
                "buy_stocks": list(stocks_to_buy),
                "keep_stocks": list(stocks_to_keep),
                "value_per_new_stock": float(value_per_new_stock),
                "actions": actions,
                "pool_size": len(valid_stocks),
                "is_topk_branch": True,
            }

        return BarPolicyDecision(
            kind="branch_topk",
            date=date,
            date_normalized=date_normalized,
            date_str=date_key,
            is_rebalance=True,
            is_skip=False,
            target_values=target_values,
            cancelled_exits=set(target_set),
            blocked_exits=blocked_sells,
            sell_instruments=executed_sells,
            hold_instruments=hold_instruments,
            order_fees=order_fees,
            snapshot_payload=snapshot_payload,
            strict_buys_skipped=strict_skipped,
            use_direct_notional_targets=True,
        )
