"""
VectorBT Backtest Runner

Core class for running portfolio backtests using VectorBT's from_orders method
with targetpercent for Top-K equal-weight rebalancing.
"""

import json
import logging
import os
import hashlib
from utils.output_paths import open_diagnostic
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ..portfolio_plugins import get_position_plugin, get_strategy_plugin

try:
    import vectorbt as vbt

    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

from .utils import (
    build_trading_calendar,
    compute_rebalance_dates,
)
from ..core.calendar_skip import (
    SkipRules,
    date_in_skip,
    describe_skip_rules,
    parse_skip_rules,
    skip_mask,
)
from .factor_loader import (
    FactorLoader,
)
from utils.strategy_kernel import model_for_date, linear_signal, rank_scores, equal_target_weights, selection_count
from .alignment_data_writer import AlignmentDataWriter
from .quality_filter import QualityFilterError, apply_pit_quality_filter

# LightGBM support
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None


class VectorBTBacktestRunner:
    """
    VectorBT 回测运行器 - Top-K 等权重组合回测

    Uses Portfolio.from_orders() with size_type='targetpercent' for
    high-performance vectorized backtesting.

    Key Features:
    - Dynamic stock pool filtering per rebalance date
    - Top-K stock selection based on factor values
    - Equal-weight allocation (1/K per stock)
    - Skip months logic (zero weights = clear positions)
    - Transaction cost modeling
    """

    def __init__(
        self,
        config: dict,
        logger: Optional[logging.Logger] = None,
        bundle_path: Optional[str] = None,
        factor_db_path: Optional[str] = None,
        factor_db_conn: Optional[Any] = None,
        price_db_path: Optional[str] = None,
        price_db_conn: Optional[Any] = None,
        save_alignment_data: bool = False,
        alignment_output_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        alignment_output_format: str = "parquet",
        return_series_output_dir: Optional[str] = None,
        return_series_trial_id: Optional[str] = None,
        return_series_metadata: Optional[Dict[str, Any]] = None,
        diagnostics_output_dir: Optional[str] = None,
        factor_load_cache: Optional[Dict[tuple, pd.DataFrame]] = None,
        synthesis_score_cache: Optional[
            Dict[
                tuple, Tuple[pd.Series, Optional[pd.DataFrame], Optional[pd.DataFrame]]
            ]
        ] = None,
        tradeability_cache: Optional[
            Dict[tuple[str, tuple[str, ...]], List[str]]
        ] = None,
    ):
        """
        Initialize backtest runner.

        Args:
            config: Configuration dict with keys:
                - top_k_strategy: {top_k, cost_rate, slippage, holding_period,
                  strict_max_actual_holdings}
                - calendar_filters: {skip_months}
            logger: Optional logger instance
            bundle_path: Path to Lasso model bundle JSON file (cloud-aligned pipeline)
            factor_db_path: Path to DuckDB factor database
            factor_db_conn: Existing DuckDB connection (preferred over path to avoid conflicts)
            price_db_path: Path to DuckDB price database (for T-1 trading day lookup)
            price_db_conn: Existing price database connection
            factor_load_cache: Optional dict shared across a multi-trial sweep to
                memoise raw factor DB reads. The per-date composite score depends
                only on (date, stock pool, bundle) and is identical across trials
                that differ solely in selection parameters (e.g. top_pct), so the
                underlying ``FactorLoader.load_factors`` reads can be reused. When
                None (default) caching is disabled and behaviour is unchanged.
            synthesis_score_cache: Optional dict shared by net/gross or top_pct
                trials to reuse exact per-date bundle scores and diagnostics.
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        # Shared (cross-trial) raw-factor read cache; passed to FactorLoader.
        self._factor_load_cache = factor_load_cache
        self._synthesis_score_cache = synthesis_score_cache
        self._tradeability_cache = tradeability_cache

        # Top-K strategy config
        top_k_cfg = config.get("top_k_strategy", {})

        # Support both top_k (fixed count) and top_pct (percentage-based)
        # Priority: top_k > top_pct > default(5)
        self.top_k = top_k_cfg.get("top_k", None)
        self.top_pct = top_k_cfg.get("top_pct", None)

        # Validate: at least one selection method should exist or use default
        if self.top_k is None and self.top_pct is None:
            self.top_k = 5  # Default fallback

        self.cost_rate = top_k_cfg.get("cost_rate", 0.003)
        self.slippage = top_k_cfg.get("slippage", 0.0)
        self.holding_period = top_k_cfg.get("holding_period", 5)
        self.nth_trading_day = top_k_cfg.get(
            "nth_trading_day", 3
        )  # Default: 3rd trading day
        self.enable_tday_filter = top_k_cfg.get(
            "enable_tday_filter", True
        )  # T-day limit filter
        self.pretrade_fixed_targets = bool(
            top_k_cfg.get("pretrade_fixed_targets", False)
        )
        self.score_source = str(top_k_cfg.get("score_source", "bundle"))
        self.all_equal_weight = bool(top_k_cfg.get("all_equal_weight", False))
        strategy_cfg = top_k_cfg.get("strategy_plugin") or {}
        if not isinstance(strategy_cfg, dict):
            raise ValueError("top_k_strategy.strategy_plugin must be an object")
        self.strategy_plugin_config = dict(strategy_cfg)
        self.strategy_plugin_id = str(strategy_cfg.get("plugin_id", "") or "")
        self._strategy_plugin = (
            get_strategy_plugin(self.strategy_plugin_id)
            if self.strategy_plugin_id
            else None
        )
        # When enabled, blocked exits continue to occupy selection slots.  New
        # entries may use only the slots left after executable sells complete,
        # so active positions can never be increased above effective_top_k.
        # Default False preserves the legacy target-top-k execution semantics.
        self.strict_max_actual_holdings = bool(
            top_k_cfg.get("strict_max_actual_holdings", False)
        )
        self.score_direction = float(top_k_cfg.get("score_direction", 1.0))
        score_decay = top_k_cfg.get("score_decay") or {}
        if not isinstance(score_decay, dict):
            raise ValueError("top_k_strategy.score_decay must be an object")
        enabled = score_decay.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("score_decay.enabled must be a boolean")
        self.score_decay_enabled = enabled
        self.score_decay_half_life_days = 7.0
        if self.score_decay_enabled:
            try:
                self.score_decay_half_life_days = float(
                    score_decay.get("half_life_days", 7.0)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "score_decay.half_life_days must be finite and positive"
                ) from exc
            if (
                not np.isfinite(self.score_decay_half_life_days)
                or self.score_decay_half_life_days <= 0
            ):
                raise ValueError(
                    "score_decay.half_life_days must be finite and positive"
                )
        self._score_decay_state = pd.Series(dtype=float)
        self._score_decay_date: Optional[pd.Timestamp] = None
        self.init_cash = top_k_cfg.get(
            "init_cash", 100_000
        )  # [V4] Initial capital for simulation
        self.return_series_output_dir = (
            Path(return_series_output_dir).expanduser().resolve()
            if return_series_output_dir
            else None
        )
        self.return_series_trial_id = return_series_trial_id
        self.return_series_metadata = dict(return_series_metadata or {})
        self.diagnostics_output_dir = (
            Path(diagnostics_output_dir).expanduser().resolve()
            if diagnostics_output_dir
            else None
        )
        self.portfolio_audit_enabled = bool(
            self.diagnostics_output_dir and self.return_series_trial_id
        )
        # Selection data quality is intentionally evaluated on every rebalance
        # before an order is simulated.  Defaults are observational so legacy
        # configs retain their behaviour; wind_micro enables the fail-closed
        # thresholds explicitly in its experiment template.
        selection_quality_cfg = top_k_cfg.get("selection_quality", {})
        self.selection_quality = (
            dict(selection_quality_cfg)
            if isinstance(selection_quality_cfg, dict)
            else {}
        )
        self.quality_filter = (
            dict(top_k_cfg.get("quality_filter", {}))
            if isinstance(top_k_cfg.get("quality_filter", {}), dict)
            else {}
        )
        self._quality_filter_diagnostics: Dict[str, Any] = {}
        self._selection_diagnostics: List[Dict[str, Any]] = []
        self._execution_diagnostics: Dict[str, Any] = {}
        self._mvo_diagnostics: List[Dict[str, Any]] = []

        # Calendar filters: whole months (1-12) and/or "MM-DD:MM-DD" windows
        cal_cfg = config.get("calendar_filters", {})
        self.skip_months: SkipRules = parse_skip_rules(cal_cfg.get("skip_months", []))

        # [PORTFOLIO OVERLAY] Turnover control / risk-aware weighting / endogenous tail-risk overlay.
        # All overlay features are OFF by default so the validated baseline path is untouched.
        self.overlay: Dict[str, Any] = self._parse_overlay_config(top_k_cfg)
        self.overlay["strategy_plugin_id"] = self.strategy_plugin_id or None
        external_regime = self.overlay["risk_overlay"]["external_regime"]
        if external_regime["enabled"]:
            get_position_plugin(external_regime["position_plugin_id"])
            if (
                not external_regime["signal_path"]
                or not external_regime["signal_sha256"]
            ):
                raise ValueError(
                    "external_regime requires signal_path and signal_sha256"
                )
            if not 0 <= external_regime["missing_exposure"] <= 1:
                raise ValueError("external_regime.missing_exposure must be in [0, 1]")
        self.overlay_active: bool = self._overlay_active(self.overlay)
        if self.overlay_active:
            self.logger.info(
                f"[Overlay] Portfolio overlay active: {self._overlay_summary(self.overlay)}"
            )

        # Bundle-based synthesis (cloud-aligned pipeline)
        self.bundle_path = bundle_path
        self._bundle_data: Optional[dict] = None
        self._bundle_models: Optional[List[dict]] = None
        self._parent_baseline_cache: Dict[Tuple[str, str], pd.Series] = {}

        # Factor database for loading raw factors
        self.factor_db_path = factor_db_path
        self.factor_db_conn = factor_db_conn
        self._factor_loader: Optional[FactorLoader] = None

        # Price database config (for T-1 trading day lookup)
        self.price_db_path = price_db_path
        self.price_db_conn = price_db_conn
        self._price_conn_owned = False  # Track if we created the connection

        if bundle_path and os.path.exists(bundle_path):
            self.logger.info(
                f"[VectorBT] Bundle-based synthesis enabled: {bundle_path}"
            )

        # Alignment data collection
        self.save_alignment_data = save_alignment_data
        self.alignment_output_dir = (
            alignment_output_dir or "outputs/synthesis/alignment"
        )
        self.experiment_name = experiment_name or "unnamed_experiment"
        self.alignment_output_format = alignment_output_format
        self._alignment_writer: Optional[AlignmentDataWriter] = None

        # Verbose alignment logging (default: OFF to reduce log noise)
        # Enable via config["logging"]["verbose_alignment"] = true
        logging_cfg = config.get("logging", {})
        self.verbose_alignment = logging_cfg.get("verbose_alignment", False)
        if self.diagnostics_output_dir is None and self.verbose_alignment:
            from utils.output_paths import execution_directory
            self.diagnostics_output_dir = execution_directory() / "diagnostics"


        if self.save_alignment_data:
            self._alignment_writer = AlignmentDataWriter(
                output_dir=self.alignment_output_dir,
                experiment_name=self.experiment_name,
                output_format=self.alignment_output_format,
                logger=self.logger,
            )
            self.logger.info(
                f"[VectorBT] Alignment data collection enabled -> {self.alignment_output_dir}/{self.experiment_name}"
            )

        if not VECTORBT_AVAILABLE:
            self.logger.warning("VectorBT not installed. Backtest will not run.")

    def run_backtest(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        open_df: Optional[pd.DataFrame] = None,
        factor_ratio_df: Optional[pd.DataFrame] = None,
        dynamic_pools: Optional[Dict[str, List[str]]] = None,
        rebalance_dates: Optional[List[datetime]] = None,
        rebalance_freq: str = "weekly",
    ) -> Dict[str, Any]:
        """
        Run vectorized portfolio backtest.

        JoinQuant Alignment:
        - Orders execute at OPEN price (next day after signal)
        - Portfolio valuation uses CLOSE price

        Args:
            factor_df: Wide DataFrame (Index=date, Columns=stocks) with factor values
            price_df: Wide DataFrame (Index=date, Columns=stocks) with close prices
            open_df: Wide DataFrame (Index=date, Columns=stocks) with open prices
                     If None, will use close prices for execution (less accurate)
            factor_ratio_df: Retained for compatibility and evidence hashing. Prices
                     are already PIT adjusted, so ratios must not alter shares again.
            dynamic_pools: Optional dict mapping date strings to stock lists
            rebalance_dates: Optional pre-computed rebalance dates
            rebalance_freq: Rebalancing frequency if dates not provided

        Returns:
            Dict with backtest metrics:
                - annual_return: Annualized return
                - sharpe_ratio: Sharpe ratio
                - max_drawdown: Maximum drawdown
                - total_return: Total return
                - trade_count: Number of trades
                - portfolio: VectorBT Portfolio object (for advanced analysis)
        """
        if not VECTORBT_AVAILABLE:
            self.logger.error("VectorBT not available. Cannot run backtest.")
            return {"error": "VectorBT not installed"}

        # Validate inputs
        if factor_df.empty or price_df.empty:
            self.logger.warning("Empty factor or price data")
            return {"error": "Empty input data"}

        # [JQ-ALIGNED V5] Align data using PRICE dates as master calendar
        # This ensures skip month dates are included for liquidation
        # Factor data may be missing for skip months (which is fine - we liquidate anyway)

        common_stocks = factor_df.columns.intersection(price_df.columns)
        if len(common_stocks) == 0:
            self.logger.warning("No common stocks between factor and price data")
            return {"error": "No stock overlap"}

        # Missing whole-market rows cannot become shorter holding periods.
        market_calendar = self._trading_calendar()
        expected = market_calendar.align(price_df).index
        if not pd.DatetimeIndex(price_df.index).as_unit("ns").equals(expected.as_unit("ns")):
            raise ValueError('Backtest prices must cover every market session in order; import missing dates')
        # DuckDB dates may be microseconds; VectorBT requires identical index metadata.
        price_df = price_df.set_axis(expected, axis=0)
        all_price_dates = expected
        factor_dates = factor_df.index

        # Log date coverage
        skip_month_dates_in_price = list(
            all_price_dates[skip_mask(all_price_dates, self.skip_months)]
        )
        skip_month_dates_in_factor = list(
            factor_dates[skip_mask(factor_dates, self.skip_months)]
        )

        # self.logger.info(f"[JQ-ALIGNED V5] 日期对齐:")
        # self.logger.info(f"  - 价格数据日期: {len(all_price_dates)} 天")
        # self.logger.info(f"  - 因子数据日期: {len(factor_dates)} 天")
        # self.logger.info(f"  - 价格数据中 skip_month 日期: {len(skip_month_dates_in_price)} 天")
        # self.logger.info(f"  - 因子数据中 skip_month 日期: {len(skip_month_dates_in_factor)} 天")

        # Align price data to common stocks only
        price_aligned = price_df[list(common_stocks)]

        # Reindex factor data to price dates (missing dates become NaN)
        factor_aligned = factor_df.reindex(
            index=all_price_dates, columns=list(common_stocks)
        )

        # Align open prices if provided
        open_aligned = None
        if open_df is not None:
            open_common_stocks = common_stocks.intersection(open_df.columns)
            if len(open_common_stocks) > 0:
                # [FIX] 使用排序的列表确保列顺序稳定
                common_stocks = sorted(
                    list(set(common_stocks) & set(open_common_stocks))
                )
                price_aligned = price_df[common_stocks]
                factor_aligned = factor_df.reindex(
                    index=all_price_dates, columns=common_stocks
                )
                open_aligned = open_df.reindex(
                    index=all_price_dates, columns=common_stocks
                )
                self.logger.info(f"[VectorBT] 使用开盘价执行订单 (JoinQuant对齐模式)")
            else:
                self.logger.warning(
                    "[VectorBT] Open price data alignment failed, falling back to close prices"
                )

        # Use price dates as common_dates for subsequent logic
        common_dates = all_price_dates

        self.logger.info(
            f"[VectorBT] Aligned data: {len(common_dates)} dates, {len(common_stocks)} stocks"
        )

        # [HIGH PRIORITY 1] Use dynamic_pools keys directly as rebalance dates (align with JQ Local)
        if self.quality_filter.get("enabled", False):
            quality_db_path = self.price_db_path or self.factor_db_path
            if not quality_db_path:
                raise QualityFilterError("quality filter requires price_db_path")
            dynamic_pools, self._quality_filter_diagnostics = apply_pit_quality_filter(
                dynamic_pools or {}, db_path=quality_db_path, config=self.quality_filter
            )
            self.logger.info(
                "[QUALITY_FILTER] PIT dividend quality applied: %s",
                self._quality_filter_diagnostics,
            )
        if rebalance_dates is None:
            rebalance_dates = self._load_strategy_rebalance_dates()
        if rebalance_dates is None:
            if dynamic_pools:
                # Get all pool dates sorted
                all_pool_dates = sorted(
                    [pd.to_datetime(d).to_pydatetime() for d in dynamic_pools.keys()]
                )

                # Apply holding_period: select every Nth date from pool dates
                if self.holding_period > 1:
                    rebalance_dates = [
                        d
                        for i, d in enumerate(all_pool_dates)
                        if i % self.holding_period == 0
                    ]
                    self.logger.info(
                        f"[VectorBT] Applied holding_period={self.holding_period}: "
                        f"{len(all_pool_dates)} pool dates -> {len(rebalance_dates)} rebalance dates"
                    )
                else:
                    rebalance_dates = all_pool_dates
            else:
                # Fallback: compute rebalance dates from trading calendar
                trading_calendar = build_trading_calendar(price_aligned, calendar=self._trading_calendar())
                rebalance_dates = compute_rebalance_dates(
                    trading_calendar,
                    start_date=common_dates.min(),
                    end_date=common_dates.max(),
                    rebalance_freq=rebalance_freq,
                    nth_trading_day=self.nth_trading_day,
                )

        # self.logger.info(f"[VectorBT] Rebalance dates: {len(rebalance_dates)} (nth_day={self.nth_trading_day})")

        # [ALIGNMENT] Initialize pool log file
        try:
            from datetime import datetime as dt

            with open_diagnostic("local_pools.txt", "w", directory=self.diagnostics_output_dir, enabled=self.verbose_alignment or bool(self.diagnostics_output_dir)) as f:
                f.write(
                    f"# Local Stock Pools - Generated at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"# Rebalance dates: {len(rebalance_dates)}\n")
                f.write(f"# Skip months: {describe_skip_rules(self.skip_months)}\n")
                f.write(f"# Top-K: {self.top_k}\n")
                f.write("#\n")
            # Initialize factor values log file
            with open_diagnostic("local_factor_values.txt", "w", directory=self.diagnostics_output_dir, enabled=self.verbose_alignment or bool(self.diagnostics_output_dir)) as f:
                f.write(
                    f"# Local Factor Values - Generated at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"# Top-K: {self.top_k}\n")
                f.write(f"# Format: FACTOR_ALIGNMENT|DATE|TOP10_FACTOR_VALUES\n")
                f.write("#\n")
        except Exception:
            pass

        # Build target values matrix (size_type='targetvalue')
        # Pass open_aligned for pre-simulation execution price
        target_weights = self._compute_target_weights(
            factor_aligned,
            price_aligned,
            dynamic_pools or {},
            rebalance_dates,
            open_df=open_aligned,  # [V4] For pre-simulation
            factor_ratio_df=factor_ratio_df,  # For corporate action share adjustment
        )

        selection_summary = self._selection_diagnostics_summary()
        selection_artifact = self._write_selection_diagnostics_artifact(
            selection_summary
        )
        selection_failure = self._selection_quality_failure(selection_summary)
        if selection_failure:
            return {
                "error": "selection_quality_gate_failed",
                "selection_quality_failure": selection_failure,
                "selection_diagnostics": selection_summary,
                "selection_diagnostics_artifact": selection_artifact,
            }

        if target_weights.isna().all().all():
            self.logger.warning("All target weights are NaN")
            return {"error": "All weights NaN"}

        # [DEBUG] 检查权重矩阵
        non_nan_count = target_weights.notna().sum().sum()
        total_cells = target_weights.shape[0] * target_weights.shape[1]
        nan_ratio = 1 - (non_nan_count / total_cells)
        # self.logger.info(f"[VectorBT DEBUG] 权重矩阵: {target_weights.shape}, 非NaN: {non_nan_count}, NaN比例: {nan_ratio:.2%}")

        # 检查前几个调仓日的权重
        rebalance_weights = target_weights.dropna(how="all")
        if len(rebalance_weights) > 0:
            sample_date = rebalance_weights.index[0]
            sample_weights = target_weights.loc[sample_date].dropna()
            # self.logger.info(f"[VectorBT DEBUG] 第一个调仓日 {sample_date}: {len(sample_weights)} 只股票有权重")
            # self.logger.info(f"[VectorBT DEBUG] 权重值: {sample_weights.head(10).to_dict()}")

        # Run VectorBT portfolio simulation
        # JoinQuant alignment: use open prices for order execution, close prices for valuation
        use_open = open_aligned is not None
        self.logger.info(
            f"[VectorBT] Running simulation with top_k={self.top_k}, cost={self.cost_rate}, use_open={use_open}"
        )

        try:
            portfolio = self._simulate_portfolio(
                target_weights, price_aligned, open_aligned, use_open
            )

            # [OVERLAY] Endogenous tail-risk overlay is a two-pass procedure: pass 1
            # produces the unscaled equity curve, then exposure is derived from
            # information up to the prior day and applied in pass 2 (no look-ahead).
            exposure_by_date: Optional[Dict[Any, float]] = None
            if self.overlay_active and self._risk_overlay_active(self.overlay):
                try:
                    pass1_returns = portfolio.returns()
                    if isinstance(pass1_returns, pd.DataFrame):
                        pass1_returns = (
                            pass1_returns.mean(axis=1)
                            if pass1_returns.shape[1] > 1
                            else pass1_returns.iloc[:, 0]
                        )
                    exposure_by_date = self._compute_exposure_by_date(
                        pass1_returns,
                        rebalance_dates,
                        price_aligned,
                        list(price_aligned.columns),
                    )
                    target_weights = self._compute_target_weights(
                        factor_aligned,
                        price_aligned,
                        dynamic_pools or {},
                        rebalance_dates,
                        open_df=open_aligned,
                        factor_ratio_df=factor_ratio_df,
                        exposure_by_date=exposure_by_date,
                    )
                    portfolio = self._simulate_portfolio(
                        target_weights, price_aligned, open_aligned, use_open
                    )
                except Exception as e:
                    self.logger.warning(
                        f"[Overlay] Risk-overlay pass-2 failed, using pass-1 result: {e}"
                    )
                    exposure_by_date = None
            self._last_exposure_by_date = exposure_by_date
            self._last_target_weights = target_weights

            # Both passes have finished reading. Factor inputs and published
            # results share output_db; release owned readers before any writes.
            self.close()

            # [DEBUG] 订单分析
            try:
                orders = portfolio.orders.records_readable
                buy_orders = len(orders[orders["Side"] == "Buy"])
                sell_orders = len(orders[orders["Side"] == "Sell"])
                self.logger.info(
                    f"[VectorBT DEBUG] 订单: 买入={buy_orders}, 卖出={sell_orders}, 总计={len(orders)}"
                )
            except Exception as e:
                self.logger.debug(f"[VectorBT] Failed to analyze orders: {e}")

            # [ALIGNMENT DATA] Save collected alignment data
            if (
                self._alignment_writer is not None
                and len(self._alignment_writer.snapshots) > 0
            ):
                try:
                    # Set metadata before saving - get factors list from bundle
                    bundle = self._load_bundle()
                    models = bundle.get("index", {}).get("models", [])
                    factors_list = models[0].get("feature_names", []) if models else []

                    self._alignment_writer.set_metadata(
                        config_path="",  # Will be set by caller
                        analysis_period={
                            "start_date": str(factor_df.index.min().date())
                            if not factor_df.empty
                            else "",
                            "end_date": str(factor_df.index.max().date())
                            if not factor_df.empty
                            else "",
                        },
                        factors_list=factors_list,
                    )
                    alignment_files = self._alignment_writer.save()
                    self.logger.info(
                        f"[VectorBT] Alignment data saved: {len(self._alignment_writer.snapshots)} snapshots"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"[VectorBT] Failed to save alignment data: {e}"
                    )

            results = self._extract_results(
                portfolio,
                exposure_by_date=getattr(self, "_last_exposure_by_date", None),
            )
            results["portfolio_audit_artifacts"] = (
                self._write_portfolio_audit_artifacts(
                    portfolio=portfolio,
                    target_values=target_weights,
                )
            )
            if self._mvo_diagnostics:
                results["mvo_diagnostics"] = list(self._mvo_diagnostics)
            results["selection_diagnostics"] = selection_summary
            results["selection_diagnostics_artifact"] = selection_artifact
            if self._quality_filter_diagnostics:
                results["quality_filter"] = self._quality_filter_diagnostics
            results["execution_diagnostics"] = dict(self._execution_diagnostics)
            return results

        except Exception as e:
            self.logger.error(f"[VectorBT] Simulation failed: {e}")
            import traceback

            traceback.print_exc()
            return {"error": str(e)}

    def _record_selection_diagnostic(
        self,
        *,
        date: pd.Timestamp,
        pool_count: int,
        valid_score_count: int,
        tradeable_count: int,
        ranked: pd.Series,
        selected: List[str],
        previous_holdings: Set[str],
    ) -> None:
        """Persist the selection-state inputs needed to detect degenerate ranks.

        The metrics use the post-tradability ranked universe because that is the
        population from which the strategy can actually buy.  ``max_tie_group``
        is reported both as a count and a ratio, avoiding the misleading
        impression that a high raw unique-count alone implies usable ordering.
        """
        if ranked.empty:
            return
        value_counts = ranked.value_counts(dropna=False)
        max_tie_count = int(value_counts.max()) if not value_counts.empty else 0
        unique_count = int(ranked.nunique(dropna=True))
        selected_set = set(selected)
        union = selected_set | previous_holdings
        intersection = selected_set & previous_holdings
        self._selection_diagnostics.append(
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "pool_count": int(pool_count),
                "valid_score_count": int(valid_score_count),
                "tradeable_count": int(tradeable_count),
                "unique_score_count": unique_count,
                "unique_score_ratio": unique_count / len(ranked),
                "max_tie_group": max_tie_count,
                "max_tie_group_ratio": max_tie_count / len(ranked),
                "selected_count": len(selected),
                # Keep the actual book, not only aggregate overlap counts: the
                # governed wind_micro audit compares candidate entrants with
                # baseline exits using the exact executable selections.
                "selected_stocks": sorted(selected_set),
                "previous_holding_count": len(previous_holdings),
                "previous_holdings": sorted(previous_holdings),
                "selected_overlap_count": len(intersection),
                "selected_overlap_ratio": (
                    len(intersection) / len(previous_holdings)
                    if previous_holdings
                    else None
                ),
                "selected_jaccard": len(intersection) / len(union) if union else None,
                "new_buy_count": len(selected_set - previous_holdings),
                "sell_count": len(previous_holdings - selected_set),
            }
        )

    def _selection_diagnostics_summary(self) -> Dict[str, Any]:
        """Aggregate selection diagnostics into a JSON-safe bundle payload."""
        rows = self._selection_diagnostics
        if not rows:
            return {"rebalance_count": 0, "rows": []}

        frame = pd.DataFrame(rows)
        overlap = frame["selected_overlap_ratio"].dropna()
        return {
            "rebalance_count": int(len(frame)),
            "avg_pool_count": float(frame["pool_count"].mean()),
            "avg_valid_score_count": float(frame["valid_score_count"].mean()),
            "avg_tradeable_count": float(frame["tradeable_count"].mean()),
            "avg_unique_score_ratio": float(frame["unique_score_ratio"].mean()),
            "min_unique_score_ratio": float(frame["unique_score_ratio"].min()),
            "avg_max_tie_group_ratio": float(frame["max_tie_group_ratio"].mean()),
            "max_max_tie_group_ratio": float(frame["max_tie_group_ratio"].max()),
            "avg_adjacent_overlap": float(overlap.mean())
            if not overlap.empty
            else None,
            "min_adjacent_overlap": float(overlap.min()) if not overlap.empty else None,
            "rows": rows,
        }

    def _selection_quality_failure(self, summary: Dict[str, Any]) -> Optional[str]:
        """Return the first configured data-quality breach, otherwise ``None``."""
        if not self.selection_quality.get("enabled", False):
            return None
        if not summary.get("rebalance_count"):
            return "no_rebalance_selection_diagnostics"

        checks = (
            ("min_avg_unique_score_ratio", "avg_unique_score_ratio", "<"),
            ("min_unique_score_ratio", "min_unique_score_ratio", "<"),
            ("max_avg_tie_group_ratio", "avg_max_tie_group_ratio", ">"),
            ("max_tie_group_ratio", "max_max_tie_group_ratio", ">"),
        )
        for config_key, metric_key, direction in checks:
            threshold = self.selection_quality.get(config_key)
            metric = summary.get(metric_key)
            if threshold is None or metric is None:
                continue
            if direction == "<" and float(metric) < float(threshold):
                return (
                    f"{metric_key}={metric:.6f} < {config_key}={float(threshold):.6f}"
                )
            if direction == ">" and float(metric) > float(threshold):
                return (
                    f"{metric_key}={metric:.6f} > {config_key}={float(threshold):.6f}"
                )
        return None

    def _write_selection_diagnostics_artifact(
        self, summary: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        if not self.diagnostics_output_dir:
            return None
        try:
            self.diagnostics_output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.diagnostics_output_dir / "selection_diagnostics.csv"
            json_path = self.diagnostics_output_dir / "selection_diagnostics.json"
            pd.DataFrame(summary.get("rows", [])).to_csv(csv_path, index=False)
            json_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return {"csv_path": str(csv_path), "summary_path": str(json_path)}
        except Exception as exc:
            self.logger.warning(
                "[VectorBT] selection diagnostics export failed: %s", exc
            )
            return None

    def _get_effective_top_k(self, n_valid_stocks: int) -> int:
        """
        Calculate effective top_k based on config.

        Supports both fixed count (top_k) and percentage-based (top_pct) selection.

        Priority:
        1. top_k (fixed count) if set
        2. top_pct (percentage of pool) if set
        3. Default to 5

        Args:
            n_valid_stocks: Number of valid stocks in the current pool

        Returns:
            Number of stocks to select (at least 1, at most n_valid_stocks)
        """
        return selection_count(n_valid_stocks, self.top_k, self.top_pct)

    # =====================================================================
    # [PORTFOLIO OVERLAY] Turnover control / weighting / endogenous tail risk
    # =====================================================================
    @staticmethod
    def _parse_overlay_config(top_k_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the optional portfolio-overlay block from top_k_strategy config.

        Returns a fully-defaulted overlay spec so downstream code never has to
        re-check for missing keys. Every feature defaults to disabled.
        """

        def _section(name: str) -> Dict[str, Any]:
            raw = top_k_cfg.get(name)
            return dict(raw) if isinstance(raw, dict) else {}

        banding = _section("banding")
        tranches = _section("tranches")
        partial_rebalance = _section("partial_rebalance")
        weighting = _section("weighting")
        liquidity_cap = _section("liquidity_cap")
        risk_overlay = _section("risk_overlay")
        crisis_guard = risk_overlay.get("crisis_guard")
        crisis_guard = dict(crisis_guard) if isinstance(crisis_guard, dict) else {}
        vol_targeting = risk_overlay.get("vol_targeting")
        vol_targeting = dict(vol_targeting) if isinstance(vol_targeting, dict) else {}
        drawdown_control = risk_overlay.get("drawdown_control")
        drawdown_control = (
            dict(drawdown_control) if isinstance(drawdown_control, dict) else {}
        )
        breadth_filter = risk_overlay.get("breadth_filter")
        breadth_filter = (
            dict(breadth_filter) if isinstance(breadth_filter, dict) else {}
        )
        regime_filter = risk_overlay.get("regime_filter")
        regime_filter = dict(regime_filter) if isinstance(regime_filter, dict) else {}
        external_regime = risk_overlay.get("external_regime")
        external_regime = (
            dict(external_regime) if isinstance(external_regime, dict) else {}
        )

        return {
            "banding": {
                "enabled": bool(banding.get("enabled", False)),
                "buy_rank_pct": float(banding.get("buy_rank_pct", 0.04)),
                "sell_rank_pct": float(banding.get("sell_rank_pct", 0.10)),
            },
            "tranches": {
                "enabled": bool(tranches.get("enabled", False)),
                "n_tranches": max(1, int(tranches.get("n_tranches", 1))),
            },
            "partial_rebalance": {
                # Garleanu-Pedersen style partial adjustment: each rebalance
                # trades only `trade_rate` of the gap between current holdings
                # and the fresh target book.
                "enabled": bool(partial_rebalance.get("enabled", False)),
                "trade_rate": min(
                    1.0, max(0.01, float(partial_rebalance.get("trade_rate", 0.3)))
                ),
                # Positions decayed below this weight are liquidated outright
                # so exiting names don't linger as dust forever.
                "min_weight": max(
                    0.0, float(partial_rebalance.get("min_weight", 0.02))
                ),
            },
            "weighting": {
                "scheme": str(weighting.get("scheme", "equal") or "equal"),
                "max_weight": weighting.get("max_weight"),
                "vol_lookback_days": max(
                    2, int(weighting.get("vol_lookback_days", 20))
                ),
                "mvo": VectorBTBacktestRunner._parse_mvo_config(weighting.get("mvo")),
            },
            "liquidity_cap": {
                "enabled": bool(liquidity_cap.get("enabled", False)),
                "adv_lookback_days": max(
                    1, int(liquidity_cap.get("adv_lookback_days", 20))
                ),
                "max_participation": float(liquidity_cap.get("max_participation", 0.1)),
                "capital_base": liquidity_cap.get("capital_base"),
            },
            "risk_overlay": {
                "vol_targeting": {
                    "enabled": bool(vol_targeting.get("enabled", False)),
                    "target_annual_vol": float(
                        vol_targeting.get("target_annual_vol", 0.20)
                    ),
                    "lookback_days": max(
                        5, int(vol_targeting.get("lookback_days", 20))
                    ),
                    "min_exposure": float(vol_targeting.get("min_exposure", 0.3)),
                    "max_exposure": float(vol_targeting.get("max_exposure", 1.0)),
                },
                "drawdown_control": {
                    "enabled": bool(drawdown_control.get("enabled", False)),
                    "derisk_dd": float(drawdown_control.get("derisk_dd", -0.12)),
                    "derisk_exposure": float(
                        drawdown_control.get("derisk_exposure", 0.5)
                    ),
                    "exit_dd": float(drawdown_control.get("exit_dd", -0.20)),
                    "reentry_dd": float(drawdown_control.get("reentry_dd", -0.08)),
                },
                "breadth_filter": {
                    "enabled": bool(breadth_filter.get("enabled", False)),
                    "loss_threshold": float(
                        breadth_filter.get("loss_threshold", -0.05)
                    ),
                    "breadth_threshold": float(
                        breadth_filter.get("breadth_threshold", 0.5)
                    ),
                    "derisk_exposure": float(
                        breadth_filter.get("derisk_exposure", 0.5)
                    ),
                },
                "regime_filter": {
                    "enabled": bool(regime_filter.get("enabled", False)),
                    "micro_index": str(regime_filter.get("micro_index", "399101.XSHE")),
                    "benchmark_index": str(
                        regime_filter.get("benchmark_index", "000300.XSHG")
                    ),
                    "spread_window_days": max(
                        2, int(regime_filter.get("spread_window_days", 20))
                    ),
                    "zscore_window_days": max(
                        20, int(regime_filter.get("zscore_window_days", 250))
                    ),
                    "derisk_z": float(regime_filter.get("derisk_z", -1.5)),
                    "derisk_exposure": float(regime_filter.get("derisk_exposure", 0.5)),
                    "exit_z": float(regime_filter.get("exit_z", -2.5)),
                },
                "external_regime": {
                    "enabled": bool(external_regime.get("enabled", False)),
                    "position_plugin_id": str(
                        external_regime.get("position_plugin_id", "") or ""
                    ),
                    "signal_path": str(external_regime.get("signal_path", "") or ""),
                    "signal_sha256": str(
                        external_regime.get("signal_sha256", "") or ""
                    ),
                    "date_column": str(
                        external_regime.get("date_column", "effective_date")
                    ),
                    "observed_through_column": str(
                        external_regime.get(
                            "observed_through_column", "observed_through"
                        )
                    ),
                    "score_column": str(
                        external_regime.get("score_column", "regime_score")
                    ),
                    "min_exposure": float(external_regime.get("min_exposure", 0.5)),
                    "max_exposure": float(external_regime.get("max_exposure", 1.0)),
                    "slope": float(external_regime.get("slope", 0.5)),
                    "missing_exposure": float(
                        external_regime.get("missing_exposure", 1.0)
                    ),
                    "require_valid_until": bool(
                        external_regime.get("require_valid_until", False)
                    ),
                },
                "crisis_guard": VectorBTBacktestRunner._parse_crisis_guard_config(
                    crisis_guard
                ),
            },
        }

    @staticmethod
    def _parse_crisis_guard_config(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the composite crisis-warning block under ``risk_overlay.crisis_guard``.

        Data-driven replacement for calendar skip_months: several independent
        crisis signals vote each day, and the book is de-risked only when at
        least ``min_triggers`` of them fire simultaneously (single-signal noise
        is ignored). All signals use prior-day data only (no look-ahead).
        """
        signals = raw.get("signals")
        signals = dict(signals) if isinstance(signals, dict) else {}

        def _sig(name: str) -> Dict[str, Any]:
            value = signals.get(name)
            return dict(value) if isinstance(value, dict) else {}

        limit_down = _sig("limit_down")
        turnover_collapse = _sig("turnover_collapse")
        breadth = _sig("breadth")
        spread = _sig("spread")
        return {
            "enabled": bool(raw.get("enabled", False)),
            # >=min_triggers simultaneous signals -> exposure *= derisk_exposure
            "min_triggers": max(1, int(raw.get("min_triggers", 2))),
            "derisk_exposure": float(raw.get("derisk_exposure", 0.3)),
            # >=hard_exit_triggers simultaneous signals -> full exit (exposure 0)
            "hard_exit_triggers": max(1, int(raw.get("hard_exit_triggers", 3))),
            # Stay de-risked for this many trading days after the last trigger
            # to avoid whipsawing back into a crashing market.
            "cooldown_days": max(0, int(raw.get("cooldown_days", 3))),
            "signals": {
                "limit_down": {
                    "enabled": bool(limit_down.get("enabled", True)),
                    # Fraction of the candidate pool closing at the limit-down
                    # price the prior day.
                    "fraction_threshold": float(
                        limit_down.get("fraction_threshold", 0.05)
                    ),
                },
                "turnover_collapse": {
                    "enabled": bool(turnover_collapse.get("enabled", True)),
                    "short_window": max(
                        1, int(turnover_collapse.get("short_window", 5))
                    ),
                    "long_window": max(
                        5, int(turnover_collapse.get("long_window", 60))
                    ),
                    # Pool aggregate turnover (money) short MA / long MA below
                    # this ratio = liquidity is evaporating.
                    "ratio_threshold": float(
                        turnover_collapse.get("ratio_threshold", 0.6)
                    ),
                },
                "breadth": {
                    "enabled": bool(breadth.get("enabled", True)),
                    "loss_threshold": float(breadth.get("loss_threshold", -0.05)),
                    "breadth_threshold": float(breadth.get("breadth_threshold", 0.4)),
                },
                "spread": {
                    "enabled": bool(spread.get("enabled", True)),
                    "micro_index": str(spread.get("micro_index", "399101.XSHE")),
                    "benchmark_index": str(
                        spread.get("benchmark_index", "000300.XSHG")
                    ),
                    "spread_window_days": max(
                        2, int(spread.get("spread_window_days", 20))
                    ),
                    "zscore_window_days": max(
                        20, int(spread.get("zscore_window_days", 250))
                    ),
                    "z_threshold": float(spread.get("z_threshold", -1.5)),
                },
            },
        }

    @staticmethod
    def _parse_mvo_config(raw: Any) -> Dict[str, Any]:
        """Parse the optional mean-variance-optimizer block under ``weighting.mvo``."""
        mvo = dict(raw) if isinstance(raw, dict) else {}

        def _optional_limit(key: str, default: Optional[float]) -> Optional[float]:
            value = mvo.get(key, default)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        return {
            "risk_aversion": float(mvo.get("risk_aversion", 8.0)),
            "turnover_penalty": float(mvo.get("turnover_penalty", 0.05)),
            "cov_lookback_days": max(20, int(mvo.get("cov_lookback_days", 120))),
            "cov_min_obs": max(10, int(mvo.get("cov_min_obs", 40))),
            # Constrain-don't-neutralize: portfolio style/industry exposure may
            # deviate from the equal-weight basket by at most these limits.
            # None disables the corresponding constraint family.
            "style_dev_limit": _optional_limit("style_dev_limit", 0.5),
            "industry_dev_limit": _optional_limit("industry_dev_limit", 0.02),
            "maxiter": max(50, int(mvo.get("maxiter", 200))),
        }

    @staticmethod
    def _overlay_active(overlay: Dict[str, Any]) -> bool:
        """Return True if any overlay feature deviates from the baseline."""
        if overlay["banding"]["enabled"]:
            return True
        if overlay["tranches"]["enabled"] and overlay["tranches"]["n_tranches"] > 1:
            return True
        if overlay["partial_rebalance"]["enabled"]:
            return True
        weighting = overlay["weighting"]
        if weighting["scheme"] not in {"equal", ""}:
            return True
        if weighting["max_weight"] is not None:
            return True
        if overlay["liquidity_cap"]["enabled"]:
            return True
        if VectorBTBacktestRunner._risk_overlay_active(overlay):
            return True
        return False

    @staticmethod
    def _risk_overlay_active(overlay: Dict[str, Any]) -> bool:
        risk = overlay["risk_overlay"]
        return bool(
            risk["vol_targeting"]["enabled"]
            or risk["drawdown_control"]["enabled"]
            or risk["breadth_filter"]["enabled"]
            or risk["regime_filter"]["enabled"]
            or risk["external_regime"]["enabled"]
            or risk["crisis_guard"]["enabled"]
        )

    @staticmethod
    def _overlay_summary(overlay: Dict[str, Any]) -> Dict[str, Any]:
        risk = overlay["risk_overlay"]
        return {
            "banding": overlay["banding"]["enabled"],
            "tranches": overlay["tranches"]["n_tranches"]
            if overlay["tranches"]["enabled"]
            else 1,
            "partial_rebalance": (
                overlay["partial_rebalance"]["trade_rate"]
                if overlay["partial_rebalance"]["enabled"]
                else None
            ),
            "weighting": overlay["weighting"]["scheme"],
            "max_weight": overlay["weighting"]["max_weight"],
            "liquidity_cap": overlay["liquidity_cap"]["enabled"],
            "vol_targeting": risk["vol_targeting"]["enabled"],
            "drawdown_control": risk["drawdown_control"]["enabled"],
            "breadth_filter": risk["breadth_filter"]["enabled"],
            "regime_filter": risk["regime_filter"]["enabled"],
            "external_regime": (
                risk["external_regime"]["position_plugin_id"]
                if risk["external_regime"]["enabled"]
                else None
            ),
            "strategy_plugin": overlay.get("strategy_plugin_id"),
            "crisis_guard": risk["crisis_guard"]["enabled"],
        }

    def _select_with_overlays(
        self,
        ranked: pd.Series,
        effective_top_k: int,
        held: Set[str],
    ) -> List[str]:
        """Apply banding (hysteresis) and tranche turnover throttle to selection.

        Without overlays this is equivalent to ``ranked.head(effective_top_k)``.
        ``ranked`` is the score series already sorted best-first.
        """
        ranked_list = list(ranked.index)
        n = len(ranked_list)
        if n == 0 or effective_top_k <= 0:
            return ranked_list[:effective_top_k]
        rank_of = {stock: idx for idx, stock in enumerate(ranked_list)}

        banding = self.overlay["banding"]
        if banding["enabled"]:
            # The entry band must always be wide enough to establish a full
            # Top-K book. Otherwise a small universe starts under-invested.
            buy_n = max(effective_top_k, int(round(n * banding["buy_rank_pct"])))
            sell_n = max(buy_n, int(round(n * banding["sell_rank_pct"])))
            sell_n = min(sell_n, n)
            # Retain held names still inside the (wider) sell band.
            kept = [s for s in ranked_list if s in held and rank_of[s] < sell_n]
            kept = kept[:effective_top_k]
            # New entrants must be inside the (tighter) buy band.
            buy_candidates = [s for s in ranked_list[:buy_n] if s not in held]
            free_slots = max(0, effective_top_k - len(kept))
            selected = kept + buy_candidates[:free_slots]
        else:
            selected = ranked_list[:effective_top_k]

        tranches = self.overlay["tranches"]
        if tranches["enabled"] and tranches["n_tranches"] > 1:
            import math

            max_new = max(1, math.ceil(effective_top_k / tranches["n_tranches"]))
            kept_selected = [s for s in selected if s in held]
            new_selected = [s for s in selected if s not in held]
            if len(new_selected) > max_new:
                allowed_new = new_selected[:max_new]
                # Backfill freed slots with next-best held names still tradeable.
                backfill = [
                    s
                    for s in ranked_list
                    if s in held and s not in kept_selected and s not in allowed_new
                ]
                free = effective_top_k - len(kept_selected) - len(allowed_new)
                selected = kept_selected + allowed_new + backfill[: max(0, free)]

        # Preserve best-first ordering.
        selected = sorted(set(selected), key=lambda s: rank_of.get(s, n))
        return selected[:effective_top_k]

    def _compute_overlay_weights(
        self,
        selected: List[str],
        score_series: pd.Series,
        price_df: pd.DataFrame,
        as_of_date: Any,
        prev_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute normalized intra-portfolio weights for the selected names.

        Schemes: equal | inverse_vol | signal | raw_score | mvo. Applies an optional
        single-name ``max_weight`` concentration cap with iterative
        redistribution. ``mvo`` solves max w'a - lambda*w'Sw - c*|w - w_prev|
        and falls back to signal weighting when it cannot run.
        """
        if not selected:
            return {}
        scheme = self.overlay["weighting"]["scheme"]
        raw: Dict[str, float] = {}
        if scheme == "mvo":
            mvo_weights = self._mvo_weights(
                selected, score_series, price_df, as_of_date, prev_weights or {}
            )
            if mvo_weights is not None:
                raw = mvo_weights
                self._mvo_diagnostics.append(
                    {"date": str(as_of_date), "state": "optimized"}
                )
            else:
                self._mvo_diagnostics.append(
                    {"date": str(as_of_date), "state": "fallback_signal"}
                )
                self.logger.debug(
                    f"[Overlay] MVO unavailable at {as_of_date}; falling back to signal weights"
                )
                k = len(selected)
                raw = {stock: float(k - idx) for idx, stock in enumerate(selected)}
        elif scheme == "inverse_vol":
            vols = self._trailing_volatility(selected, price_df, as_of_date)
            for stock in selected:
                vol = vols.get(stock)
                raw[stock] = (1.0 / vol) if (vol is not None and vol > 1e-9) else 0.0
            if sum(raw.values()) <= 0:
                raw = {stock: 1.0 for stock in selected}
        elif scheme == "raw_score":
            # Preserve the economic magnitude of an input score (for example,
            # circulating market cap) instead of converting it to rank weights.
            values = pd.to_numeric(score_series.reindex(selected), errors="coerce")
            raw = {
                stock: max(0.0, float(value)) if pd.notna(value) else 0.0
                for stock, value in values.items()
            }
            if sum(raw.values()) <= 0:
                raw = {stock: 1.0 for stock in selected}
        elif scheme == "signal":
            # Rank-based positive weights (best name gets highest weight).
            k = len(selected)
            for idx, stock in enumerate(selected):
                raw[stock] = float(k - idx)
        else:  # equal
            raw = {stock: 1.0 for stock in selected}

        total = sum(raw.values())
        if total <= 0:
            weights = equal_target_weights(selected)
        else:
            weights = {stock: value / total for stock, value in raw.items()}

        max_weight = self.overlay["weighting"]["max_weight"]
        if max_weight is not None:
            if float(max_weight) * len(weights) < 1.0 - 1e-9:
                # Infeasible cap (e.g. 2% cap on a 5-name book) silently destroys
                # the weighting scheme -- surface it loudly so operators notice.
                self.logger.warning(
                    f"[Overlay] max_weight={max_weight} infeasible for {len(weights)} names "
                    f"({float(max_weight) * len(weights):.2f} < 1.0): falling back to EQUAL "
                    f"weight; the '{scheme}' scheme is NOT applied"
                )
            weights = self._apply_max_weight_cap(weights, float(max_weight))
        return weights

    @staticmethod
    def _apply_max_weight_cap(
        weights: Dict[str, float], max_weight: float
    ) -> Dict[str, float]:
        """Cap each weight at max_weight, redistributing excess to uncapped names."""
        if max_weight <= 0 or not weights:
            return weights
        n = len(weights)
        if max_weight * n < 1.0 - 1e-9:
            # Cap is infeasible (too tight); fall back to equal weight.
            return {stock: 1.0 / n for stock in weights}
        result = dict(weights)
        for _ in range(100):
            capped = {s: w for s, w in result.items() if w > max_weight + 1e-12}
            if not capped:
                break
            excess = sum(result[s] - max_weight for s in capped)
            for s in capped:
                result[s] = max_weight
            uncapped = [s for s in result if s not in capped]
            uncapped_total = sum(result[s] for s in uncapped)
            if not uncapped or uncapped_total <= 0:
                break
            for s in uncapped:
                result[s] += excess * (result[s] / uncapped_total)
        return result

    def _blend_partial_rebalance(
        self,
        target_map: Dict[str, float],
        prev_weight_map: Dict[str, float],
    ) -> Dict[str, float]:
        """Garleanu-Pedersen style partial adjustment toward the target book.

        new_w = (1 - trade_rate) * current_w + trade_rate * target_w over the
        union of current holdings and fresh targets, so each rebalance only
        trades ``trade_rate`` of the gap. Exiting names decay geometrically and
        are liquidated outright once below ``min_weight`` (dust control); the
        freed weight is redistributed proportionally so the book stays fully
        invested. Both books are normalized to sum 1 first, which keeps the
        trading-speed decision independent of the risk-overlay exposure scaler.
        """
        cfg = self.overlay["partial_rebalance"]
        rate = float(cfg["trade_rate"])
        min_weight = float(cfg["min_weight"])
        prev_total = sum(w for w in prev_weight_map.values() if w > 0)
        if prev_total <= 1e-12:
            # Empty book (first rebalance / post-liquidation): jump to target.
            return dict(target_map)
        prev_norm = {s: w / prev_total for s, w in prev_weight_map.items() if w > 0}
        names = sorted(set(prev_norm) | set(target_map))
        blended = {
            s: (1.0 - rate) * prev_norm.get(s, 0.0) + rate * target_map.get(s, 0.0)
            for s in names
        }
        blended = {s: w for s, w in blended.items() if w >= max(min_weight, 1e-9)}
        total = sum(blended.values())
        if total <= 1e-12:
            return dict(target_map)
        return {s: w / total for s, w in blended.items()}

    def _apply_liquidity_cap(
        self,
        weights: Dict[str, float],
        as_of_date_str: str,
        deployed_capital: float,
    ) -> Dict[str, float]:
        """Cap each weight at its ADV participation limit (liquidity-aware weighting).

        weight_i <= max_participation * ADV_i * holding_period / capital, where ADV
        is the trailing average daily traded value (``money``) up to T-1. Excess is
        redistributed to names that still have headroom; if all names are capped the
        residual stays in cash (weights sum < 1). Names without ADV data keep their
        weight (data gap should not silently zero a position).
        """
        liq = self.overlay["liquidity_cap"]
        if not weights:
            return weights
        capital_base = liq.get("capital_base")
        capital = float(capital_base) if capital_base else float(deployed_capital)
        if capital <= 0:
            return weights
        adv = self._average_daily_value(
            sorted(weights), as_of_date_str, liq["adv_lookback_days"]
        )
        max_participation = float(liq["max_participation"])
        holding_period = max(1, int(self.holding_period or 1))
        caps: Dict[str, float] = {}
        for stock in weights:
            stock_adv = adv.get(stock)
            if stock_adv is None or stock_adv <= 0:
                continue  # no ADV data -> leave uncapped
            caps[stock] = max_participation * stock_adv * holding_period / capital

        result = dict(weights)
        for _ in range(100):
            over = {
                s: w for s, w in result.items() if s in caps and w > caps[s] + 1e-12
            }
            if not over:
                break
            excess = sum(result[s] - caps[s] for s in over)
            for s in over:
                result[s] = caps[s]
            headroom = {
                s: (caps[s] - result[s]) if s in caps else float("inf")
                for s in result
                if s not in over and (s not in caps or result[s] < caps[s] - 1e-12)
            }
            free_total = sum(result[s] for s in headroom)
            if not headroom or free_total <= 0:
                break  # everyone capped -> residual goes to cash
            for s in list(headroom):
                add = excess * (result[s] / free_total)
                result[s] += min(add, headroom[s])
        capped_names = [
            s
            for s in result
            if s in caps
            and result[s] <= caps[s] + 1e-9
            and weights[s] > result[s] + 1e-12
        ]
        if capped_names:
            self.logger.debug(
                f"[Overlay] liquidity_cap clipped {len(capped_names)} names at {as_of_date_str} "
                f"(capital={capital:,.0f}, participation={max_participation})"
            )
        return result

    def _average_daily_value(
        self,
        stocks: List[str],
        as_of_date_str: str,
        lookback_days: int,
    ) -> Dict[str, float]:
        """Trailing average daily traded value (yuan) per stock up to as_of_date (inclusive)."""
        conn = self.price_db_conn
        if conn is None or not stocks:
            return {}
        cache_key = (as_of_date_str, int(lookback_days))
        cache = getattr(self, "_adv_cache", None)
        if cache is None:
            cache = {}
            self._adv_cache = cache
        cached = cache.get(cache_key)
        if cached is None:
            cached = {}
            cache[cache_key] = cached
        missing = [s for s in stocks if s not in cached]
        if missing:
            stocks_str = "', '".join(missing)
            sql = f"""
            SELECT stock_code, AVG(money) AS adv
            FROM (
                SELECT stock_code, money,
                       ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY date DESC) AS rn
                FROM price_data_daily
                WHERE date <= '{as_of_date_str}'
                  AND stock_code IN ('{stocks_str}')
                  AND money IS NOT NULL AND money > 0
            )
            WHERE rn <= {int(lookback_days)}
            GROUP BY stock_code
            """
            try:
                rows = conn.execute(sql).fetchall()
                for stock_code, adv in rows:
                    cached[str(stock_code)] = float(adv)
            except Exception as e:
                self.logger.warning(
                    f"[Overlay] ADV query failed at {as_of_date_str}: {e}"
                )
            for s in missing:
                cached.setdefault(s, None)
        return {s: cached.get(s) for s in stocks if cached.get(s) is not None}

    # =====================================================================
    # [PORTFOLIO OVERLAY] Mean-variance optimizer (weighting.scheme == "mvo")
    # =====================================================================
    def _mvo_weights(
        self,
        selected: List[str],
        score_series: pd.Series,
        price_df: pd.DataFrame,
        as_of_date: Any,
        prev_weights: Dict[str, float],
    ) -> Optional[Dict[str, float]]:
        """Solve max w'alpha - lambda*w'Sigma*w - c*sum|w - w_prev| over the basket.

        Long-only, fully invested (sum w = 1), per-name ``max_weight`` bound.
        Style/industry exposures are *constrained* relative to the equal-weight
        basket (not neutralized) so the small-cap alpha survives. Returns None
        when the optimizer cannot run (missing scipy, short history, solver
        failure); the caller falls back to signal weighting.
        """
        cfg = self.overlay["weighting"]["mvo"]
        n = len(selected)
        if n < 2:
            return None
        try:
            from scipy.optimize import minimize
        except ImportError:
            self.logger.warning("[Overlay] MVO requires scipy; not installed")
            return None

        sigma = self._shrunk_covariance(selected, price_df, as_of_date, cfg)
        if sigma is None:
            return None

        # Alpha: cross-sectional z-score of the composite signal within the basket.
        scores = np.array(
            [float(score_series.get(s, np.nan)) for s in selected], dtype=float
        )
        if np.isnan(scores).any():
            fill = np.nanmedian(scores) if np.isfinite(np.nanmedian(scores)) else 0.0
            scores = np.where(np.isnan(scores), fill, scores)
        spread = float(scores.std())
        alpha = (scores - scores.mean()) / spread if spread > 1e-12 else np.zeros(n)

        prev = np.array(
            [float(prev_weights.get(s, 0.0)) for s in selected], dtype=float
        )
        w_eq = np.full(n, 1.0 / n)

        max_weight = self.overlay["weighting"]["max_weight"]
        upper = 1.0
        if max_weight is not None and float(max_weight) * n >= 1.0 - 1e-9:
            upper = float(max_weight)
        bounds = [(0.0, upper)] * n
        constraints: List[Dict[str, Any]] = [
            {
                "type": "eq",
                "fun": lambda w: float(np.sum(w) - 1.0),
                "jac": lambda w: np.ones(n),
            }
        ]
        constraints.extend(
            self._mvo_exposure_constraints(selected, as_of_date, w_eq, cfg)
        )

        lam = float(cfg["risk_aversion"])
        cost = float(cfg["turnover_penalty"])
        eps = 1e-6

        def objective(w: np.ndarray) -> float:
            delta = w - prev
            return float(
                -(w @ alpha)
                + lam * (w @ sigma @ w)
                + cost * np.sum(np.sqrt(delta**2 + eps**2))
            )

        def gradient(w: np.ndarray) -> np.ndarray:
            delta = w - prev
            return (
                -alpha
                + 2.0 * lam * (sigma @ w)
                + cost * delta / np.sqrt(delta**2 + eps**2)
            )

        result = minimize(
            objective,
            w_eq,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": int(cfg["maxiter"]), "ftol": 1e-10},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            self.logger.debug(
                f"[Overlay] MVO solver failed at {as_of_date}: {result.message}"
            )
            return None
        weights = np.clip(result.x, 0.0, None)
        total = float(weights.sum())
        if total <= 0:
            return None
        weights = weights / total
        return {stock: float(w) for stock, w in zip(selected, weights)}

    def _shrunk_covariance(
        self,
        selected: List[str],
        price_df: pd.DataFrame,
        as_of_date: Any,
        cfg: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """Annualized Ledoit-Wolf (identity-target) shrunk covariance of the basket."""
        cols = [s for s in selected if s in price_df.columns]
        if len(cols) != len(selected):
            return None
        try:
            window = price_df.loc[:as_of_date, cols].tail(
                int(cfg["cov_lookback_days"]) + 1
            )
        except (KeyError, TypeError):
            window = price_df[cols].tail(int(cfg["cov_lookback_days"]) + 1)
        rets = window.pct_change().iloc[1:]
        rets = rets.dropna(how="all")
        if len(rets) < int(cfg["cov_min_obs"]):
            return None
        x = rets.to_numpy(dtype=float)
        col_mean = np.nanmean(x, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        nan_mask = ~np.isfinite(x)
        if nan_mask.any():
            x[nan_mask] = np.take(col_mean, np.nonzero(nan_mask)[1])
        t, n = x.shape
        xc = x - x.mean(axis=0)
        sample = (xc.T @ xc) / max(t - 1, 1)
        mu = float(np.trace(sample)) / n
        target = mu * np.eye(n)
        d2 = float(np.sum((sample - target) ** 2))
        if d2 <= 1e-18:
            shrunk = sample
        else:
            b2 = 0.0
            for row in xc:
                outer = np.outer(row, row)
                b2 += float(np.sum((outer - sample) ** 2))
            b2 /= t * t
            delta = float(np.clip(b2 / d2, 0.0, 1.0))
            shrunk = delta * target + (1.0 - delta) * sample
        return shrunk * 252.0

    def _mvo_exposure_constraints(
        self,
        selected: List[str],
        as_of_date: Any,
        w_eq: np.ndarray,
        cfg: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Style/industry deviation constraints vs the equal-weight basket.

        For each style factor k: |(w - w_eq) @ X_k| <= style_dev_limit.
        For each industry j:     |(w - w_eq) @ G_j| <= industry_dev_limit.
        Missing exposure data degrades to "no constraint" rather than failing.
        """
        constraints: List[Dict[str, Any]] = []
        as_of_str = (
            str(pd.Timestamp(as_of_date).date())
            if not isinstance(as_of_date, str)
            else as_of_date
        )

        def _band(vector: np.ndarray, limit: float) -> None:
            base = float(w_eq @ vector)
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, v=vector, b=base, lim=limit: (
                        lim - float(w @ v) + b
                    ),
                    "jac": lambda w, v=vector: -v,
                }
            )
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, v=vector, b=base, lim=limit: (
                        lim + float(w @ v) - b
                    ),
                    "jac": lambda w, v=vector: v,
                }
            )

        style_limit = cfg["style_dev_limit"]
        if style_limit is not None:
            exposures = self._style_exposures_for(selected, as_of_str)
            if exposures is not None:
                for column in exposures.columns:
                    _band(exposures[column].to_numpy(dtype=float), float(style_limit))

        industry_limit = cfg["industry_dev_limit"]
        if industry_limit is not None:
            groups = self._industry_groups_for(selected, as_of_str)
            if groups:
                for industry_code in sorted(set(groups.values())):
                    indicator = np.array(
                        [
                            1.0 if groups.get(s) == industry_code else 0.0
                            for s in selected
                        ]
                    )
                    _band(indicator, float(industry_limit))
        return constraints

    def _style_exposures_for(
        self, stocks: List[str], as_of_date_str: str
    ) -> Optional[pd.DataFrame]:
        """Latest style exposures (<= as_of_date) per stock; NaN filled with 0."""
        conn = self.price_db_conn
        if conn is None:
            return None
        cache = getattr(self, "_style_exposure_cache", None)
        if cache is None:
            cache = {}
            self._style_exposure_cache = cache
        key = as_of_date_str
        if key not in cache:
            try:
                stocks_str = "', '".join(sorted(set(stocks)))
                frame = conn.execute(
                    f"""
                    SELECT * FROM style_factor_exposure
                    WHERE date = (SELECT MAX(date) FROM style_factor_exposure WHERE date <= '{as_of_date_str}')
                      AND stock_code IN ('{stocks_str}')
                    """
                ).fetchdf()
            except Exception as e:
                self.logger.debug(
                    f"[Overlay] style exposure query failed at {as_of_date_str}: {e}"
                )
                cache[key] = None
                return None
            if frame.empty:
                cache[key] = None
            else:
                frame = frame.set_index("stock_code").drop(
                    columns=["date"], errors="ignore"
                )
                cache[key] = frame
        base = cache[key]
        if base is None:
            return None
        return base.reindex(stocks).fillna(0.0)

    def _industry_groups_for(
        self, stocks: List[str], as_of_date_str: str
    ) -> Dict[str, str]:
        """Latest SW level-1 industry membership (<= as_of_date) per stock."""
        conn = self.price_db_conn
        if conn is None:
            return {}
        cache = getattr(self, "_industry_cache", None)
        if cache is None:
            cache = {}
            self._industry_cache = cache
        key = as_of_date_str
        if key not in cache:
            try:
                stocks_str = "', '".join(sorted(set(stocks)))
                rows = conn.execute(
                    f"""
                    SELECT stock_code, sw_l1_code FROM industry_membership
                    WHERE date = (SELECT MAX(date) FROM industry_membership WHERE date <= '{as_of_date_str}')
                      AND stock_code IN ('{stocks_str}')
                    """
                ).fetchall()
                cache[key] = {str(code): str(ind) for code, ind in rows if ind}
            except Exception as e:
                self.logger.debug(
                    f"[Overlay] industry query failed at {as_of_date_str}: {e}"
                )
                cache[key] = {}
        return {s: cache[key][s] for s in stocks if s in cache[key]}

    def _trailing_volatility(
        self,
        stocks: List[str],
        price_df: pd.DataFrame,
        as_of_date: Any,
    ) -> Dict[str, float]:
        """Trailing daily-return volatility per stock using closes up to as_of_date."""
        lookback = self.overlay["weighting"]["vol_lookback_days"]
        result: Dict[str, float] = {}
        try:
            window = price_df.loc[:as_of_date].tail(lookback + 1)
        except (KeyError, TypeError):
            window = price_df.tail(lookback + 1)
        for stock in stocks:
            if stock not in window.columns:
                result[stock] = None
                continue
            rets = window[stock].pct_change().dropna()
            result[stock] = float(rets.std()) if len(rets) >= 2 else None
        return result

    def _compute_exposure_by_date(
        self,
        returns: pd.Series,
        rebalance_dates: List[Any],
        price_df: pd.DataFrame,
        pool_stocks: List[str],
    ) -> Dict[Any, float]:
        """Compute a per-rebalance-date exposure multiplier from endogenous signals.

        Uses only information available up to the prior day (no look-ahead):
        - vol_targeting: target_vol / realized_vol over a trailing window
        - drawdown_control: hysteretic de-risk / exit / re-entry on the equity curve
        - breadth_filter: fraction of pool names with a large prior-day loss
        - regime_filter: micro-cap vs large-cap index spread z-score de-risk
        """
        risk = self.overlay["risk_overlay"]
        returns = returns.fillna(0.0)
        equity = (1.0 + returns).cumprod()
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0

        vt = risk["vol_targeting"]
        dd = risk["drawdown_control"]
        bf = risk["breadth_filter"]
        rf = risk["regime_filter"]
        external_regime = risk["external_regime"]
        cg = risk["crisis_guard"]

        external_signal: Optional[pd.DataFrame] = None
        position_plugin = None
        if external_regime["enabled"]:
            external_signal = self._load_external_regime_signal(external_regime)
            position_plugin = get_position_plugin(external_regime["position_plugin_id"])

        # Prior-day breadth of large losses across the pool.
        breadth_series: Optional[pd.Series] = None
        if bf["enabled"]:
            pool_cols = [s for s in pool_stocks if s in price_df.columns]
            if pool_cols:
                daily_rets = price_df[pool_cols].pct_change()
                breadth_series = (daily_rets < bf["loss_threshold"]).mean(axis=1)

        # Micro-cap vs large-cap regime spread z-score (data-driven replacement
        # for calendar-based skip-month heuristics).
        regime_z: Optional[pd.Series] = None
        if rf["enabled"]:
            regime_z = self._regime_spread_zscore(rf)

        # Composite crisis-warning vote series (limit-downs + turnover collapse
        # + loss breadth + micro/large spread). Each is a daily boolean Series.
        crisis_votes: List[pd.Series] = []
        if cg["enabled"]:
            crisis_votes = self._crisis_guard_vote_series(cg, pool_stocks, price_df)

        rebalance_norm = sorted({pd.Timestamp(d).normalize() for d in rebalance_dates})
        exposure_by_date: Dict[Any, float] = {}
        derisk_state = False  # drawdown_control latch
        crisis_cooldown = 0  # crisis_guard cooldown counter (rebalance steps)
        for reb_date in rebalance_norm:
            # Use data strictly before this rebalance date.
            prior = equity.index[equity.index < reb_date]
            exposure = 1.0
            if vt["enabled"] and len(prior) > vt["lookback_days"]:
                recent = returns.loc[prior].tail(vt["lookback_days"])
                realized = float(recent.std()) * np.sqrt(252.0)
                if realized > 1e-9:
                    exposure *= float(
                        np.clip(
                            vt["target_annual_vol"] / realized,
                            vt["min_exposure"],
                            vt["max_exposure"],
                        )
                    )
            if dd["enabled"] and len(prior) > 0:
                cur_dd = float(drawdown.loc[prior].iloc[-1])
                if derisk_state:
                    if cur_dd >= dd["reentry_dd"]:
                        derisk_state = False
                else:
                    if cur_dd <= dd["exit_dd"] or cur_dd <= dd["derisk_dd"]:
                        derisk_state = True
                if derisk_state:
                    exposure *= (
                        0.0 if cur_dd <= dd["exit_dd"] else dd["derisk_exposure"]
                    )
            if bf["enabled"] and breadth_series is not None:
                prior_b = breadth_series.index[breadth_series.index < reb_date]
                if len(prior_b) > 0:
                    cur_breadth = float(breadth_series.loc[prior_b].iloc[-1])
                    if cur_breadth >= bf["breadth_threshold"]:
                        exposure *= bf["derisk_exposure"]
            if rf["enabled"] and regime_z is not None:
                prior_z = regime_z.index[regime_z.index < reb_date]
                if len(prior_z) > 0:
                    cur_z = float(regime_z.loc[prior_z].iloc[-1])
                    if cur_z <= rf["exit_z"]:
                        exposure = 0.0
                    elif cur_z <= rf["derisk_z"]:
                        exposure *= rf["derisk_exposure"]
            if external_signal is not None and position_plugin is not None:
                eligible = external_signal.loc[
                    (external_signal["effective_date"] <= reb_date)
                    & (external_signal["observed_through"] < reb_date)
                ]
                regime_exposure = external_regime["missing_exposure"]
                if not eligible.empty:
                    latest = eligible.iloc[-1]
                    # Do not fall back to an older vintage if the latest expired.
                    if "valid_until" not in eligible or latest["valid_until"] >= reb_date:
                        regime_exposure = position_plugin(
                            float(latest["score"]), external_regime
                        )
                exposure *= regime_exposure
            if cg["enabled"] and crisis_votes:
                votes = 0
                for sig in crisis_votes:
                    prior_s = sig.index[sig.index < reb_date]
                    if len(prior_s) > 0 and bool(sig.loc[prior_s].iloc[-1]):
                        votes += 1
                if votes >= cg["min_triggers"]:
                    crisis_cooldown = cg["cooldown_days"]
                    exposure = (
                        0.0
                        if votes >= cg["hard_exit_triggers"]
                        else exposure * cg["derisk_exposure"]
                    )
                elif crisis_cooldown > 0:
                    crisis_cooldown -= 1
                    exposure *= cg["derisk_exposure"]
            exposure_by_date[reb_date] = max(0.0, exposure)
        return exposure_by_date

    def _load_external_regime_signal(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Load and validate one hash-bound PIT regime signal artifact."""
        cached = getattr(self, "_external_regime_signal_cache", None)
        cache_key = (
            config["signal_path"], config["signal_sha256"], config["date_column"],
            config["observed_through_column"], config["score_column"],
            config.get("require_valid_until", False),
        )
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        path = Path(config["signal_path"]).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"external regime signal not found: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != config["signal_sha256"]:
            raise ValueError(f"external regime signal hash mismatch: {path}")

        source_columns = [
            config["date_column"],
            config["observed_through_column"],
            config["score_column"],
        ]
        raw = pd.read_parquet(path)
        if config.get("require_valid_until", False) and "valid_until" not in raw:
            raise ValueError("external regime signal requires valid_until")
        if "valid_until" in raw:
            source_columns.append("valid_until")
        frame = raw[source_columns].rename(
            columns={
                config["date_column"]: "effective_date",
                config["observed_through_column"]: "observed_through",
                config["score_column"]: "score",
            }
        )
        frame["effective_date"] = pd.to_datetime(
            frame["effective_date"], errors="coerce"
        ).dt.normalize()
        frame["observed_through"] = pd.to_datetime(
            frame["observed_through"], errors="coerce"
        ).dt.normalize()
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        if "valid_until" in frame:
            frame["valid_until"] = pd.to_datetime(
                frame["valid_until"], errors="coerce"
            ).dt.normalize()
            if (frame["valid_until"] < frame["effective_date"]).any():
                raise ValueError("external regime signal expires before effective_date")
        if (
            frame.empty
            or frame.isna().any().any()
            or frame["effective_date"].duplicated().any()
            or not np.isfinite(frame["score"]).all()
            or (frame["observed_through"] >= frame["effective_date"]).any()
        ):
            raise ValueError("external regime signal violates uniqueness or PIT timing")
        frame = frame.sort_values("effective_date").reset_index(drop=True)
        self._external_regime_signal_cache = (cache_key, frame)
        return frame

    def _load_strategy_rebalance_dates(self) -> Optional[List[datetime]]:
        """Load an optional hash-bound schedule owned by the strategy plugin."""
        source = self.strategy_plugin_config.get("rebalance_dates_source") or {}
        if not source:
            return None
        if not isinstance(source, dict):
            raise ValueError("strategy_plugin.rebalance_dates_source must be an object")
        path = Path(str(source.get("path", ""))).expanduser().resolve()
        expected_hash = str(source.get("sha256", ""))
        if not path.is_file() or not expected_hash:
            raise ValueError("strategy rebalance-date source is missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("strategy rebalance-date source hash mismatch")
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        for key in str(source.get("field", "assumptions.rebalance_dates")).split("."):
            if not isinstance(payload, dict) or key not in payload:
                raise ValueError("strategy rebalance-date field is missing")
            payload = payload[key]
        dates = pd.DatetimeIndex(pd.to_datetime(payload, errors="coerce")).normalize()
        if dates.hasnans or dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("strategy rebalance dates must be unique and increasing")
        return [date.to_pydatetime() for date in dates]

    def _regime_spread_zscore(self, rf: Dict[str, Any]) -> Optional[pd.Series]:
        """Z-score of the micro-cap minus large-cap rolling return spread.

        Built from ``index_price_daily`` (e.g. 399101.XSHE vs 000300.XSHG). The
        spread is the difference of trailing ``spread_window_days`` returns and
        the z-score is computed against a trailing ``zscore_window_days`` window,
        so each value only uses information available on that day.
        """
        cached = getattr(self, "_regime_z_cache", None)
        cache_key = (
            rf["micro_index"],
            rf["benchmark_index"],
            rf["spread_window_days"],
            rf["zscore_window_days"],
        )
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        conn = self.price_db_conn
        if conn is None:
            self.logger.warning(
                "[Overlay] regime_filter enabled but no price DB connection; skipping"
            )
            self._regime_z_cache = (cache_key, None)
            return None
        try:
            sql = f"""
            SELECT date, index_code, close
            FROM index_price_daily
            WHERE index_code IN ('{rf["micro_index"]}', '{rf["benchmark_index"]}')
            ORDER BY date
            """
            frame = conn.execute(sql).fetchdf()
        except Exception as e:
            self.logger.warning(
                f"[Overlay] regime_filter index query failed ({e}); skipping"
            )
            self._regime_z_cache = (cache_key, None)
            return None
        if frame.empty:
            self.logger.warning(
                "[Overlay] regime_filter: no index_price_daily rows; skipping"
            )
            self._regime_z_cache = (cache_key, None)
            return None
        wide = frame.pivot(
            index="date", columns="index_code", values="close"
        ).sort_index()
        wide.index = pd.to_datetime(wide.index)
        if (
            rf["micro_index"] not in wide.columns
            or rf["benchmark_index"] not in wide.columns
        ):
            self.logger.warning(
                f"[Overlay] regime_filter: missing index series "
                f"(have {list(wide.columns)}, need {rf['micro_index']} and {rf['benchmark_index']}); skipping"
            )
            self._regime_z_cache = (cache_key, None)
            return None
        window = rf["spread_window_days"]
        z_window = rf["zscore_window_days"]
        micro_ret = wide[rf["micro_index"]].pct_change(window)
        bench_ret = wide[rf["benchmark_index"]].pct_change(window)
        spread = (micro_ret - bench_ret).dropna()
        rolling_mean = spread.rolling(
            z_window, min_periods=max(20, z_window // 5)
        ).mean()
        rolling_std = spread.rolling(z_window, min_periods=max(20, z_window // 5)).std()
        z = (
            ((spread - rolling_mean) / rolling_std)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        self._regime_z_cache = (cache_key, z)
        return z

    def _crisis_guard_vote_series(
        self,
        cg: Dict[str, Any],
        pool_stocks: List[str],
        price_df: pd.DataFrame,
    ) -> List[pd.Series]:
        """Build the daily boolean vote series for each enabled crisis signal.

        Signals (all computed from data available at the close of each day, and
        consumed with a one-day lag by the rebalance loop):
        - limit_down: fraction of pool names closing at the limit-down price
        - turnover_collapse: pool aggregate money short-MA / long-MA ratio
        - breadth: fraction of pool names with a large daily loss
        - spread: micro-cap minus large-cap rolling return z-score
        """
        sigs = cg["signals"]
        votes: List[pd.Series] = []
        if sigs["limit_down"]["enabled"]:
            frac = self._pool_limit_down_fraction(pool_stocks)
            if frac is not None:
                votes.append(frac >= sigs["limit_down"]["fraction_threshold"])
        if sigs["turnover_collapse"]["enabled"]:
            tc = sigs["turnover_collapse"]
            ratio = self._pool_turnover_ratio(
                pool_stocks, tc["short_window"], tc["long_window"]
            )
            if ratio is not None:
                votes.append(ratio <= tc["ratio_threshold"])
        if sigs["breadth"]["enabled"]:
            br = sigs["breadth"]
            pool_cols = [s for s in pool_stocks if s in price_df.columns]
            if pool_cols:
                daily_rets = price_df[pool_cols].pct_change()
                breadth = (daily_rets < br["loss_threshold"]).mean(axis=1)
                votes.append(breadth >= br["breadth_threshold"])
        if sigs["spread"]["enabled"]:
            sp = dict(sigs["spread"])
            z = self._regime_spread_zscore(sp)
            if z is not None:
                votes.append(z <= sp["z_threshold"])
        self.logger.info(
            f"[Overlay] crisis_guard active with {len(votes)} signal series "
            f"(min_triggers={cg['min_triggers']}, derisk={cg['derisk_exposure']}, "
            f"hard_exit={cg['hard_exit_triggers']}, cooldown={cg['cooldown_days']}d)"
        )
        return votes

    def _pool_limit_down_fraction(self, pool_stocks: List[str]) -> Optional[pd.Series]:
        """Daily fraction of pool names that closed at the limit-down price.

        Uses ``price_data_daily.low_limit`` (exact exchange limit price), so no
        return-threshold approximation is needed. Paused names are excluded.
        """
        conn = self.price_db_conn
        if conn is None or not pool_stocks:
            self.logger.warning(
                "[Overlay] crisis_guard limit_down: no price DB connection; skipping"
            )
            return None
        try:
            in_list = ", ".join(f"'{s}'" for s in sorted(set(pool_stocks)))
            sql = f"""
            SELECT date,
                   AVG(CASE WHEN close <= low_limit * 1.0001 THEN 1.0 ELSE 0.0 END) AS frac
            FROM price_data_daily
            WHERE stock_code IN ({in_list})
              AND (paused IS NULL OR paused = 0)
              AND low_limit IS NOT NULL AND low_limit > 0
            GROUP BY date
            ORDER BY date
            """
            frame = conn.execute(sql).fetchdf()
        except Exception as e:
            self.logger.warning(
                f"[Overlay] crisis_guard limit_down query failed ({e}); skipping"
            )
            return None
        if frame.empty:
            return None
        series = pd.Series(frame["frac"].values, index=pd.to_datetime(frame["date"]))
        return series.sort_index()

    def _pool_turnover_ratio(
        self, pool_stocks: List[str], short_window: int, long_window: int
    ) -> Optional[pd.Series]:
        """Short-MA / long-MA ratio of pool aggregate traded value (money).

        A ratio well below 1 means liquidity in the candidate pool is drying up
        much faster than its recent norm (the Jan-2024 microcap crisis showed
        this pattern ahead of the worst drawdown days).
        """
        conn = self.price_db_conn
        if conn is None or not pool_stocks:
            self.logger.warning(
                "[Overlay] crisis_guard turnover_collapse: no price DB connection; skipping"
            )
            return None
        try:
            in_list = ", ".join(f"'{s}'" for s in sorted(set(pool_stocks)))
            sql = f"""
            SELECT date, SUM(money) AS pool_money
            FROM price_data_daily
            WHERE stock_code IN ({in_list})
              AND (paused IS NULL OR paused = 0)
            GROUP BY date
            ORDER BY date
            """
            frame = conn.execute(sql).fetchdf()
        except Exception as e:
            self.logger.warning(
                f"[Overlay] crisis_guard turnover query failed ({e}); skipping"
            )
            return None
        if frame.empty:
            return None
        money = pd.Series(
            frame["pool_money"].values, index=pd.to_datetime(frame["date"])
        ).sort_index()
        short_ma = money.rolling(short_window, min_periods=short_window).mean()
        long_ma = money.rolling(long_window, min_periods=long_window).mean()
        ratio = (short_ma / long_ma).replace([np.inf, -np.inf], np.nan).dropna()
        return ratio

    def _load_bundle(self) -> dict:
        """
        Load and cache Lasso model bundle - mirrors cloud _load_lasso_bundle().

        Bundle structure:
        {
            "index": {
                "models": [
                    {
                        "train_end": "2023-09-18",
                        "weights": {"factor1": 0.5, ...},
                        "feature_names": ["factor1", ...],
                        "blend_only": false,
                        "blend_with_original_ratio": 0.3,
                        "blend_weights": {"Factor1": 1.0},
                        "standardization_method": {...}
                    },
                    ...
                ]
            }
        }

        Returns:
            Bundle data dict, or empty dict if not available
        """
        if self._bundle_data is not None:
            return self._bundle_data

        if not self.bundle_path or not os.path.exists(self.bundle_path):
            self.logger.warning(f"[BUNDLE] Bundle not found: {self.bundle_path}")
            self._bundle_data = {}
            self._bundle_models = []
            return self._bundle_data

        try:
            with open(self.bundle_path, "r", encoding="utf-8") as f:
                self._bundle_data = json.load(f)

            # Cache sorted models list
            index = self._bundle_data.get("index", {})
            self._bundle_models = index.get("models", [])
            if index.get("requires_parent_baseline", False):
                expected_count = int(index.get("expected_model_count", -1))
                if expected_count != len(self._bundle_models):
                    raise ValueError(
                        "incomplete parent bundle: "
                        f"expected {expected_count} models, found {len(self._bundle_models)}"
                    )
                missing_parent = [
                    str(model.get("train_end", "unknown"))
                    for model in self._bundle_models
                    if not isinstance(model.get("parent_baseline"), dict)
                    or not model.get("parent_baseline")
                ]
                if missing_parent:
                    raise ValueError(
                        "incomplete parent bundle: parent_baseline missing for "
                        + ", ".join(missing_parent)
                    )

            # Log bundle info (match cloud format)
            model_dates = [m.get("train_end") for m in self._bundle_models]
            blend_only_count = sum(
                1 for m in self._bundle_models if m.get("blend_only", False)
            )

            self.logger.info(
                f"[BUNDLE_LOADED] n_models={len(self._bundle_models)}, blend_only_models={blend_only_count}"
            )
            self.logger.info(f"[BUNDLE_INFO] model_train_ends={model_dates}")

        except ValueError as e:
            self.logger.error(f"[BUNDLE_ERROR] Failed to load bundle: {e}")
            self._bundle_data = {}
            self._bundle_models = []
            raise
        except Exception as e:
            self.logger.error(f"[BUNDLE_ERROR] Failed to load bundle: {e}")
            self._bundle_data = {}
            self._bundle_models = []

        return self._bundle_data

    def _get_model_for_date(self, current_date_str: str) -> Optional[dict]:
        """
        Get model for date - EXACT mirror of cloud _get_lasso_model_info_for_date().

        Selection logic: find latest model where train_end < current_date
        This ensures we use the model trained with data available at that point.

        Args:
            current_date_str: Current date in 'YYYY-MM-DD' format (T-1 factor date)

        Returns:
            Model dict with weights, feature_names, blend params, or None
        """
        self._load_bundle()

        if not self._bundle_models:
            self.logger.warning(
                f"[MODEL_SELECT] No models in bundle for date={current_date_str}"
            )
            return None

        best_model = model_for_date(self._bundle_models, current_date_str)

        # Log selection (match cloud format)
        if best_model:
            is_blend_only = best_model.get("blend_only", False)
            n_features = len(best_model.get("feature_names", []))
            if self.verbose_alignment:
                self.logger.info(
                    f"[MODEL_SELECT] date={current_date_str}|"
                    f"selected_train_end={best_model.get('train_end')}|"
                    f"n_features={n_features}|blend_only={is_blend_only}"
                )
        else:
            if self.verbose_alignment:
                self.logger.warning(
                    f"[MODEL_SELECT] date={current_date_str}|no_model_available "
                    f"(all train_end >= current_date)"
                )

        return best_model

    def _parent_baseline_scores_for_model(
        self, model_info: dict, date_str: str, stock_list: List[str]
    ) -> Optional[pd.Series]:
        """Load the frozen parent composite used by a leaf model.

        Relative paths are resolved beside the bundle so the deployment
        artifact is portable.  A declared parent is fail-closed: hash, schema,
        duplicate, missing-date, or overlap errors must never degrade to a
        residual-only portfolio.
        """
        contract = model_info.get("parent_baseline")
        if not isinstance(contract, dict) or not contract:
            return None
        path_value = str(contract.get("path", "")).strip()
        expected_sha256 = str(contract.get("sha256", "")).strip()
        if not path_value or not expected_sha256:
            raise ValueError("parent_baseline requires path and sha256")
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            if not self.bundle_path:
                raise ValueError("relative parent baseline requires bundle_path")
            path = Path(self.bundle_path).expanduser().resolve().parent / path
        path = path.resolve()
        cache_key = (str(path), expected_sha256)
        baseline = self._parent_baseline_cache.get(cache_key)
        if baseline is None:
            if not path.is_file():
                raise FileNotFoundError(f"parent baseline not found: {path}")
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(f"parent baseline hash mismatch: {path}")
            frame = pd.read_parquet(path)
            if "composite_value" not in frame or not isinstance(
                frame.index, pd.MultiIndex
            ):
                raise ValueError(f"invalid parent baseline cache: {path}")
            if (
                "date" not in frame.index.names
                or "stock_code" not in frame.index.names
            ):
                raise ValueError(f"invalid parent baseline index: {path}")
            baseline = pd.to_numeric(frame["composite_value"], errors="coerce")
            if baseline.index.duplicated().any() or baseline.isna().any():
                raise ValueError(f"invalid parent baseline values: {path}")
            baseline = baseline.sort_index()
            self._parent_baseline_cache[cache_key] = baseline

        try:
            scores = baseline.xs(pd.Timestamp(date_str), level="date")
        except KeyError as exc:
            raise ValueError(
                f"parent baseline has no scores for replay date {date_str}"
            ) from exc
        scores = pd.to_numeric(scores.reindex(stock_list), errors="coerce").dropna()
        if scores.empty:
            raise ValueError(
                f"parent baseline has no stock overlap for replay date {date_str}"
            )
        return scores

    def _is_lightgbm_bundle(self) -> bool:
        """
        Check if the loaded bundle is a LightGBM bundle.

        LightGBM bundles have 'models' dict with model file contents,
        while Lasso bundles have 'weights' in each model entry.
        """
        bundle = self._load_bundle()

        # Check if bundle has 'models' dict (LightGBM format)
        if "models" in bundle and isinstance(bundle.get("models"), dict):
            # Also verify index.models don't have 'weights' (Lasso format)
            index = bundle.get("index", {})
            models = index.get("models", [])
            if models and not models[0].get("weights"):
                return True

        return False

    def _load_lgb_booster_for_date(self, date_str: str) -> Optional["lgb.Booster"]:
        """
        Load LightGBM booster for a given date from bundle.

        Args:
            date_str: Date string in 'YYYY-MM-DD' format

        Returns:
            LightGBM Booster or None if not available
        """
        if not LIGHTGBM_AVAILABLE:
            self.logger.warning("[LGB_LOAD] LightGBM not available")
            return None

        model_info = self._get_model_for_date(date_str)
        if not model_info:
            return None

        model_file = model_info.get("model_file")
        if not model_file:
            self.logger.warning(
                f"[LGB_LOAD] No model_file in model_info for {date_str}"
            )
            return None

        bundle = self._load_bundle()
        models_dict = bundle.get("models", {})

        model_content = models_dict.get(model_file)
        if not model_content:
            self.logger.warning(
                f"[LGB_LOAD] Model file {model_file} not found in bundle"
            )
            return None

        try:
            booster = lgb.Booster(model_str=model_content)

            if self.verbose_alignment:
                self.logger.info(
                    f"[LGB_LOAD] Loaded booster for {date_str} from {model_file}"
                )

            return booster

        except Exception as e:
            self.logger.error(f"[LGB_LOAD] Failed to load booster: {e}")
            return None

    def _compute_lgb_prediction(
        self, date_str: str, stock_list: List[str], return_raw_factors: bool = False
    ) -> Tuple[pd.Series, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Compute LightGBM predictions for a given date.

        Args:
            date_str: Date string in 'YYYY-MM-DD' format (T-1 date)
            stock_list: List of stock codes
            return_raw_factors: If True, also return processed and raw factors

        Returns:
            Tuple of (prediction Series, processed factors DataFrame, raw factors DataFrame)
        """
        model_info = self._get_model_for_date(date_str)
        if not model_info:
            return pd.Series(dtype=float), None, None

        booster = self._load_lgb_booster_for_date(date_str)
        if booster is None:
            return pd.Series(dtype=float), None, None

        # Get feature names from model_info
        feature_names = model_info.get("feature_names", [])
        if not feature_names:
            self.logger.warning(
                f"[LGB_PREDICT] No feature_names in model_info for {date_str}"
            )
            return pd.Series(dtype=float), None, None

        # Load factors
        if self._factor_loader is None:
            if self.factor_db_conn is not None:
                self._factor_loader = FactorLoader(
                    existing_conn=self.factor_db_conn,
                    logger=self.logger,
                    shared_cache=self._factor_load_cache,
                )
            elif self.factor_db_path:
                self._factor_loader = FactorLoader(
                    duckdb_path=self.factor_db_path,
                    logger=self.logger,
                    shared_cache=self._factor_load_cache,
                )
            else:
                self.logger.warning("[LGB_PREDICT] No factor database connection")
                return pd.Series(dtype=float), None, None

        # Dynamic-pool members can originate from a set/SQL result and therefore
        # have no stable order.  Cross-sectional floating-point reductions feed
        # tree thresholds, so a different stock order can flip a boundary row
        # even when the model files are byte-identical.  Canonicalise before
        # loading, preprocessing, prediction, and score/index reconstruction.
        canonical_stocks = sorted(set(stock_list))
        factor_df = self._factor_loader.load_factors(
            date_str, canonical_stocks, feature_names
        )

        if factor_df.empty:
            self.logger.debug(f"[LGB_PREDICT] No factor data for {date_str}")
            return pd.Series(dtype=float), None, None

        # Import preprocessing functions (single-date versions from factor_loader)
        from .factor_loader import (
            cross_sectional_raw_zscore,
            cross_sectional_zscore,
            cross_sectional_rank_zscore,
            cross_sectional_binning,
            cross_sectional_winsorize_zscore,
        )

        # Get preprocessing config
        preprocess = model_info.get("standardization_method", {})
        default_mode = (
            preprocess.get("ALL", "rank_zscore")
            if isinstance(preprocess, dict)
            else "rank_zscore"
        )
        n_bins = (
            preprocess.get("_binning_n_bins", 5) if isinstance(preprocess, dict) else 5
        )
        winsorize_pct = (
            preprocess.get("_winsorize_percentile", 0.025)
            if isinstance(preprocess, dict)
            else 0.025
        )

        # Apply preprocessing
        processed = pd.DataFrame(index=factor_df.index)
        for fname in feature_names:
            if fname in factor_df.columns:
                raw = factor_df[fname]
                mode = (
                    preprocess.get(fname, default_mode)
                    if isinstance(preprocess, dict)
                    else default_mode
                )

                if mode == "raw_zscore":
                    processed[fname] = cross_sectional_raw_zscore(raw)
                elif mode == "rank_zscore":
                    processed[fname] = cross_sectional_rank_zscore(raw)
                elif mode == "binning":
                    processed[fname] = cross_sectional_binning(raw, n_bins=n_bins)
                elif mode == "winsorize_zscore":
                    processed[fname] = cross_sectional_winsorize_zscore(
                        raw, percentile=winsorize_pct
                    )
                else:
                    if mode not in ("zscore", "none", None, ""):
                        self.logger.warning(
                            f"[LGB_PREPROCESS] Unknown method '{mode}' for {fname}, falling back to zscore"
                        )
                    processed[fname] = cross_sectional_zscore(raw)
            else:
                processed[fname] = 0.0

        # Ensure column order matches feature_names
        processed = processed.reindex(columns=feature_names, fill_value=0.0)

        # Fill NaN with 0 for prediction
        X = processed.fillna(0.0).values

        # Run prediction
        try:
            predictions = booster.predict(X)
            score = pd.Series(predictions, index=processed.index, name="lgb_score")

            if self.verbose_alignment:
                self.logger.info(
                    f"[LGB_PREDICT] {date_str}: {len(score)} predictions generated"
                )

        except Exception as e:
            self.logger.error(f"[LGB_PREDICT] Prediction failed: {e}")
            return pd.Series(dtype=float), None, None

        # Apply blend if configured
        blend_ratio = model_info.get("blend_with_original_ratio", 0.0)
        blend_weights = model_info.get("blend_weights", {})

        if blend_ratio > 0 and blend_weights:
            original_composite = pd.Series(0.0, index=score.index)

            for fname, bw in blend_weights.items():
                blend_factor_df = self._factor_loader.load_factors(
                    date_str, list(score.index), [fname]
                )

                if not blend_factor_df.empty and fname in blend_factor_df.columns:
                    raw = blend_factor_df[fname].reindex(score.index)
                    preprocess_method = (
                        preprocess.get(fname, "zscore")
                        if isinstance(preprocess, dict)
                        else "zscore"
                    )

                    if preprocess_method == "raw_zscore":
                        std_series = cross_sectional_raw_zscore(raw)
                    elif preprocess_method == "rank_zscore":
                        std_series = cross_sectional_rank_zscore(raw)
                    elif preprocess_method == "binning":
                        std_series = cross_sectional_binning(raw, n_bins=n_bins)
                    elif preprocess_method == "winsorize_zscore":
                        std_series = cross_sectional_winsorize_zscore(
                            raw, percentile=winsorize_pct
                        )
                    else:
                        if preprocess_method not in ("zscore", "none", None, ""):
                            self.logger.warning(
                                f"[LGB_BLEND] Unknown method '{preprocess_method}' for {fname}, falling back to zscore"
                            )
                        std_series = cross_sectional_zscore(raw)

                    original_composite += std_series.fillna(0.0) * bw

            # Apply blend
            score = score * (1 - blend_ratio) + original_composite * blend_ratio

            if self.verbose_alignment:
                self.logger.info(f"[LGB_BLEND] Applied blend_ratio={blend_ratio}")

        # Apply postprocess standardization if configured
        if model_info.get("postprocess_cs_standardize", False):
            score = cross_sectional_zscore(score)

        if return_raw_factors:
            return score, processed, factor_df
        return score, None, None

    def _apply_score_decay(
        self, scores: pd.Series, score_date: Any
    ) -> pd.Series:
        """Smooth final composite scores without changing Top-K execution."""
        if not self.score_decay_enabled:
            return scores

        current = pd.to_numeric(scores, errors="coerce").dropna().astype(float)
        current_date = pd.Timestamp(score_date).normalize()
        if self._score_decay_date is None or self._score_decay_state.empty:
            smoothed = current
        else:
            elapsed_days = (current_date - self._score_decay_date).days
            if elapsed_days <= 0:
                raise ValueError("score decay dates must be strictly increasing")
            retention = 2.0 ** (
                -elapsed_days / self.score_decay_half_life_days
            )
            previous = self._score_decay_state.reindex(current.index)
            smoothed = current.copy()
            overlap = previous.notna()
            smoothed.loc[overlap] = (
                retention * previous.loc[overlap]
                + (1.0 - retention) * current.loc[overlap]
            )

        self._score_decay_state = smoothed.copy()
        self._score_decay_date = current_date
        return smoothed

    def _compute_target_weights(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        dynamic_pools: Dict[str, List[str]],
        rebalance_dates: List[datetime],
        open_df: pd.DataFrame = None,
        factor_ratio_df: pd.DataFrame = None,
        exposure_by_date: Optional[Dict[Any, float]] = None,
    ) -> pd.DataFrame:
        """
        Compute target VALUE matrix (dates x stocks) for size_type='targetvalue'.

        [JQ-ALIGNED V4] 模拟 JoinQuant 的精确仓位管理逻辑:
        - 使用预模拟计算每个调仓日的现金和持仓
        - 卖出股票: 按当日开盘价计算释放的现金
        - 买入股票: value = cash / num_stocks_to_buy (JQ 的精确逻辑)
        - 保持股票: NaN (不操作，VectorBT 保持当前股数)

        Corporate actions are already represented by PIT-adjusted prices. Applying
        factor ratios to shares here would double count splits and cash dividends.

        Non-rebalance dates: NaN (hold existing positions)
        Skip months: 0 (clear all positions)

        Returns:
            DataFrame with same shape as factor_df, containing TARGET VALUES (not percentages)
        """
        all_dates = factor_df.index
        self._score_decay_state = pd.Series(dtype=float)
        self._score_decay_date = None
        self._input_factor_df = factor_df
        all_stocks = factor_df.columns.tolist()
        self._backtest_previous_dates = {
            pd.Timestamp(all_dates[idx]).strftime("%Y-%m-%d"): pd.Timestamp(
                all_dates[idx - 1]
            ).strftime("%Y-%m-%d")
            for idx in range(1, len(all_dates))
        }
        self._factor_replay_dates = sorted({
            self._backtest_previous_dates[pd.Timestamp(date).strftime("%Y-%m-%d")]
            for date in rebalance_dates
            if pd.Timestamp(date).strftime("%Y-%m-%d") in self._backtest_previous_dates
        })
        all_stocks_set = set(all_stocks)
        # Dynamic pools are constant across a benchmark suite.  Parsing and
        # sorting all pool dates inside every seed/rebalance made a 100-seed
        # random reference unnecessarily quadratic.  Build the as-of lookup
        # once while retaining exactly the same "latest pool <= trade date"
        # semantics as ``filter_stocks_by_pool``.
        normalized_pools = {
            pd.Timestamp(key).normalize(): set(values)
            for key, values in dynamic_pools.items()
        }
        sorted_pool_dates = sorted(normalized_pools)

        # Initialize with NaN (means "no action" in VectorBT)
        weights = pd.DataFrame(
            np.nan,
            index=all_dates,
            columns=all_stocks,
            dtype=np.float64,
        )

        # Convert rebalance dates to set for fast lookup
        rebalance_set = set(pd.to_datetime(d).normalize() for d in rebalance_dates)

        # [JQ-ALIGNED V5] 构建 Skip 清仓日期集合（支持整月与 MM-DD:MM-DD 窗口）
        # 逻辑：进入每个 skip 区间的第一个交易日作为清仓日
        skip_month_liquidation_dates = set()
        if self.skip_months:
            dates_idx = pd.DatetimeIndex(all_dates)
            in_skip = skip_mask(dates_idx, self.skip_months)
            if in_skip.any():
                prev_in_skip = np.concatenate(([False], in_skip[:-1]))
                entry_dates = dates_idx.normalize()[in_skip & ~prev_in_skip]
                skip_month_liquidation_dates = set(entry_dates)

                for d in entry_dates:
                    self.logger.debug(
                        f"[Skip] 识别清仓日: {pd.Timestamp(d).strftime('%Y-%m-%d')}"
                    )

        # Normalize skip_month_liquidation_dates to pd.Timestamp for consistent comparison
        skip_month_liquidation_dates = set(
            pd.Timestamp(d).normalize() for d in skip_month_liquidation_dates
        )

        if skip_month_liquidation_dates:
            sorted_liq_dates = sorted(
                [
                    pd.Timestamp(d).strftime("%Y-%m-%d")
                    for d in skip_month_liquidation_dates
                ]
            )
            # self.logger.info(
            #     f"[JQ-ALIGNED V5] Skip Month 清仓日期: {len(skip_month_liquidation_dates)} 个, "
            #     f"日期: {sorted_liq_dates[:5]}{'...' if len(sorted_liq_dates) > 5 else ''}"
            # )

        # [JQ-ALIGNED V4] 预模拟：追踪持仓和现金
        # 使用与 JQ 相同的逻辑计算每个调仓日的买入金额
        sim_cash = float(self.init_cash)  # 模拟现金
        sim_positions = {}  # {stock_code: shares}  股数
        pending_sells: set[str] = set()
        pending_sell_since: Dict[str, pd.Timestamp] = {}
        blocked_exit_notional = 0.0
        completed_exit_delays: List[int] = []
        blocked_exit_events = 0
        strict_position_buys_skipped = 0
        buy_cost_multiplier = (
            1.0 + max(float(self.cost_rate), 0.0) + max(float(self.slippage), 0.0)
        )
        sell_proceeds_multiplier = max(
            0.0, 1.0 - max(float(self.cost_rate), 0.0) - max(float(self.slippage), 0.0)
        )

        # 使用开盘价进行模拟（如果提供了开盘价数据）
        exec_price_df = open_df if open_df is not None else price_df

        def mark_price(current_date: Any, stock: str) -> Optional[float]:
            """Return an observable mark without inventing an executable price."""
            for frame in (exec_price_df, price_df):
                if stock in frame.columns and current_date in frame.index:
                    value = frame.loc[current_date, stock]
                    if pd.notna(value) and float(value) > 0:
                        return float(value)
            if stock in price_df.columns:
                history = price_df.loc[:current_date, stock].dropna()
                if not history.empty and float(history.iloc[-1]) > 0:
                    return float(history.iloc[-1])
            return None

        def execution_price(current_date: Any, stock: str) -> Optional[float]:
            """Return only the price VectorBT can actually execute at."""
            if (
                stock not in exec_price_df.columns
                or current_date not in exec_price_df.index
            ):
                return None
            value = exec_price_df.loc[current_date, stock]
            if pd.isna(value) or float(value) <= 0:
                return None
            return float(value)

        def block_exit(current_date: Any, stock: str) -> None:
            nonlocal blocked_exit_events, blocked_exit_notional
            if stock not in pending_sells:
                blocked_exit_events += 1
                pending_sell_since[stock] = pd.Timestamp(current_date).normalize()
                px = mark_price(current_date, stock)
                if px is not None and stock in sim_positions:
                    blocked_exit_notional += sim_positions[stock] * px
            pending_sells.add(stock)

        def finish_exit(current_date: Any, stock: str) -> None:
            started = pending_sell_since.pop(stock, None)
            if started is not None:
                completed_exit_delays.append(
                    max(0, int((pd.Timestamp(current_date).normalize() - started).days))
                )
            pending_sells.discard(stock)

        def cancel_exit(stock: str) -> None:
            pending_sells.discard(stock)
            pending_sell_since.pop(stock, None)

        def liquidate_targetless(current_date: Any, date_str: str) -> None:
            """Attempt an empty-target liquidation without inventing fills."""
            nonlocal sim_cash
            held = set(sim_positions)
            sellable = set(
                self._filter_tradeable_stocks(
                    date_str, sorted(held), side="sell", require_non_st=False
                )
            )
            for stock in sorted(held):
                sell_price = execution_price(current_date, stock)
                if stock not in sellable or sell_price is None:
                    weights.loc[current_date, stock] = np.nan
                    block_exit(current_date, stock)
                    continue
                weights.loc[current_date, stock] = 0.0
                sim_cash += sim_positions[stock] * sell_price * sell_proceeds_multiplier
                sim_positions.pop(stock, None)
                finish_exit(current_date, stock)

        for date in all_dates:
            date_normalized = pd.to_datetime(date).normalize()

            # A failed exit remains a real holding and consumes NAV.  Retry it
            # on every non-rebalance trading day; rebalance days reconsider the
            # T-1 target before deciding whether the exit is still desired.
            if pending_sells and date_normalized not in rebalance_set:
                date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
                sellable = set(
                    self._filter_tradeable_stocks(
                        date_str,
                        sorted(pending_sells),
                        side="sell",
                        require_non_st=False,
                    )
                )
                for stock in sorted(pending_sells & sellable):
                    sell_price = execution_price(date, stock)
                    if sell_price is None or stock not in sim_positions:
                        continue
                    weights.loc[date, stock] = 0.0
                    sim_cash += (
                        sim_positions[stock] * sell_price * sell_proceeds_multiplier
                    )
                    sim_positions.pop(stock, None)
                    finish_exit(date, stock)

            # [JQ-ALIGNED V4] 先检查 Skip Month 清仓日期
            if date_normalized in skip_month_liquidation_dates:
                date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
                held = set(sim_positions)
                weights.loc[date, :] = 0.0
                sellable = set(
                    self._filter_tradeable_stocks(
                        date_str, sorted(held), side="sell", require_non_st=False
                    )
                )
                for stock in sorted(held):
                    if stock not in sellable:
                        weights.loc[date, stock] = np.nan
                        block_exit(date, stock)
                        continue
                    sell_price = execution_price(date, stock)
                    if sell_price is None:
                        weights.loc[date, stock] = np.nan
                        block_exit(date, stock)
                        continue
                    weights.loc[date, stock] = 0.0
                    sim_cash += (
                        sim_positions[stock] * sell_price * sell_proceeds_multiplier
                    )
                    sim_positions.pop(stock, None)

                # self.logger.info(f"📅 Skip Month 清仓: {date_normalized.strftime('%Y-%m-%d')} - 清仓价值={liquidation_value:,.0f}, 现金={sim_cash:,.0f}")
                continue

            # Check if this is a rebalance date
            if date_normalized not in rebalance_set:
                continue

            # Check skip rules (fallback, should not reach here if skip_month_liquidation_dates works)
            if date_in_skip(date_normalized, self.skip_months):
                held = set(sim_positions)
                weights.loc[date, :] = 0.0
                sellable = set(
                    self._filter_tradeable_stocks(
                        pd.Timestamp(date).strftime("%Y-%m-%d"),
                        sorted(held),
                        side="sell",
                        require_non_st=False,
                    )
                )
                for stock in sorted(held):
                    sell_price = execution_price(date, stock)
                    if stock not in sellable or sell_price is None:
                        weights.loc[date, stock] = np.nan
                        block_exit(date, stock)
                        continue
                    weights.loc[date, stock] = 0.0
                    sim_cash += (
                        sim_positions[stock] * sell_price * sell_proceeds_multiplier
                    )
                    sim_positions.pop(stock, None)
                continue

            # Get valid stocks from dynamic pool
            if not sorted_pool_dates:
                valid_stocks = all_stocks_set
            else:
                pool_pos = bisect_right(sorted_pool_dates, date_normalized) - 1
                valid_stocks = (
                    normalized_pools[sorted_pool_dates[pool_pos]] & all_stocks_set
                    if pool_pos >= 0
                    else set()
                )

            # [ALIGNMENT] Enhanced pool logging - save pre-filter pool and per-filter removals
            date_str_log = pd.to_datetime(date).strftime("%Y-%m-%d")
            try:

                with open_diagnostic("local_pools.txt", "a", directory=self.diagnostics_output_dir, enabled=self.verbose_alignment or bool(self.diagnostics_output_dir)) as f:
                    f.write(f"\n{'=' * 60}\n")
                    f.write(f"DATE: {date_str_log}\n")
                    f.write(f"POOL_PRE_FILTER: {len(valid_stocks)}\n")
                    f.write(f"POOL_STOCKS: {','.join(sorted(valid_stocks))}\n")
            except Exception as e:
                self.logger.warning(f"Failed to dump pool: {e}")

            if not valid_stocks:
                liquidate_targetless(date, pd.Timestamp(date).strftime("%Y-%m-%d"))
                continue

            # [HIGH PRIORITY 2] Get factor values for valid stocks using T-1 data (avoid look-ahead bias)
            # Use database query to get previous trading day (align with JQ Local)
            date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
            if self.score_source == "input_factor":
                position = all_dates.get_loc(date)
                if isinstance(position, slice):
                    position = position.start
                prev_date = all_dates[int(position) - 1] if int(position) > 0 else None
                prev_date = (
                    pd.Timestamp(prev_date).strftime("%Y-%m-%d")
                    if prev_date is not None
                    else None
                )
            else:
                prev_date = self._backtest_previous_dates.get(
                    date_str
                ) or self._get_previous_trading_day(date_str)

            if prev_date is None:
                # First date or no previous trading day found, skip
                self.logger.debug(f"[VectorBT] No T-1 found for {date_str}, skipping")
                weights.loc[date, :] = 0.0
                continue
            prev_date_str = prev_date

            # [CLOUD-ALIGNED] Always use bundle-based synthesis when bundle is available
            raw_factors_df = None  # For logging

            self.logger.debug(
                f"[DEBUG SCORE] Date={date_str}, T-1={prev_date_str}, "
                f"Mode=BUNDLE_SYNTHESIS, bundle_path={self.bundle_path}"
            )

            score_series, processed_factors_df, raw_factors_df = (
                self._get_synthesis_score(prev_date_str, list(valid_stocks))
            )
            score_series = self._apply_score_decay(score_series, prev_date_str)

            # [DEBUG] Log score statistics for verification
            if len(score_series) > 0:
                self.logger.debug(
                    f"[DEBUG SCORE] Computed {len(score_series)} scores, "
                    f"mean={score_series.mean():.4f}, std={score_series.std():.4f}, "
                    f"top3={score_series.nlargest(3).to_dict()}"
                )

                # [DIAG] FACTOR_ALIGNMENT: 输出 Top10 composite scores 用于云端-本地对齐验证
                if self.verbose_alignment and len(score_series) > 0:
                    ranked_scores = score_series.sort_values(ascending=False)
                    top10 = ranked_scores.head(10)
                    top10_str = "|".join([f"{s}:{v:.6f}" for s, v in top10.items()])
                    self.logger.info(
                        f"FACTOR_ALIGNMENT|{date_str}|TOP10|n={len(score_series)}|{top10_str}"
                    )

            # Calculate effective top_k for early validation
            min_required = self._get_effective_top_k(len(score_series))

            if len(score_series) < min_required:
                # Not enough stocks with valid factor values
                self.logger.debug(
                    f"[VectorBT] {date}: Only {len(score_series)} valid stocks (T-1={prev_date}), need ~{min_required}"
                )
                if len(score_series) == 0:
                    liquidate_targetless(date, date_str)
                    continue

            # Rank stocks (ascending=False: higher factor value is better)
            ranked = score_series.reindex(rank_scores(score_series))

            # [DIAG] TOP10_FINAL 日志 (与云端对齐)
            if self.verbose_alignment:
                self.logger.info(
                    f"TOP10_FINAL|{date_str}|"
                    + "|".join(
                        [
                            f"{i + 1}:{s}:{score_series[s]:.6f}"
                            for i, s in enumerate(ranked.index[:10])
                        ]
                    )
                )

            # Track stocks removed by score filter (for alignment debugging)
            stocks_removed_by_score = set(valid_stocks) - set(score_series.index)

            # =====================================================
            # [REBALANCE DEBUG] 调仓日因子诊断日志
            # 输出每个调仓日的因子rank和raw值用于云端-本地对齐验证
            # =====================================================
            self._log_rebalance_debug(
                date_str=date_str,
                prev_date_str=prev_date_str,
                ranked=ranked,
                raw_factors_df=raw_factors_df,
                score_series=score_series,
            )

            # [T-DAY FILTER] A held name may remain selected while trading is
            # blocked.  A new name must be buyable.  Directional order checks
            # below decide whether an existing position can be resized/exited.
            # Controlled by config: enable_tday_filter (default: true)
            if self.enable_tday_filter:
                ranked_names = ranked.index.tolist()
                buyable = set(
                    self._filter_tradeable_stocks(
                        date_str, ranked_names, side="buy", require_non_st=True
                    )
                )
                held_eligible = set(
                    self._filter_tradeable_stocks(
                        date_str,
                        [s for s in ranked_names if s in sim_positions],
                        side="hold",
                        require_non_st=True,
                    )
                )
                tradeable_stocks = [
                    s for s in ranked_names if s in buyable or s in held_eligible
                ]
                stocks_removed_by_tday = set(ranked.index) - set(tradeable_stocks)
                if not self.pretrade_fixed_targets:
                    ranked = ranked.loc[
                        [s for s in ranked.index if s in tradeable_stocks]
                    ]
            else:
                # Skip T-day filter for alignment debugging
                tradeable_stocks = ranked.index.tolist()
                stocks_removed_by_tday = set()

            # Calculate effective top_k based on pool size (supports both top_k and top_pct)
            effective_top_k = self._get_effective_top_k(len(ranked))

            if len(ranked) < effective_top_k:
                self.logger.debug(
                    f"[VectorBT] {date_str}: Only {len(ranked)} tradeable stocks after T-day filter, "
                    f"effective_top_k={effective_top_k}"
                )

            strategy_weight_map: Optional[Dict[Any, float]] = None
            if self._strategy_plugin is not None:
                strategy_weight_map = self._strategy_plugin(ranked, effective_top_k)
                top_k_stocks = list(strategy_weight_map)
            elif self.all_equal_weight:
                top_k_stocks = ranked.index.tolist()
                effective_top_k = len(top_k_stocks)
            elif self.overlay_active:
                # [OVERLAY] Banding hysteresis + tranche turnover throttle on selection.
                top_k_stocks = self._select_with_overlays(
                    ranked, effective_top_k, set(sim_positions.keys())
                )
            else:
                top_k_stocks = ranked.head(effective_top_k).index.tolist()

            self._record_selection_diagnostic(
                date=date_normalized,
                pool_count=len(valid_stocks),
                valid_score_count=len(score_series),
                tradeable_count=len(tradeable_stocks),
                ranked=ranked,
                selected=top_k_stocks,
                previous_holdings=set(sim_positions),
            )

            # [ALIGNMENT] Append per-filter removal info to file
            try:
                with open_diagnostic("local_pools.txt", "a", directory=self.diagnostics_output_dir, enabled=self.verbose_alignment or bool(self.diagnostics_output_dir)) as f:
                    f.write(
                        f"FILTER_SCORE_REMOVED: {len(stocks_removed_by_score)} ({','.join(sorted(stocks_removed_by_score)) if stocks_removed_by_score else 'none'})\n"
                    )
                    f.write(
                        f"FILTER_TDAY_REMOVED: {len(stocks_removed_by_tday)} ({','.join(sorted(stocks_removed_by_tday)) if stocks_removed_by_tday else 'none'})\n"
                    )
                    f.write(f"FINAL_POOL: {len(ranked)}\n")
                    f.write(
                        f"EFFECTIVE_TOP_K: {effective_top_k} (top_k={self.top_k}, top_pct={self.top_pct})\n"
                    )
                    f.write(f"TOP_K_SELECTED: {','.join(top_k_stocks)}\n")
            except Exception:
                pass

            if self.overlay_active:
                # =====================================================
                # [OVERLAY] Full target-value rebalance with risk-aware
                # weights + per-rebalance exposure multiplier.
                # Every selected name is assigned an absolute target value so
                # VectorBT rebalances exactly to the desired weights each period.
                # =====================================================
                target_set = set(top_k_stocks)
                prev_holdings_set = set(sim_positions.keys())
                pending_sells.difference_update(target_set)
                for stock in target_set:
                    cancel_exit(stock)
                # Mark current book to market at the execution price.
                nav = sim_cash
                for s, shares in sorted(sim_positions.items(), key=lambda x: x[0]):
                    px = mark_price(date, s)
                    if px is not None:
                        nav += shares * px
                exposure = 1.0
                if exposure_by_date:
                    exposure = float(exposure_by_date.get(date_normalized, 1.0))
                # Current holdings as weight fractions (for the MVO turnover penalty).
                prev_weight_map: Dict[str, float] = {}
                if nav > 0:
                    for s, shares in sorted(sim_positions.items(), key=lambda x: x[0]):
                        px = mark_price(date, s)
                        if px is not None:
                            prev_weight_map[s] = shares * px / nav
                weight_map = (
                    strategy_weight_map
                    if strategy_weight_map is not None
                    else self._compute_overlay_weights(
                        top_k_stocks,
                        score_series,
                        price_df,
                        prev_date_str,
                        prev_weight_map,
                    )
                )
                if self.overlay["partial_rebalance"]["enabled"]:
                    # [OVERLAY] G-P partial adjustment: only move trade_rate of
                    # the way toward the fresh target. The traded book becomes
                    # the blended union of current holdings and targets.
                    weight_map = self._blend_partial_rebalance(
                        weight_map, prev_weight_map
                    )
                    target_set = set(weight_map)
                if self.overlay["liquidity_cap"]["enabled"]:
                    # Cap weights at the ADV participation limit against the
                    # capital actually deployed at this rebalance (nav * exposure)
                    # unless an explicit capital_base is configured.
                    weight_map = self._apply_liquidity_cap(
                        weight_map, prev_date_str, nav * exposure
                    )
                order_names = sorted(prev_holdings_set | target_set)
                buyable = set(
                    self._filter_tradeable_stocks(
                        date_str, order_names, side="buy", require_non_st=True
                    )
                )
                buyable = {
                    stock
                    for stock in buyable
                    if execution_price(date, stock) is not None
                }
                sellable = set(
                    self._filter_tradeable_stocks(
                        date_str, order_names, side="sell", require_non_st=False
                    )
                )
                sellable = {
                    stock
                    for stock in sellable
                    if execution_price(date, stock) is not None
                }
                if self.strict_max_actual_holdings:
                    executable_exits = (prev_holdings_set - target_set) & sellable
                    surviving_holdings = prev_holdings_set - executable_exits
                    ranked_new_targets = [
                        stock
                        for stock in top_k_stocks
                        if stock in target_set and stock not in prev_holdings_set
                    ]
                    available_slots = max(
                        0, effective_top_k - len(surviving_holdings)
                    )
                    allowed_new = set(ranked_new_targets[:available_slots])
                    skipped_new = set(ranked_new_targets) - allowed_new
                    strict_position_buys_skipped += len(skipped_new)
                    target_set.difference_update(skipped_new)
                    weight_map = {
                        stock: weight
                        for stock, weight in weight_map.items()
                        if stock in target_set
                    }
                    order_names = sorted(prev_holdings_set | target_set)
                current_values: Dict[str, float] = {}
                for stock in prev_holdings_set:
                    px = mark_price(date, stock)
                    if px is not None:
                        current_values[stock] = sim_positions[stock] * px
                base_capital = nav * exposure / buy_cost_multiplier
                desired = {
                    stock: weight_map.get(stock, 0.0) * base_capital
                    for stock in target_set
                }
                forced: Dict[str, float] = {}
                for stock in order_names:
                    current = current_values.get(stock, 0.0)
                    target = desired.get(stock, 0.0)
                    if stock not in target_set and stock not in sellable:
                        forced[stock] = current
                        block_exit(date, stock)
                    elif target > current and stock not in buyable:
                        forced[stock] = current
                    elif target < current and stock not in sellable:
                        forced[stock] = current
                if not self.pretrade_fixed_targets:
                    flexible_targets = [s for s in target_set if s not in forced]
                    available = max(0.0, base_capital - sum(forced.values()))
                    flexible_weight = sum(
                        weight_map.get(s, 0.0) for s in flexible_targets
                    )
                    if flexible_targets and flexible_weight > 0:
                        for stock in flexible_targets:
                            desired[stock] = (
                                available
                                * weight_map.get(stock, 0.0)
                                / flexible_weight
                            )

                for stock in order_names:
                    current = current_values.get(stock, 0.0)
                    target = forced.get(stock, desired.get(stock, 0.0))
                    if stock in forced:
                        weights.loc[date, stock] = np.nan
                        continue
                    px = mark_price(date, stock)
                    if px is None:
                        if stock not in target_set:
                            block_exit(date, stock)
                        continue
                    if target < current:
                        sim_cash += (current - target) * sell_proceeds_multiplier
                    elif target > current:
                        sim_cash -= (target - current) * buy_cost_multiplier
                    weights.loc[date, stock] = target
                    if target > 0:
                        sim_positions[stock] = target / px
                    else:
                        sim_positions.pop(stock, None)
                        finish_exit(date, stock)
                sim_cash = max(0.0, sim_cash)
                if self._alignment_writer is not None:
                    self._alignment_writer.add_rebalance_snapshot(
                        rebalance_date=date_str,
                        factor_date=prev_date_str,
                        stock_pool=valid_stocks,
                        raw_factors=raw_factors_df,
                        processed_factors=processed_factors_df,
                        composite_scores=score_series,
                        target_holdings=[s for s in top_k_stocks if s in target_set],
                        target_values={s: forced.get(s, desired.get(s, 0.0))
                                       for s in sorted(target_set | set(forced))},
                        pool_size=len(valid_stocks),
                        nav=nav / self.init_cash,
                    )
                continue

            # =====================================================
            # [JQ-ALIGNED V4] 预模拟 + targetvalue 逻辑
            #
            # 关键改动：
            # - 使用预模拟追踪持仓和现金
            # - 卖出股票：按开盘价计算释放的现金
            # - 买入股票：value = cash / num_to_buy (JQ 精确逻辑)
            # - 保持股票：NaN（VectorBT 保持当前股数）
            # =====================================================
            target_set = set(top_k_stocks)
            prev_holdings_set = set(sim_positions.keys())
            pending_sells.difference_update(target_set)
            for stock in target_set:
                cancel_exit(stock)

            # [FIX] 分类股票 - 排序以确保可重复性
            stocks_to_sell = sorted(list(prev_holdings_set - target_set))  # 需要卖出
            stocks_to_keep = sorted(list(prev_holdings_set & target_set))  # 继续持有
            stocks_to_buy = sorted(list(target_set - prev_holdings_set))  # 需要买入
            sellable = set(
                self._filter_tradeable_stocks(
                    date_str, stocks_to_sell, side="sell", require_non_st=False
                )
            )
            blocked_sells = set(stocks_to_sell) - sellable
            for stock in blocked_sells:
                block_exit(date, stock)

            # [JQ-ALIGNED V4] 模拟卖出：计算释放的现金。A tradeability
            # lookup alone is insufficient: VectorBT executes at open_df, so a
            # missing/non-positive next-open price must remain a blocked exit.
            released_cash = 0.0
            executed_sells: set[str] = set()
            for stock in stocks_to_sell:
                if stock not in sellable:
                    continue
                if stock in sim_positions and sim_positions[stock] > 0:
                    if (
                        stock not in exec_price_df.columns
                        or date not in exec_price_df.index
                    ):
                        block_exit(date, stock)
                        continue
                    sell_price = exec_price_df.loc[date, stock]
                    if pd.isna(sell_price) or sell_price <= 0:
                        block_exit(date, stock)
                        continue
                    sell_value = sim_positions[stock] * sell_price
                    net_sell_value = sell_value * sell_proceeds_multiplier
                    released_cash += net_sell_value
                    sim_cash += net_sell_value
                    del sim_positions[stock]
                    executed_sells.add(stock)
                    finish_exit(date, stock)

            # A blocked exit remains an actual holding and therefore consumes
            # one of the effective_top_k slots.  Under the strict contract,
            # buy only the highest-ranked new targets that fit after completed
            # sells; a later successful exit releases its slot for a future
            # rebalance rather than triggering an unscheduled replacement.
            if self.strict_max_actual_holdings and stocks_to_buy:
                ranked_new_targets = [
                    stock for stock in top_k_stocks if stock in stocks_to_buy
                ]
                available_slots = max(0, effective_top_k - len(sim_positions))
                skipped_count = max(0, len(ranked_new_targets) - available_slots)
                strict_position_buys_skipped += skipped_count
                stocks_to_buy = ranked_new_targets[:available_slots]

            intended_buy_count = len(stocks_to_buy)
            if (
                self.pretrade_fixed_targets
                and self.enable_tday_filter
                and stocks_to_buy
            ):
                buyable = set(
                    self._filter_tradeable_stocks(
                        date_str, stocks_to_buy, side="buy", require_non_st=True
                    )
                )
                stocks_to_buy = [
                    stock
                    for stock in stocks_to_buy
                    if stock in buyable and execution_price(date, stock) is not None
                ]

            # [JQ-ALIGNED V4] 计算每只新股的买入金额
            # JQ 逻辑: value = context.portfolio.cash / (target_num - position_count)
            if not prev_holdings_set:
                # 第一次建仓: 用全部现金均分
                value_per_new_stock = (
                    sim_cash / effective_top_k / buy_cost_multiplier
                    if effective_top_k > 0
                    else 0.0
                )
            elif stocks_to_buy:
                # 后续换仓: 用当前现金（释放的+剩余的）均分给新股
                # JQ: value = cash / num_to_buy
                value_per_new_stock = (
                    sim_cash
                    / (
                        intended_buy_count
                        if self.pretrade_fixed_targets
                        else len(stocks_to_buy)
                    )
                    / buy_cost_multiplier
                )
            else:
                value_per_new_stock = 0.0

            # [DEBUG] 详细日志
            # [DEBUG] 详细日志
            # self.logger.info(f"📅 Rebalance Date: {date_str} (T-1: {prev_date_str})")
            # self.logger.info(f"  Pool Size: {len(valid_stocks)} (Raw), {len(score_series)} (Valid Scores), {len(ranked)} (Tradeable)")
            # self.logger.info(f"  [JQ-ALIGNED V4] 预模拟状态:")
            # self.logger.info(f"    - 现金: {sim_cash:,.0f} (释放: {released_cash:,.0f})")
            # self.logger.info(f"    - 卖出: {list(stocks_to_sell)}")
            # self.logger.info(f"    - 保持: {list(stocks_to_keep)} (NaN)")
            # self.logger.info(f"    - 买入: {list(stocks_to_buy)} -> 每只 {value_per_new_stock:,.0f} 元")

            # 构建目标市值矩阵 (size_type='targetvalue')

            # 1. 只有按 next-open 成功预模拟的卖出设为 0；其余保持。
            for stock in stocks_to_sell:
                weights.loc[date, stock] = 0.0 if stock in executed_sells else np.nan

            # 2. 保持的股票：设为 NaN（不操作）
            for stock in stocks_to_keep:
                weights.loc[date, stock] = np.nan

            # 3. 新买入的股票：设置目标市值
            for stock in stocks_to_buy:
                weights.loc[date, stock] = value_per_new_stock
                # 模拟买入：扣除现金，增加持仓
                if stock in exec_price_df.columns and date in exec_price_df.index:
                    buy_price = exec_price_df.loc[date, stock]
                    if pd.notna(buy_price) and buy_price > 0:
                        shares = value_per_new_stock / buy_price
                        sim_positions[stock] = shares
                        sim_cash -= value_per_new_stock * buy_cost_multiplier

            # [ALIGNMENT] 记录持仓变动到日志文件
            try:
                with open_diagnostic("local_pools.txt", "a", directory=self.diagnostics_output_dir, enabled=self.verbose_alignment or bool(self.diagnostics_output_dir)) as f:
                    f.write(
                        f"JQ_ALIGNED_V4_SELL: {','.join(sorted(stocks_to_sell)) if stocks_to_sell else 'none'} (释放现金={released_cash:,.0f})\n"
                    )
                    f.write(
                        f"JQ_ALIGNED_V4_KEEP: {','.join(sorted(stocks_to_keep)) if stocks_to_keep else 'none'} (NaN)\n"
                    )
                    f.write(
                        f"JQ_ALIGNED_V4_BUY: {','.join(sorted(stocks_to_buy)) if stocks_to_buy else 'none'} (每只={value_per_new_stock:,.0f}元)\n"
                    )
            except Exception:
                pass

            # [ALIGNMENT DATA COLLECTION] Collect snapshot for alignment analysis
            if self._alignment_writer is not None:
                # Build actions dict
                actions = {}
                for s in stocks_to_sell:
                    actions[s] = "SELL"
                for s in stocks_to_keep:
                    actions[s] = "HOLD"
                for s in stocks_to_buy:
                    actions[s] = "BUY"

                # Build target values dict
                target_values = {}
                for s in top_k_stocks:
                    if s in stocks_to_buy:
                        target_values[s] = value_per_new_stock
                    else:
                        # For HOLD stocks, estimate current value
                        if (
                            s in sim_positions
                            and s in exec_price_df.columns
                            and date in exec_price_df.index
                        ):
                            price = exec_price_df.loc[date, s]
                            if pd.notna(price) and price > 0:
                                target_values[s] = sim_positions[s] * price

                # Calculate current NAV
                # [FIX] 排序sim_positions以确保可重复性
                current_nav = sim_cash
                for s, shares in sorted(sim_positions.items(), key=lambda x: x[0]):
                    if s in exec_price_df.columns and date in exec_price_df.index:
                        price = exec_price_df.loc[date, s]
                        if pd.notna(price) and price > 0:
                            current_nav += shares * price

                self._alignment_writer.add_rebalance_snapshot(
                    rebalance_date=date_str,
                    factor_date=prev_date_str,
                    stock_pool=valid_stocks,
                    raw_factors=raw_factors_df,
                    processed_factors=processed_factors_df.fillna(0.0)
                    if processed_factors_df is not None
                    else None,
                    composite_scores=score_series,
                    target_holdings=top_k_stocks,
                    target_values=target_values,
                    actions=actions,
                    pool_size=len(valid_stocks),
                    nav=current_nav / self.init_cash,  # Normalized NAV
                )

        # [JQ-ALIGNED V2] 不使用 ffill！
        # 非调仓日保持 NaN，VectorBT 会保持当前持仓不变
        # 只填充第一个调仓日之前的 NaN 为 0（初始无持仓）

        # 找到第一个有非 NaN 值的行
        has_values = weights.notna().any(axis=1)
        if has_values.any():
            first_valid_idx = has_values.idxmax()
            # 只填充第一个有效行之前的行（不包括第一个有效行本身）
            before_first = weights.index < first_valid_idx
            weights.loc[before_first] = 0.0

        # [DEBUG] 打印权重矩阵统计
        # self.logger.debug(f"[VectorBT] 权重矩阵 NaN 分布:")
        # self.logger.debug(f"  总单元格: {weights.shape[0] * weights.shape[1]}")
        # self.logger.debug(f"  非 NaN: {weights.notna().sum().sum()}")

        self._execution_diagnostics = {
            "strict_max_actual_holdings": self.strict_max_actual_holdings,
            "strict_position_buys_skipped": strict_position_buys_skipped,
            "blocked_exit_events": blocked_exit_events,
            "blocked_exit_notional": blocked_exit_notional,
            "completed_blocked_exits": len(completed_exit_delays),
            "open_blocked_exits_at_end": len(pending_sells),
            "average_calendar_days_to_exit": (
                float(np.mean(completed_exit_delays)) if completed_exit_delays else None
            ),
            "max_calendar_days_to_exit": (
                int(max(completed_exit_delays)) if completed_exit_delays else None
            ),
        }
        return weights

    def _get_synthesis_score(
        self, date_str: str, stock_list: List[str]
    ) -> Tuple[pd.Series, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        if self._synthesis_score_cache is None or self.score_source == "input_factor":
            return self._compute_synthesis_score(
                date_str, stock_list, return_raw_factors=True
            )

        key = (self.bundle_path, date_str, tuple(stock_list))
        cached = self._synthesis_score_cache.get(key)
        if cached is None:
            cached = self._compute_synthesis_score(
                date_str, stock_list, return_raw_factors=True
            )
            self._synthesis_score_cache[key] = tuple(
                value.copy() if value is not None else None for value in cached
            )
        return tuple(value.copy() if value is not None else None for value in cached)

    def _compute_synthesis_score(
        self, date_str: str, stock_list: List[str], return_raw_factors: bool = False
    ) -> Tuple[pd.Series, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Compute composite factor score by loading raw factors and applying preprocessing.

        [CLOUD-ALIGNED] Uses bundle-based model selection:
        - Calls _get_model_for_date() to get model for the given date
        - Extracts weights, feature_names, blend params from model_info
        - Matches cloud _get_lasso_model_info_for_date() + _select_stocks() logic

        Args:
            date_str: Date string in 'YYYY-MM-DD' format (T-1 date)
            stock_list: List of stock codes to score
            return_raw_factors: If True, also return processed and raw factor DataFrames

        Returns:
            Tuple of (composite score Series, processed factors DataFrame, raw factors DataFrame)
            When return_raw_factors is False, both DataFrames are None.
        """
        if self.score_source == "input_factor":
            source = getattr(self, "_input_factor_df", None)
            if source is None:
                return pd.Series(dtype=float), None, None
            try:
                row = source.loc[pd.Timestamp(date_str)]
            except KeyError:
                return pd.Series(dtype=float), None, None
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            scores = pd.to_numeric(row.reindex(stock_list), errors="coerce").dropna()
            scores = scores * self.score_direction
            return scores, None, None

        # [LightGBM Support] Check if this is a LightGBM bundle
        if self._is_lightgbm_bundle():
            return self._compute_lgb_prediction(
                date_str, stock_list, return_raw_factors
            )

        # [CLOUD-ALIGNED] Get model from bundle (Lasso path)
        model_info = self._get_model_for_date(date_str)

        if not model_info:
            self.logger.debug(
                f"[DEBUG _compute_synthesis_score] No model available for {date_str}"
            )
            return pd.Series(dtype=float), None, None

        # Extract replay parameters before opening the factor DB.  Parent-only
        # fallback windows need no raw factor data and must return the frozen
        # parent score exactly.
        weights = model_info.get("weights", {})
        preprocess = model_info.get("standardization_method", {})
        blend_ratio = float(model_info.get("blend_with_original_ratio", 0.0))
        blend_weights = model_info.get("blend_weights", {})
        is_blend_only = model_info.get("blend_only", False)
        parent_scores = self._parent_baseline_scores_for_model(
            model_info, date_str, stock_list
        )

        if (not weights or is_blend_only) and parent_scores is not None:
            if blend_ratio <= 0:
                raise ValueError(
                    "parent-only bundle model requires a positive blend ratio"
                )
            return parent_scores, None, None

        if not self.factor_db_conn and not self.factor_db_path:
            self.logger.debug(
                "[DEBUG _compute_synthesis_score] No factor_db_conn/path, returning empty"
            )
            return pd.Series(dtype=float), None, None

        # Lazy initialize only when uncached scores actually need raw factors.
        if self._factor_loader is None:
            if self.factor_db_conn is not None:
                self._factor_loader = FactorLoader(
                    existing_conn=self.factor_db_conn,
                    logger=self.logger,
                    shared_cache=self._factor_load_cache,
                )
            else:
                self._factor_loader = FactorLoader(
                    duckdb_path=self.factor_db_path,
                    logger=self.logger,
                    shared_cache=self._factor_load_cache,
                )
            dates = getattr(self, "_factor_replay_dates", [])
            if dates:
                try:
                    factors = set()
                    for model in self._bundle_models or []:
                        factors.update(model.get("weights", {}))
                        if float(model.get("blend_with_original_ratio", 0.0)) > 0 and not model.get("parent_baseline"):
                            factors.update(model.get("blend_weights", {}))
                    self._factor_loader.prefetch(dates, list(self._input_factor_df.columns), sorted(factors))
                except Exception as exc:
                    self.logger.warning("[FactorLoader] Batch read failed; using per-date queries: %s", exc)

        needed = set(weights) | (set(blend_weights) if blend_ratio > 0 and parent_scores is None else set())
        raw = self._factor_loader.load_factors(date_str, stock_list, sorted(needed)).reindex(stock_list)
        score, processed = linear_signal(raw, model_info, parent_scores)
        return (score, processed, raw) if return_raw_factors else (score, None, None)

    def _trading_calendar(self):
        from utils.trading_calendar import load_trading_calendar
        return load_trading_calendar(self.price_db_conn, duckdb_path=self.price_db_path)

    def _get_previous_trading_day(self, current_date_str: str) -> Optional[str]:
        return self._trading_calendar().previous_sessions(current_date_str, 1)[0].strftime('%Y-%m-%d')

    def _filter_tradeable_stocks(
        self,
        date_str: str,
        stock_list: List[str],
        *,
        side: str = "both",
        require_non_st: bool = True,
    ) -> List[str]:
        """
        Filter stocks by T-day directional tradeability conditions.

        JoinQuant Alignment:
        - Buy: limit-up blocks; limit-down does not.
        - Sell: limit-down blocks; limit-up does not.
        - Hold: requires an observable T-day row but no executable order.
        - Suspension (停牌): From price_data_daily (paused)
        - ST status: From factor_data (is_st)

        These checks must be on T-day (trade execution day), not T-1.

        Args:
            date_str: T-day date string in 'YYYY-MM-DD' format
            stock_list: List of stock codes to filter

        Returns:
            List of tradeable stock codes
        """
        if not stock_list:
            return []
        if side not in {"buy", "sell", "both", "hold"}:
            raise ValueError(f"unsupported trade side: {side}")
        cache_key = (
            "directional",
            date_str,
            tuple(sorted(stock_list)),
            side,
            require_non_st,
        )
        if (
            self._tradeability_cache is not None
            and cache_key in self._tradeability_cache
        ):
            return list(self._tradeability_cache[cache_key])

        conn = self.price_db_conn
        if conn is None and self.price_db_path:
            try:
                import duckdb
                from utils.data_store import connect_read_only as _connect_data_read_only

                conn = _connect_data_read_only(self.price_db_path)
                self._price_conn_owned = True
                self.price_db_conn = conn
            except Exception as exc:
                self.logger.warning(f"[VectorBT] Cannot connect to price DB: {exc}")
                return []
        if conn is None:
            # Callers without a market-status connection retain the historical
            # fallback.  Production/research paths provide the connection and
            # therefore fail closed on missing rows below.
            return stock_list

        row_cache = self._tradeability_cache
        unresolved = [
            stock
            for stock in stock_list
            if row_cache is None
            or ("directional_row", date_str, stock) not in row_cache
        ]

        try:
            if unresolved:
                stocks_str = "', '".join(unresolved)
                result_daily = conn.execute(
                    f"""
                    SELECT stock_code, open, high_limit, low_limit, paused
                    FROM price_data_daily
                    WHERE date = '{date_str}'
                      AND stock_code IN ('{stocks_str}')
                    """
                ).fetchdf()
                try:
                    result_st = conn.execute(
                        f"""
                        SELECT stock_code, is_st
                        FROM factor_data
                        WHERE date = '{date_str}'
                          AND stock_code IN ('{stocks_str}')
                        """
                    ).fetchdf()
                    if result_st.empty:
                        non_st = set(unresolved)
                    else:
                        normalized = (
                            result_st["is_st"].astype(str).str.strip().str.lower()
                        )
                        non_st = set(
                            result_st[
                                result_st["is_st"].isna()
                                | normalized.isin({"0", "false", "f", "no", "n", ""})
                            ]["stock_code"]
                        )
                except Exception as exc:
                    self.logger.debug(f"[VectorBT] ST查询失败: {exc}")
                    non_st = set(unresolved)

                daily_rows = {
                    str(row.stock_code): row
                    for row in result_daily.itertuples(index=False)
                }
                for stock in unresolved:
                    row = daily_rows.get(stock)
                    valid = bool(
                        row is not None
                        and pd.notna(row.open)
                        and float(row.open) > 0
                        and pd.notna(row.high_limit)
                        and pd.notna(row.low_limit)
                    )
                    active = (
                        valid and pd.notna(row.paused) and int(row.paused) == 0
                        if row is not None
                        else False
                    )
                    state = {
                        "buy": bool(active and float(row.open) < float(row.high_limit))
                        if valid
                        else False,
                        "sell": bool(active and float(row.open) > float(row.low_limit))
                        if valid
                        else False,
                        "both": bool(
                            active
                            and float(row.open) < float(row.high_limit)
                            and float(row.open) > float(row.low_limit)
                        )
                        if valid
                        else False,
                        "hold": valid,
                        "non_st": stock in non_st,
                    }
                    if row_cache is not None:
                        row_cache[("directional_row", date_str, stock)] = state

            allowed: List[str] = []
            for stock in stock_list:
                state = (
                    row_cache.get(("directional_row", date_str, stock))
                    if row_cache is not None
                    else None
                )
                if state is None:
                    # No shared cache: resolve this one request from the query
                    # results materialized above.
                    row = daily_rows.get(stock)
                    valid = bool(
                        row is not None
                        and pd.notna(row.open)
                        and float(row.open) > 0
                        and pd.notna(row.high_limit)
                        and pd.notna(row.low_limit)
                    )
                    active = (
                        valid and pd.notna(row.paused) and int(row.paused) == 0
                        if row is not None
                        else False
                    )
                    state = {
                        "buy": bool(active and float(row.open) < float(row.high_limit))
                        if valid
                        else False,
                        "sell": bool(active and float(row.open) > float(row.low_limit))
                        if valid
                        else False,
                        "both": bool(
                            active
                            and float(row.open) < float(row.high_limit)
                            and float(row.open) > float(row.low_limit)
                        )
                        if valid
                        else False,
                        "hold": valid,
                        "non_st": stock in non_st,
                    }
                if state[side] and (not require_non_st or state["non_st"]):
                    allowed.append(stock)

            if row_cache is not None:
                row_cache[cache_key] = list(allowed)
            return allowed
        except Exception as e:
            self.logger.warning(f"[VectorBT] T-day tradeability check failed: {e}")
            return []

    def _log_rebalance_debug(
        self,
        date_str: str,
        prev_date_str: str,
        ranked: pd.Series,
        raw_factors_df: Optional[pd.DataFrame],
        score_series: pd.Series,
    ) -> None:
        """
        输出调仓日的因子诊断日志，用于云端-本地对齐验证。

        日志格式:
        - FACTOR_ALIGNMENT: Top 10 股票及其合成分数
        - FACTOR_DETAIL: 每只股票的原始因子值、权重和贡献
        - BLEND_DEBUG: 混合因子计算过程（如果有）

        Args:
            date_str: 调仓日期 (T日)
            prev_date_str: 因子日期 (T-1日)
            ranked: 排序后的股票分数 Series
            raw_factors_df: 原始因子值 DataFrame (可选，synthesis mode)
            score_series: 合成分数 Series
        """
        # Skip verbose logging if not enabled
        if not self.verbose_alignment:
            return

        try:


            # =====================================================
            # FACTOR_ALIGNMENT: Top 10 股票及合成分数
            # =====================================================
            self.logger.info(
                f"FACTOR_ALIGNMENT|{date_str}|T-1={prev_date_str}|n_stocks={len(ranked)}"
            )
            self.logger.info(f"FACTOR_ALIGNMENT|{date_str}|TOP10_COMPOSITE_SCORES")

            top10 = ranked.head(10)
            for i, (stock, score_val) in enumerate(top10.items()):
                self.logger.info(f"  [{i + 1}] {stock}: CompositeScore={score_val:.6f}")

            self.logger.info(f"FACTOR_ALIGNMENT|{date_str}|END")

            # =====================================================
            # FACTOR_DETAIL: 每只股票的原始因子值和权重贡献
            # =====================================================
            if raw_factors_df is not None and not raw_factors_df.empty:
                factor_cols = list(raw_factors_df.columns)
                n_factors = len(factor_cols)

                self.logger.info(
                    f"FACTOR_DETAIL|{date_str}|START|n_factors={n_factors}"
                )
                self.logger.info(
                    f"FACTOR_DETAIL|{date_str}|FACTORS|{','.join(factor_cols[:10])}"
                )

                # 获取synthesis权重 (用于计算贡献) - from bundle
                # [FIX] 使用 T-1 日期选择模型，与实际计算逻辑一致
                weights = {}
                model_info = self._get_model_for_date(prev_date_str)
                if model_info:
                    weights = model_info.get("weights", {})
                    if not weights:
                        weights = model_info.get("blend_weights", {})

                # 输出 Top 5 股票的详细因子分解
                for i, stock in enumerate(top10.index[:5]):
                    if stock not in raw_factors_df.index:
                        continue

                    self.logger.info(
                        f"FACTOR_DETAIL|{date_str}|STOCK|{stock}|rank={i + 1}"
                    )

                    stock_factors = raw_factors_df.loc[stock]
                    total_contrib = 0.0

                    for factor_name in factor_cols[:8]:  # 限制输出前8个因子
                        raw_val = stock_factors.get(factor_name, float("nan"))
                        weight = weights.get(factor_name, 0.0)
                        contrib = raw_val * weight if pd.notna(raw_val) else 0.0
                        total_contrib += contrib

                        self.logger.info(
                            f"    FACTOR|{factor_name}|raw={raw_val:.6f}|weight={weight:.4f}|contrib={contrib:.6f}"
                        )

                    # 输出总贡献与最终分数对比
                    final_score = score_series.get(stock, float("nan"))
                    self.logger.info(
                        f"    TOTAL|weighted_sum={total_contrib:.6f}|final_score={final_score:.6f}"
                    )

                self.logger.info(f"FACTOR_DETAIL|{date_str}|END")

                # =====================================================
                # BLEND_DEBUG: 混合因子计算过程（如果有blend配置）
                # =====================================================
                if model_info:
                    blend_ratio = model_info.get("blend_with_original_ratio", 0.0)
                    blend_weights = model_info.get("blend_weights", {})

                    if blend_ratio > 0 and blend_weights:
                        self.logger.info(
                            f"BLEND_DEBUG|{date_str}|ratio={blend_ratio}|weights={blend_weights}"
                        )

                        # 输出前3只股票的blend计算
                        for i, stock in enumerate(top10.index[:3]):
                            if stock not in raw_factors_df.index:
                                continue

                            stock_factors = raw_factors_df.loc[stock]
                            blend_score = 0.0

                            for fname, w in blend_weights.items():
                                if fname in stock_factors.index:
                                    raw_val = stock_factors[fname]
                                    blend_score += raw_val * w

                            model_score = (
                                score_series.get(stock, 0.0) - blend_score * blend_ratio
                            )
                            final_score = score_series.get(stock, float("nan"))

                            self.logger.info(
                                f"BLEND_DEBUG|{date_str}|{stock}|"
                                f"model={model_score:.6f}|blend_factor={blend_score:.6f}|"
                                f"ratio={blend_ratio}|final={final_score:.6f}"
                            )

            # 同时写入文件（用于后续分析）
            with open_diagnostic("local_factor_values.txt", "a", directory=self.diagnostics_output_dir, enabled=self.verbose_alignment or bool(self.diagnostics_output_dir)) as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"REBALANCE_DATE: {date_str} (T-1: {prev_date_str})\n")
                f.write(f"POOL_SIZE: {len(ranked)}\n")
                f.write(f"FACTOR_ALIGNMENT|{date_str}|TOP10_COMPOSITE_SCORES\n")

                for i, (stock, score_val) in enumerate(top10.items()):
                    # Include raw factor values if available
                    if raw_factors_df is not None and stock in raw_factors_df.index:
                        factor_vals = raw_factors_df.loc[stock]
                        factor_str = "|".join(
                            [
                                f"{c}={factor_vals.get(c, float('nan')):.4f}"
                                for c in list(raw_factors_df.columns)[:5]
                            ]
                        )
                        f.write(
                            f"  [{i + 1}] {stock}: Score={score_val:.6f}|{factor_str}\n"
                        )
                    else:
                        f.write(f"  [{i + 1}] {stock}: Score={score_val:.6f}\n")

                f.write(f"FACTOR_ALIGNMENT|{date_str}|END\n")

        except Exception as e:
            self.logger.debug(f"[REBALANCE DEBUG] 日志记录失败: {e}")

    def close(self):
        """Close any open connections."""
        if self._factor_loader is not None:
            self._factor_loader.close()
            self._factor_loader = None

        # Close price connection if we own it
        if self._price_conn_owned and self.price_db_conn is not None:
            try:
                self.price_db_conn.close()
            except Exception:
                pass
            self.price_db_conn = None
            self._price_conn_owned = False

    def _simulate_portfolio(
        self,
        target_weights: pd.DataFrame,
        price_aligned: pd.DataFrame,
        open_aligned: Optional[pd.DataFrame],
        use_open: bool,
    ):
        """Run the VectorBT target-value simulation. Extracted for two-pass reuse."""
        if use_open:
            return vbt.Portfolio.from_orders(
                close=price_aligned,
                price=open_aligned,  # Execution price = open (not close)
                size=target_weights,
                size_type="targetvalue",
                fees=self.cost_rate,
                slippage=self.slippage,
                cash_sharing=True,
                call_seq="auto",
                freq="1D",
                init_cash=self.init_cash,
                log=self.portfolio_audit_enabled,
            )
        return vbt.Portfolio.from_orders(
            close=price_aligned,
            size=target_weights,
            size_type="targetvalue",
            fees=self.cost_rate,
            slippage=self.slippage,
            cash_sharing=True,
            call_seq="auto",
            freq="1D",
            init_cash=self.init_cash,
            log=self.portfolio_audit_enabled,
        )

    def _compute_turnover_metrics(
        self,
        portfolio,
        exposure_by_date: Optional[Dict[Any, float]],
    ) -> Dict[str, Any]:
        """Best-effort annualized one-way turnover + average exposure diagnostics."""
        metrics: Dict[str, Any] = {
            "annual_turnover": None,
            "avg_holding_count": None,
            "max_holding_count": None,
            "holding_count_violation_days": 0,
            "max_excess_holding_notional": 0.0,
            "holding_count_violation_dates": [],
            "holding_count_violation_details": [],
            "avg_exposure": None,
        }
        orders_frame: Optional[pd.DataFrame] = None
        try:
            orders_frame = portfolio.orders.records_readable
            traded_value = float(
                (orders_frame["Size"] * orders_frame["Price"]).abs().sum()
            )
            value_series = portfolio.value()
            if isinstance(value_series, pd.DataFrame):
                value_series = value_series.mean(axis=1)
            avg_value = float(value_series.mean())
            n_days = int(len(value_series))
            years = max(n_days / 252.0, 1e-9)
            if avg_value > 0:
                # buys ~= sells, so half of total traded value approximates one-way turnover
                metrics["annual_turnover"] = traded_value / avg_value / years / 2.0
        except Exception as e:
            self.logger.debug(f"[Overlay] turnover metric failed: {e}")
        try:
            assets = portfolio.asset_value(group_by=False)
            if isinstance(assets, pd.DataFrame):
                daily_count = (assets.abs() > 1e-9).sum(axis=1)
                # Average over invested days only; including liquidated skip-month
                # days would understate the actual book size (e.g. 7 -> 5.25).
                active = daily_count[daily_count > 0]
                metrics["avg_holding_count"] = (
                    float(active.mean()) if len(active) else 0.0
                )
                metrics["max_holding_count"] = (
                    int(daily_count.max()) if len(daily_count) else 0
                )
                if self.top_k is not None:
                    cap = max(0, int(self.top_k))
                    violations = daily_count[daily_count > cap]
                    metrics["holding_count_violation_days"] = int(len(violations))
                    metrics["holding_count_violation_dates"] = [
                        pd.Timestamp(value).strftime("%Y-%m-%d")
                        for value in violations.index[:20]
                    ]
                    if len(violations):
                        excess_values = assets.abs().apply(
                            lambda row: float(
                                row.sort_values(ascending=False).iloc[cap:].sum()
                            ),
                            axis=1,
                        )
                        metrics["max_excess_holding_notional"] = float(
                            excess_values.loc[violations.index].max()
                        )
                        details: List[Dict[str, Any]] = []
                        for violation_date in violations.index[:3]:
                            values = assets.loc[violation_date].abs()
                            values = values[values > 1e-9].sort_values(ascending=False)
                            excess_names = list(values.index[cap:])
                            order_history: List[Dict[str, Any]] = []
                            if orders_frame is not None and not orders_frame.empty:
                                for excess_name in excess_names:
                                    normalized_name = (
                                        excess_name[-1]
                                        if isinstance(excess_name, tuple)
                                        else excess_name
                                    )
                                    column_labels = {
                                        str(excess_name),
                                        str(normalized_name),
                                    }
                                    matching = orders_frame[
                                        orders_frame["Column"]
                                        .astype(str)
                                        .isin(column_labels)
                                        & (
                                            pd.to_datetime(orders_frame["Timestamp"])
                                            <= pd.Timestamp(violation_date)
                                        )
                                    ].tail(5)
                                    for record in matching.to_dict(orient="records"):
                                        order_history.append(
                                            {
                                                str(key): (
                                                    pd.Timestamp(value).strftime(
                                                        "%Y-%m-%d"
                                                    )
                                                    if isinstance(value, pd.Timestamp)
                                                    else value
                                                )
                                                for key, value in record.items()
                                            }
                                        )
                            details.append(
                                {
                                    "date": pd.Timestamp(violation_date).strftime(
                                        "%Y-%m-%d"
                                    ),
                                    "holdings": {
                                        str(stock): float(value)
                                        for stock, value in values.items()
                                    },
                                    "excess_names": [
                                        str(stock) for stock in excess_names
                                    ],
                                    "excess_order_history": order_history,
                                }
                            )
                        metrics["holding_count_violation_details"] = details
        except Exception as e:
            self.logger.debug(f"[Overlay] holding-count metric failed: {e}")
        if exposure_by_date:
            values = [float(v) for v in exposure_by_date.values()]
            if values:
                metrics["avg_exposure"] = float(np.mean(values))
        elif self.overlay_active:
            metrics["avg_exposure"] = 1.0
        return metrics

    def _extract_results(
        self, portfolio, exposure_by_date: Optional[Dict[Any, float]] = None
    ) -> Dict[str, Any]:
        """
        Extract performance metrics from VectorBT portfolio.

        Uses VectorBT's native methods for accurate metric calculation.
        With freq='1B', ann_factor=252 (business day convention).

        Returns:
            Dict with key metrics
        """

        def _scalar(val: Any, default: float = 0.0) -> float:
            """Extract scalar from VectorBT result (may be Series or scalar)."""
            if isinstance(val, pd.Series):
                val = val.iloc[0] if len(val) > 0 else default
            return float(val) if not pd.isna(val) else default

        try:
            # Core metrics - VectorBT native methods
            # year_freq='252D' corrects annualization: freq='1D' uses ann_factor=365.25
            # but data only has trading days (~244/yr). year_freq='252D' tells VectorBT
            # "1 year = 252 data points", matching the business-day convention.
            total_ret = _scalar(portfolio.total_return())
            annual_ret = _scalar(portfolio.annualized_return(year_freq="252D"))
            sharpe = _scalar(portfolio.sharpe_ratio(year_freq="252D"), default=np.nan)
            max_dd = _scalar(portfolio.max_drawdown())

            # Additional risk-adjusted metrics
            try:
                calmar = _scalar(
                    portfolio.calmar_ratio(year_freq="252D"), default=np.nan
                )
            except Exception:
                calmar = np.nan
            try:
                sortino = _scalar(
                    portfolio.sortino_ratio(year_freq="252D"), default=np.nan
                )
            except Exception:
                sortino = np.nan

            # Annual returns breakdown - VectorBT native
            annual_rets: Dict[int, float] = {}
            try:
                ar = portfolio.annual_returns()
                if isinstance(ar, pd.Series) and not ar.empty:
                    for k, v in ar.items():
                        # Key may be Timestamp or int depending on VectorBT version
                        year = k.year if hasattr(k, "year") else int(k)
                        annual_rets[year] = float(v)
                else:
                    self.logger.warning(
                        f"[VectorBT] annual_returns() returned empty: type={type(ar)}, value={ar}"
                    )
            except Exception as e:
                self.logger.warning(f"[VectorBT] annual_returns() failed: {e}")

            # Trade statistics
            trade_count = 0
            win_rate = 0.0
            try:
                trades = portfolio.trades
                tc = trades.count() if hasattr(trades, "count") else 0
                wr = trades.win_rate() if hasattr(trades, "win_rate") else 0.0
                trade_count = int(_scalar(tc, default=0))
                win_rate = _scalar(wr)
            except Exception:
                pass

            return_series_artifact = self._write_return_series_artifact(portfolio)

            results = {
                "annual_return": float(annual_ret),
                "total_return": float(total_ret),
                "sharpe_ratio": float(sharpe) if not np.isnan(sharpe) else 0.0,
                "calmar_ratio": float(calmar) if not np.isnan(calmar) else 0.0,
                "sortino_ratio": float(sortino) if not np.isnan(sortino) else 0.0,
                "max_drawdown": float(max_dd),
                "win_rate": float(win_rate),
                "trade_count": trade_count,
                "annual_returns": annual_rets,
                "portfolio": portfolio,
                "top_pct": self.top_pct,
                "top_k": self.top_k,
                "return_series_artifact": return_series_artifact,
            }
            # [OVERLAY] Turnover / exposure diagnostics + overlay echo.
            results.update(self._compute_turnover_metrics(portfolio, exposure_by_date))
            if self.overlay_active:
                results["overlay"] = self._overlay_summary(self.overlay)
            return results

        except Exception as e:
            self.logger.error(f"[VectorBT] Failed to extract results: {e}")
            return {
                "error": str(e),
                "portfolio": portfolio,
            }

    def _write_return_series_artifact(self, portfolio) -> Optional[Dict[str, Any]]:
        """Write per-period portfolio returns as CSV plus metadata."""
        if not self.return_series_output_dir or not self.return_series_trial_id:
            return None
        return self._write_return_series_artifact_body(portfolio)

    def _write_portfolio_audit_artifacts(
        self,
        *,
        portfolio,
        target_values: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        """Export the order instructions, fills, and realized daily book."""
        if not self.diagnostics_output_dir or not self.return_series_trial_id:
            return None
        try:
            safe_trial_id = "".join(
                ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
                for ch in str(self.return_series_trial_id)
            ).strip("._") or "trial"
            output_dir = self.diagnostics_output_dir / "portfolio_audit"
            output_dir.mkdir(parents=True, exist_ok=True)

            value = portfolio.value()
            if isinstance(value, pd.DataFrame):
                value = value.mean(axis=1)
            value = pd.Series(value)
            value.index = pd.to_datetime(value.index)

            requested = target_values.stack(dropna=True).rename("target_value").reset_index()
            requested.columns = ["date", "stock_code", "target_value"]
            requested["date"] = pd.to_datetime(requested["date"])
            requested["portfolio_value"] = requested["date"].map(value)
            requested["target_weight"] = requested["target_value"] / requested["portfolio_value"]
            requested_path = output_dir / f"{safe_trial_id}.target_instructions.csv"
            requested.to_csv(requested_path, index=False)

            assets = portfolio.asset_value(group_by=False)
            if isinstance(assets, pd.Series):
                assets = assets.to_frame()
            assets = assets.copy()
            assets.index = pd.to_datetime(assets.index)
            assets.columns = [
                str(column[-1] if isinstance(column, tuple) else column)
                for column in assets.columns
            ]
            actual = assets.div(value.reindex(assets.index), axis=0)
            actual.index.name = "date"
            actual = actual.stack().rename("actual_weight").reset_index()
            actual.columns = ["date", "stock_code", "actual_weight"]
            actual = actual[actual["actual_weight"].abs() > 1e-12]
            actual_path = output_dir / f"{safe_trial_id}.actual_weights.csv"
            actual.to_csv(actual_path, index=False)

            orders = portfolio.orders.records_readable.copy()
            orders["stock_code"] = orders["Column"].map(
                lambda column: str(column[-1] if isinstance(column, tuple) else column)
            )
            orders["gross_notional"] = (
                pd.to_numeric(orders["Size"], errors="coerce")
                * pd.to_numeric(orders["Price"], errors="coerce")
            ).abs()
            orders["modeled_price_impact"] = orders["gross_notional"] * max(
                float(self.slippage), 0.0
            )
            orders_path = output_dir / f"{safe_trial_id}.orders.csv"
            orders.to_csv(orders_path, index=False)
            logs = portfolio.logs.records_readable.copy()
            logs = logs[logs["Result Status"].isin(["Filled", "Rejected"])].copy()
            logs_path = output_dir / f"{safe_trial_id}.execution_logs.csv"
            logs.to_csv(logs_path, index=False)
            rejected = logs[logs["Result Status"] == "Rejected"].copy()
            rejected["unfilled_notional"] = (
                pd.to_numeric(rejected["Request Size"], errors="coerce")
                - (
                    pd.to_numeric(rejected["Position"], errors="coerce")
                    * pd.to_numeric(rejected["Val Price"], errors="coerce")
                )
            ).abs()
            status_counts = {
                str(status): int(count)
                for status, count in logs["Result Status"].value_counts().items()
            }
            mvo_path = output_dir / f"{safe_trial_id}.mvo_diagnostics.json"
            mvo_path.write_text(
                json.dumps(self._mvo_diagnostics, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )

            manifest = {
                "schema_version": "1.0",
                "target_semantics": "daily target-value instructions; NaN means hold",
                "actual_weight_semantics": "close-valued realized asset weight",
                "order_count": int(len(orders)),
                "execution_log_count": int(len(logs)),
                "unfilled_order_count": int(len(rejected)),
                "unfilled_notional": float(rejected["unfilled_notional"].sum()),
                "execution_status_counts": status_counts,
                "blocked_exit_events": int(
                    self._execution_diagnostics.get("blocked_exit_events", 0) or 0
                ),
                "slippage_rate": float(self.slippage),
                "target_instructions_path": str(requested_path),
                "actual_weights_path": str(actual_path),
                "orders_path": str(orders_path),
                "execution_logs_path": str(logs_path),
                "mvo_diagnostics_path": str(mvo_path),
            }
            manifest_path = output_dir / f"{safe_trial_id}.manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            from utils.data_store import publish_outputs
            manifest["data_refs"] = publish_outputs({
                "backtest.requested_targets": requested_path, "backtest.actual_weights": actual_path,
                "backtest.orders": orders_path, "backtest.logs": logs_path, "meta.audit_manifest": manifest_path,
            })
            from utils.local_outputs import write_json
            write_json(manifest_path, manifest)
            return {**manifest, "manifest_path": str(manifest_path)}
        except Exception as exc:
            self.logger.warning(
                f"[VectorBT] portfolio audit artifact export failed: {exc}"
            )
            from utils.data_store import binding_contract
            if binding_contract().get("output_db"):
                raise
            return None

    def _write_return_series_artifact_body(
        self, portfolio
    ) -> Optional[Dict[str, Any]]:
        try:
            returns = portfolio.returns()
            if isinstance(returns, pd.DataFrame):
                if returns.empty:
                    return None
                if returns.shape[1] == 1:
                    series = returns.iloc[:, 0]
                else:
                    series = returns.mean(axis=1)
            else:
                series = pd.Series(returns)

            series.index = pd.to_datetime(series.index)
            series = pd.to_numeric(series, errors="coerce").dropna()
            if series.empty:
                return None

            self.return_series_output_dir.mkdir(parents=True, exist_ok=True)
            safe_trial_id = "".join(
                ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
                for ch in str(self.return_series_trial_id)
            ).strip("._")
            if not safe_trial_id:
                safe_trial_id = "trial"
            csv_path = self.return_series_output_dir / f"{safe_trial_id}.csv"
            metadata_path = (
                self.return_series_output_dir / f"{safe_trial_id}.metadata.json"
            )

            frame = pd.DataFrame(
                {
                    "date": series.index.strftime("%Y-%m-%d"),
                    "net_return": series.astype(float).values,
                }
            )
            frame.to_csv(csv_path, index=False)
            sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()

            metadata: Dict[str, Any] = {
                "schema_version": "1.0",
                "run_id": self.return_series_metadata.get("run_id", ""),
                "trial_id": safe_trial_id,
                "selected_config_path": self.return_series_metadata.get(
                    "selected_config_path", ""
                ),
                "selection_mode": "top_pct" if self.top_pct is not None else "top_k",
                "top_pct": self.top_pct,
                "top_k": self.top_k if self.top_pct is None else None,
                "start_date": frame["date"].iloc[0],
                "end_date": frame["date"].iloc[-1],
                "row_count": int(len(frame)),
                "sha256": sha256,
                "created_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "producer": "factor_synthesis.vectorbt_adapter",
            }
            metadata.update(
                {
                    key: value
                    for key, value in self.return_series_metadata.items()
                    if key not in metadata and value is not None
                }
            )
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            from utils.data_store import publish_outputs
            refs = publish_outputs({"backtest.returns": csv_path})
            metadata["data_refs"] = refs
            from utils.local_outputs import write_json
            write_json(metadata_path, metadata)
            refs = {**refs, **publish_outputs({"meta.return_metadata": metadata_path})}
            return {
                "data_refs": refs,
                "artifact_type": "algorithm_return_series",
                "format": "csv",
                "path": str(csv_path),
                "metadata_path": str(metadata_path),
                "trial_id": safe_trial_id,
                "selection_mode": metadata["selection_mode"],
                "top_pct": metadata["top_pct"],
                "top_k": metadata["top_k"],
                "frequency": "daily",
                "return_column": "net_return",
            }
        except Exception as exc:
            self.logger.warning(
                f"[VectorBT] return series artifact export failed: {exc}"
            )
            from utils.data_store import binding_contract
            if binding_contract().get("output_db"):
                raise
            return None

    def format_results(self, results: Dict[str, Any]) -> str:
        """
        Format results for logging. Includes all key metrics and yearly breakdown.

        Args:
            results: Results dict from run_backtest

        Returns:
            Formatted string for logging
        """
        if "error" in results:
            return f"📊 VectorBT 回测失败: {results['error']}"

        # Build header with top_pct/top_k label for multi-run distinguishability
        top_pct = results.get("top_pct")
        top_k = results.get("top_k")
        if top_pct is not None:
            header = f"📊 VectorBT 回测结果 (top_pct={top_pct}):"
        elif top_k is not None:
            header = f"📊 VectorBT 回测结果 (top_k={top_k}):"
        else:
            header = "📊 VectorBT 回测结果:"

        lines = [
            header,
            f"  年化收益: {results.get('annual_return', 0) * 100:.1f}%",
            f"  夏普比率: {results.get('sharpe_ratio', 0):.2f}",
            f"  Calmar比率: {results.get('calmar_ratio', 0):.2f}",
            f"  Sortino比率: {results.get('sortino_ratio', 0):.2f}",
            f"  最大回撤: {results.get('max_drawdown', 0) * 100:.1f}%",
            f"  总收益: {results.get('total_return', 0) * 100:.1f}%",
            f"  交易次数: {results.get('trade_count', 0):.0f}",
        ]

        # Annual returns breakdown (from VectorBT native annual_returns())
        annual_rets = results.get("annual_returns", {})
        if annual_rets:
            lines.append("📅 年度收益分解:")
            for year in sorted(annual_rets.keys()):
                lines.append(f"  {year}: {annual_rets[year] * 100:.1f}%")

        return "\n".join(lines)
