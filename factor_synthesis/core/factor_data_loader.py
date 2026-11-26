"""
Factor Data Loader Module

Data loading utilities for factor synthesis, including:
- Factor data loading with Parquet cache support
- Price data retrieval for analysis
- Market cap data loading for neutralization
- PIT (Point-In-Time) dynamic adjustment support

Extracted from FactorSynthesizer for modularity and reuse.

Author: QuantationAI
Date: 2026-01-12
Updated: 2026-01-13 - Added PIT adjustment support
"""

import hashlib
import logging
import os
from tempfile import NamedTemporaryFile
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from utils.market_factor import MarketFactorValueStore
from utils.market_factor.schema import FactorReadPlan
from utils.slots import MarketSlot
from ..factor_reader import (
    DEFAULT_SLOT_ID, market_factor_inputs, plan_market_factor_values,
    read_market_factor_values, resolve_market_input,
)
from .calendar_skip import SkipRules, skip_mask

logger = logging.getLogger(__name__)


def _sql_literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_in(values: List[str]) -> str:
    return ", ".join(_sql_literal(value) for value in values)


def _empty_factor_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "stock_code", "factor_name", "factor_value"])


def _read_factor_rows(
    conn,
    select_clause: str,
    table_name: str,
    where_clause: str,
    stock_filter: Optional[Set[str]],
    log: logging.Logger,
) -> pd.DataFrame:
    if stock_filter and len(stock_filter) > 0:
        log.info(f"[内存优化] 使用SQL级别股票过滤，池内股票数: {len(stock_filter)}")
        chunk_size = 500
        # Dynamic pools originate as sets.  A canonical order keeps chunk SQL,
        # query plans, and failures reproducible across identical replays.
        stock_list = sorted(stock_filter)
        all_dfs = []

        for i in range(0, len(stock_list), chunk_size):
            chunk = stock_list[i:i + chunk_size]
            log.info(
                "[内存优化] 加载因子股票块 %d-%d/%d",
                i + 1,
                min(i + chunk_size, len(stock_list)),
                len(stock_list),
            )
            sql = f"""
            {select_clause}
            FROM {table_name}
            WHERE {where_clause}
            AND stock_code IN ({_sql_in(chunk)})
            ORDER BY date, stock_code, factor_name
            """
            chunk_df = conn.execute(sql).fetchdf()
            if not chunk_df.empty:
                all_dfs.append(chunk_df)

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return _empty_factor_rows()

    sql = f"""
    {select_clause}
    FROM {table_name}
    WHERE {where_clause}
    ORDER BY date, stock_code, factor_name
    """
    return conn.execute(sql).fetchdf()


# =============================================================================
# PIT Adjustment Helpers
# =============================================================================


class PITValidationError(ValueError):
    """Enabled PIT cannot produce a validated adjusted price panel."""


def _pit_dates(index, context):
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans or str(index.tz) not in ("None", "UTC"):
        raise PITValidationError(f"PIT {context}: dates must be finite local dates or UTC timestamps")


def _pit_series_index(series, context):
    index = series.index
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise PITValidationError(f"PIT {context}: expected (asset, date) or (stock_code, date) index")
    asset = 'asset' if 'asset' in index.names else 'stock_code'
    if set(index.names) != {asset, 'date'}:
        raise PITValidationError(f"PIT {context}: index must name the stock and date levels")
    dates, stocks = index.get_level_values('date'), index.get_level_values(asset)
    _pit_dates(dates, context)
    if stocks.isna().any():
        raise PITValidationError(f"PIT {context}: missing stock keys")
    canonical = pd.MultiIndex.from_arrays([stocks, dates], names=['asset', 'date'])
    if canonical.has_duplicates:
        raise PITValidationError(f"PIT {context}: duplicate stock/date keys")
    return canonical


def _pit_wide_axes(frame, context):
    _pit_dates(frame.index, context)
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise PITValidationError(f"PIT {context}: duplicate dates or stock columns")
    if isinstance(frame.columns, pd.MultiIndex) or frame.columns.hasnans:
        raise PITValidationError(f"PIT {context}: expected non-null stock columns")


def _pit_bad_values(frame, bad, reason):
    count = np.count_nonzero(bad)
    if not count:
        return
    examples = []
    # Scan bounded slices: do not allocate coordinates for every invalid cell,
    # or flatten-copy a non-contiguous mask. Keep row-major sample order.
    for start in range(0, bad.size, 4096):
        offsets = np.flatnonzero(bad.flat[start:start + 4096])
        for offset in offsets[:5 - len(examples)]:
            position = start + int(offset)
            if isinstance(frame, pd.Series):
                key = str(frame.index[position])
                value = frame.iloc[position]
            else:
                row, column = divmod(position, bad.shape[1])
                key = f"{frame.columns[column]!r} / {frame.index[row]}"
                value = frame.iloc[row, column]
            examples.append(f"{key}: value={value!r}")
        if len(examples) == 5:
            break
    raise PITValidationError(f"PIT {reason}: {count} invalid observations; " + '; '.join(examples))


def _pit_numeric(frame, observed, role, *, positive=False):
    """Validate observed cells only; missing-price cells cannot supply anchors."""
    raw = frame.to_numpy()
    if raw.dtype.kind in 'iuf':
        numeric = raw.astype(float)
        type_ok = np.ones(raw.shape, dtype=bool)
    else:
        from numbers import Real
        type_ok = np.array([isinstance(v, Real) and not isinstance(v, (bool, np.bool_))
                            for v in raw.flat]).reshape(raw.shape)
        numeric = np.full(raw.shape, np.nan)
        valid = type_ok & observed
        if valid.any():
            numeric[valid] = raw[valid].astype(float)
    invalid = ~type_ok | ~np.isfinite(numeric)
    if positive:
        invalid |= numeric <= 0
    _pit_bad_values(frame, observed & invalid, f"invalid {role}")
    numeric = np.where(observed, numeric, np.nan)
    if isinstance(frame, pd.Series):
        return pd.Series(numeric, index=frame.index, name=frame.name)
    return pd.DataFrame(numeric, index=frame.index, columns=frame.columns)


def _validated_pit_factors(prices, aligned_factors):
    observed = prices.notna().to_numpy()
    _pit_numeric(prices, observed, 'price')
    return _pit_numeric(aligned_factors, observed, 'factor (finite positive number required)', positive=True)


def _pit_adjusted_prices(prices, factors, anchors):
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        ratios = factors / anchors if isinstance(prices, pd.Series) else factors.div(anchors, axis=1)
        result = prices * ratios
    observed = prices.notna().to_numpy()
    _pit_numeric(ratios, observed, 'factor ratio', positive=True)
    _pit_numeric(result, observed, 'adjusted price')
    return result


def validate_pit_price_rows(frame, price_column):
    """Validate before intraday aggregation can discard an invalid factor/key."""
    required = {'stock_code', 'date', price_column, 'factor'}
    missing = required - set(frame.columns)
    if missing:
        raise PITValidationError(f"PIT missing columns: {sorted(missing)}")
    if frame.empty:
        return
    try:
        dates = pd.to_datetime(frame.date)
    except (TypeError, ValueError) as exc:
        raise PITValidationError("PIT price rows: invalid dates") from exc
    index = pd.MultiIndex.from_arrays([frame.stock_code, dates],
                                     names=['asset', 'date'])
    prices = pd.Series(frame[price_column].to_numpy(), index=index)
    _pit_series_index(prices, 'price rows')
    factors = pd.Series(frame.factor.to_numpy(), index=index)
    _validated_pit_factors(prices, factors)


def apply_pit_adjustment_factor_end(
    price_series: pd.Series,
    factor_series: pd.Series,
    custom_logger: Optional[logging.Logger] = None,
) -> pd.Series:
    """Adjust by each stock's latest observed-date factor; preserve input order.

    Requires unique named stock/date keys and finite positive numeric factors
    wherever prices exist. Missing prices stay missing and cannot supply an
    anchor. Invalid inputs raise PITValidationError; no unadjusted fallback.
    """
    if price_series.empty:
        return price_series
    index = _pit_series_index(price_series, 'prices')
    prices = price_series.copy()
    prices.index = index
    if factor_series.empty:
        aligned = pd.Series(np.nan, index=index)
    else:
        factors = factor_series.copy()
        factors.index = _pit_series_index(factors, 'factors')
        aligned = factors.reindex(index)
    factors = _validated_pit_factors(prices, aligned)
    chronological = factors.sort_index(level='date')
    anchors = chronological.groupby(level='asset').transform('last').reindex(index)
    result = _pit_adjusted_prices(prices, factors, anchors)
    result.index = price_series.index
    result.name = price_series.name
    result.attrs = price_series.attrs.copy()
    (custom_logger or logger).info(f"[PIT] 已应用 factor_end 复权，处理 {len(result)} 条价格记录")
    return result


def apply_pit_adjustment_factor_end_df(
    price_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    custom_logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Adjust a wide panel using each stock's latest observed-price anchor.

    Missing-price cells remain missing. Dates and stock columns must be unique;
    factors at observed prices must be finite positive numbers. Failures raise.
    """
    if price_df.empty:
        return price_df
    _pit_wide_axes(price_df, 'prices')
    if not factor_df.empty:
        _pit_wide_axes(factor_df, 'factors')
    factors = _validated_pit_factors(price_df, factor_df.reindex_like(price_df))
    anchors = factors.sort_index().ffill().iloc[-1]
    result = _pit_adjusted_prices(price_df, factors, anchors)
    result.attrs = price_df.attrs.copy()
    (custom_logger or logger).info(f"[PIT] 已应用 factor_end 复权，处理 {len(result)} 日期 × {len(result.columns)} 股票")
    return result


# =============================================================================
# Factor Data Loading
# =============================================================================

def _write_factor_cache(frame: pd.DataFrame, target: str, *, index: bool | None = None) -> None:
    path = Path(target)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        frame.to_parquet(pending, index=index)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _with_market_factor_inputs(
    frame: pd.DataFrame, identities: dict[str, str], plan: FactorReadPlan | None,
    slot_id: str, frequency: str,
) -> pd.DataFrame:
    if plan is not None:
        frame.attrs["market_factor_inputs"] = market_factor_inputs(
            identities, plan, slot_id, frequency,
        )
    return frame


PANEL_DATE_ALIGNMENT_ATTR = "panel_date_alignment"
PANEL_DATE_ALIGNMENT_TRADE_DATE = "trade_date_aligned"
PANEL_DATE_ALIGNMENT_RAW = "raw_signal"


def load_factor_data(
    factor_conn,
    factor_table: str,
    factor_names: List[str],
    start_date: str,
    end_date: str,
    stock_filter: Optional[Set[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    price_conn=None,
    price_table: Optional[str] = None,
    neutralization_config: Optional[Dict[str, Any]] = None,
    custom_logger: Optional[logging.Logger] = None,
    *,
    factor_database_path: str | Path | None = None,
    slot: MarketSlot | None = None,
    slot_id: str | None = None,
    frequency: str | None = None,
    factor_values_store: MarketFactorValueStore | str | Path | None = None,
    factor_identities: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    加载方案 B 因子面板并按插槽网格进行 T−1 对齐。

    Args:
        factor_conn: 冻结制品所用的只读 DuckDB 连接
        factor_table: 冻结制品的逻辑表名
        factor_names: 因子名称列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        stock_filter: 可选的股票代码集合，用于SQL级别过滤以减少内存占用
        config: 配置字典，包含 cache_settings、missing_data_strategy 等
        price_conn: 价格数据库连接（用于获取交易日历）
        price_table: 价格表名
        neutralization_config: 插槽中性化配置。冻结绑定沿用已计算的制品。
        custom_logger: 可选的自定义 logger
        factor_database_path: 默认方案 B 目录所对应的 DuckDB 路径

    Returns:
        因子数据DataFrame，MultiIndex (date, stock_code)
    """
    from utils.local_outputs import cache_lease
    with cache_lease((config or {}).get("cache_settings", {}).get("cache_dir", "./cache/factor_data")):
        log = custom_logger or logger
        cfg = config or {}
        neu_cfg = neutralization_config or {}
        from utils.data_store import binding_contract, canonical, output_context
        bindings = binding_contract(cfg.get("data_bindings"))
        bound = bool(bindings)
        if slot is None:
            slot = cfg.get("slot")
        slot_id, frequency = resolve_market_input(
            slot, slot_id if slot_id is not None else cfg.get("slot_id"),
            frequency if frequency is not None else cfg.get("frequency", "1d"),
        )
        configured_store = factor_values_store if factor_values_store is not None else cfg.get("factor_values_store")
        pinned = factor_identities if factor_identities is not None else cfg.get("factor_identities") or {}
        output_aliases = {alias.removeprefix("main.") for alias in bindings.get("outputs", {})}
        use_precomputed_neutralization = bound and bool(neu_cfg.get("enabled", False))
        if bound:
            if configured_store is not None or pinned or slot_id != DEFAULT_SLOT_ID or frequency != "1d":
                raise ValueError("Frozen DuckDB factor bindings already select the daily synthesis inputs")
            if not use_precomputed_neutralization and factor_table.removeprefix("main.") not in output_aliases:
                raise ValueError(f"Frozen factor table binding is missing: {factor_table}")
        market_identities, market_plan = {}, None
        if not bound:
            if isinstance(configured_store, MarketFactorValueStore):
                store = configured_store
            elif configured_store is not None:
                store = MarketFactorValueStore(configured_store)
            elif factor_database_path is not None:
                store = MarketFactorValueStore.beside_results_database(factor_database_path)
            else:
                raise ValueError("Supply factor_values_store or factor_database_path for Scheme B synthesis")
            market_identities, market_plan = plan_market_factor_values(
                store, factor_names, pinned, start_date, end_date,
                sorted(stock_filter) if stock_filter else None,
                slot_id=slot_id, frequency=frequency, slot=slot,
            )
            if not market_identities:
                return pd.DataFrame()
        if slot is None and slot_id != DEFAULT_SLOT_ID:
            raise ValueError("Pass a MarketSlot to align a non-default synthesis panel")
        from .pool_manager import synthesis_calendar
        calendar, grid = synthesis_calendar(slot, price_conn, frequency, start_date, end_date)

        # === Parquet 缓存逻辑 ===
        cache_dir = cfg.get('cache_settings', {}).get('cache_dir', './cache/factor_data')
        use_cache = (
            cfg.get('cache_settings', {}).get('use_parquet_cache', True)
            and not use_precomputed_neutralization
            and not neu_cfg.get("enabled", False)
        )
        if use_precomputed_neutralization and cfg.get('cache_settings', {}).get('use_parquet_cache', True):
            log.info("[中性化输入] 已启用预计算中性化输入，绕过 Parquet 因子缓存以避免源映射陈旧")

        cache_path = None
        processed_cache_path = None
        df = None
        raw_cache_hit = False

        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)

            # 生成缓存键（基于因子名、日期范围、股票过滤器）
            stock_filter_hash = hashlib.md5(
                str(sorted(stock_filter) if stock_filter else 'ALL').encode()
            ).hexdigest()[:8]
            from utils.data_revision import connection_revision
            source_stamp = (
                canonical({
                    "outputs": bindings.get("outputs", {}), "relation": factor_table,
                    "revision": connection_revision(factor_conn),
                    "rows": factor_conn.execute(
                        f"SELECT COUNT(*), BIT_XOR(HASH(date, stock_code, factor_name, factor_value)) "
                        f"FROM {factor_table} WHERE factor_name = ANY(?) AND date BETWEEN ? AND ?",
                        [factor_names, start_date, end_date],
                    ).fetchone(),
                }) if bound else canonical([slot_id, frequency, sorted(market_identities.items()),
                                            [str(path) for path in market_plan.shard_paths]])
            )
            factor_hash = hashlib.md5(
                canonical([sorted(set(factor_names)), source_stamp]).encode()
            ).hexdigest()[:8]
            cache_key = f"factors_v4_{factor_hash}_{start_date}_{end_date}_{stock_filter_hash}"
            cache_path = os.path.join(cache_dir, f"{cache_key}.parquet")

            price_source_stamp = ""
            if bound:
                price_source_stamp = canonical(bindings.get("inputs", {}))
            elif price_conn is not None:
                try:
                    price_source_files = [
                        str(row[-1])
                        for row in price_conn.execute("PRAGMA database_list").fetchall()
                        if row[-1]
                    ]
                    price_source_stamp = "|".join(
                        f"{path}:{os.stat(path).st_mtime_ns}" for path in price_source_files
                    )
                except Exception:
                    pass
            if price_conn is not None:
                price_source_stamp += canonical(connection_revision(price_conn))
            calendar_hash = calendar.fingerprint if slot is None else calendar.calendar_hash
            processing_hash = hashlib.md5(
                repr(
                    (
                        cfg.get("missing_data_strategy", "drop"),
                        cfg.get("ffill_limit", 30),
                        price_table,
                        price_source_stamp,
                        calendar_hash,
                        list(factor_names),
                    )
                ).encode()
            ).hexdigest()[:8]
            processed_cache_path = os.path.join(
                cache_dir, f"{cache_key}_processed_calendar_v3_{processing_hash}.parquet"
            )

            if os.path.exists(processed_cache_path):
                try:
                    log.info(f"[缓存] 从已处理因子缓存加载: {processed_cache_path}")
                    cached = pd.read_parquet(processed_cache_path)
                    try:
                        stamp = os.stat(processed_cache_path)
                        os.utime(processed_cache_path, ns=(time.time_ns(), stamp.st_mtime_ns))
                    except OSError as exc:
                        log.warning("Cache use timestamp could not be refreshed: %s", exc)
                    cached.attrs[PANEL_DATE_ALIGNMENT_ATTR] = PANEL_DATE_ALIGNMENT_TRADE_DATE
                    return _with_market_factor_inputs(cached, market_identities, market_plan, slot_id, frequency)
                except Exception as e:
                    log.warning(f"[缓存] 已处理因子缓存读取失败，回退到原始缓存: {e}")

            # 尝试从缓存加载
            if os.path.exists(cache_path):
                try:
                    log.info(f"[缓存] 从 Parquet 缓存加载: {cache_path}")
                    df = pd.read_parquet(cache_path)
                    try:
                        stamp = os.stat(cache_path)
                        os.utime(cache_path, ns=(time.time_ns(), stamp.st_mtime_ns))
                    except OSError as exc:
                        log.warning("Cache use timestamp could not be refreshed: %s", exc)
                    raw_cache_hit = True
                    log.info(f"[缓存] 加载成功，{len(df)} 行数据")
                except Exception as e:
                    log.warning(f"[缓存] 读取失败，回退到SQL: {e}")
                    df = None

        raw_needed = list(factor_names)
        frames = []
        if df is None and use_precomputed_neutralization:
            skip_factors = set(neu_cfg.get("skip_factors") or [])
            candidates = [name for name in factor_names if name not in skip_factors]
            raw_needed = [name for name in factor_names if name in skip_factors]
            if candidates:
                from .result_persistence import check_table_exists
                if ("neutralized_factor_values" not in output_aliases
                        or not check_table_exists(factor_conn, "neutralized_factor_values")):
                    raise ValueError("Frozen neutralized_factor_values binding is missing")
                pool = neu_cfg.get("pool_id") or cfg.get("pool_id") or output_context().get("pool_id")
                where = (
                    f"raw_factor_name IN ({_sql_in(candidates)}) "
                    f"AND factor_name IN ({_sql_in([f'{name}__neu' for name in candidates])}) "
                    f"AND date BETWEEN {_sql_literal(start_date)} AND {_sql_literal(end_date)}"
                )
                if pool:
                    where += f" AND pool_id={_sql_literal(pool)}"
                pools = factor_conn.execute(
                    f"SELECT DISTINCT pool_id FROM neutralized_factor_values WHERE {where}"
                ).fetchall()
                if not pools:
                    raise ValueError(f"Frozen neutralized factors are unavailable: {candidates}")
                if len(pools) != 1 or not pools[0][0]:
                    raise ValueError("Frozen neutralization mixes pools or has no pool identity")
                rows = _read_factor_rows(
                    factor_conn, "SELECT date, stock_code, raw_factor_name AS factor_name, factor_value",
                    "neutralized_factor_values", where, stock_filter, log,
                )
                # Coverage is checked on the bound pool independently of an instrument subset.
                if stock_filter:
                    names = factor_conn.execute(
                        f"SELECT DISTINCT raw_factor_name FROM neutralized_factor_values WHERE {where}"
                    ).fetchnumpy()["raw_factor_name"]
                else:
                    names = rows.factor_name
                missing = set(candidates) - set(names)
                if missing:
                    raise ValueError(f"Frozen neutralized factors are unavailable: {sorted(missing)}")
                frames.append(rows)
        if df is None and market_plan is not None:
            frames.append(read_market_factor_values(market_plan, frequency))
        if df is None and bound and raw_needed:
            if factor_table.removeprefix("main.") not in output_aliases:
                raise ValueError(f"Frozen factor table binding is missing: {factor_table}")
            where = (
                f"factor_name IN ({_sql_in(raw_needed)}) "
                f"AND date BETWEEN {_sql_literal(start_date)} AND {_sql_literal(end_date)}"
            )
            frames.append(_read_factor_rows(
                factor_conn, "SELECT date, stock_code, factor_name, factor_value",
                factor_table, where, stock_filter, log,
            ))

        if df is None:
            non_empty_frames = [frame for frame in frames if not frame.empty]
            df = (
                pd.concat(non_empty_frames, ignore_index=True)
                if non_empty_frames
                else _empty_factor_rows()
            )

        # 保存到 Parquet 缓存
        if use_cache and not raw_cache_hit and not df.empty:
            try:
                _write_factor_cache(df, cache_path, index=False)
                log.info(f"[缓存] 已保存到 Parquet: {cache_path}")
            except Exception as e:
                log.warning(f"[缓存] 保存失败: {e}")

        if neu_cfg.get("enabled", False) and not bound and not df.empty:
            from .factor_ops import neutralize_market_features
            raw_panel = _pivot_raw_factor_df(df, factor_names, log)
            if not raw_panel.empty:
                df = neutralize_market_features(
                    raw_panel, neu_cfg, slot=slot, price_conn=price_conn, frequency=frequency,
                    start_date=start_date, end_date=end_date, lag_exposures=False,
                )
        processed = process_raw_factor_df(
            df, factor_names, start_date, end_date,
            config=cfg, price_conn=price_conn, price_table=price_table,
            custom_logger=log, calendar=calendar, grid=grid,
        )
        if use_cache and processed_cache_path and not processed.empty:
            try:
                _write_factor_cache(processed, processed_cache_path)
                log.info(f"[缓存] 已保存已处理因子面板: {processed_cache_path}")
            except Exception as e:
                log.warning(f"[缓存] 已处理因子面板保存失败: {e}")
        return _with_market_factor_inputs(processed, market_identities, market_plan, slot_id, frequency)


def _pivot_raw_factor_df(
    df: pd.DataFrame,
    factor_names: List[str],
    log: logging.Logger,
) -> pd.DataFrame:
    """Pivot long raw factor rows into a (date, stock_code) panel preserving original timestamps."""
    factor_df = df.pivot_table(
        index=['date', 'stock_code'],
        columns='factor_name',
        values='factor_value',
    ).reindex(columns=factor_names)
    raw_dates = factor_df.index.get_level_values('date')
    if not isinstance(raw_dates, pd.DatetimeIndex):
        try:
            factor_df.index = pd.MultiIndex.from_tuples(
                [(pd.to_datetime(str(date)), stock_code) for date, stock_code in factor_df.index],
                names=['date', 'stock_code'],
            )
        except Exception as e:
            log.error(f"日期索引转换失败: {e}")
            factor_df.index.names = ['date', 'stock_code']
    return factor_df


def process_raw_factor_df(
    df: pd.DataFrame,
    factor_names: List[str],
    start_date: str,
    end_date: str,
    config: Optional[Dict[str, Any]] = None,
    price_conn=None,
    price_table: Optional[str] = None,
    custom_logger: Optional[logging.Logger] = None,
    *,
    calendar=None,
    grid: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    处理原始因子DataFrame：pivot、日期对齐、shift等

    Args:
        df: 原始长表格式的因子数据
        factor_names: 因子名称列表
        start_date: 开始日期
        end_date: 结束日期
        config: 配置字典，包含 missing_data_strategy、ffill_limit 等
        price_conn: 价格数据库连接（用于获取交易日历）
        price_table: 价格表名
        custom_logger: 可选的自定义 logger

    Returns:
        处理后的因子DataFrame，MultiIndex (date, stock_code)
    """
    log = custom_logger or logger
    cfg = config or {}

    try:
        if df.empty:
            log.warning(f"未找到因子数据: {factor_names}")
            return pd.DataFrame()

        if isinstance(df.index, pd.MultiIndex) and list(df.index.names) == ['date', 'stock_code']:
            factor_df = df.reindex(columns=factor_names)
        else:
            # Preserve the configured order and report missing columns at the pivot.
            factor_df = _pivot_raw_factor_df(df, factor_names, log)

        log.info(f"成功加载因子数据，形状: {factor_df.shape}，因子数: {len(factor_names)}")

        # [ALIGNMENT] Robust Shift Logic (Dense Date Alignment)
        # 1. Ensure dense index (all trading days) to handle gaps in factor data
        # 2. Optional FFill based on missing_data_strategy config
        # 3. Shift(1) to align T with T-1 info

        # Read missing data strategy from config
        missing_strategy = cfg.get("missing_data_strategy", "drop")
        ffill_limit = cfg.get("ffill_limit", 30)

        if missing_strategy == "ffill":
            log.info(f"应用T-1数据对齐: Dense Reindex -> FFill({ffill_limit}) -> Shift(1)")
        else:
            log.info(f"应用T-1数据对齐: Dense Reindex -> Shift(1) (missing_data_strategy={missing_strategy})")

        from utils.trading_calendar import load_trading_calendar
        factor_wide = factor_df.reset_index().pivot(
            index='date', columns='stock_code', values=factor_names)
        if grid is None:
            calendar = calendar if calendar is not None else load_trading_calendar(price_conn)
            factor_wide = calendar.align(factor_wide, start_date, end_date)
        else:
            if not factor_wide.index.isin(grid).all():
                raise ValueError("Scheme B factor timestamps must belong to the synthesis grid")
            factor_wide = factor_wide.reindex(grid.rename("date"))
        if missing_strategy == 'ffill':
            factor_wide = factor_wide.ffill(limit=ffill_limit)
        factor_df = factor_wide.shift(1).stack(level='stock_code')
        factor_df.index.names = ['date', 'stock_code']

        # Remove rows with NaNs introduced by shift or gaps
        factor_df = factor_df.dropna(how='all')
        factor_df.attrs[PANEL_DATE_ALIGNMENT_ATTR] = PANEL_DATE_ALIGNMENT_TRADE_DATE

        return factor_df

    except Exception as e:
        log.error(f"处理因子数据失败: {e}")
        raise


# =============================================================================
# Price Data Loading
# =============================================================================

def _price_rows(
    price_conn, price_table, stocks, start, end, field, *, slot, frequency,
    pit_enabled, for_label=False, fallback=None, batch_size=None,
) -> pd.DataFrame:
    from .pool_manager import read_synthesis_bars

    fields = {field: field}
    if pit_enabled:
        fields["factor"] = "factor"
    try:
        rows = read_synthesis_bars(
            slot, price_conn, price_table, stocks, start, end, frequency=frequency,
            fields=fields, daily_open_label=for_label, batch_size=batch_size,
        )
        if not rows.empty or fallback is None or slot is not None or pit_enabled:
            return rows
    except Exception as exc:
        if pit_enabled:
            raise PITValidationError(f"PIT price query failed for {price_table}: {exc}") from exc
        if fallback is None or slot is not None:
            raise
    table, column = fallback
    return read_synthesis_bars(
        slot, price_conn, table, stocks, start, end, frequency=frequency, fields={field: column},
    )


def _prepare_price_rows(
    rows: pd.DataFrame, field: str, *, pit_enabled: bool, legacy_daily: bool, first: bool = False,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    if first:
        rows = rows.sort_values(["stock_code", "date"], kind="mergesort")
        if legacy_daily:
            rows = rows.assign(date=rows.date.dt.normalize()).drop_duplicates(["date", "stock_code"])
    if pit_enabled:
        validate_pit_price_rows(rows, field)
        rows = rows.assign(factor=rows.factor.where(rows[field].notna()))
    if legacy_daily and field == "close":
        days = rows.date.dt.normalize()
        if rows.assign(day=days).duplicated(["stock_code", "day"]).any():
            fields = [field, "factor"] if pit_enabled else [field]
            rows = rows.assign(date=days).groupby(["stock_code", "date"], as_index=False)[fields].last()
    return rows


def _pricing_window(slot, price_conn, frequency, dates, future_periods):
    from .pool_manager import market_bar_grid
    from utils.session_calendar import CalendarCoverageError
    from utils.trading_calendar import load_trading_calendar

    start, end = dates.min(), dates.max()
    calendar = slot.calendar if slot is not None else load_trading_calendar(price_conn)
    if frequency == "1d":
        calendar.sessions(start, end)
        if future_periods:
            future = calendar.sessions(end, None)
            future = future[future > end]
            if len(future) < future_periods:
                raise CalendarCoverageError(f"Calendar has fewer than {future_periods} sessions after {end}")
            end = future[future_periods - 1]
    elif slot is None:
        raise ValueError("A MarketSlot is required for non-daily synthesis prices")
    elif future_periods:
        grid = market_bar_grid(slot, frequency, end, None)
        future = grid.timestamp_utc[grid.timestamp_utc > end]
        if len(future) < future_periods:
            raise CalendarCoverageError(f"Calendar has fewer than {future_periods} bars after {end}")
        end = future.iloc[future_periods - 1]
    return calendar, start, end


def _adjust_equity_prices(slot: MarketSlot | None, pit_enabled: bool | None) -> bool:
    return (slot is None or slot.slot_id == DEFAULT_SLOT_ID) if pit_enabled is None else pit_enabled


def get_pricing_data(
    price_conn,
    price_table: str,
    stocks: List[str],
    dates: pd.Index,
    skip_months: Optional[SkipRules] = None,
    skip_months_evaluation: bool = False,
    custom_logger: Optional[logging.Logger] = None,
    pit_enabled: bool | None = None,
    *,
    future_periods: int = 21,
    slot: MarketSlot | None = None,
    frequency: str = "1d",
) -> pd.Series:
    """Return close prices on the synthesis index, including the requested forward window."""
    if not len(dates) or not len(stocks):
        return pd.Series(dtype=float)
    _, frequency = resolve_market_input(slot, None, frequency)
    calendar, start, end = _pricing_window(slot, price_conn, frequency, dates, future_periods)
    pit_enabled = _adjust_equity_prices(slot, pit_enabled)
    rows = _price_rows(
        price_conn, price_table, stocks, start, end, "close", slot=slot, frequency=frequency,
        pit_enabled=pit_enabled, fallback=("factor_data", "close_price"),
    )
    rows = _prepare_price_rows(rows, "close", pit_enabled=pit_enabled, legacy_daily=slot is None)
    rows = rows.loc[rows.close.gt(0)].set_index(["stock_code", "date"]).rename_axis(["asset", "date"])
    prices = rows["close"]
    if pit_enabled and not prices.empty:
        prices = apply_pit_adjustment_factor_end(prices, rows.factor, custom_logger)
    if skip_months_evaluation and skip_months:
        prices = prices[~skip_mask(prices.index.get_level_values("date"), skip_months)]
    prices.attrs["trading_calendar"] = calendar
    return prices


def _wide_prices(
    price_conn, price_table, stocks, start, end, field, *, slot, frequency,
    pit_enabled, custom_logger, for_label=False,
) -> pd.DataFrame | None:
    if not len(stocks):
        return None
    pit_enabled = _adjust_equity_prices(slot, pit_enabled)
    try:
        rows = _price_rows(
            price_conn, price_table, stocks, start, end, field, slot=slot, frequency=frequency,
            pit_enabled=pit_enabled, for_label=for_label, batch_size=500 if field == "close" else None,
        )
    except Exception as exc:
        if pit_enabled or slot is not None:
            raise
        (custom_logger or logger).warning("Could not read %s prices: %s", field, exc)
        return None
    if rows.empty:
        return None
    rows = _prepare_price_rows(rows, field, pit_enabled=pit_enabled, legacy_daily=slot is None, first=for_label)
    keep = rows[field].gt(0)
    if for_label:
        keep |= rows[field].isna()
    rows = rows.loc[keep]
    wide = rows.pivot(index="date", columns="stock_code", values=field).sort_index()
    if pit_enabled:
        factors = rows.pivot(index="date", columns="stock_code", values="factor").sort_index()
        wide = apply_pit_adjustment_factor_end_df(wide, factors, custom_logger)
    return wide


def get_open_price_data(
    price_conn,
    price_table: str,
    stocks: List[str],
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    custom_logger: Optional[logging.Logger] = None,
    pit_enabled: bool | None = None,
    *,
    for_label: bool = False,
    slot: MarketSlot | None = None,
    frequency: str = "1d",
) -> pd.DataFrame | None:
    """Return wide open prices. Preserve the first observation for historical daily labels."""
    return _wide_prices(
        price_conn, price_table, stocks, start_date, end_date, "open", slot=slot, frequency=frequency,
        pit_enabled=pit_enabled, custom_logger=custom_logger, for_label=for_label,
    )


def get_complete_price_data(
    price_conn,
    price_table: str,
    stocks: List[str],
    start_date: str,
    end_date: str,
    custom_logger: Optional[logging.Logger] = None,
    pit_enabled: bool | None = None,
    *,
    slot: MarketSlot | None = None,
    frequency: str = "1d",
) -> pd.DataFrame | None:
    """Return wide close prices across the full requested window."""
    return _wide_prices(
        price_conn, price_table, stocks, start_date, end_date, "close", slot=slot, frequency=frequency,
        pit_enabled=pit_enabled, custom_logger=custom_logger,
    )


def get_factor_ratio_data(
    price_conn,
    price_table: str,
    stocks: List[str],
    start_date: str,
    end_date: str,
    custom_logger: Optional[logging.Logger] = None,
    *,
    slot: MarketSlot | None = None,
    frequency: str = "1d",
) -> pd.DataFrame | None:
    """Return consecutive corporate-action factor ratios where the feed provides factors."""
    from .pool_manager import read_synthesis_bars

    try:
        rows = read_synthesis_bars(
            slot, price_conn, price_table, stocks, start_date, end_date,
            frequency=frequency, fields={"factor": "factor"}, batch_size=500,
        )
    except Exception as exc:
        if slot is not None:
            raise
        (custom_logger or logger).warning("Could not read adjustment factors: %s", exc)
        return None
    if "factor" not in rows or not rows.factor.notna().any():
        return None
    observed = rows.factor.notna()
    if (observed & ((rows.factor <= 0) | ~np.isfinite(rows.factor))).any():
        raise ValueError("Observed adjustment factors must be positive and finite")
    if slot is None:
        days = rows.date.dt.normalize()
        if rows.assign(day=days).duplicated(["stock_code", "day"]).any():
            rows = rows.assign(date=days).groupby(["stock_code", "date"], as_index=False).factor.last()
    factors = rows.pivot(index="date", columns="stock_code", values="factor").sort_index()
    return (factors / factors.shift(1)).fillna(1.0)


def load_market_cap_data(
    price_conn,
    fundamental_table: str,
    start_date: str,
    end_date: str,
    stock_filter: Optional[Set[str]] = None,
    custom_logger: Optional[logging.Logger] = None,
    *,
    trading_calendar=None,
    slot: MarketSlot | None = None,
    frequency: str = "1d",
) -> pd.Series:
    """Read the slot's log market cap on the T−1 feature grid."""
    from .pool_manager import (
        MARKET_CAP_FEATURE, aligned_aux_feature, legacy_aux_slot, legacy_cap_instruments,
    )

    _, frequency = resolve_market_input(slot, None, frequency)
    sessions = None
    if slot is None:
        if trading_calendar is None:
            raise TypeError("Supply trading_calendar or a MarketSlot for log market cap")
        sessions = trading_calendar.sessions(start_date, end_date)
        slot = legacy_aux_slot(price_conn, fundamental_table)
        instruments = sorted(stock_filter) if stock_filter else legacy_cap_instruments(
            price_conn, fundamental_table, start_date, end_date,
        )
    elif stock_filter is None:
        raise ValueError("Supply the instruments for MarketSlot market-cap reads")
    else:
        instruments = sorted(stock_filter)
    return aligned_aux_feature(
        slot, MARKET_CAP_FEATURE, instruments, start_date, end_date, frequency=frequency, sessions=sessions,
    ).rename("log_market_cap")
