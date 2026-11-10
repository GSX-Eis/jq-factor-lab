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
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from .calendar_skip import SkipRules, describe_skip_rules, skip_mask

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


def _neutralized_factor_table_exists(conn) -> bool:
    try:
        conn.execute("SELECT 1 FROM neutralized_factor_values LIMIT 1").fetchdf()
        return True
    except Exception:
        return False


def _factor_source_stamp(
    conn,
    table_name: str,
    factor_names: List[str],
    start_date: str,
    end_date: str,
) -> str:
    """Fingerprint only the factor rows that can affect this cache entry."""
    row = conn.execute(
        f"""
        SELECT COUNT(*), BIT_XOR(HASH(date, stock_code, factor_name, factor_value))
        FROM {table_name}
        WHERE factor_name IN ({_sql_in(factor_names)})
          AND date BETWEEN {_sql_literal(start_date)} AND {_sql_literal(end_date)}
        """
    ).fetchone()
    return f"{table_name}:{int(row[0])}:{int(row[1] or 0)}"


# =============================================================================
# PIT Adjustment Helpers
# =============================================================================


class PITValidationError(ValueError):
    """Enabled PIT cannot produce a validated adjusted price panel."""


def _pit_dates(index, context):
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.tz is not None:
        raise PITValidationError(f"PIT {context}: dates must be finite timezone-naive timestamps")


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
    custom_logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    从数据库加载因子数据（支持 Parquet 缓存加速和预计算中性化输入）

    Args:
        factor_conn: 因子数据库连接
        factor_table: 因子表名
        factor_names: 因子名称列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        stock_filter: 可选的股票代码集合，用于SQL级别过滤以减少内存占用
        config: 配置字典，包含 cache_settings、missing_data_strategy 等
        price_conn: 价格数据库连接（用于获取交易日历）
        price_table: 价格表名
        neutralization_config: 中性化配置；enabled=True 时优先读取预计算中性化因子
        custom_logger: 可选的自定义 logger

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
        if not bound:
            import duckdb
            if isinstance(factor_conn, duckdb.DuckDBPyConnection) and factor_conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema='meta' AND table_name='artifacts'"
            ).fetchone()[0]:
                raise ValueError("Frozen factor bindings are required for a versioned output store")
        output_aliases = {alias.removeprefix("main.") for alias in bindings.get("outputs", {})}
        neutralization_pool = str(neu_cfg.get("pool_id") or cfg.get("pool_id") or output_context().get("pool_id") or "") if bound else ""
        use_precomputed_neutralization = bool(neu_cfg.get("enabled", False))
        if bound and not use_precomputed_neutralization and factor_table.removeprefix("main.") not in output_aliases:
            raise ValueError(f"Frozen factor table binding is missing: {factor_table}")

        # === Parquet 缓存逻辑 ===
        cache_dir = cfg.get('cache_settings', {}).get('cache_dir', './cache/factor_data')
        use_cache = (
            cfg.get('cache_settings', {}).get('use_parquet_cache', True)
            and not use_precomputed_neutralization
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
            source_stamp = ""
            try:
                source_stamp = canonical({"outputs": bindings.get("outputs", {}), "relation": factor_table}) if bound else _factor_source_stamp(
                    factor_conn,
                    factor_table,
                    factor_names,
                    start_date,
                    end_date,
                )
            except Exception:
                if bound:
                    raise
                source_files = [
                    str(row[-1])
                    for row in factor_conn.execute("PRAGMA database_list").fetchall()
                    if row[-1]
                ]
                source_stamp = "|".join(
                    f"{path}:{os.stat(path).st_mtime_ns}" for path in source_files
                )
            from utils.data_revision import connection_revision
            source_stamp += canonical(connection_revision(factor_conn))
            factor_hash = hashlib.md5(
                canonical([sorted(set(factor_names)), source_stamp]).encode()
            ).hexdigest()[:8]
            cache_key = f"factors_v3_{factor_hash}_{start_date}_{end_date}_{stock_filter_hash}"
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
            from utils.trading_calendar import load_trading_calendar
            calendar = load_trading_calendar(price_conn)
            calendar.validate_coverage(start_date, end_date)
            processing_hash = hashlib.md5(
                repr(
                    (
                        cfg.get("missing_data_strategy", "drop"),
                        cfg.get("ffill_limit", 30),
                        price_table,
                        price_source_stamp,
                        calendar.fingerprint,
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
                    return cached
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

        raw_factor_names = list(factor_names)
        raw_needed = list(raw_factor_names)
        neutralized_needed: List[str] = []

        if use_precomputed_neutralization:
            skip_factors = set(neu_cfg.get("skip_factors") or [])
            candidates = [name for name in raw_factor_names if name not in skip_factors]
            raw_needed = [name for name in raw_factor_names if name in skip_factors]
            if bound and candidates and "neutralized_factor_values" not in output_aliases:
                raise ValueError("Frozen neutralized_factor_values binding is missing")

            available_neutralized: Set[str] = set()
            if candidates and _neutralized_factor_table_exists(factor_conn):
                try:
                    candidate_factor_names = [f"{name}__neu" for name in candidates]
                    available_sql = f"""
                    SELECT DISTINCT raw_factor_name
                    FROM neutralized_factor_values
                    WHERE raw_factor_name IN ({_sql_in(candidates)})
                    AND factor_name IN ({_sql_in(candidate_factor_names)})
                    AND date BETWEEN {_sql_literal(start_date)} AND {_sql_literal(end_date)}
                    """
                    if neutralization_pool:
                        available_sql += f" AND pool_id={_sql_literal(neutralization_pool)}"
                    available_df = factor_conn.execute(available_sql).fetchdf()
                    if not available_df.empty:
                        available_neutralized = set(available_df["raw_factor_name"].astype(str))
                except Exception as e:
                    if bound:
                        raise
                    log.warning(f"[中性化输入] 查询 neutralized_factor_values 失败，回退 raw: {e}")
            elif candidates:
                if bound:
                    raise ValueError("Frozen neutralized_factor_values binding is missing")
                log.warning("[中性化输入] neutralized_factor_values 表不存在，回退 raw")

            missing_neutralized = [name for name in candidates if name not in available_neutralized]
            if bound and missing_neutralized:
                raise ValueError(f"Frozen neutralized factors are unavailable: {missing_neutralized}")
            for name in missing_neutralized:
                log.warning(f"[中性化输入缺失] raw={name} expected={name}__neu fallback=raw")

            neutralized_needed = [name for name in candidates if name in available_neutralized]
            raw_needed.extend(missing_neutralized)

        frames = []
        if df is None and neutralized_needed:
            neutralized_where = (
                f"raw_factor_name IN ({_sql_in(neutralized_needed)}) "
                f"AND factor_name IN ({_sql_in([f'{name}__neu' for name in neutralized_needed])}) "
                f"AND date BETWEEN {_sql_literal(start_date)} AND {_sql_literal(end_date)}"
            )
            if neutralization_pool:
                neutralized_where += f" AND pool_id={_sql_literal(neutralization_pool)}"
            if bound:
                pools = factor_conn.execute(f"SELECT DISTINCT pool_id FROM neutralized_factor_values WHERE {neutralized_where}").fetchall()
                if len(pools) != 1 or not pools[0][0]:
                    raise ValueError("Frozen neutralization mixes pools or has no pool identity")
            frames.append(
                _read_factor_rows(
                    factor_conn,
                    "SELECT date, stock_code, raw_factor_name AS factor_name, factor_value",
                    "neutralized_factor_values",
                    neutralized_where,
                    stock_filter,
                    log,
                )
            )

        if df is None and raw_needed:
            if bound and factor_table.removeprefix("main.") not in output_aliases:
                raise ValueError(f"Frozen factor table binding is missing: {factor_table}")
            raw_where = (
                f"factor_name IN ({_sql_in(raw_needed)}) "
                f"AND date BETWEEN {_sql_literal(start_date)} AND {_sql_literal(end_date)}"
            )
            frames.append(
                _read_factor_rows(
                    factor_conn,
                    "SELECT date, stock_code, factor_name, factor_value",
                    factor_table,
                    raw_where,
                    stock_filter,
                    log,
                )
            )

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
                df.to_parquet(cache_path, index=False)
                log.info(f"[缓存] 已保存到 Parquet: {cache_path}")
            except Exception as e:
                log.warning(f"[缓存] 保存失败: {e}")

        processed = process_raw_factor_df(
            df, factor_names, start_date, end_date,
            config=cfg, price_conn=price_conn, price_table=price_table,
            custom_logger=log
        )
        if use_cache and processed_cache_path and not processed.empty:
            try:
                processed.to_parquet(processed_cache_path)
                log.info(f"[缓存] 已保存已处理因子面板: {processed_cache_path}")
            except Exception as e:
                log.warning(f"[缓存] 已处理因子面板保存失败: {e}")
        return processed


def process_raw_factor_df(
    df: pd.DataFrame,
    factor_names: List[str],
    start_date: str,
    end_date: str,
    config: Optional[Dict[str, Any]] = None,
    price_conn=None,
    price_table: Optional[str] = None,
    custom_logger: Optional[logging.Logger] = None
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

        # 转换为pivot格式
        factor_df = df.pivot_table(
            index=['date', 'stock_code'],
            columns='factor_name',
            values='factor_value'
        )

        # 转换日期索引，处理可能的字符串日期格式
        try:
            factor_df.index = pd.MultiIndex.from_tuples(
                [(pd.to_datetime(str(date)), stock_code) for date, stock_code in factor_df.index],
                names=['date', 'stock_code']
            )
        except Exception as e:
            log.error(f"日期索引转换失败: {e}")
            # 如果转换失败，保持原始索引但重命名
            factor_df.index.names = ['date', 'stock_code']

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
        calendar = load_trading_calendar(price_conn)
        factor_wide = factor_df.reset_index().pivot(
            index='date', columns='stock_code', values=factor_names)
        factor_wide = calendar.align(factor_wide, start_date, end_date)
        if missing_strategy == 'ffill':
            factor_wide = factor_wide.ffill(limit=ffill_limit)
        factor_df = factor_wide.shift(1).stack(level='stock_code')
        factor_df.index.names = ['date', 'stock_code']

        # Remove rows with NaNs introduced by shift or gaps
        factor_df = factor_df.dropna(how='all')

        return factor_df

    except Exception as e:
        log.error(f"处理因子数据失败: {e}")
        raise


# =============================================================================
# Price Data Loading
# =============================================================================

def get_pricing_data(
    price_conn,
    price_table: str,
    stocks: List[str],
    dates: pd.Index,
    skip_months: Optional[SkipRules] = None,
    skip_months_evaluation: bool = False,
    custom_logger: Optional[logging.Logger] = None,
    # === PIT 调整参数 ===
    pit_enabled: bool = True,
    *,
    future_periods: int = 21,
) -> pd.Series:
    """
    获取价格数据用于alphalens分析，返回MultiIndex Series

    PIT 复权公式: adjusted_price = price * (factor / factor_end)
    其中 factor_end = 每只股票窗口内最后一个有价格日期的复权因子。
    PIT 启用时，查询失败或非法复权因子直接抛错，不回退未复权行情。

    Args:
        price_conn: 价格数据库连接
        price_table: 价格表名
        stocks: 股票代码列表
        dates: 日期索引
        skip_months: 可选的跳过月份集合
        skip_months_evaluation: 是否在评估阶段应用 skip_months
        custom_logger: 可选的自定义 logger
        pit_enabled: 是否启用 PIT 复权（默认 True）
        future_periods: 结束日之后额外读取的交易日数量，仅限关键字传入。
            最后读取的交易日包含整天，以次日午夜为半开上界。

    Returns:
        MultiIndex Series (asset, date) 格式的价格数据（已应用 PIT 调整，如果启用）
    """
    log = custom_logger or logger

    stocks_str = "', '".join(stock.replace("'", "''") for stock in stocks)
    from utils.trading_calendar import load_trading_calendar
    calendar = load_trading_calendar(price_conn)
    calendar.validate_coverage(dates.min(), dates.max())
    start_date = dates.min()
    end_date = calendar.next_sessions(dates.max(), future_periods)[-1] if future_periods else dates.max()

    log.info(f"获取价格数据: {len(stocks)}只股票，时间范围 {start_date} 到 {end_date}")

    # 根据是否启用 PIT 决定查询的列
    if pit_enabled:
        select_columns = "date, stock_code, close, factor"
        log.info("[PIT] 启用动态复权 (factor_end 方式)，同时获取 factor 列")
    else:
        select_columns = "date, stock_code, close"

    # 首先尝试从price_data表获取
    sql_price = f"""
    SELECT {select_columns}
    FROM {price_table}
    WHERE stock_code IN ('{stocks_str}')
    AND date >= '{start_date.strftime('%Y-%m-%d')}'
    AND date < CAST('{end_date.strftime('%Y-%m-%d')}' AS DATE) + INTERVAL 1 DAY
    ORDER BY stock_code, date
    """

    try:
        price_df = price_conn.execute(sql_price).fetchdf()
        if price_df.empty and not pit_enabled:
            raise ValueError("price_data表中未找到数据")
        log.info(f"从price_data表获取到 {len(price_df)} 条价格记录")
    except Exception as e:
        if pit_enabled:
            raise PITValidationError(f"PIT price query failed for {price_table}: {e}") from e
        # 备选：从factor_data表获取close_price价格
        log.info(f"price_data表无数据 ({e})，从factor_data表获取价格数据")
        # 仅显式关闭 PIT 时允许从无 factor 的价格表回退。
        sql_factor = f"""
        SELECT date, stock_code, close_price AS close
        FROM factor_data
        WHERE stock_code IN ('{stocks_str}')
        AND date >= '{start_date.strftime('%Y-%m-%d')}'
        AND date < CAST('{end_date.strftime('%Y-%m-%d')}' AS DATE) + INTERVAL 1 DAY
        ORDER BY stock_code, date
        """
        price_df = price_conn.execute(sql_factor).fetchdf()
        log.info(f"从factor_data表获取到 {len(price_df)} 条价格记录")

    if price_df.empty:
        log.error("未找到任何价格数据")
        return pd.Series()

    # 检查价格数据质量
    nan_count = price_df['close'].isna().sum()
    zero_count = (price_df['close'] == 0).sum()
    log.info(f"价格数据质量检查: NaN数量={nan_count}, 零值数量={zero_count}")

    has_factor_column = 'factor' in price_df.columns
    if pit_enabled:
        validate_pit_price_rows(price_df, 'close')
        price_df.loc[price_df['close'].isna(), 'factor'] = np.nan

    # 处理5分钟数据聚合为日度数据
    price_df['datetime'] = pd.to_datetime(price_df['date'])
    price_df['date_only'] = price_df['datetime'].dt.date

    raw_record_count = len(price_df)
    unique_daily_count = len(price_df.groupby(['stock_code', 'date_only']))

    if raw_record_count > unique_daily_count:
        log.info(f"检测到高频数据: {raw_record_count}条记录 > {unique_daily_count}个交易日，聚合为日度数据")
        # 聚合时同时处理 factor 列
        agg_dict = {'close': 'last'}
        if has_factor_column:
            agg_dict['factor'] = 'last'
        price_df = price_df.groupby(['stock_code', 'date_only']).agg(agg_dict).reset_index()
        price_df['date'] = pd.to_datetime(price_df['date_only'])
    else:
        price_df['date'] = price_df['datetime']
        log.info("数据已为日度频率，无需聚合")

    # 移除无效价格
    original_len = len(price_df)
    price_df = price_df.dropna(subset=['close'])
    price_df = price_df[price_df['close'] > 0]
    final_len = len(price_df)

    if final_len < original_len:
        log.warning(f"清理价格数据: 从 {original_len} 条减少到 {final_len} 条（移除NaN和零值）")

    # 转换为alphalens需要的MultiIndex Series格式: (asset, date)
    price_df['date'] = pd.to_datetime(price_df['date'])
    pricing_series = price_df.set_index(['stock_code', 'date'])['close']
    pricing_series.index.names = ['asset', 'date']

    # === PIT 调整 (使用 factor_end 方式) ===
    if pit_enabled:
        # 提取复权因子 Series
        factors_series = price_df.set_index(['stock_code', 'date'])['factor']
        factors_series.index.names = ['asset', 'date']

        # 应用 PIT 调整 (factor_end 方式)
        pricing_series = apply_pit_adjustment_factor_end(
            pricing_series, factors_series, log
        )

    # 可选：在评估阶段从价格数据中剔除指定月份/日期窗口
    if skip_months_evaluation and skip_months:
        date_values = pd.to_datetime(pricing_series.index.get_level_values('date'))
        mask = ~skip_mask(date_values, skip_months)
        kept = int(mask.sum())
        total = len(pricing_series)
        if kept < total and total > 0:
            log.info(
                f"在价格数据中应用skip_months过滤: 保留 {kept}/{total} 行 ({kept/total:.2%})，"
                f"跳过规则={describe_skip_rules(skip_months)}"
            )
            pricing_series = pricing_series[mask]

        if pricing_series.empty:
            log.warning("应用skip_months后价格数据为空")
            return pricing_series

    pricing_series.attrs['trading_calendar'] = calendar
    log.info(f"转换为MultiIndex Series: {len(pricing_series)} 条价格记录")
    log.info(f"价格数据时间范围: {pricing_series.index.get_level_values('date').min()} 到 {pricing_series.index.get_level_values('date').max()}")
    log.info(f"涵盖股票数量: {len(pricing_series.index.get_level_values('asset').unique())}")

    return pricing_series


def get_open_price_data(
    price_conn,
    price_table: str,
    stocks: List[str],
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    custom_logger: Optional[logging.Logger] = None,
    # === PIT 调整参数 ===
    pit_enabled: bool = True,
    *,
    for_label: bool = False,
) -> Optional[pd.DataFrame]:
    """
    获取开盘价数据用于VectorBT回测 (JoinQuant对齐)

    JoinQuant的日级别回测使用开盘价撮合订单，此方法获取开盘价数据
    以便VectorBT使用相同的执行方式。

    PIT 复权公式: adjusted_price = price * (factor / factor_end)
    其中 factor_end = 每只股票窗口内最后一个有价格日期的复权因子

    Args:
        price_conn: 价格数据库连接
        price_table: 价格表名
        stocks: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        custom_logger: 可选的自定义 logger
        pit_enabled: 是否启用 PIT 复权（默认 True）
        for_label: 标签专用日开盘；优先当前连接日表，否则保留首bar（包括缺失）

    Returns:
        Wide DataFrame (Index=date, Columns=stocks) with open prices（已应用 PIT 调整，如果启用）,
        or None if there are no observations. With PIT enabled, invalid factors
        or query failures raise PITValidationError instead of returning raw prices.
    """
    log = custom_logger or logger

    if not stocks:
        return None

    stocks_str = "', '".join(stock.replace("'", "''") for stock in stocks)

    # 转换日期格式
    if hasattr(start_date, 'strftime'):
        start_str = start_date.strftime('%Y-%m-%d')
    else:
        start_str = str(start_date)[:10]

    if hasattr(end_date, 'strftime'):
        end_str = end_date.strftime('%Y-%m-%d')
    else:
        end_str = str(end_date)[:10]

    end_exclusive = (pd.Timestamp(end_str) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    log.info(f"[开盘价] 获取 {len(stocks)} 只股票的开盘价: {start_str} ~ {end_str}")

    # 根据是否启用 PIT 决定查询的列
    if pit_enabled:
        select_columns = "date, stock_code, open, factor"
        log.info("[PIT] 启用动态复权 (factor_end 方式)，同时获取 factor 列")
    else:
        select_columns = "date, stock_code, open"

    if for_label:
        from .result_persistence import check_table_exists
        if check_table_exists(price_conn, 'price_data_daily'):
            price_table = 'price_data_daily'

    # 从price_data_daily表获取open价格
    sql = f"""
    SELECT {select_columns}
    FROM {price_table}
    WHERE stock_code IN ('{stocks_str}')
    AND date >= '{start_str}' AND date < '{end_exclusive}'
    ORDER BY stock_code, date
    """

    try:
        df = price_conn.execute(sql).fetchdf()

        if df.empty:
            log.warning("[开盘价] 未找到开盘价数据")
            return None

        log.info(f"[开盘价] 获取到 {len(df)} 条开盘价记录")

        # 转换为Wide格式
        df['date'] = pd.to_datetime(df['date'])
        if for_label:
            # Select the original first observation before filtering NaN prices.
            df = df.sort_values(['stock_code', 'date'], kind='mergesort')
            df['date'] = df['date'].dt.normalize()
            df = df.drop_duplicates(['date', 'stock_code'], keep='first')

        if pit_enabled:
            validate_pit_price_rows(df, 'open')

        # 移除NaN和零值
        if for_label:
            df = df[df['open'].isna() | (df['open'] > 0)]
        else:
            df = df.dropna(subset=['open'])
            df = df[df['open'] > 0]

        # Pivot成Wide格式: Index=date, Columns=stock
        open_wide = df.pivot(index='date', columns='stock_code', values='open')
        open_wide = open_wide.sort_index()

        # === PIT 调整 (使用 factor_end 方式) ===
        if pit_enabled:
            # Pivot 复权因子
            factor_wide = df.pivot(index='date', columns='stock_code', values='factor')
            factor_wide = factor_wide.sort_index()

            # 应用 PIT 调整 (factor_end 方式)
            open_wide = apply_pit_adjustment_factor_end_df(
                open_wide, factor_wide, log
            )

        log.info(f"[开盘价] 转换为Wide格式: {open_wide.shape[0]} 日期, {open_wide.shape[1]} 股票")

        return open_wide

    except Exception as e:
        if pit_enabled:
            if isinstance(e, PITValidationError):
                raise
            raise PITValidationError(f"PIT open-price query/processing failed for {price_table}: {e}") from e
        log.error(f"[开盘价] 获取开盘价数据失败: {e}")
        return None


def get_complete_price_data(
    price_conn,
    price_table: str,
    stocks: List[str],
    start_date: str,
    end_date: str,
    custom_logger: Optional[logging.Logger] = None,
    # === PIT 调整参数 ===
    pit_enabled: bool = True,
) -> Optional[pd.DataFrame]:
    """
    [JQ-ALIGNED V5] 获取完整收盘价数据，不过滤 skip_months

    VectorBT 需要完整的价格数据来:
    1. 正确追踪 skip month 期间的持仓价值
    2. 在 skip month 第一个交易日执行清仓
    3. 在 skip month 结束后正确恢复交易

    PIT 复权公式: adjusted_price = price * (factor / factor_end)
    其中 factor_end = 每只股票窗口内最后一个有价格日期的复权因子

    Args:
        price_conn: 价格数据库连接
        price_table: 价格表名
        stocks: 股票代码列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        custom_logger: 可选的自定义 logger
        pit_enabled: 是否启用 PIT 复权（默认 True）

    Returns:
        Wide DataFrame (Index=date, Columns=stocks) with close prices（已应用 PIT 调整，如果启用）。
        PIT 启用时，非法复权因子或查询分片失败会抛出 PITValidationError。
    """
    log = custom_logger or logger

    if not stocks:
        return None

    # 根据是否启用 PIT 决定查询的列
    if pit_enabled:
        select_columns = "date, stock_code, close, factor"
        log.info("[PIT] 启用动态复权 (factor_end 方式)，同时获取 factor 列")
    else:
        select_columns = "date, stock_code, close"

    # 分批次查询以避免SQL语句过长
    chunk_size = 500
    all_dfs = []

    for i in range(0, len(stocks), chunk_size):
        chunk = stocks[i:i + chunk_size]
        stocks_str = "', '".join(stock.replace("'", "''") for stock in chunk)

        sql = f"""
        SELECT {select_columns}
        FROM {price_table}
        WHERE stock_code IN ('{stocks_str}')
        AND date >= CAST('{start_date}' AS DATE)
        AND date < CAST('{end_date}' AS DATE) + INTERVAL 1 DAY
        ORDER BY stock_code, date
        """

        try:
            chunk_df = price_conn.execute(sql).fetchdf()
            if not chunk_df.empty:
                all_dfs.append(chunk_df)
        except Exception as e:
            if pit_enabled:
                raise PITValidationError(f"PIT price batch {i//chunk_size} failed for {price_table}: {e}") from e
            log.warning(f"[完整价格] 查询分片 {i//chunk_size} 失败: {e}")

    if not all_dfs:
        log.warning("[完整价格] 未找到价格数据")
        return None

    df = pd.concat(all_dfs, ignore_index=True)
    log.info(f"[完整价格] 获取到 {len(df)} 条收盘价记录")

    has_factor_column = 'factor' in df.columns
    if pit_enabled:
        validate_pit_price_rows(df, 'close')
        df.loc[df['close'].isna(), 'factor'] = np.nan

    # 处理日期格式（可能是5分钟数据需要聚合）
    df['datetime'] = pd.to_datetime(df['date'])
    df['date_only'] = df['datetime'].dt.date

    # Only detect duplicate stock-days; len(groupby) materializes every group.
    if df.duplicated(['stock_code', 'date_only']).any():
        log.info("[完整价格] 检测到高频数据，聚合为日度")
        # 聚合时同时处理 factor 列
        agg_dict = {'close': 'last'}
        if has_factor_column:
            agg_dict['factor'] = 'last'
        df = df.groupby(['stock_code', 'date_only']).agg(agg_dict).reset_index()
        df['date'] = pd.to_datetime(df['date_only'])
    else:
        df['date'] = df['datetime']

    # 移除无效价格
    df = df.dropna(subset=['close'])
    df = df[df['close'] > 0]

    # Pivot成Wide格式: Index=date, Columns=stock
    price_wide = df.pivot(index='date', columns='stock_code', values='close')
    price_wide = price_wide.sort_index()

    # === PIT 调整 (使用 factor_end 方式) ===
    if pit_enabled:
        # Pivot 复权因子
        factor_wide = df.pivot(index='date', columns='stock_code', values='factor')
        factor_wide = factor_wide.sort_index()

        # 应用 PIT 调整 (factor_end 方式)
        price_wide = apply_pit_adjustment_factor_end_df(
            price_wide, factor_wide, log
        )

    log.info(
        f"[完整价格] Wide格式: {price_wide.shape[0]} 日期, {price_wide.shape[1]} 股票, "
        f"范围: {price_wide.index.min()} ~ {price_wide.index.max()}"
    )

    return price_wide


# =============================================================================
# Factor Ratio Data (Corporate Action Detection)
# =============================================================================

def get_factor_ratio_data(
    price_conn,
    price_table: str,
    stocks: List[str],
    start_date: str,
    end_date: str,
    custom_logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    """
    获取除权因子比率数据 (factor[T] / factor[T-1])

    用于 VectorBT 回测中检测除权事件并调整持仓股数。
    当 ratio != 1.0 时，表示该股票在当天发生了除权事件（拆股/合股/派息等）。

    数据结构说明:
    - 用户数据库中 factor 以最后一天为基准 (factor[latest] = 1.0)
    - 之前的 factor < 1.0
    - 拆股示例: 2:1 拆股时，factor 从 0.5 变为 1.0，ratio = 1.0/0.5 = 2.0
    - 此时应将持仓股数乘以 2.0

    Args:
        price_conn: 价格数据库连接
        price_table: 价格表名 (应包含 date, stock_code, factor 列)
        stocks: 股票代码列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        custom_logger: 可选的自定义 logger

    Returns:
        Wide DataFrame (Index=date, Columns=stocks) with factor ratios.
        值 != 1.0 表示当天发生了除权事件。
        如果无法获取 factor 数据则返回 None。
    """
    log = custom_logger or logger

    if not stocks:
        return None

    end_exclusive = (pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # 分批次查询以避免SQL语句过长
    chunk_size = 500
    all_dfs = []

    for i in range(0, len(stocks), chunk_size):
        chunk = stocks[i:i + chunk_size]
        stocks_str = "', '".join(stock.replace("'", "''") for stock in chunk)

        sql = f"""
        SELECT date, stock_code, factor
        FROM {price_table}
        WHERE stock_code IN ('{stocks_str}')
        AND date >= '{start_date}' AND date < '{end_exclusive}'
        ORDER BY stock_code, date
        """

        try:
            chunk_df = price_conn.execute(sql).fetchdf()
            if not chunk_df.empty:
                all_dfs.append(chunk_df)
        except Exception as e:
            log.warning(f"[除权因子] 查询分片 {i//chunk_size} 失败: {e}")

    if not all_dfs:
        log.warning("[除权因子] 未找到 factor 数据，无法计算除权比率")
        return None

    df = pd.concat(all_dfs, ignore_index=True)

    # 检查 factor 列是否存在且有效
    if 'factor' not in df.columns:
        log.warning("[除权因子] 价格表中无 factor 列，无法计算除权比率")
        return None

    observed = df['factor'].notna()
    if (observed & ((df['factor'] <= 0) | ~np.isfinite(df['factor']))).any():
        raise ValueError('Observed adjustment factors must be positive and finite')

    factor_valid_count = df['factor'].notna().sum()
    if factor_valid_count == 0:
        log.warning("[除权因子] factor 列全为 NaN，无法计算除权比率")
        return None

    log.info(f"[除权因子] 获取到 {len(df)} 条 factor 记录，有效值: {factor_valid_count}")

    # 处理日期格式
    df['datetime'] = pd.to_datetime(df['date'])
    df['date_only'] = df['datetime'].dt.date

    # 检查是否为高频数据，需要聚合
    raw_record_count = len(df)
    unique_daily_count = len(df.groupby(['stock_code', 'date_only']))

    if raw_record_count > unique_daily_count:
        log.info("[除权因子] 检测到高频数据，聚合为日度 (取每日最后一个 factor 值)")
        df = df.groupby(['stock_code', 'date_only']).agg({'factor': 'last'}).reset_index()
        df['date'] = pd.to_datetime(df['date_only'])
    else:
        df['date'] = df['datetime']

    # Pivot 成 Wide 格式: Index=date, Columns=stock_code
    factor_wide = df.pivot(index='date', columns='stock_code', values='factor')
    factor_wide = factor_wide.sort_index()

    # 计算除权比率: ratio = factor[T] / factor[T-1]
    # 正常情况下 ratio = 1.0
    # 拆股时 ratio > 1.0 (如 2:1 拆股，ratio = 2.0)
    # 合股时 ratio < 1.0 (如 1:2 合股，ratio = 0.5)
    factor_ratio = factor_wide / factor_wide.shift(1)

    # 第一行无法计算 ratio，填充为 1.0 (无调整)
    factor_ratio = factor_ratio.fillna(1.0)

    # 统计除权事件
    non_one_count = ((factor_ratio != 1.0) & factor_ratio.notna()).sum().sum()
    if non_one_count > 0:
        log.info(f"[除权因子] 检测到 {non_one_count} 个除权事件 (ratio != 1.0)")

        # 可选: 输出除权事件详情 (debug 级别)
        for date in factor_ratio.index:
            for stock in factor_ratio.columns:
                ratio = factor_ratio.loc[date, stock]
                if pd.notna(ratio) and ratio != 1.0:
                    log.debug(
                        f"[除权事件] {date.strftime('%Y-%m-%d')} {stock}: ratio={ratio:.4f}"
                    )
    else:
        log.info("[除权因子] 未检测到除权事件 (所有 ratio = 1.0)")

    log.info(
        f"[除权因子] Wide格式: {factor_ratio.shape[0]} 日期, {factor_ratio.shape[1]} 股票, "
        f"范围: {factor_ratio.index.min()} ~ {factor_ratio.index.max()}"
    )

    return factor_ratio


# =============================================================================
# Market Cap Data Loading
# =============================================================================

def load_market_cap_data(
    price_conn,
    fundamental_table: str,
    start_date: str,
    end_date: str,
    stock_filter: Optional[Set[str]] = None,
    custom_logger: Optional[logging.Logger] = None,
    *,
    trading_calendar,
) -> pd.Series:
    """
    Load log market cap data for neutralization from DuckDB.

    IMPORTANT: Applies shift(1) to align with factor data.
    At time T, use the previous bound market session within the research window.
    Missing observations stay missing; no outside-window warmup is loaded.

    Args:
        price_conn: 价格/基本面数据库连接
        fundamental_table: 基本面表名
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        stock_filter: Optional set of stock codes to filter
        custom_logger: 可选的自定义 logger
        trading_calendar: Required verified calendar covering the research window

    Returns:
        pd.Series with MultiIndex (date, stock_code), T-1 aligned
    """
    log = custom_logger or logger
    sessions = trading_calendar.sessions(start_date, end_date)

    try:
        # Query natural_log_of_market_cap from factor_data table (wide format)
        if stock_filter and len(stock_filter) > 0:
            # Chunked query to avoid SQL length limits
            chunk_size = 500
            stock_list = list(stock_filter)
            all_dfs = []

            for i in range(0, len(stock_list), chunk_size):
                chunk = stock_list[i:i + chunk_size]
                stocks_str = "', '".join(stock.replace("'", "''") for stock in chunk)

                sql = f"""
                SELECT date, stock_code, natural_log_of_market_cap
                FROM {fundamental_table}
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
                AND stock_code IN ('{stocks_str}')
                AND natural_log_of_market_cap IS NOT NULL
                ORDER BY date, stock_code
                """

                chunk_df = price_conn.execute(sql).fetchdf()
                if not chunk_df.empty:
                    all_dfs.append(chunk_df)

            if all_dfs:
                df = pd.concat(all_dfs, ignore_index=True)
            else:
                df = pd.DataFrame()
        else:
            sql = f"""
            SELECT date, stock_code, natural_log_of_market_cap
            FROM {fundamental_table}
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            AND natural_log_of_market_cap IS NOT NULL
            ORDER BY date, stock_code
            """
            df = price_conn.execute(sql).fetchdf()

        if df.empty:
            log.warning("[中性化] 未找到市值数据")
            return pd.Series(dtype=float)

        # Convert to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Pivot to wide format for shift operation
        wide_df = df.pivot(index='date', columns='stock_code', values='natural_log_of_market_cap')

        # Restore every bound market session before taking the previous session.
        wide_df = wide_df.reindex(sessions.as_unit(wide_df.index.unit).rename(wide_df.index.name))

        # Apply shift(1) - T uses T-1 market cap (same alignment as factors)
        wide_df = wide_df.shift(1)

        # Stack back to long format
        series = wide_df.stack()
        series.index.names = ['date', 'stock_code']
        series.name = 'log_market_cap'

        log.info(
            f"[中性化] 加载市值数据: {len(series)} 条记录，"
            f"日期范围: {series.index.get_level_values('date').min()} ~ "
            f"{series.index.get_level_values('date').max()}"
        )

        return series

    except Exception as e:
        log.error(f"[中性化] 加载市值数据失败: {e}")
        raise
