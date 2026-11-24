"""
Pool Manager Module

Dynamic stock pool management utilities for factor synthesis.
Handles stock pool filtering, computation, and skip-month logic.

Extracted from FactorSynthesizer for modularity and reuse.

Author: QuantationAI
Date: 2026-01-12
"""

import logging
from dataclasses import replace
from collections.abc import Sequence
from datetime import timedelta
from functools import lru_cache
from numbers import Number
from typing import Any, Dict, List, Optional, Set, Union, Mapping

import pandas as pd

from utils.slots import MarketSlot
from utils.time_axis import BarSchedule, Frequency

from utils.dynamic_stock_pool import (
    DynamicStockPoolManager,
    StockFilter,
    apply_date_level_pool_filter_to_long_df,
)

from .calendar_skip import SkipRules, describe_skip_rules, skip_mask

logger = logging.getLogger(__name__)


MARKET_CAP_FEATURE = "factor_data.natural_log_of_market_cap"
INDUSTRY_FEATURE = "industry_membership.sw_l1_code"


def _bar_bucket_strategy(slot: MarketSlot, frequency: Frequency) -> str | None:
    if frequency.kind != "intraday":
        return None
    from utils.factor_core.bar_bridge import _bucket_strategy

    return _bucket_strategy(slot.bar_feed, frequency, getattr(slot.bar_feed, "root", None))


def synthesis_bar_schedule(slot: MarketSlot, frequency: str) -> BarSchedule:
    parsed = Frequency.parse(frequency)
    return BarSchedule(
        getattr(slot.calendar, "default", slot.calendar), parsed,
        view=getattr(slot.bar_feed, "view", "ETH"), bucket_strategy=_bar_bucket_strategy(slot, parsed),
    )


def market_bar_grid(slot: MarketSlot, frequency: str, start, end) -> pd.DataFrame:
    # Slot providers can configure bucket alignment independently of the exchange calendar.
    feed = slot.bar_feed
    parsed = Frequency.parse(frequency)
    bounds = [pd.Timestamp(value) if value is not None else None for value in (start, end)]
    view = getattr(feed, "view", "ETH")
    days = []
    for bound in bounds:
        session = (slot.calendar.session_at(bound, view=view, frequency=parsed)
                   if bound is not None and bound.tz is not None else None)
        days.append(session.session_date if session is not None else None if bound is None else bound.date())
    grid = slot.calendar.bar_grid(
        parsed, *days, view=view,
        bucket_strategy=_bar_bucket_strategy(slot, parsed),
    )
    if bounds[0] is not None and bounds[0].tz is not None:
        grid = grid.loc[grid.timestamp_utc.ge(bounds[0])]
    if bounds[1] is not None and bounds[1].tz is not None:
        grid = grid.loc[grid.timestamp_utc.le(bounds[1])]
    return grid


def synthesis_calendar(slot: MarketSlot | None, price_conn, frequency: str, start, end):
    if slot is None:
        if frequency != "1d":
            raise ValueError("Synthesis panel alignment requires a MarketSlot for non-daily frequencies")
        from utils.trading_calendar import load_trading_calendar
        calendar = load_trading_calendar(price_conn)
        calendar.validate_coverage(start, end)
        return calendar, None
    if frequency == "1d":
        return slot.calendar, slot.calendar.sessions(start, end)
    bars = market_bar_grid(slot, frequency, start, end)
    return slot.calendar, pd.DatetimeIndex(bars.timestamp_utc)


def _relation(name: str) -> str:
    from utils.data_store import quote, table_name

    return table_name(name) if "." in name else quote(name)


class _BoundAShareBarFeed:
    """Adapt a configured DuckDB relation while preserving historical local-clock rows."""

    def __init__(self, connection, table: str, fields: dict[str, str]) -> None:
        self.connection, self.table, self.fields = connection, table, fields

    def read_execution_bars(
        self, instruments: Sequence[str], frequency: Frequency, *, start_utc, end_utc,
    ) -> pd.DataFrame:
        from slots.jq_ashare_local.feed import TIMEZONE
        from utils.data_store import quote

        start = start_utc.tz_convert(TIMEZONE).tz_localize(None)
        end = end_utc.tz_convert(TIMEZONE).tz_localize(None)
        projection = ", ".join(f"{quote(source)} AS {quote(field)}" for field, source in self.fields.items())
        rows = self.connection.execute(
            f"SELECT CAST(date AS TIMESTAMP_NS) AS date, stock_code AS instrument_id, {projection} "
            f"FROM {_relation(self.table)} WHERE stock_code = ANY(?) "
            "AND CAST(date AS TIMESTAMP_NS) >= ? AND CAST(date AS TIMESTAMP_NS) < ? ORDER BY stock_code, date",
            [list(instruments), start, end],
        ).fetchdf()
        return rows.assign(
            timestamp_utc=rows.date.dt.tz_localize(TIMEZONE).dt.tz_convert("UTC"),
            session_id=rows.date.dt.strftime("%Y-%m-%d"),
        ).drop(columns="date")

    def read_signal_bars(self, instruments, frequency, *, start_utc, end_utc, as_of_utc) -> pd.DataFrame:
        return self.read_execution_bars(instruments, frequency, start_utc=start_utc, end_utc=min(end_utc, as_of_utc))


def read_synthesis_bars(
    slot: MarketSlot | None, price_conn, table: str, stocks: Sequence[str], start, end,
    *, frequency: str, fields: dict[str, str], daily_open_label: bool = False, batch_size: int | None = None,
) -> pd.DataFrame:
    parsed = Frequency.parse(frequency)
    if not parsed.is_valid:
        raise ValueError(f"Invalid synthesis frequency: {frequency!r}")
    if slot is not None:
        grid = market_bar_grid(slot, parsed.canonical, start, end)
        if grid.empty or not stocks:
            return pd.DataFrame(columns=["date", "stock_code", *fields])
        rows = slot.bar_feed.read_execution_bars(
            stocks, parsed, start_utc=grid.timestamp_utc.min(), end_utc=grid.timestamp_utc.max(),
        )
        dates = pd.to_datetime(rows.session_id) if parsed.kind == "session" else rows.timestamp_utc
    else:
        if parsed.kind != "session":
            raise ValueError("Pass a MarketSlot to request non-daily synthesis prices")
        from slots.jq_ashare_local.feed import PRICE_TABLES, TIMEZONE
        from .result_persistence import check_table_exists

        if daily_open_label and check_table_exists(price_conn, PRICE_TABLES["1d"]):
            table = PRICE_TABLES["1d"]
        slot = replace(legacy_aux_slot(price_conn), bar_feed=_BoundAShareBarFeed(price_conn, table, fields))
        first = pd.Timestamp(start).normalize().tz_localize(TIMEZONE).tz_convert("UTC")
        stop = (pd.Timestamp(end).normalize() + pd.Timedelta(days=1)).tz_localize(TIMEZONE).tz_convert("UTC")
        step = batch_size or len(stocks) or 1
        chunks = [
            slot.bar_feed.read_execution_bars(stocks[i:i + step], parsed, start_utc=first, end_utc=stop)
            for i in range(0, len(stocks), step)
        ]
        if not chunks:
            return pd.DataFrame(columns=["date", "stock_code", *fields])
        rows = pd.concat(chunks, ignore_index=True)
        dates = rows.timestamp_utc.dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    return rows.assign(date=dates, stock_code=rows.instrument_id).sort_values(["stock_code", "date"])


class _BoundAShareCapFeatures:
    """Resolve legacy fundamental-table aliases through the same auxiliary interface."""

    def __init__(self, connection, table: str) -> None:
        self.connection, self.table = connection, table

    def read_aux_features(self, grid: pd.DataFrame, *, features: Sequence[str]) -> pd.DataFrame:
        rows = self.connection.execute(
            f"SELECT CAST(date AS DATE)::VARCHAR AS session_id, stock_code AS instrument_id, "
            f'natural_log_of_market_cap AS "{MARKET_CAP_FEATURE}" FROM {_relation(self.table)} '
            "WHERE CAST(date AS DATE) BETWEEN ?::DATE AND ?::DATE AND stock_code = ANY(?)",
            [grid.session_id.min(), grid.session_id.max(), list(grid.instrument_id.unique())],
        ).fetchdf()
        return grid.merge(rows, on=["session_id", "instrument_id"]).assign(available_at_utc=lambda data: data.timestamp_utc)


@lru_cache(maxsize=1)
def _legacy_ashare_calendar():
    from utils.session_calendar import SessionCalendar

    return SessionCalendar.from_pmc()


def legacy_aux_slot(connection, fundamental_table: str = "factor_data") -> MarketSlot:
    from slots.jq_ashare_local.aux import AShareAuxFeatures
    from slots.jq_ashare_local.feed import AShareBarFeed
    from slots.jq_ashare_local.microstructure import AShareMicrostructure
    from slots.jq_ashare_local.universe import AShareUniverse
    from ..factor_reader import DEFAULT_SLOT_ID

    calendar = _legacy_ashare_calendar()
    auxiliary = (AShareAuxFeatures(connection, calendar) if fundamental_table.removeprefix("main.") == "factor_data"
                 else _BoundAShareCapFeatures(connection, fundamental_table))
    return MarketSlot(
        DEFAULT_SLOT_ID, calendar, AShareBarFeed(connection, calendar), AShareUniverse(connection, calendar),
        auxiliary, AShareMicrostructure(connection, calendar),
    )


def legacy_cap_instruments(connection, fundamental_table: str, start, end) -> list[str]:
    rows = connection.execute(
        f"SELECT DISTINCT stock_code FROM {_relation(fundamental_table)} "
        "WHERE CAST(date AS DATE) BETWEEN ?::DATE AND ?::DATE ORDER BY stock_code",
        [start, end],
    ).fetchall()
    return [row[0] for row in rows]


def aligned_aux_feature(
    slot: MarketSlot, feature: str, instruments: Sequence[str], start, end, *, frequency: str = "1d",
    sessions: pd.DatetimeIndex | None = None, shift: int = 1,
) -> pd.Series:
    """Read each publication on the native clock, then align with the feature lag."""
    from slots.jq_ashare_local.aux import AShareAuxFeatures

    calendar_grid = market_bar_grid(slot, frequency, start, end)
    if sessions is not None and isinstance(slot.aux_features, _BoundAShareCapFeatures):
        # Aliased historical inputs can carry a bound date-only test/replay calendar.
        calendar_grid = pd.DataFrame({
            "session_id": sessions.strftime("%Y-%m-%d"),
            "timestamp_utc": sessions.tz_localize("Asia/Shanghai").tz_convert("UTC") + pd.Timedelta(hours=15),
        })
    if calendar_grid.empty or not instruments:
        return pd.Series(dtype=float)
    daily_published = (
        frequency != "1d"
        and feature == MARKET_CAP_FEATURE
        and (slot.slot_id == "jq_ashare_local" or isinstance(slot.aux_features, (AShareAuxFeatures, _BoundAShareCapFeatures)))
    )
    query_calendar_grid = calendar_grid
    prior_session_map: dict[str, str] = {}
    if daily_published:
        first_session = pd.Timestamp(calendar_grid.session_id.min())
        last_session = pd.Timestamp(calendar_grid.session_id.max())
        pub_start = first_session
        if first_session > slot.calendar.start:
            prior = slot.calendar.sessions(slot.calendar.start, first_session)
            prior = prior[prior < first_session]
            if len(prior):
                pub_start = prior[-1]
        query_calendar_grid = market_bar_grid(slot, "1d", pub_start, last_session)
        ordered_sessions = list(
            query_calendar_grid.sort_values("timestamp_utc")["session_id"].drop_duplicates()
        )
        prior_session_map = {
            ordered_sessions[idx]: ordered_sessions[idx - 1]
            for idx in range(1, len(ordered_sessions))
        }
    dates = pd.to_datetime(calendar_grid.session_id) if frequency == "1d" else calendar_grid.timestamp_utc
    parts = []
    for offset in range(0, len(instruments), 500):
        inst_df = pd.DataFrame({"instrument_id": instruments[offset:offset + 500]})
        grid = query_calendar_grid.merge(inst_df, how="cross")
        try:
            rows = slot.read_aux_features(grid, features=[feature])
        except (KeyError, NotImplementedError):
            return pd.Series(dtype=float)
        except ValueError as exc:
            # Current futures and declarative providers reject feature names outside their capability.
            if str(exc).startswith(("Futures features must be", "Unknown auxiliary features:")):
                return pd.Series(dtype=float)
            raise
        if feature not in rows or rows.empty:
            continue
        rows = rows.dropna(subset=[feature]).copy()
        if rows.empty:
            continue
        if daily_published:
            pubs = rows.dropna(subset=["available_at_utc"])[
                ["instrument_id", "session_id", "available_at_utc", feature]
            ].rename(columns={"session_id": "pub_session_id"}).copy()
            if pubs.empty:
                continue
            target = calendar_grid[["timestamp_utc", "session_id"]].merge(inst_df, how="cross")
            target["instrument_id"] = target["instrument_id"].astype(object)
            target["timestamp_utc"] = pd.to_datetime(target["timestamp_utc"], utc=True).astype("datetime64[ns, UTC]")
            pubs["instrument_id"] = pubs["instrument_id"].astype(object)
            pubs["available_at_utc"] = pd.to_datetime(pubs["available_at_utc"], utc=True).astype("datetime64[ns, UTC]")
            pubs = pubs.drop_duplicates(["instrument_id", "available_at_utc"], keep="last").sort_values("available_at_utc")
            rows = pd.merge_asof(
                target.sort_values("timestamp_utc"),
                pubs,
                by="instrument_id",
                left_on="timestamp_utc",
                right_on="available_at_utc",
                direction="backward",
            ).dropna(subset=[feature])
            # A daily session-close publication is valid only through the next trading session.
            rows = rows.loc[
                (rows["pub_session_id"] == rows["session_id"])
                | (rows["pub_session_id"] == rows["session_id"].map(prior_session_map))
            ]
            if rows.empty:
                continue
            rows["date"] = rows["timestamp_utc"]
        else:
            rows["date"] = pd.to_datetime(rows.session_id) if frequency == "1d" else rows.timestamp_utc
        wide = rows.pivot(index="date", columns="instrument_id", values=feature)
        aligned = wide.reindex(pd.DatetimeIndex(dates, name="date"))
        if shift:
            aligned = aligned.shift(shift)
        parts.append(aligned.stack().rename_axis(["date", "stock_code"]))
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


# =============================================================================
# Filter Construction Utilities
# =============================================================================


def build_stock_filters(
    stock_pool_cfg: Dict[str, Any],
    price_conn: Any,
    fundamental_table: str,
    price_table: str,
    db_backend: str,
    duckdb_path: Optional[str] = None,
    include_basic: bool = True,
) -> List[StockFilter]:
    """
    根据配置构建 StockFilter 列表
    
    Args:
        stock_pool_cfg: stock_pool 配置节
        price_conn: 价格数据库连接
        fundamental_table: 基础数据表名
        price_table: 价格表名
        db_backend: 数据库后端类型
        duckdb_path: DuckDB 文件路径
        include_basic: 是否包含基础过滤器
        
    Returns:
        StockFilter 对象列表
    """
    if not isinstance(stock_pool_cfg, Mapping):
        raise ValueError("stock_pool must be a mapping")
    configured_filters = stock_pool_cfg.get('filters', [])
    if not isinstance(configured_filters, (list, tuple)):
        raise ValueError("stock_pool.filters must be a list or tuple")

    filters: List[StockFilter] = []
    for index, condition in enumerate(configured_filters):
        if not isinstance(condition, Mapping):
            raise ValueError(f"stock_pool.filters[{index}] must be a mapping")
        field = condition.get('field')
        operator = condition.get('operator')
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"stock_pool.filters[{index}].field must be a nonempty string")
        if not isinstance(operator, str) or not operator.strip():
            raise ValueError(f"stock_pool.filters[{index}].operator must be a nonempty string")
        filters.append(StockFilter(field, operator, condition.get('value'), condition.get('description', '')))

    # Validate custom entries before initializing a manager; basic filters still run first.
    if include_basic and stock_pool_cfg.get('include_basic_filters', False):
        tmp_manager = DynamicStockPoolManager(
            price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
        )
        filters = tmp_manager.add_basic_filters() + filters

    return filters


@lru_cache(maxsize=8)
def _create_pool_manager(
    price_conn: Any,
    fundamental_table: str,
    price_table: str,
    db_backend: str,
    duckdb_path: Optional[str] = None,
) -> DynamicStockPoolManager:
    """Internal helper to create DynamicStockPoolManager with common parameters."""
    return DynamicStockPoolManager(
        price_conn,
        fundamental_table=fundamental_table,
        price_table=price_table,
        db_backend=db_backend,
        duckdb_path=duckdb_path,
    )


# =============================================================================
# Dynamic Stock Pool Filtering
# =============================================================================


def apply_dynamic_stock_pool_filter(
    df: pd.DataFrame,
    price_conn: Any,
    config: Dict[str, Any],
    fundamental_table: str,
    price_table: str,
    db_backend: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    custom_logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    按配置的调仓周期，可选地应用动态股票池过滤。

    Args:
        df: MultiIndex (date, stock_code) 的 DataFrame
        price_conn: 价格数据库连接
        config: 完整配置字典
        fundamental_table: 基础数据表名
        price_table: 价格表名
        db_backend: 数据库后端
        start_date: 股票池生成开始日期；缺省时取面板最早日期
        end_date: 股票池生成结束日期；缺省时取面板最晚日期
        custom_logger: 可选自定义 logger

    Returns:
        过滤后的 DataFrame

    启用过滤后，数据源、日期和选股错误直接传播；合法空池返回空面板。
    """
    log = custom_logger or logger

    stock_pool_cfg = config.get('stock_pool', {})
    use_dynamic = bool(stock_pool_cfg.get('use_dynamic_pools', False))
    if not use_dynamic:
        return df

    if df.empty:
        return df

    # 计算过滤所需的时间范围
    if start_date is None or end_date is None:
        idx_dates = df.index.get_level_values('date')
        # 兼容字符串日期
        idx_dates = pd.to_datetime(idx_dates, errors='raise')
        period_start = start_date if start_date is not None else idx_dates.min().strftime('%Y-%m-%d')
        period_end = end_date if end_date is not None else idx_dates.max().strftime('%Y-%m-%d')
    else:
        period_start, period_end = start_date, end_date

    # 构建过滤器
    duckdb_path = config.get('database', {}).get('price_db', {}).get('duckdb_path')
    filters = build_stock_filters(
        stock_pool_cfg=stock_pool_cfg,
        price_conn=price_conn,
        fundamental_table=fundamental_table,
        price_table=price_table,
        db_backend=db_backend,
        duckdb_path=duckdb_path,
        include_basic=True,
    )

    manager = _create_pool_manager(
        price_conn=price_conn,
        fundamental_table=fundamental_table,
        price_table=price_table,
        db_backend=db_backend,
        duckdb_path=duckdb_path,
    )
    stock_pools = manager.get_stock_pools_for_period(
        start_date=period_start,
        end_date=period_end,
        filters=filters,
        max_stocks=stock_pool_cfg.get('max_stocks'),
        sort_by=stock_pool_cfg.get('sort_by'),
        ascending=bool(stock_pool_cfg.get('ascending', False)),
        rebalance_period_months=int(stock_pool_cfg.get('rebalance_period_months', 1)),
        rebalance_schedule=stock_pool_cfg.get('rebalance_schedule', 'quarterly'),
        schedule=stock_pool_cfg.get('schedule'),
    )

    if not stock_pools:
        log.warning("动态股票池结果为空，返回空面板")
        return df.iloc[:0].copy()

    # 使用共享工具函数进行日期级别动态股票池过滤
    total = len(df)
    filtered_df = apply_date_level_pool_filter_to_long_df(df, stock_pools)
    kept = len(filtered_df)

    log.info(f"应用动态股票池过滤: 保留 {kept}/{total} 行 ({kept/total:.2%})")
    if kept == 0:
        log.warning("动态股票池过滤后无数据，请检查过滤条件与时间范围")

    return filtered_df


def get_all_pool_stocks(
    price_conn: Any,
    config: Dict[str, Any],
    fundamental_table: str,
    price_table: str,
    db_backend: str,
    start_date: str,
    end_date: str,
    custom_logger: Optional[logging.Logger] = None,
) -> Optional[Set[str]]:
    """
    预计算所有期间股票池的并集，用于 SQL 级别过滤以减少内存占用。
    日期排除由训练、评估或回测阶段处理，不在此处缩小读取范围。
    
    Args:
        price_conn: 价格数据库连接
        config: 完整配置字典
        fundamental_table: 基础数据表名
        price_table: 价格表名
        db_backend: 数据库后端
        start_date: 开始日期
        end_date: 结束日期
        custom_logger: 可选自定义 logger
        
    Returns:
        所有池内股票代码的集合，如果未启用动态池则返回 None
    """
    log = custom_logger or logger
    
    stock_pool_cfg = config.get('stock_pool', {})
    if not stock_pool_cfg.get('use_dynamic_pools', False):
        return None
    
    try:
        # 构建过滤器
        duckdb_path = config.get('database', {}).get('price_db', {}).get('duckdb_path')
        filters = build_stock_filters(
            stock_pool_cfg=stock_pool_cfg,
            price_conn=price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
            include_basic=True,
        )
        
        manager = _create_pool_manager(
            price_conn=price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
        )
        
        monthly_pools = manager.get_stock_pools_for_period(
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            max_stocks=stock_pool_cfg.get('max_stocks'),
            sort_by=stock_pool_cfg.get('sort_by'),
            ascending=bool(stock_pool_cfg.get('ascending', False)),
            rebalance_period_months=int(stock_pool_cfg.get('rebalance_period_months', 1)),
            rebalance_schedule=stock_pool_cfg.get('rebalance_schedule', 'quarterly'),
            schedule=stock_pool_cfg.get('schedule'),
        )
        
        if not monthly_pools:
            log.warning("预计算动态股票池为空")
            return None
        
        # 计算所有期间股票的并集
        all_stocks: Set[str] = set()
        for stocks in monthly_pools.values():
            if stocks:
                all_stocks.update(stocks)
        
        # Debug: 显示日期范围和样本池信息
        pool_dates = sorted(monthly_pools.keys())
        sample_counts = [len(monthly_pools[d]) for d in pool_dates[:3]]
        log.info(f"[内存优化] 预计算动态股票池: {len(monthly_pools)} 个调仓期，共 {len(all_stocks)} 只唯一股票")
        log.info(f"[内存优化] 池日期范围: {pool_dates[0]} ~ {pool_dates[-1]}, 请求范围: {start_date} ~ {end_date}")
        log.info(f"[内存优化] 前3个池股票数: {sample_counts}")
        
        return all_stocks
        
    except Exception as e:
        log.error(f"预计算动态股票池失败: {e}")
        raise


def compute_predict_period_dynamic_pools(
    price_conn: Any,
    config: Dict[str, Any],
    fundamental_table: str,
    price_table: str,
    db_backend: str,
    train_end,
    predict_window_days: int,
    method_name: str = "synthesis",
    custom_logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Set[str]]]:
    """
    计算预测期的动态股票池。
    
    供 rolling_window_synthesis 中的 LASSO/LightGBM/SizeConditionalElasticNet 共用。
    
    Args:
        price_conn: 价格数据库连接
        config: 完整配置字典
        fundamental_table: 基础数据表名
        price_table: 价格表名
        db_backend: 数据库后端
        train_end: 训练结束日期 (datetime)
        predict_window_days: 相对预测开始日的自然日结束偏移量，沿用上下游两端包含约定
        method_name: 方法名称，用于日志
        custom_logger: 可选自定义 logger
        
    Returns:
        动态股票池字典 {date_str: set(stock_codes)}，未启用时返回 None。
        有交易日但没有生效期时抛错；纯休市窗口返回显式空池。
        某个生效期明确为空成员仍合法。
    """
    log = custom_logger or logger
    
    predict_start = train_end + timedelta(days=1)
    predict_end = predict_start + timedelta(days=predict_window_days)
    
    stock_pool_cfg = config.get('stock_pool', {})
    if not stock_pool_cfg.get('use_dynamic_pools', False):
        return None
    
    try:
        duckdb_path = config.get('database', {}).get('price_db', {}).get('duckdb_path')
        
        # Preserve the prediction policy: validate custom conditions without basic filters.
        filters = build_stock_filters(
            stock_pool_cfg=stock_pool_cfg,
            price_conn=price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
            include_basic=False,
        )

        manager = _create_pool_manager(
            price_conn=price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
        )
        
        raw_pools = manager.get_stock_pools_for_period(
            start_date=predict_start.strftime('%Y-%m-%d'),
            end_date=predict_end.strftime('%Y-%m-%d'),
            filters=filters,
            max_stocks=stock_pool_cfg.get('max_stocks'),
            sort_by=stock_pool_cfg.get('sort_by'),
            ascending=bool(stock_pool_cfg.get('ascending', False)),
            rebalance_period_months=int(stock_pool_cfg.get('rebalance_period_months', 1)),
            rebalance_schedule=stock_pool_cfg.get('rebalance_schedule', 'weekly'),
            schedule=stock_pool_cfg.get('schedule'),
        )
        
        if not raw_pools:
            # A holiday-only final window has no predictions; keep it explicitly empty.
            if manager._trading_calendar().sessions(predict_start, predict_end).empty:
                return {predict_start.strftime('%Y-%m-%d'): set()}
            raise ValueError(
                f"Dynamic prediction pool has no effective periods: "
                f"{predict_start:%Y-%m-%d} ~ {predict_end:%Y-%m-%d}"
            )
        return {k: set(v) for k, v in raw_pools.items()}
        
    except Exception as e:
        log.warning(f"[动态池] {method_name}预测期池计算失败: {e}")
        raise


def compute_dynamic_pools_for_topk(
    price_conn: Any,
    config: Dict[str, Any],
    fundamental_table: str,
    price_table: str,
    db_backend: str,
    start_date: str,
    end_date: str,
    custom_logger: Optional[logging.Logger] = None,
) -> Dict[str, List[str]]:
    """
    为 Top-K 策略评估计算动态股票池
    
    与 get_all_pool_stocks 使用相同的配置参数，确保合成和回测阶段使用一致的股票池。
    
    Args:
        price_conn: 价格数据库连接
        config: 完整配置字典
        fundamental_table: 基础数据表名
        price_table: 价格表名
        db_backend: 数据库后端
        start_date: 开始日期
        end_date: 结束日期
        custom_logger: 可选自定义 logger
        
    Returns:
        {date_str: [stock_codes]} 按生效日期的候选池，未启用时返回 {}。
        有交易日但没有生效期时抛错；纯休市窗口返回显式空池。
        跳月规则由回测器单独处理，不删除本函数的池历史。
    """
    log = custom_logger or logger
    
    stock_pool_cfg = config.get('stock_pool', {})
    use_dynamic = stock_pool_cfg.get('use_dynamic_pools', False)
    
    if not use_dynamic:
        log.info("[Top-K动态池] 配置未启用动态股票池，跳过")
        return {}
    
    try:
        # 构建过滤器 - 与 get_all_pool_stocks 保持一致
        duckdb_path = config.get('database', {}).get('price_db', {}).get('duckdb_path')
        filters = build_stock_filters(
            stock_pool_cfg=stock_pool_cfg,
            price_conn=price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
            include_basic=True,
        )
        
        # 从配置读取所有参数
        rebalance_schedule = stock_pool_cfg.get('rebalance_schedule', 'quarterly')
        rebalance_period_months = int(stock_pool_cfg.get('rebalance_period_months', 1))
        max_stocks = stock_pool_cfg.get('max_stocks', None)
        sort_by = stock_pool_cfg.get('sort_by')
        ascending = bool(stock_pool_cfg.get('ascending', False))
        
        # 创建 DynamicStockPoolManager
        manager = _create_pool_manager(
            price_conn=price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
        )
        
        pools = manager.get_stock_pools_for_period(
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            max_stocks=max_stocks,
            sort_by=sort_by,
            ascending=ascending,
            rebalance_period_months=rebalance_period_months,
            rebalance_schedule=rebalance_schedule,
            schedule=stock_pool_cfg.get('schedule'),
        )
        
        if not pools:
            # Keep an enabled empty window distinct from disabled membership filtering.
            if manager._trading_calendar().sessions(start_date, end_date).empty:
                return {pd.Timestamp(start_date).strftime('%Y-%m-%d'): []}
            raise ValueError(
                f"Dynamic Top-K pool has no effective periods: {start_date} ~ {end_date}"
            )
        
        return pools
        
    except Exception as e:
        log.error(f"[Top-K动态池] 计算失败: {e}")
        raise


# =============================================================================
# Skip Months Filtering
# =============================================================================


def apply_skip_months(
    obj: Union[pd.DataFrame, pd.Series, None],
    skip_months: Optional[SkipRules],
    custom_logger: Optional[logging.Logger] = None,
) -> Union[pd.DataFrame, pd.Series, None]:
    """
    按配置跳过指定月份/日期窗口的数据。
    
    支持：
    - index 为 MultiIndex(date, stock_code) 的 DataFrame/Series
    - index 为 DatetimeIndex 或可转换为日期的单层索引
    日期缺失、数字索引或无法解析的日期会报错；不在失败后返回未过滤数据。
    
    Args:
        obj: DataFrame 或 Series
        skip_months: 跳过规则集合：整月 (1-12) 或 "MM-DD:MM-DD" 窗口字符串
        custom_logger: 可选自定义 logger
        
    Returns:
        过滤后的对象
    """
    log = custom_logger or logger
    
    if obj is None:
        return obj
    if not isinstance(obj, (pd.Series, pd.DataFrame)):
        raise TypeError("apply_skip_months expects a DataFrame or Series")
    if not skip_months or obj.empty:
        return obj

    idx = obj.index
    if isinstance(idx, pd.MultiIndex):
        if idx.names.count('date') != 1:
            raise ValueError("Calendar filtering requires exactly one 'date' index level")
        idx = idx.get_level_values('date')
    # Numbers would otherwise be interpreted as nanoseconds since the Unix epoch.
    if not isinstance(idx, pd.DatetimeIndex) and any(isinstance(value, Number) for value in idx):
        raise ValueError("Calendar filtering does not accept numeric dates")
    dates = pd.DatetimeIndex(pd.to_datetime(idx))
    if dates.hasnans:
        raise ValueError("Calendar filtering requires dates without NaT or missing values")

    mask = ~skip_mask(dates, skip_months)
    keep_count = int(mask.sum())
    total = len(mask)
    if keep_count < total:
        log.info(
            f"应用skip_months过滤: 保留 {keep_count}/{total} 行 ({keep_count/total:.2%})，"
            f"跳过规则={describe_skip_rules(skip_months)}"
        )
    return obj.iloc[mask]
