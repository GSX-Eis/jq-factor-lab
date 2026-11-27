"""Point-in-time projections of local A-share feature tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from utils.bars.catalog import identifier, table_exists
from utils.market_db import MarketDbProfile
from utils.market_db.validation import calendar_key, market_db_calendar_bounds
from utils.session_calendar import SessionCalendar

from .feed import BAR_KEYS, local_day, table_columns
from .panel import TIMEZONE


@dataclass(frozen=True)
class _FeatureTable:
    clock: str
    columns: tuple[str, ...]
    instrument: tuple[str, ...] = ('stock_code',)


_TABLES = {
    **{table: _FeatureTable('session_close', ('date',)) for table in (
        'factor_data', 'valuation_daily', 'style_factor_exposure',
        'money_flow_daily', 'northbound_holding_daily', 'margin_trading_daily', 'billboard_events',
    )},
    **{table: _FeatureTable('published', ('pub_date',)) for table in (
        'earnings_forecast_events', 'dividend_quality_financial_events', 'earned_equity_financial_events',
        'statutory_cash_claim_events', 'fund_portfolio_stock_events', 'shareholder_structure',
        'capital_change_events', 'holder_num', 'shares_pledge_events', 'shares_frozen_events',
        'shareholder_share_change_events', 'shareholder_top10_events',
    )},
    **{table: _FeatureTable('published', ('pub_date',), ('code', 'stock_code')) for table in (
        'raw_financial_statements', 'raw_income_statement_events', 'raw_balance_statement_events',
        'raw_cashflow_statement_events', 'listing_compliance_status_events',
    )},
    'industry_membership': _FeatureTable('snapshot_close', ('date',)),
    'index_price_daily': _FeatureTable('session_close', ('date',), ('index_code',)),
    'audit_opinion_events': _FeatureTable('effective', ('activation_date',)),
    'locked_shares_events': _FeatureTable('published', ('date',)),
}
_DIVIDEND_PAYOUT = _FeatureTable('published', ('a_ex_dividend_date', 'implementation_pub_date'))
_DIVIDEND_FIELDS = {
    **{field: _FeatureTable('published', (clock,)) for clock, fields in (
        ('board_plan_pub_date', ('board_plan_pub_date', 'board_plan_bonusnote')),
        ('shareholders_plan_pub_date', ('shareholders_plan_pub_date', 'shareholders_plan_bonusnote')),
        ('implementation_pub_date', ('implementation_pub_date', 'implementation_bonusnote',
                                     'a_registration_date', 'a_bonus_date')),
        ('bonus_cancel_pub_date', ('bonus_cancel_pub_date',)),
    ) for field in fields},
    # The source stores the final status only, so it requires the latest dated phase.
    'plan_progress': _FeatureTable('published', (
        'a_ex_dividend_date', 'board_plan_pub_date', 'shareholders_plan_pub_date',
        'implementation_pub_date', 'bonus_cancel_pub_date')),
}
_REGIME_TABLES = {
    'regime_cninfo_': _FeatureTable('utc', ('announcement_timestamp',), ('issuer_stock_code',)),
    'regime_csrc_': _FeatureTable('local', ('publication_timestamp',), ('issuer_stock_code',)),
    'regime_exchange_': _FeatureTable('local', ('publication_timestamp',), ('issuer_stock_code',)),
}
_AVAILABLE = 'available_at_utc'
_RESULT_KEYS = [*BAR_KEYS, _AVAILABLE]


def _source(table: str, field: str) -> _FeatureTable:
    if table == 'dividend_events':
        return _DIVIDEND_FIELDS.get(field, _DIVIDEND_PAYOUT)
    if table in _TABLES:
        return _TABLES[table]
    for prefix, spec in _REGIME_TABLES.items():
        if table.startswith(prefix):
            return spec
    raise ValueError(f'Unsupported A-share auxiliary table {table!r}')


class AShareAuxFeatures:
    def __init__(self, connection: duckdb.DuckDBPyConnection, calendar: SessionCalendar, *,
                 market_db_profile: MarketDbProfile | None = None) -> None:
        self.connection = connection
        self.calendar = calendar
        self.market_db_profile = market_db_profile

    def _feature_groups(self, features: Sequence[str]) -> dict[tuple[str, _FeatureTable], dict[str, str]]:
        if isinstance(features, (str, bytes)):
            raise ValueError('features must be a sequence of field names')
        groups: dict[tuple[str, _FeatureTable], dict[str, str]] = {}
        for feature in features:
            if not isinstance(feature, str):
                raise ValueError('features must be strings')
            table, separator, field = feature.partition('.')
            if not separator:
                table, field = 'factor_data', table
            if feature in _RESULT_KEYS:
                raise ValueError(f'Reserved A-share feature name {feature!r}')
            identifier(table)
            identifier(field)
            groups.setdefault((table, _source(table, field)), {})[feature] = field
        return groups

    def _read_industry(self, grid: pd.DataFrame, fields: dict[str, str], projection: str) -> pd.DataFrame:
        start, end = local_day(grid.timestamp_utc.min()), local_day(grid.timestamp_utc.max())
        closes = self.calendar.bar_grid('1d', start, end)
        dates = self.connection.execute(
            """SELECT DISTINCT CAST(date AS DATE) AS snapshot_day FROM industry_membership
               WHERE CAST(date AS DATE) <= ?
                 AND CAST(date AS DATE) >= COALESCE(
                     (SELECT max(CAST(date AS DATE)) FROM industry_membership WHERE CAST(date AS DATE) < ?), ?)
               ORDER BY snapshot_day""", [end, start, start]).fetchnumpy()['snapshot_day']
        if len(dates) and pd.Timestamp(dates[0]) < start:
            previous = pd.Timestamp(dates[0])
            profile = self.market_db_profile
            if profile is None or previous >= market_db_calendar_bounds(profile)[0]:
                calendar_id = calendar_key(self.calendar.calendar_id, market_domain='cn_equity')
                calendar = (self.calendar if previous >= self.calendar.start
                            else SessionCalendar(calendar_id, previous, previous))
                closes = pd.concat([calendar.bar_grid('1d', previous, previous), closes], ignore_index=True)

        snapshot_days = pd.DatetimeIndex(dates).strftime('%Y-%m-%d')
        receipts = closes.loc[closes.session_id.isin(snapshot_days)].rename(
            columns={'session_id': 'snapshot_id', 'timestamp_utc': _AVAILABLE})
        matched = pd.merge_asof(grid.sort_values('timestamp_utc'), receipts.sort_values(_AVAILABLE),
                                left_on='timestamp_utc', right_on=_AVAILABLE)
        snapshots = self.connection.execute(
            f'SELECT stock_code AS instrument_id, CAST(date AS DATE)::VARCHAR AS snapshot_id, {projection} '
            'FROM industry_membership WHERE stock_code IN (SELECT unnest(?)) '
            'AND CAST(date AS DATE) IN (SELECT unnest(?)::DATE)',
            [list(grid.instrument_id.unique()), list(snapshot_days)]).fetchdf()
        # Membership comes from the latest complete market snapshot, including additions and removals.
        return matched.merge(snapshots, on=['instrument_id', 'snapshot_id'])[[*_RESULT_KEYS, *fields]]

    def _read_table(self, grid: pd.DataFrame, table: str, spec: _FeatureTable,
                    fields: dict[str, str]) -> pd.DataFrame:
        columns = table_columns(self.connection, table)
        if spec.clock == 'effective' and spec.columns[0] not in columns:
            spec = _FeatureTable('published', ('pub_date',))
        missing = set(fields.values()) - columns.keys()
        key = next((key for key in spec.instrument if key in columns), None)
        if key is None or spec.columns[0] not in columns or missing:
            raise ValueError(f'{table} lacks instrument, availability or requested fields {sorted(missing)}')
        projection = ', '.join(f'{identifier(field)} AS "{name}"' for name, field in fields.items())
        if spec.clock == 'snapshot_close':
            return self._read_industry(grid, fields, projection)
        clocks = [identifier(value) for value in spec.columns if value in columns]
        clock = clocks[0] if len(clocks) == 1 else f'greatest({", ".join(clocks)})'
        if spec.clock == 'session_close':
            clock = f'CAST({clock} AS DATE)'
        ids = list(grid.instrument_id.unique())
        cutoff = grid.timestamp_utc.max()
        lower = ''
        params = [ids]
        if spec.clock == 'session_close':
            lower = f'AND {clock} >= ?'
            params.append(grid.session_id.min())
        params.append(cutoff if 'WITH TIME ZONE' in columns[spec.columns[0]]
                      else (cutoff.tz_localize(None) if spec.clock == 'utc'
                            else cutoff.tz_convert(TIMEZONE).tz_localize(None)))
        rows = self.connection.execute(
            f'SELECT {identifier(key)} AS instrument_id, {clock} AS observed_at, {projection} '
            f'FROM {identifier(table)} WHERE {identifier(key)} IN (SELECT unnest(?)) '
            f'{lower} AND {clock} <= ?', params).fetchdf()
        rows['instrument_id'] = rows.instrument_id.astype(object)
        observed = pd.to_datetime(rows.pop('observed_at'))
        if spec.clock == 'session_close':
            rows['session_id'] = observed.dt.strftime('%Y-%m-%d')
            start = local_day(grid.timestamp_utc.min())
            end = local_day(cutoff)
            closes = self.calendar.bar_grid('1d', start, end).rename(columns={'timestamp_utc': _AVAILABLE})
            rows = rows.merge(closes, on='session_id')
            aligned = grid.merge(rows, on=['instrument_id', 'session_id'])
            return aligned.loc[aligned[_AVAILABLE].le(aligned.timestamp_utc), [*_RESULT_KEYS, *fields]]
        if spec.clock == 'published':
            if 'TIMESTAMP' not in columns[spec.columns[0]]:
                observed += pd.Timedelta(days=1)
        if observed.dt.tz is None:
            observed = observed.dt.tz_localize('UTC' if spec.clock == 'utc' else TIMEZONE)
        rows[_AVAILABLE] = observed.dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
        rows = rows.loc[rows[_AVAILABLE].le(cutoff)]
        # Retain every record at the most recently available publication instant.
        receipts = rows[['instrument_id', _AVAILABLE]].drop_duplicates().sort_values(_AVAILABLE)
        matched = pd.merge_asof(grid.sort_values('timestamp_utc'), receipts, by='instrument_id',
                                left_on='timestamp_utc', right_on=_AVAILABLE)
        return matched.merge(rows, on=['instrument_id', _AVAILABLE])[[*_RESULT_KEYS, *fields]]

    def read_aux_features(self, grid: pd.DataFrame, *, features: Sequence[str]) -> pd.DataFrame:
        """Return table-specific rows, with daily snapshots and latest eligible event batches."""
        groups = self._feature_groups(features)
        keyed = grid[BAR_KEYS].copy()
        keyed['instrument_id'] = keyed.instrument_id.astype(object)
        keyed['timestamp_utc'] = keyed.timestamp_utc.astype('datetime64[ns, UTC]')
        frames = [self._read_table(keyed, table, spec, fields) for (table, spec), fields in groups.items()
                  if not keyed.empty and table_exists(self.connection, table)]
        if not frames:
            empty = keyed.iloc[:0].assign(available_at_utc=pd.DatetimeIndex([], tz='UTC'))
            return empty.reindex(columns=[*_RESULT_KEYS, *dict.fromkeys(features)])
        return pd.concat(frames, ignore_index=True).reindex(
            columns=[*_RESULT_KEYS, *dict.fromkeys(features)])

    def list_table_features(self, table: str) -> list[str]:
        """Return qualified feature names for an auxiliary table if present."""
        identifier(table)
        if not table_exists(self.connection, table):
            return []
        excluded = {'stock_code', 'code', 'issuer_stock_code', 'index_code', 'date', 'pub_date', 'activation_date'}
        return [f'{table}.{col}' for col in table_columns(self.connection, table) if col not in excluded]

    @staticmethod
    def default_regime_indices() -> tuple[str, str]:
        """Return default A-share micro-cap and benchmark index instruments."""
        return ('399101.XSHE', '000300.XSHG')

    @property
    def universe_filter_plugin(self) -> AShareDividendQualityPlugin:
        """Return the A-share PIT dividend-quality UniverseFilterPlugin."""
        return AShareDividendQualityPlugin(self.connection)

    def read_dividend_quality_history(
        self,
        stocks: Sequence[str],
        start_date: Any,
        end_date: Any,
    ) -> dict[str, pd.DataFrame]:
        """Load multi-year PIT financial, dividend, valuation and industry histories."""
        return _read_dividend_quality_history(self.connection, stocks, start_date, end_date)


def _read_dividend_quality_history(
    connection: duckdb.DuckDBPyConnection,
    stocks: Sequence[str],
    start_date: Any,
    end_date: Any,
) -> dict[str, pd.DataFrame]:
    """Read multi-year PIT histories for A-share dividend quality screening."""
    stock_list = sorted({str(s) for s in stocks})
    if not stock_list:
        return {
            'financials': pd.DataFrame(),
            'dividends': pd.DataFrame(),
            'valuation': pd.DataFrame(),
            'industry': pd.DataFrame(),
        }
    placeholders = ','.join('?' for _ in stock_list)
    financials = connection.execute(
        f"""
        SELECT statement_type, stock_code, pub_date, start_date, end_date,
               net_profit, net_operate_cash_flow, total_liability, total_assets
        FROM dividend_quality_financial_events
        WHERE stock_code IN ({placeholders}) AND pub_date <= ?
        """,
        [*stock_list, end_date],
    ).df()
    dividend_sql = f"""
        SELECT stock_code, report_date, implementation_pub_date,
               board_plan_pub_date, shareholders_plan_pub_date,
               bonus_type, bonus_amount_rmb, bonus_ratio_rmb,
               dividend_ratio, transfer_ratio
        FROM dividend_events
        WHERE stock_code IN ({placeholders})
          AND COALESCE(board_plan_pub_date, shareholders_plan_pub_date,
                       implementation_pub_date) IS NOT NULL
          AND COALESCE(board_plan_pub_date, shareholders_plan_pub_date,
                       implementation_pub_date) <= ?
          AND bonus_type = '年度分红'
          AND COALESCE(bonus_amount_rmb, 0) > 0
    """
    try:
        dividends = connection.execute(dividend_sql, [*stock_list, end_date]).df()
    except duckdb.BinderException:
        # Legacy fixtures/DBs predate the early announcement columns.
        dividends = connection.execute(
            f"""
            SELECT stock_code, report_date, implementation_pub_date,
                   bonus_type, bonus_amount_rmb, bonus_ratio_rmb
            FROM dividend_events
            WHERE stock_code IN ({placeholders})
              AND implementation_pub_date IS NOT NULL
              AND implementation_pub_date <= ?
              AND bonus_type = '年度分红'
              AND COALESCE(bonus_amount_rmb, 0) > 0
            """,
            [*stock_list, end_date],
        ).df()
        for column in ('board_plan_pub_date', 'shareholders_plan_pub_date', 'dividend_ratio', 'transfer_ratio'):
            dividends[column] = pd.NaT if column.endswith('_pub_date') else 0.0
    valuation = connection.execute(
        f"""
        SELECT stock_code, date, market_cap
        FROM valuation_daily
        WHERE stock_code IN ({placeholders}) AND date BETWEEN ? AND ?
        """,
        [*stock_list, start_date, end_date],
    ).df()
    industry = connection.execute(
        f"""
        SELECT stock_code, date, sw_l1_name
        FROM industry_membership
        WHERE stock_code IN ({placeholders}) AND date BETWEEN ? AND ?
        """,
        [*stock_list, start_date, end_date],
    ).df()
    return {
        'financials': financials,
        'dividends': dividends,
        'valuation': valuation,
        'industry': industry,
    }


class AShareDividendQualityPlugin:
    """PIT dividend-quality UniverseFilterPlugin backed by local A-share DuckDB history."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection

    def filter_universe(
        self,
        dynamic_pools: Mapping[str, list[str]],
        *,
        config: Mapping[str, Any],
    ) -> tuple[dict[str, list[str]], dict[str, Any]]:
        cfg = dict(config or {})
        if not bool(cfg.get('enabled', False)):
            return {str(k): list(v or []) for k, v in dynamic_pools.items()}, {
                'enabled': False,
                'qualified_counts': {},
            }
        if not dynamic_pools:
            raise RuntimeError('quality filter enabled but dynamic pool is empty')
        dates = sorted(pd.Timestamp(k).normalize() for k in dynamic_pools)
        stocks = sorted({str(s) for values in dynamic_pools.values() for s in (values or [])})
        if not stocks:
            raise RuntimeError('quality filter enabled but dynamic pool has no stocks')
        start_date, end_date = dates[0].date(), dates[-1].date()
        history = _read_dividend_quality_history(self.connection, stocks, start_date, end_date)
        financials = history['financials']
        dividends = history['dividends']
        valuation = history['valuation']
        industry = history['industry']

        for frame, cols in (
            (financials, ['pub_date', 'start_date', 'end_date']),
            (dividends, [
                'implementation_pub_date', 'board_plan_pub_date',
                'shareholders_plan_pub_date', 'report_date',
            ]),
            (valuation, ['date']),
            (industry, ['date']),
        ):
            for col in cols:
                if col in frame:
                    frame[col] = pd.to_datetime(frame[col])

        financials = financials.sort_values(['stock_code', 'pub_date', 'end_date'])
        dividends['announcement_date'] = dividends[
            ['board_plan_pub_date', 'shareholders_plan_pub_date', 'implementation_pub_date']
        ].bfill(axis=1).iloc[:, 0]
        dividends['comparable_dps'] = (
            pd.to_numeric(dividends['bonus_ratio_rmb'], errors='coerce') / 10.0
        ) * (
            1.0
            + pd.to_numeric(dividends['dividend_ratio'], errors='coerce').fillna(0.0) / 10.0
            + pd.to_numeric(dividends['transfer_ratio'], errors='coerce').fillna(0.0) / 10.0
        )
        dividends = dividends.sort_values(['stock_code', 'report_date', 'announcement_date'])
        valuation = valuation.sort_values(['stock_code', 'date'])
        industry = industry.sort_values(['stock_code', 'date'])
        fin_by_stock = {k: v for k, v in financials.groupby('stock_code', sort=False)}
        div_by_stock = {k: v for k, v in dividends.groupby('stock_code', sort=False)}
        val_by_stock = {k: v for k, v in valuation.groupby('stock_code', sort=False)}
        ind_by_stock = {k: v for k, v in industry.groupby('stock_code', sort=False)}

        if str(cfg.get('mode', '') or '') == 'dividend_cut_veto':
            return _apply_dividend_cut_veto(dynamic_pools, div_by_stock, cfg)

        filtered: dict[str, list[str]] = {}
        diagnostics: dict[str, Any] = {'enabled': True, 'qualified_counts': {}, 'eligible_counts': {}}
        for raw_date, pool in dynamic_pools.items():
            asof = pd.Timestamp(raw_date).normalize()
            rows = []
            for stock in pool or []:
                row = _quality_features_for_stock(
                    asof,
                    fin_by_stock.get(stock),
                    div_by_stock.get(stock),
                    val_by_stock.get(stock),
                    ind_by_stock.get(stock),
                )
                if row is not None:
                    row['stock_code'] = stock
                    rows.append(row)
            frame = pd.DataFrame(rows)
            if frame.empty:
                filtered[str(raw_date)] = []
                diagnostics['qualified_counts'][str(raw_date)] = 0
                diagnostics['eligible_counts'][str(raw_date)] = 0
                continue
            hard = (
                (frame['dividend_years'] >= 3)
                & frame['payout_mean'].between(0.10, 1.00, inclusive='both')
                & (frame['roe_count'] >= 12)
            )
            eligible = frame.loc[hard].copy()
            diagnostics['eligible_counts'][str(raw_date)] = int(len(eligible))
            if eligible.empty:
                filtered[str(raw_date)] = []
                diagnostics['qualified_counts'][str(raw_date)] = 0
                continue
            eligible['is_finance'] = eligible['industry_name'].fillna('').str.contains(
                '银行|非银金融|金融', regex=True
            )
            score_parts = ['roe_mean', 'roe_change', 'roe_stability', 'dividend_yield']
            for group_key, group in eligible.groupby('is_finance', sort=False):
                idx = group.index
                parts = [group[col].rank(pct=True) for col in score_parts]
                if not bool(group_key):
                    parts.append(group['cfo_liability'].rank(pct=True))
                eligible.loc[idx, 'quality_score'] = pd.concat(parts, axis=1).mean(axis=1)
            threshold = float(cfg.get('min_score_percentile', 0.5))
            selected = eligible.loc[eligible['quality_score'] >= threshold, 'stock_code']
            filtered[str(raw_date)] = selected.astype(str).tolist()
            diagnostics['qualified_counts'][str(raw_date)] = int(len(selected))

        diagnostics['min_qualified_count'] = min(diagnostics['qualified_counts'].values(), default=0)
        diagnostics['min_eligible_count'] = min(diagnostics['eligible_counts'].values(), default=0)
        if diagnostics['min_qualified_count'] < int(cfg.get('min_qualified_count', 1)):
            raise RuntimeError(
                f"quality filter produced too few names: min={diagnostics['min_qualified_count']}"
            )
        return filtered, diagnostics


def _apply_dividend_cut_veto(
    dynamic_pools: Mapping[str, list[str]],
    div_by_stock: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    min_growth = float(cfg.get('min_comparable_dps_growth', 0.0))
    filtered: dict[str, list[str]] = {}
    diagnostics: dict[str, Any] = {
        'enabled': True,
        'mode': 'dividend_cut_veto',
        'min_comparable_dps_growth': min_growth,
        'qualified_counts': {},
        'veto_counts': {},
        'unknown_counts': {},
    }
    for raw_date, pool in dynamic_pools.items():
        asof = pd.Timestamp(raw_date).normalize()
        selected: list[str] = []
        vetoed = unknown = 0
        for stock in pool or []:
            growth = _latest_disclosed_dps_growth(div_by_stock.get(stock), asof)
            if growth is None:
                unknown += 1
                selected.append(stock)
            elif growth < min_growth:
                vetoed += 1
            else:
                selected.append(stock)
        filtered[str(raw_date)] = selected
        diagnostics['qualified_counts'][str(raw_date)] = len(selected)
        diagnostics['veto_counts'][str(raw_date)] = vetoed
        diagnostics['unknown_counts'][str(raw_date)] = unknown
    diagnostics['min_qualified_count'] = min(diagnostics['qualified_counts'].values(), default=0)
    if diagnostics['min_qualified_count'] < int(cfg.get('min_qualified_count', 1)):
        raise RuntimeError(
            f"dividend cut veto produced too few names: min={diagnostics['min_qualified_count']}"
        )
    return filtered, diagnostics


def _latest_disclosed_dps_growth(dividends: pd.DataFrame | None, asof: pd.Timestamp) -> float | None:
    if dividends is None or dividends.empty:
        return None
    disclosed = dividends.loc[dividends['announcement_date'] <= asof].copy()
    if disclosed.empty:
        return None
    disclosed['year'] = disclosed['report_date'].dt.year
    yearly = disclosed.groupby('year')['comparable_dps'].sum().sort_index()
    if len(yearly) < 2 or int(yearly.index[-1]) != int(yearly.index[-2]) + 1:
        return None
    prior = float(yearly.iloc[-2])
    return float(yearly.iloc[-1]) / prior - 1.0 if prior > 0 else None


def _quality_features_for_stock(
    asof: pd.Timestamp,
    financials: pd.DataFrame | None,
    dividends: pd.DataFrame | None,
    valuation: pd.DataFrame | None,
    industry: pd.DataFrame | None,
) -> dict[str, Any] | None:
    if financials is None or dividends is None or financials.empty or dividends.empty:
        return None
    asof_ns = asof.value
    nat = pd.NaT.value
    fin_pub = financials['pub_date'].to_numpy(dtype='datetime64[ns]').view('int64')
    fin_mask = (fin_pub != nat) & (fin_pub <= asof_ns)
    fin = financials.iloc[fin_mask]

    div_ann = dividends['announcement_date'].to_numpy(dtype='datetime64[ns]').view('int64')
    div_mask = (div_ann != nat) & (div_ann <= asof_ns)
    if not div_mask.any():
        return None
    div = dividends.iloc[div_mask].copy()

    # Preserve exact pandas sort/drop_duplicates tie-break & groupby aggregation semantics.
    annual = fin.loc[(fin['statement_type'] == 'income') & fin['end_date'].dt.month.eq(12)]
    annual = annual.sort_values('pub_date').drop_duplicates('end_date', keep='last')
    div['year'] = div['report_date'].dt.year
    yearly = div.groupby('year', as_index=False).agg(
        bonus_amount_rmb=('bonus_amount_rmb', 'sum'),
        comparable_dps=('comparable_dps', 'sum'),
    )
    annual_years = set(annual['end_date'].dt.year.astype(int))
    available_years = sorted(set(yearly['year'].astype(int)) & annual_years)
    recent_years = _latest_three_consecutive_years(available_years)
    yearly = yearly.loc[yearly['year'].isin(recent_years)]
    if len(recent_years) != 3:
        return None

    annual_profit: dict[int, float] = {}
    for end_dt, net_p in zip(annual['end_date'], annual['net_profit']):
        annual_profit[int(end_dt.year)] = float(net_p)
    payout = []
    for yr, bonus_rmb in zip(yearly['year'], yearly['bonus_amount_rmb']):
        profit = annual_profit.get(int(yr))
        if profit is None or profit <= 0:
            continue
        payout.append(float(bonus_rmb) * 10000.0 / profit)
    if len(payout) != 3:
        return None

    quarters = fin.loc[fin['statement_type'] == 'income'].sort_values('pub_date')
    quarters = quarters.drop_duplicates('end_date', keep='last')
    balances = fin.loc[fin['statement_type'] == 'balance'].sort_values('pub_date')
    balances = balances.drop_duplicates('end_date', keep='last')

    q_map: dict[pd.Timestamp, tuple[float, float]] = {}
    for end_dt, profit, cfo in zip(
        quarters['end_date'], quarters['net_profit'], quarters['net_operate_cash_flow']
    ):
        if pd.notna(end_dt):
            q_map[pd.Timestamp(end_dt)] = (float(profit), float(cfo))

    b_map: dict[pd.Timestamp, tuple[float, float]] = {}
    for end_dt, assets, liab in zip(
        balances['end_date'], balances['total_assets'], balances['total_liability']
    ):
        if pd.notna(end_dt):
            b_map[pd.Timestamp(end_dt)] = (float(assets), float(liab))

    roe_values = []
    cfo_values = []
    for end in sorted(q_map)[-16:]:
        if end > asof or end.month not in (3, 6, 9, 12):
            continue
        cur_profit, cur_cfo = q_map[end]
        if end.month == 12:
            ttm_profit = cur_profit
            ttm_cfo = cur_cfo
        else:
            prior_annual = q_map.get(pd.Timestamp(year=end.year - 1, month=12, day=31))
            prior_same = q_map.get(pd.Timestamp(year=end.year - 1, month=end.month, day=end.day))
            if prior_annual is None or prior_same is None:
                continue
            ttm_profit = cur_profit + prior_annual[0] - prior_same[0]
            ttm_cfo = cur_cfo + prior_annual[1] - prior_same[1]
        bal = b_map.get(end)
        if bal is None:
            continue
        equity = float(bal[0] - bal[1])
        prior_end = end - pd.offsets.QuarterEnd()
        prior_bal = b_map.get(prior_end)
        prior_equity = float(prior_bal[0] - prior_bal[1]) if prior_bal is not None else equity
        if equity <= 0 or prior_equity <= 0 or not np.isfinite(ttm_profit):
            continue
        roe_values.append(float(ttm_profit) / ((equity + prior_equity) / 2.0))
        cfo_values.append(float(ttm_cfo) / max(abs(float(bal[1])), 1e-12))
    if len(roe_values) < 12:
        return None
    market_cap = _asof_value(valuation, asof, 'market_cap')
    if market_cap is None or market_cap <= 0:
        return None
    dividend_yield = float(yearly['bonus_amount_rmb'].sum()) / float(market_cap) / 10000.0
    roe = np.asarray(roe_values[-12:], dtype=float)
    cfo = np.asarray(cfo_values[-12:], dtype=float)
    industry_name = ''
    if industry is not None and not industry.empty:
        ind_dates = industry['date'].to_numpy(dtype='datetime64[ns]').view('int64')
        if (ind_dates == nat).any():
            valid_idx = np.flatnonzero((ind_dates != nat) & (ind_dates <= asof_ns))
            pos = int(valid_idx[-1]) if valid_idx.size else -1
        else:
            pos = int(np.searchsorted(ind_dates, asof_ns, side='right')) - 1
        if pos >= 0:
            raw_ind = industry['sw_l1_name'].iloc[pos]
            industry_name = str(raw_ind or '')
    return {
        'dividend_years': len(yearly),
        'payout_mean': float(np.mean(payout)),
        'roe_count': len(roe),
        'roe_mean': float(np.mean(roe)),
        'roe_change': float(np.mean(roe[-4:]) - np.mean(roe[:4])),
        'roe_stability': float(-np.std(roe)),
        'cfo_liability': float(np.nanmean(cfo)) if np.isfinite(cfo).any() else 0.0,
        'dividend_yield': dividend_yield,
        'industry_name': industry_name,
    }


def _asof_value(frame: pd.DataFrame | None, asof: pd.Timestamp, column: str) -> float | None:
    if frame is None or frame.empty:
        return None
    dates = frame['date'].to_numpy(dtype='datetime64[ns]').view('int64')
    nat = pd.NaT.value
    asof_ns = asof.value
    if (dates == nat).any():
        valid_idx = np.flatnonzero((dates != nat) & (dates <= asof_ns))
        pos = int(valid_idx[-1]) if valid_idx.size else -1
    else:
        pos = int(np.searchsorted(dates, asof_ns, side='right')) - 1
    if pos < 0:
        return None
    value = frame[column].iloc[pos]
    return float(value) if pd.notna(value) else None


def _latest_three_consecutive_years(years: list[int]) -> list[int]:
    """Return the latest disclosed consecutive three-year block."""
    ordered = sorted(set(int(year) for year in years))
    blocks = [ordered[i : i + 3] for i in range(max(0, len(ordered) - 2))]
    blocks = [block for block in blocks if block[-1] - block[0] == 2]
    return blocks[-1] if blocks else []


def read_legacy_daily_execution_rows(
    connection: duckdb.DuckDBPyConnection,
    stocks: Sequence[str],
    date_str: str,
) -> pd.DataFrame:
    """Compatibility helper for minimal A-share DuckDB fixtures lacking full bar columns."""
    stock_list = list(stocks)
    if not stock_list or not table_exists(connection, 'price_data_daily'):
        return pd.DataFrame()
    present = table_columns(connection, 'price_data_daily')
    wanted = [col for col in ('open', 'close', 'high_limit', 'low_limit', 'paused', 'money') if col in present]
    if not wanted:
        return pd.DataFrame()
    placeholders = ', '.join('?' for _ in stock_list)
    cols_sql = ', '.join(identifier(col) for col in wanted)
    return connection.execute(
        f'SELECT stock_code AS instrument_id, {cols_sql} FROM price_data_daily '
        f'WHERE CAST(date AS DATE) = ? AND stock_code IN ({placeholders})',
        [date_str, *stock_list],
    ).fetchdf()


def query_legacy_stub(connection: Any, kind: str, *args: Any) -> Any:
    """Invoke historical SQL on non-DuckDB test double connections outside Core."""
    if kind == 'adv':
        stocks_str, as_of_date_str, lookback_days = args
        return connection.execute(
            f"SELECT stock_code, AVG(money) AS adv FROM (SELECT stock_code, money, "
            f"ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY date DESC) AS rn "
            f"FROM price_data_daily WHERE date <= '{as_of_date_str}' AND stock_code IN ('{stocks_str}') "
            f"AND money IS NOT NULL AND money > 0) WHERE rn <= {int(lookback_days)} GROUP BY stock_code"
        ).fetchall()
    if kind == 'style':
        stocks_str, as_of_date_str = args
        return connection.execute(
            f"SELECT * FROM style_factor_exposure WHERE date = "
            f"(SELECT MAX(date) FROM style_factor_exposure WHERE date <= '{as_of_date_str}') "
            f"AND stock_code IN ('{stocks_str}')"
        ).fetchdf()
    if kind == 'industry':
        stocks_str, as_of_date_str = args
        return connection.execute(
            f"SELECT stock_code, sw_l1_code FROM industry_membership WHERE date = "
            f"(SELECT MAX(date) FROM industry_membership WHERE date <= '{as_of_date_str}') "
            f"AND stock_code IN ('{stocks_str}')"
        ).fetchall()
    if kind == 'index':
        micro_index, benchmark_index = args
        return connection.execute(
            f"SELECT date, index_code, close FROM index_price_daily "
            f"WHERE index_code IN ('{micro_index}', '{benchmark_index}') ORDER BY date"
        ).fetchdf()
    raise ValueError(f'Unknown legacy stub query kind: {kind!r}')
