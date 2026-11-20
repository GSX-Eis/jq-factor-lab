"""Frequency routing and the Session-based Bar interface for FactorBase."""
from typing import Optional

import numpy as np
import pandas as pd


def initialize_frequency_route(
    self,
    data_frequency: str,
    analysis_mode: Optional[str],
    session_view: str,
    output_frequency: Optional[str],
    label_config: Optional[dict],
    data_version: Optional[str],
    session_calendar,
    source_revision: Optional[str],
) -> None:
    self.data_frequency = data_frequency
    self.analysis_mode = analysis_mode
    self.session_view = session_view
    self.output_frequency = output_frequency or data_frequency
    self.label_config = dict(label_config or {})
    self.data_version = data_version
    self.session_calendar = session_calendar
    self.source_revision = source_revision
    from utils.time_axis.frequency import FactorFrequencyRoute, Frequency
    self._frequency_route = FactorFrequencyRoute.resolve(data_frequency, analysis_mode)
    if analysis_mode is not None and analysis_mode not in ('time_series', 'cross_sectional'):
        raise ValueError('analysis_mode must be time_series or cross_sectional')
    if self._frequency_route.explicit_interface and analysis_mode is None:
        raise ValueError('analysis_mode must be explicit for the new Bar interface')
    if session_view not in ('ETH', 'RTH'):
        raise ValueError('session_view must be ETH or RTH')
    if self._frequency_route.bar_grid:
        if self.output_frequency in ('daily', '1d'):
            if analysis_mode != 'cross_sectional':
                raise ValueError('daily Session output requires cross_sectional analysis_mode')
        elif Frequency.parse(self.output_frequency).canonical != self._frequency_route.frequency.canonical:
            raise ValueError('Bar output_frequency must match data_frequency or be daily/1d')


def _fetch_market_bars(self) -> pd.DataFrame:
    """Read UTC Bar starts; start/end dates denote calendar Session dates.

    No daily PIT transformation is applied: minute factors consume the
    explicitly versioned source prices, including unadjusted futures.
    """
    from utils.bars.source import BarSource
    from utils.session_calendar import SessionCalendar
    from backtest_and_analysis.time_series.factor_bridge import session_grid

    if not self._frequency_route.bar_grid:
        raise ValueError('market Bars require a minute data_frequency and explicit analysis_mode')
    frequency = self._frequency_route.frequency.canonical
    with BarSource(self.db_connection, legacy_snapshot=True,
            config=self._get_local_data_config().as_dict(), duckdb_path=(
            None if self.db_connection is not None else self.duckdb_path)) as reader:
        metadata = reader.get_instrument_metadata(self.all_stocks)
        calendars = self.session_calendar
        if calendars is None:
            from data_pipeline.loaders.market_bar_importer import calendar_name
            calendars = {row['instrument_id']: SessionCalendar(
                calendar_name(row), self.start_date, self.end_date,
                rth_reference_calendar=row['rth_reference_calendar'])
                for row in metadata.to_dict('records')}
        elif isinstance(calendars, SessionCalendar):
            calendars = {instrument: calendars for instrument in self.all_stocks}
        if not isinstance(calendars, dict) or any(
                not isinstance(calendars.get(instrument), SessionCalendar) for instrument in self.all_stocks):
            raise ValueError('session_calendar must supply a SessionCalendar for every instrument')
        calendars = {instrument: calendars[instrument] for instrument in self.all_stocks}
        parts, coverage = [], {}
        for instrument, calendar in calendars.items():
            if not calendar.start <= pd.Timestamp(self.start_date) <= pd.Timestamp(self.end_date) <= calendar.end:
                raise ValueError('session_calendar must cover the requested Session date range')
            # Each market has its own overnight UTC bounds; never let another
            # market's opening pull uncovered previous sessions into a read.
            times = pd.DatetimeIndex([], tz='UTC')
            for day in calendar.sessions():
                if self.start_date <= day.strftime('%Y-%m-%d') <= self.end_date:
                    times = times.append(calendar.expected_bar_opens(day.strftime('%Y-%m-%d'), frequency))
            start = times.min() if len(times) else pd.Timestamp(self.start_date, tz='UTC')
            end = times.max() if len(times) else start
            part = reader.read([instrument], frequency, view=self.session_view, start=start, end=end)
            parts.append(part)
            for name in ('raw_coverage_by_instrument', 'raw_coverage_by_key'):
                coverage.setdefault(name, {}).update(part.attrs.get(name, {}))
        bars = pd.concat(parts, ignore_index=True)
        bounds = list(coverage['raw_coverage_by_instrument'].values())
        bars.attrs = dict(coverage, session_view=self.session_view,
            calendar_hash={instrument: calendar.calendar_hash for instrument, calendar in calendars.items()},
            raw_coverage_start_utc=min((a for a, _ in bounds), default=pd.NaT),
            raw_coverage_end_utc=max((b for _, b in bounds), default=pd.NaT))
    keep = []
    sessions = []
    for row in bars.itertuples():
        session = calendars[row.instrument_id].session_at(row.timestamp_utc, self.session_view)
        day = session.session_date
        keep.append(self.start_date <= day <= self.end_date)
        sessions.append(day)
    bars['session_id'] = sessions
    bars = bars.loc[keep].reset_index(drop=True)
    if self.data_version is not None and not bars.data_version.eq(self.data_version).all():
        raise ValueError('market Bars do not match the requested data_version')
    self.instrument_metadata = metadata
    self.market_calendars = calendars
    self.calendar_hash = {instrument: calendar.calendar_hash for instrument, calendar in calendars.items()}
    self.rth_reference_calendar = dict(zip(metadata.instrument_id, metadata.rth_reference_calendar))
    bars.attrs['instrument_metadata'] = metadata.to_dict('records')
    self.market_bars = bars
    self.market_bar_grid = session_grid(bars, calendars, frequency, view=self.session_view,
                                        start_date=self.start_date, end_date=self.end_date)
    return self.market_bar_grid


def session_rolling(self, data: pd.DataFrame, column: str, window: int, **kwargs) -> pd.Series:
    """Roll the expected grid returned by _fetch_market_bars, preserving gaps."""
    from backtest_and_analysis.time_series.factor_bridge import session_rolling
    return session_rolling(data, column, window, **kwargs)


def _get_market_factor_values(self, force_recalculate: bool = False) -> pd.DataFrame:
    from backtest_and_analysis.time_series.factor_bridge import bar_factor_frame, session_daily_frame

    raw = self._fetch_market_bars()
    self.last_raw_data = raw
    key = self._get_cache_key(raw) if self.cache_dir else None
    if key and not force_recalculate:
        cached = self._load_from_cache(key)
        if cached is not None:
            return cached
    values = bar_factor_frame(self.calculate_factor(raw))
    expected = raw[['timestamp_utc', 'instrument_id']]
    checked = values.merge(expected, on=['timestamp_utc', 'instrument_id'], how='left', indicator=True)
    if (checked._merge.eq('left_only') & checked.factor_value.notna()).any():
        raise ValueError('factor output contains keys outside the Session grid')
    # Reinsert omitted/absent slots instead of compressing rolling history.
    values = expected.merge(values, on=['timestamp_utc', 'instrument_id'], how='left',
                            validate='one_to_one')
    values.loc[~raw.observed.to_numpy(), 'factor_value'] = np.nan
    if self.output_frequency in ('daily', '1d'):
        wide = session_daily_frame(values, raw)
    else:
        wide = values.pivot(index='timestamp_utc', columns='instrument_id', values='factor_value')
        wide.index.name = 'date'  # Existing disk-cache wide-frame contract.
    if key:
        self._save_to_cache(wide, key)
    return wide


def get_bar_factor_frame(self, force_recalculate: bool = False) -> pd.DataFrame:
    """Return UTC (timestamp_utc, instrument_id, factor_value), retaining NaNs."""
    from backtest_and_analysis.time_series.factor_bridge import bar_factor_frame
    if not self._frequency_route.bar_grid or self.output_frequency in ('daily', '1d'):
        raise ValueError('get_bar_factor_frame requires minute Bar output')
    values = bar_factor_frame(self._get_market_factor_values(force_recalculate))
    return self.market_bar_grid[['timestamp_utc', 'instrument_id']].merge(
        values, on=['timestamp_utc', 'instrument_id'], how='left', validate='one_to_one'
    ).sort_values(['timestamp_utc', 'instrument_id']).reset_index(drop=True)


def get_session_daily_pricing(self) -> pd.Series:
    """Return MultiIndex(asset, date) prices bound to the Session calendar."""
    from backtest_and_analysis.time_series.factor_bridge import session_daily_pricing
    if not self._frequency_route.bar_grid:
        raise ValueError('Session pricing requires minute Bar input')
    if not hasattr(self, 'market_bar_grid'):
        self._fetch_market_bars()
    return session_daily_pricing(self.market_bar_grid, self.market_calendars)


def evaluate_bar_factor(self, **evaluator_options):
    """Label and evaluate using the same read snapshot and explicit mode."""
    from backtest_and_analysis.time_series.labels import BarReturnLabeler
    from backtest_and_analysis.time_series.evaluator import TimeSeriesEvaluator, BarCrossSectionalEvaluator
    factors = self.get_bar_factor_frame()
    labels = BarReturnLabeler(self.market_calendars, view=self.session_view,
                              **self.label_config).label(self.market_bars, factors)
    evaluator = TimeSeriesEvaluator if self.analysis_mode == 'time_series' else BarCrossSectionalEvaluator
    return evaluator(analysis_mode=self.analysis_mode, **evaluator_options).evaluate(labels, factors)
