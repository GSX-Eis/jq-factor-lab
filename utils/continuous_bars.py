"""Read-time continuous futures views; raw concrete-contract Bars stay immutable."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pandas as pd

from backtest_and_analysis.time_series.contracts import PRICE_COLUMNS, _utc
from backtest_and_analysis.time_series.csv_reader import _record_raw_coverage
from utils.bar_builder import instrument_calendar
from utils.bar_catalog import BarCatalog, table_exists
from utils.bar_quality import warn_quality
from utils.bar_source import BarSource, MissingBarDataError
from utils.futures_roll import product_metadata, session_close
from utils.time_axis import BarSchedule, Frequency

ADJUSTMENTS = ('none', 'ratio_back_adjust', 'diff_back_adjust')

if TYPE_CHECKING:
    from backtest_and_analysis.time_series.labels import BarReturnLabeler


class ContinuousBarSource(BarSource):
    def read(self, product_root: str, frequency: str, start, end, view: str = 'ETH', *,
             adjust: str) -> pd.DataFrame:
        if adjust not in ADJUSTMENTS:
            raise ValueError(f'adjust must explicitly specify one of {ADJUSTMENTS}')
        parsed = Frequency.parse(frequency)
        if not parsed.is_valid:
            raise MissingBarDataError(f'Invalid frequency={frequency!r}')
        frequency = parsed.canonical
        if view not in ('ETH', 'RTH'):
            raise ValueError('view must be ETH or RTH')
        try:
            metadata = product_metadata(self.connection, product_root)
        except ValueError as exc:
            raise MissingBarDataError(f'Missing frequency={frequency}: {exc}') from exc
        by_id = metadata.set_index('instrument_id').to_dict('index')
        meta = metadata.iloc[0].to_dict()
        dates = []
        for name, value in (('start', start), ('end', end)):
            if value is None:
                raise ValueError('start and end are required')
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None:
                if timestamp != timestamp.normalize():
                    raise ValueError('start/end must be UTC timestamps or Session dates')
                dates.append(timestamp.strftime('%Y-%m-%d'))
            else:
                _utc(timestamp, name)
                dates.append(timestamp.tz_convert(meta['timezone']).strftime('%Y-%m-%d'))
        if (bool(pd.Timestamp(start).tzinfo) == bool(pd.Timestamp(end).tzinfo)
                and pd.Timestamp(start) > pd.Timestamp(end)):
            raise ValueError('start must not follow end')
        # UTC bounds can land on a Friday night belonging to Monday's Session.
        utc_request = any(pd.Timestamp(value).tzinfo is not None for value in (start, end))
        cal = instrument_calendar(meta,
            (pd.Timestamp(dates[0])-pd.Timedelta(days=7)).strftime('%Y-%m-%d') if utc_request else dates[0],
            (pd.Timestamp(dates[1])+pd.Timedelta(days=7)).strftime('%Y-%m-%d') if utc_request else dates[1])
        if not table_exists(self.connection, 'futures_roll_schedule'):
            raise MissingBarDataError(f'Missing frequency={frequency} roll schedule for {product_root}; '
                                      'run build-roll-schedule --db <path>')
        plans = self.connection.execute('SELECT * FROM futures_roll_schedule WHERE product_root=? '
            'ORDER BY effective_session_id', [product_root]).fetchdf()
        plans = {row['effective_session_id']: row for row in plans.to_dict('records')}
        catalog = BarCatalog(self.connection, config=self.config)
        frames, requests, used, strategies = [], [], [], set()
        lower = pd.Timestamp(start) if pd.Timestamp(start).tzinfo else None
        upper = pd.Timestamp(end) if pd.Timestamp(end).tzinfo else None
        for day in cal.sessions():
            session_id = day.strftime('%Y-%m-%d')
            session = cal.session_on(session_id)
            a, b = session.segments[0][0], session.segments[-1][1]-pd.Timedelta(nanoseconds=1)
            if (lower is not None and b < lower) or (upper is not None and a > upper):
                continue
            # Date bounds refer to Session labels, even for mixed date/UTC requests.
            if (lower is None and session_id < dates[0]) or (upper is None and session_id > dates[1]):
                continue
            plan = plans.get(session_id)
            if plan is None:
                raise MissingBarDataError(f'Missing frequency={frequency} roll schedule: '
                                          f'{product_root} Session={session_id}')
            contract = plan['to_contract']
            if contract not in by_id:
                raise MissingBarDataError(f'Missing frequency={frequency} metadata for contract={contract}')
            if not plan['decision_session_id'] < session_id:
                raise ValueError(f'Invalid roll schedule for {product_root} Session={session_id}')
            try:
                entry = catalog.resolve(frequency, contract, asset_class='futures')
            except LookupError as exc:
                raise self._missing(frequency, contract, catalog, 'futures', 'no catalog entry') from exc
            if entry.table_name != parsed.table_name:
                raise MissingBarDataError(f'Missing physical frequency={frequency} for {contract}')
            strategies.add(entry.bucket_strategy)
            left, right = max(a, lower) if lower is not None else a, min(b, upper) if upper is not None else b
            # Selection is direct SQL through the existing reader; never aggregate or fall back.
            frame = self.read_bars([contract], frequency, start_utc=left, end_utc=right,
                                   source_table=entry.table_name, view='ETH', calendar=cal)
            if view == 'RTH':
                frame = frame.loc[[cal.session_at(ts, 'RTH') is not None for ts in frame.timestamp_utc]].copy()
            slots = BarSchedule(cal, frequency, view=view,
                bucket_strategy=entry.bucket_strategy).for_session(session_id)
            expected = slots.loc[slots.start_utc.between(left, right)]
            if frame.empty and not expected.empty:
                raise self._missing(frequency, contract, catalog, 'futures', f'no Bars in Session={session_id}')
            if not expected.empty and entry.quality_status == 'scanned':
                scanned = table_exists(self.connection, 'bar_quality') and self.connection.execute(
                    'SELECT EXISTS(SELECT 1 FROM bar_quality WHERE frequency=? AND instrument_id=? '
                    'AND session_id=?)', [frequency, contract, session_id]).fetchone()[0]
                if not scanned:
                    raise self._missing(frequency, contract, catalog, 'futures', 'unbuilt or unaudited Session')
            is_roll = pd.notna(plan['from_contract']) and plan['from_contract'] != contract
            frame = frame.assign(contract_id=contract, instrument_id=product_root,
                                 session_id=session_id, is_roll_session=is_roll)
            frame.attrs = {}
            frames.append(frame)
            requests.append((contract, entry, left, right))
            used.append(plan)
        if not frames:
            raise MissingBarDataError(f'Missing frequency={frequency}: no Sessions in requested range')
        if len(strategies) != 1:
            raise ValueError(f'Continuous frequency={frequency} requires one shared bucket_strategy')
        result = pd.concat(frames, ignore_index=True).sort_values('timestamp_utc').reset_index(drop=True)
        if adjust != 'none':
            # Only effective Sessions selected above enter the adjustment chain.
            for plan in reversed(used):
                old, new = plan['from_contract'], plan['to_contract']
                if pd.isna(old) or old == new:
                    continue
                prices = []
                for contract in (old, new):
                    row = dict(by_id[contract], instrument_id=contract)
                    prices.append(session_close(self.connection, row, plan['decision_session_id'], 'close')['value'])
                if min(prices) <= 0:
                    raise ValueError('Roll adjustment requires positive decision closes')
                mask = result.session_id.lt(plan['effective_session_id'])
                if adjust == 'ratio_back_adjust':
                    result.loc[mask, list(PRICE_COLUMNS)] *= prices[1]/prices[0]
                else:
                    result.loc[mask, list(PRICE_COLUMNS)] += prices[1]-prices[0]
        _record_raw_coverage(result)
        result.attrs.update(session_view=view, calendar_hash={product_root: cal.calendar_hash},
            continuous=dict(product_root=product_root, adjust=adjust, frequency=frequency,
                start=str(start), end=str(end), calendar_start=cal.start.strftime('%Y-%m-%d'),
                calendar_end=cal.end.strftime('%Y-%m-%d'), bucket_strategy=next(iter(strategies))),
            roll_build_ids={row['effective_session_id']: row['build_id'] for row in used})
        warn_quality(self.connection, frequency, requests)
        return result

    def get_calendars(self, bars: pd.DataFrame) -> dict:
        request = bars.attrs['continuous']
        root = request['product_root']
        metadata = product_metadata(self.connection, root).iloc[0].to_dict()
        return {root: instrument_calendar(metadata, request['calendar_start'], request['calendar_end'])}

    def label(self, bars: pd.DataFrame, labeler: BarReturnLabeler,
              factors: pd.DataFrame | None = None, *,
              signal_timestamps=None) -> pd.DataFrame:
        """Expand each signal's concrete contract through the requested raw coverage."""
        request = bars.attrs.get('continuous')
        if not isinstance(request, Mapping):
            raise ValueError('Continuous labels require the source request metadata')
        root, frequency = request['product_root'], request['frequency']
        calendar = self.get_calendars(bars)[root]
        tables = []
        for contract, group in bars.groupby('contract_id'):
            entry = BarCatalog(self.connection, config=self.config).resolve(frequency, contract, asset_class='futures')
            raw = self.read_bars([contract], frequency, start_utc=bars.timestamp_utc.min(),
                end_utc=bars.timestamp_utc.max(), view='ETH', source_table=entry.table_name, calendar=calendar)
            requests = [(contract, entry, bars.timestamp_utc.min(), bars.timestamp_utc.max())]
            warn_quality(self.connection, frequency, requests)
            raw = raw.assign(instrument_id=root)
            # The request bounds classify a hole after rolling as missing_bar, not a new contract.
            raw.attrs = dict(bars.attrs)
            if factors is None:
                raw = raw.merge(group[['timestamp_utc', 'factor_value']], on='timestamp_utc', how='left')
                raw.attrs = dict(bars.attrs)
            signals = pd.DatetimeIndex(group.timestamp_utc)
            if signal_timestamps is not None:
                signals = signals.intersection(pd.DatetimeIndex(signal_timestamps))
            from backtest_and_analysis.time_series.labels import BarReturnLabeler
            concrete_labeler = BarReturnLabeler(calendar, view=labeler.view,
                horizons=labeler.horizons, entry_offset_bars=labeler.entry_offset_bars,
                boundary_policy=labeler.boundary_policy, bucket_strategy=labeler.bucket_strategy,
                continuous_source=self, include_scheduled_exit=labeler.include_scheduled_exit)
            # Raw Bars deliberately drop the continuous marker to prevent recursive dispatch.
            raw = raw.drop(columns='contract_id', errors='ignore')
            tables.append(concrete_labeler.label(raw, factors, signal_timestamps=signals))
        return pd.concat(tables, ignore_index=True).sort_values(
            ['instrument_id', 'frequency', 'signal_timestamp', 'horizon_bars']).reset_index(drop=True)
