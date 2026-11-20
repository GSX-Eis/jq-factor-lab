"""Small synthetic concrete-contract datasets; never open a configured database."""
from collections.abc import Iterator
from functools import cache
import json

import numpy as np
import pandas as pd
import pandas_market_calendars as pmc
import pytest

from data_pipeline.loaders.market_bar_importer import import_market_bars
from utils import session_calendar
from utils.session_calendar import SessionCalendar
from utils.time_axis import BarSchedule


@pytest.fixture(scope='module')
def warm_market_calendars() -> Iterator[None]:
    get_calendar = pmc.get_calendar
    build_schedule = session_calendar._schedule
    calendar_names = {'SHFE_CU', 'SHFE_Day'}

    @cache
    def holiday_data(name: str) -> tuple[str, tuple[np.datetime64, ...], np.busdaycalendar]:
        offset = get_calendar(name).holidays()
        return offset.weekmask, offset.holidays, offset.calendar

    def fresh_calendar(name: str, *args, **kwargs) -> pmc.MarketCalendar:
        calendar = get_calendar(name, *args, **kwargs)
        if name in calendar_names and not (args or kwargs):
            weekmask, holidays, business_days = holiday_data(name)
            # NumPy's business-day calendar is immutable. The PMC calendar and
            # its offset are recreated for each factory call.
            calendar._holidays = pd.offsets.CustomBusinessDay(
                weekmask=weekmask, holidays=holidays, calendar=business_days,
            )
        return calendar

    @cache
    def schedule_data(
        name: str, start: pd.Timestamp, end: pd.Timestamp, overrides_json: str,
    ) -> tuple[tuple[tuple[str, tuple[session_calendar.Segment, ...]], ...], str]:
        sessions, timezone = build_schedule(name, start, end, json.loads(overrides_json))
        return tuple(sessions.items()), timezone

    def fresh_schedule(
        name: str, start: pd.Timestamp, end: pd.Timestamp, overrides: dict,
    ) -> tuple[dict[str, tuple[session_calendar.Segment, ...]], str]:
        if name not in calendar_names:
            return build_schedule(name, start, end, overrides)
        # Session maps are rebuilt per caller from immutable date and segment tuples.
        sessions, timezone = schedule_data(name, start, end, json.dumps(overrides, sort_keys=True))
        return dict(sessions), timezone

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(pmc, 'get_calendar', fresh_calendar)
        monkeypatch.setattr(session_calendar, '_schedule', fresh_schedule)
        yield


def futures_sample(connection, *, start='2026-01-05', end='2026-01-16', frequency='1d',
                   oi=None, root='CU', calendar_id='SHFE_CU'):
    exchange = calendar_id.split('_')[0]
    cal = SessionCalendar(calendar_id, start, end, rth_reference_calendar=exchange+'_Day')
    schedule = BarSchedule(cal, frequency, bucket_strategy='segment_anchored')
    starts = [schedule.for_session(day.strftime('%Y-%m-%d')).start_utc for day in cal.sessions()]
    frames, metadata = [], []
    months = [(pd.Timestamp(start)+pd.offsets.MonthBegin(i+1)).strftime('%Y-%m') for i in range(3)]
    for index, month in enumerate(months):
        instrument = root + month[2:].replace('-', '')
        metadata.append(dict(instrument_id=instrument, asset_class='futures', exchange=exchange,
            timezone='Asia/Shanghai', contract_expiry=month, contract_month=month,
            price_adjustment='none', point_value=5., calendar_id=calendar_id,
            product_root=root, rth_reference_calendar=exchange+'_Day'))
        for day_index, timestamps in enumerate(starts):
            prices = 100. + index*20 + day_index*2 + np.arange(len(timestamps))*.01
            interest = oi[day_index][index] if oi is not None else 300. - index*100
            frames.append(pd.DataFrame(dict(instrument_id=instrument, timestamp_utc=timestamps,
                frequency=frequency, open=prices, high=prices+2., low=prices-1., close=prices+1.,
                volume=10., money=100., factor=1., open_interest=interest,
                source='synthetic', data_version='v1')))
    bars, meta = pd.concat(frames, ignore_index=True), pd.DataFrame(metadata)
    import_market_bars(connection, bars, meta, bucket_strategy='segment_anchored', legacy_compatibility=False)
    return bars, meta, cal
