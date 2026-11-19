"""Small synthetic concrete-contract datasets; never open a configured database."""
import numpy as np
import pandas as pd

from data_pipeline.loaders.market_bar_importer import import_market_bars
from utils.session_calendar import SessionCalendar
from utils.time_axis import BarSchedule


def futures_sample(connection, *, start='2026-01-05', end='2026-01-16', frequency='1d',
                   oi=None, root='CU', calendar_id='SHFE_CU'):
    exchange = calendar_id.split('_')[0]
    cal = SessionCalendar(calendar_id, start, end, rth_reference_calendar=exchange+'_Day')
    days = [day.strftime('%Y-%m-%d') for day in cal.sessions()]
    frames, metadata = [], []
    months = [(pd.Timestamp(start)+pd.offsets.MonthBegin(i+1)).strftime('%Y-%m') for i in range(3)]
    for index, month in enumerate(months):
        instrument = root + month[2:].replace('-', '')
        metadata.append(dict(instrument_id=instrument, asset_class='futures', exchange=exchange,
            timezone='Asia/Shanghai', contract_expiry=month, contract_month=month,
            price_adjustment='none', point_value=5., calendar_id=calendar_id,
            product_root=root, rth_reference_calendar=exchange+'_Day'))
        for day_index, day in enumerate(days):
            grid = BarSchedule(cal, frequency, bucket_strategy='segment_anchored').for_session(day)
            n = len(grid)
            prices = 100. + index*20 + day_index*2 + np.arange(n)*.01
            interest = oi[day_index][index] if oi is not None else 300. - index*100
            frames.append(pd.DataFrame(dict(instrument_id=instrument, timestamp_utc=grid.start_utc,
                frequency=frequency, open=prices, high=prices+2., low=prices-1., close=prices+1.,
                volume=10., money=100., factor=1., open_interest=interest,
                source='synthetic', data_version='v1')))
    bars, meta = pd.concat(frames, ignore_index=True), pd.DataFrame(metadata)
    import_market_bars(connection, bars, meta, bucket_strategy='segment_anchored', legacy_compatibility=False)
    return bars, meta, cal
