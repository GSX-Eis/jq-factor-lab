import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.forward_returns_calculator import newey_west_mean_test
from utils.duckdb_results_writer import DuckDBResultsWriter
from utils.trading_calendar import TradingCalendar


def test_newey_west_reduces_overlapping_sample_significance() -> None:
    rng = np.random.default_rng(3)
    innovations = rng.normal(size=200)
    values = np.empty_like(innovations)
    values[0] = 0.1 + innovations[0] * 0.01
    for index in range(1, len(values)):
        values[index] = 0.1 + 0.9 * (values[index - 1] - 0.1) + innovations[index] * 0.01

    hac_t, _ = newey_west_mean_test(values, max_lag=19)
    naive_t = values.mean() / (values.std(ddof=1) / np.sqrt(len(values)))

    assert hac_t is not None
    assert abs(hac_t) < abs(naive_t)


def monthly_frame(values=(.1, .2, .3, .4), period='2D'):
    dates = pd.bdate_range('2024-01-02', periods=len(values))
    frame = pd.DataFrame({period: values}, index=dates)
    frame.attrs['trading_calendar'] = TradingCalendar(dates, dates.min(), dates.max())
    return frame


def summarize(frame):
    return object.__new__(DuckDBResultsWriter)._calculate_monthly_metrics(frame)


def test_monthly_hac_sorts_dates_and_preserves_input():
    frame = monthly_frame()
    expected = summarize(frame)
    shuffled = frame.iloc[[0, 2, 1, 3]].copy()
    shuffled.index = shuffled.index.strftime('%Y-%m-%d')
    original = shuffled.copy(deep=True)
    pd.testing.assert_frame_equal(summarize(shuffled), expected)
    pd.testing.assert_frame_equal(shuffled, original)
    row = expected.iloc[0]
    assert row.monthly_ic == pytest.approx(.25)
    assert row.ic_std == pytest.approx(.12909944487358055)
    assert row.ic_count == 4
    assert row.ic_t_stat == pytest.approx(4.)
    assert row.ic_p_value == pytest.approx(6.334248366623973e-5)


@pytest.mark.parametrize('values', [[.1]*20, [0.]*20, [-.1]*20, [.1]])
def test_constant_or_single_ic_has_no_significance(values):
    row = summarize(monthly_frame(values)).iloc[0]
    assert pd.isna(row.monthly_ir)
    assert pd.isna(row.ic_t_stat) and pd.isna(row.ic_p_value)
    assert row.ic_std == 0 if len(values) > 1 else pd.isna(row.ic_std)
    assert newey_west_mean_test(values, 19) == (None, None)


@pytest.mark.parametrize('mode', ['nan', 'omitted', 'unknown', 'coverage', 'edge_nan'])
@pytest.mark.parametrize('period', ['1D', '2D'])
def test_missing_or_unverified_sessions_retain_descriptive_statistics(mode, period):
    frame = monthly_frame(period=period)
    if mode == 'nan':
        frame.iloc[1, 0] = np.nan
    elif mode == 'omitted':
        frame = frame.drop(frame.index[1])
    elif mode == 'unknown':
        frame.attrs.clear()
    elif mode == 'edge_nan':
        frame.iloc[-1, 0] = np.nan
    else:
        frame.attrs['trading_calendar'] = TradingCalendar(frame.index[:2], frame.index[0], frame.index[1])
    row = summarize(frame).iloc[0]
    assert row.monthly_ic == pytest.approx(frame.iloc[:, 0].mean())
    assert row.ic_count == frame.iloc[:, 0].count()
    assert pd.isna(row.ic_t_stat) and pd.isna(row.ic_p_value)


def test_market_holiday_is_not_a_missing_session():
    frame = monthly_frame()
    frame.index = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-05', '2024-01-08'])
    frame.attrs['trading_calendar'] = TradingCalendar(frame.index, frame.index.min(), frame.index.max())
    assert summarize(frame).iloc[0].ic_t_stat == pytest.approx(4.)


@pytest.mark.parametrize('bad', [np.inf, -np.inf, 1.1, -1.1, True, '0.1', 1+2j])
def test_invalid_ic_is_not_silently_dropped(bad):
    frame = monthly_frame().astype(object)
    frame.iloc[1, 0] = bad
    with pytest.raises(ValueError, match='Invalid IC'):
        summarize(frame)


@pytest.mark.parametrize('kind', ['duplicate', 'nat', 'intraday', 'timezone', 'numeric', 'non_session'])
def test_invalid_daily_ic_dates_raise(kind):
    frame = monthly_frame()
    if kind == 'duplicate':
        frame.index = [frame.index[0]] * len(frame)
    elif kind == 'nat':
        frame.index = pd.DatetimeIndex([pd.NaT, *frame.index[1:]])
    elif kind == 'intraday':
        frame.index += pd.Timedelta(hours=1)
    elif kind == 'timezone':
        frame.index = frame.index.tz_localize('Asia/Shanghai')
    elif kind == 'numeric':
        frame.index = range(len(frame))
    else:
        frame.attrs['trading_calendar'] = TradingCalendar(frame.index.delete(1), frame.index.min(), frame.index.max())
    with pytest.raises(ValueError):
        summarize(frame)


@pytest.mark.parametrize('value', [1, np.int64(5), '1', '20D'])
def test_period_accepts_actual_callsite_formats(value):
    assert object.__new__(DuckDBResultsWriter)._parse_period(value) == int(str(value).removesuffix('D'))


@pytest.mark.parametrize('value', ['bad', '1W', 'badD', '', '0D', '-2D', 0, -2, 1., 1.9, True, np.bool_(True), (), ('5D', 1), None])
def test_period_rejects_invalid_labels_without_defaulting(value):
    with pytest.raises(ValueError, match='Invalid period'):
        object.__new__(DuckDBResultsWriter)._parse_period(value)


def test_duplicate_normalized_periods_rejected():
    frame = monthly_frame()
    frame[2] = frame['2D']
    with pytest.raises(ValueError, match='distinct periods'):
        summarize(frame)


@pytest.mark.parametrize('values,lag', [([.1, np.nan, .2], 1), ([.1, np.inf], 1), ([[.1, .2]], 1), ([.1, .2], -1), ([.1, .2], 1.5), ([.1, .2], True)])
def test_hac_rejects_invalid_sample_or_lag(values, lag):
    with pytest.raises(ValueError):
        newey_west_mean_test(values, lag)


@pytest.mark.parametrize('method', ['write_monthly_pool_results'])
def test_unknown_calendar_persists_null_significance(tmp_path, method):
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), export_parquet=False)
    frame = monthly_frame()
    frame.attrs.clear()
    assert getattr(writer, method)('example', {'ic': frame}) == 1
    table = writer.monthly_output_table
    with writer._get_connection(read_only=True) as conn:
        row = conn.execute(f'SELECT monthly_ic, ic_count, ic_t_stat, ic_p_value FROM {table}').fetchone()
    assert row == (.25, 4, None, None)


def _quantile_analysis_results(ic_values):
    return {
        'ic': pd.DataFrame({'1D':ic_values}, index=pd.to_datetime(['2024-01-02','2024-01-03'])),
        'ir': pd.DataFrame({'ir':[.5]},index=['1D']),
        'quantile_returns': pd.DataFrame(),
    }


def _quantile_panel():
    index = pd.MultiIndex.from_product(
        [[1, 2, 3], pd.to_datetime(['2024-01-02', '2024-02-01'])],
        names=['factor_quantile', 'date'],
    )
    return pd.DataFrame({'1D': [.01, .04, .02, .06, .03, .08]}, index=index)


def test_monthly_missing_quantile_is_unknown_not_zero():
    import numpy as np
    writer = object.__new__(DuckDBResultsWriter)
    frame = _quantile_panel().drop((3, pd.Timestamp('2024-02-01')))
    result = writer._calculate_monthly_returns(frame)['1D']
    assert result['monthly_return']['2024-01'] == pytest.approx(.02)
    assert pd.isna(result['monthly_return']['2024-02'])
    assert pd.isna(result['long_short_return']['2024-02'])
    # An explicit NaN and a structurally absent group have identical meaning.
    explicit = _quantile_panel()
    explicit.loc[(3, pd.Timestamp('2024-02-01')), '1D'] = np.nan
    actual = writer._calculate_monthly_returns(explicit)['1D']
    for name in result:
        pd.testing.assert_series_equal(result[name], actual[name])


@pytest.mark.parametrize('shape', ['long', 'wide', 'flat'])
@pytest.mark.parametrize('period', ['1D', '1', 1])
def test_monthly_quantile_formats_and_periods_are_equivalent(shape, period):
    writer = object.__new__(DuckDBResultsWriter)
    frame = _quantile_panel().rename(columns={'1D': period})
    if shape == 'wide':
        frame = frame.unstack('factor_quantile')
    elif shape == 'flat':
        frame = frame.reset_index()
    before = frame.copy(deep=True)
    actual = writer._calculate_monthly_returns(frame)
    expected = writer._calculate_monthly_returns(_quantile_panel())
    assert actual.keys() == expected.keys()
    for name in expected['1D']:
        pd.testing.assert_series_equal(actual['1D'][name], expected['1D'][name])
    pd.testing.assert_frame_equal(before, frame)


@pytest.mark.parametrize('kind', ['string', 'inf', 'bool', 'complex', 'date', 'nat', 'intraday', 'quantile', 'duplicate', 'duplicate_period', 'shape'])
def test_invalid_quantile_data_raises_instead_of_silent_missing(kind):
    import numpy as np
    frame = _quantile_panel().reset_index()
    if kind in ('string', 'inf', 'bool', 'complex'):
        frame['1D'] = frame['1D'].astype(object)
        frame.loc[0, '1D'] = {'string':'bad', 'inf':np.inf, 'bool':True, 'complex':1+2j}[kind]
    elif kind == 'date':
        frame['date'] = range(len(frame))
    elif kind == 'nat':
        frame.loc[0, 'date'] = pd.NaT
    elif kind == 'intraday':
        frame.loc[0, 'date'] += pd.Timedelta(hours=1)
    elif kind == 'quantile':
        frame.loc[0, 'factor_quantile'] = 0
    elif kind == 'duplicate':
        frame = pd.concat([frame, frame.iloc[:1]])
    elif kind == 'duplicate_period':
        frame[1] = frame['1D']
    else:
        frame = frame.drop(columns='factor_quantile')
    with pytest.raises(ValueError):
        object.__new__(DuckDBResultsWriter)._calculate_monthly_returns(frame)


@pytest.mark.parametrize('method', ['write_monthly_pool_results'])
def test_monthly_write_accepts_wide_numeric_period_and_preserves_rows_on_error(tmp_path, method):
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), export_parquet=False)
    results = _quantile_analysis_results([.1, .2])
    results['ic'].index = pd.to_datetime(['2024-01-02','2024-02-01'])
    results['quantile_returns'] = _quantile_panel().rename(columns={'1D':1}).unstack('factor_quantile')
    assert getattr(writer, method)('factor', results) == 2
    table = writer.monthly_output_table
    with writer._get_connection(read_only=True) as conn:
        before = conn.execute(f'SELECT * FROM {table} ORDER BY year_month').fetchall()
        means = conn.execute(f'SELECT monthly_return FROM {table} ORDER BY year_month').fetchall()
    assert [r[0] for r in means] == pytest.approx([.02, .06])
    results['quantile_returns'] = _quantile_panel().astype(object)
    results['quantile_returns'].iloc[0, 0] = 'bad'
    with pytest.raises(ValueError):
        getattr(writer, method)('factor', results)
    with writer._get_connection(read_only=True) as conn:
        assert conn.execute(f'SELECT * FROM {table} ORDER BY year_month').fetchall() == before


def test_daily_quantile_write_keeps_missing_returns_out_of_rows(tmp_path):
    import numpy as np
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), export_parquet=False)
    results = _quantile_analysis_results([.1, .2])
    results['quantile_returns'] = _quantile_panel()
    results['quantile_returns'].iloc[0, 0] = np.nan
    writer.write_factor_analysis_results('factor', results)
    with writer._get_connection(read_only=True) as conn:
        rows = conn.execute('SELECT COUNT(*), COUNT("return") FROM factor_analysis_results WHERE quantile IS NOT NULL').fetchone()
    assert rows == (5, 5)


def test_missing_middle_group_does_not_hide_observed_extreme_spread():
    frame = _quantile_panel().drop((2, pd.Timestamp('2024-02-01')))
    result = object.__new__(DuckDBResultsWriter)._calculate_monthly_returns(frame)['1D']
    assert pd.isna(result['monthly_return']['2024-02'])
    assert result['long_short_return']['2024-02'] == pytest.approx(.04)


def test_all_missing_group_is_preserved_and_real_zero_is_valid():
    import numpy as np
    frame = _quantile_panel()
    frame.loc[3, '1D'] = np.nan
    writer = object.__new__(DuckDBResultsWriter)
    result = writer._calculate_monthly_returns(frame)['1D']
    assert result['monthly_return'].isna().all()
    assert result['long_short_return'].isna().all()
    frame.loc[3, '1D'] = 0.
    result = writer._calculate_monthly_returns(frame)['1D']
    assert result['monthly_return']['2024-02'] == pytest.approx((.04+.06)/3)
    assert result['long_short_return']['2024-02'] == pytest.approx(-.04)


@pytest.mark.parametrize('dtype', ['object', 'Float64'])
def test_quantile_missing_nullable_values_are_supported(dtype):
    frame = _quantile_panel().astype(dtype)
    frame.loc[(3, pd.Timestamp('2024-02-01')), '1D'] = pd.NA
    result = object.__new__(DuckDBResultsWriter)._calculate_monthly_returns(frame)['1D']
    assert pd.isna(result['monthly_return']['2024-02'])
    assert pd.isna(result['long_short_return']['2024-02'])


def test_invalid_daily_quantile_write_keeps_previous_rows(tmp_path):
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), export_parquet=False)
    results = _quantile_analysis_results([.1,.2])
    results['quantile_returns'] = _quantile_panel()
    writer.write_factor_analysis_results('factor', results)
    with writer._get_connection(read_only=True) as conn:
        before = conn.execute('SELECT * FROM factor_analysis_results ORDER BY code').fetchall()
    results['quantile_returns'] = results['quantile_returns'].astype(object)
    results['quantile_returns'].iloc[0,0] = 'bad'
    with pytest.raises(ValueError):
        writer.write_factor_analysis_results('factor', results)
    with writer._get_connection(read_only=True) as conn:
        assert conn.execute('SELECT * FROM factor_analysis_results ORDER BY code').fetchall() == before


@pytest.mark.parametrize('kind', ['factor', 'factor_range', 'analysis', 'analysis_range', 'pool'])
@pytest.mark.parametrize('borrowed', [False, True])
def test_write_probe_failure_preserves_all_rows(tmp_path, monkeypatch, kind, borrowed):
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), export_parquet=False,
                                monthly_output_table='monthly_pool_results')
    dates = pd.to_datetime(['2024-01-02', '2024-02-01'])
    values = pd.DataFrame({'S':[1.,2.]},index=dates)
    results = {'ic':pd.DataFrame({'1D':[.1,.2]},index=dates)}
    writer.save_factor_values(values, 'factor')
    writer.save_factor_values(values*3, 'other')
    writer.write_factor_analysis_results('factor', results)
    writer.write_monthly_pool_results('factor', results)
    tables = ['factor_values','factor_analysis_results','factor_monthly_results','monthly_pool_results']
    def snapshot():
        with writer._get_connection(read_only=True) as conn:
            return {table:conn.execute(f'SELECT * FROM {table} ORDER BY ALL').fetchall() for table in tables}
    before = snapshot()
    failure = RuntimeError('IO Error: Corrupt database file: checksum mismatch in block')
    def fail_probe(*args):
        raise failure
    monkeypatch.setattr(writer, '_count_factor_rows', fail_probe)
    def write(conn=None):
        options = {'duckdb_conn':conn}
        if kind.endswith('_range'):
            options['date_range'] = ('2024-01-02','2024-01-02')
        if kind.startswith('factor'):
            writer.save_factor_values(values.iloc[:1]*10, 'factor', skip_parquet=True, **options)
        else:
            method = {'analysis':'write_factor_analysis_results', 'analysis_range':'write_factor_analysis_results',
                      'pool':'write_monthly_pool_results'}[kind]
            getattr(writer,method)('factor',{'ic':results['ic'].iloc[:1]*2},**options)
    if borrowed:
        with writer.acquire_write_session() as session:
            session.conn.execute('BEGIN TRANSACTION')
            try:
                with pytest.raises(RuntimeError) as caught:
                    write(session.conn)
                assert caught.value is failure
                assert session.conn.execute('SELECT 1').fetchone() == (1,)
            finally:
                session.conn.execute('ROLLBACK')
    else:
        with pytest.raises(RuntimeError) as caught:
            write()
        assert caught.value is failure
    assert snapshot() == before


def test_failed_probe_does_not_restore_stale_or_partial_parquet(tmp_path, monkeypatch):
    import hashlib
    pytest.importorskip('pyarrow')
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'),parquet_dir=str(tmp_path/'parquet'))
    values = pd.DataFrame({'S':[1.,2.]},index=pd.to_datetime(['2024-01-02','2024-01-03']))
    writer.save_factor_values(values, 'target')
    writer.save_factor_values(values*2, 'backed')
    writer.save_factor_values(values*3, 'unbacked', skip_parquet=True)
    writer.save_factor_values(values*4, 'backed', skip_parquet=True)
    def snapshot():
        with writer._get_connection(read_only=True) as conn:
            return conn.execute('SELECT * FROM factor_values ORDER BY ALL').fetchall()
    before = snapshot()
    files = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'parquet').rglob('*') if p.is_file()}
    def fail_probe(*args):
        raise RuntimeError('Corrupt database file')
    monkeypatch.setattr(writer, '_count_factor_rows', fail_probe)
    with pytest.raises(RuntimeError, match='Corrupt database file'):
        writer.save_factor_values(values*5, 'target', skip_parquet=True)
    assert snapshot() == before
    assert {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'parquet').rglob('*') if p.is_file()} == files


@pytest.mark.parametrize('case', ['duplicate_date','duplicate_period','numeric_date','missing_date','intraday','timezone','infinite_ic','out_of_bounds','bool_ic','string_ic','complex_ic','missing_ir_column','duplicate_ir','aliased_ir','infinite_ir','string_ir','bool_ir','unknown_ir'])
def test_daily_invalid_metrics_preserve_previous_results(tmp_path, case):
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), export_parquet=False)
    initial = _quantile_analysis_results([.1,.2])
    writer.write_factor_analysis_results('factor',initial)
    with writer._get_connection(read_only=True) as conn:
        before = conn.execute('SELECT * FROM factor_analysis_results ORDER BY code').fetchall()
    results = _quantile_analysis_results([.3,.4])
    ic = results['ic']
    if case == 'duplicate_date': ic.index = [ic.index[0]]*2
    elif case == 'duplicate_period': ic[1] = [.5,.6]
    elif case == 'numeric_date': ic.index = [20240102,20240103]
    elif case == 'missing_date': ic.index = pd.DatetimeIndex([ic.index[0],pd.NaT])
    elif case == 'intraday': ic.index = ic.index + pd.Timedelta(hours=1)
    elif case == 'timezone': ic.index = ic.index.tz_localize('Asia/Shanghai')
    elif case in ['infinite_ic','out_of_bounds','bool_ic','string_ic','complex_ic']:
        results['ic'] = ic.astype(object)
        results['ic'].iloc[0,0] = {'infinite_ic':np.inf,'out_of_bounds':1.1,'bool_ic':True,'string_ic':'.1','complex_ic':1+2j}[case]
    elif case == 'missing_ir_column': results['ir'] = results['ir'].rename(columns={'ir':'wrong'})
    elif case == 'duplicate_ir': results['ir'] = pd.DataFrame({'ir':[.5,.6]},index=['1D','1D'])
    elif case == 'aliased_ir': results['ir'] = pd.DataFrame({'ir':[.5,.6]},index=['1D',1])
    elif case == 'unknown_ir': results['ir'].index = ['5D']
    else:
        results['ir'] = results['ir'].astype(object)
        results['ir'].iloc[0,0] = {'infinite_ir':np.inf,'string_ir':'.5','bool_ir':True}[case]
    with pytest.raises(ValueError): writer.write_factor_analysis_results('factor',results)
    with writer._get_connection(read_only=True) as conn:
        assert conn.execute('SELECT * FROM factor_analysis_results ORDER BY code').fetchall() == before


def test_daily_nullable_missing_values_and_ir_match_correct_period(tmp_path):
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), export_parquet=False)
    ic = pd.DataFrame({'1D':[.1,pd.NA], '5D':[.2,.3]},index=['2024-01-03','2024-01-02'],dtype=object)
    original = ic.copy(deep=True)
    ir = pd.DataFrame({'ir':[.8,.4]},index=['5D',1])
    assert writer.write_factor_analysis_results('factor',{'ic':ic,'ir':ir}) == 3
    pd.testing.assert_frame_equal(ic,original)
    with writer._get_connection(read_only=True) as conn:
        rows=conn.execute('SELECT period,ir FROM factor_analysis_results ORDER BY date,period').fetchall()
    assert rows == [(5,.8),(1,.4),(5,.8)]
    summary = writer._calculate_monthly_metrics(ic)
    assert summary.ic_count.tolist() == [1,2]
    writer.write_factor_analysis_results('factor',{'ic':ic,'ir':pd.DataFrame({'ir':[pd.NA,.8]},index=['1D','5D'])})
    with writer._get_connection(read_only=True) as conn:
        assert conn.execute('SELECT ir FROM factor_analysis_results WHERE period=1').fetchone() == (None,)


@pytest.mark.parametrize('date_range', [(20240102,20240103), ('2024-01-02 12:00','2024-01-03'), ('2024-01-02',pd.NaT), '2024-01-02', ('2024-01-02',)])
def test_daily_invalid_date_range_rejected(tmp_path,date_range):
    writer= DuckDBResultsWriter(str(tmp_path/'results.duckdb'),export_parquet=False)
    with pytest.raises(ValueError):
        writer.write_factor_analysis_results('factor',_quantile_analysis_results([.1,.2]),date_range=date_range)


@pytest.mark.parametrize('entry', ['save', 'export'])
@pytest.mark.parametrize('case', [
    'numeric_date', 'missing_date', 'duplicate_date', 'intraday', 'timezone',
    'duplicate_stock', 'empty_stock', 'numeric_stock', 'infinite', 'string',
    'bool', 'complex', 'empty', 'all_missing', 'unsafe_name',
])
def test_factor_snapshot_invalid_input_preserves_storage(tmp_path, entry, case):
    import hashlib
    pytest.importorskip('pyarrow')
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), parquet_dir=str(tmp_path/'parquet'))
    values = pd.DataFrame({'A': [1., 2.]}, index=pd.to_datetime(['2024-01-02', '2024-01-03']))
    writer.save_factor_values(values, 'factor')
    def snapshot():
        with writer._get_connection(read_only=True) as conn:
            rows = conn.execute('SELECT * FROM factor_values ORDER BY ALL').fetchall()
        files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'parquet').rglob('*') if p.is_file()}
        return rows, files
    before = snapshot()
    frame = values.copy()
    name = 'factor'
    if case == 'numeric_date': frame.index = [20240102, 20240103]
    elif case == 'missing_date': frame.index = [frame.index[0], pd.NaT]
    elif case == 'duplicate_date': frame.index = [frame.index[0]]*2
    elif case == 'intraday': frame.index += pd.Timedelta(hours=1)
    elif case == 'timezone': frame.index = frame.index.tz_localize('Asia/Shanghai')
    elif case == 'duplicate_stock': frame = pd.concat([frame, frame], axis=1)
    elif case == 'empty_stock': frame.columns = [' ']
    elif case == 'numeric_stock': frame.columns = [1]
    elif case == 'empty': frame = frame.iloc[:0]
    elif case == 'all_missing': frame[:] = np.nan
    elif case == 'unsafe_name': name = '../factor'
    else:
        frame = frame.astype(object)
        frame.iloc[0, 0] = {'infinite': np.inf, 'string': '1.5', 'bool': True, 'complex': 1+2j}[case]
    method = writer.save_factor_values if entry == 'save' else writer.export_factor_values_to_parquet
    with pytest.raises(ValueError):
        method(frame, name)
    assert snapshot() == before


@pytest.mark.parametrize('dtype', ['float64', 'Float64', 'object'])
def test_factor_snapshot_normalizes_double_without_mutating_input(tmp_path, dtype):
    import pyarrow.parquet as pq
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), parquet_dir=str(tmp_path/'parquet'))
    # These stock labels collide with old reset_index/melt temporary names.
    frame = pd.DataFrame({'factor_value': [-20., pd.NA], 'date': [0., 25.]},
                         index=['2024-01-03', '2024-01-02'], dtype='Float64').astype(dtype)
    frame.index.name = 'date'
    original = frame.copy(deep=True)
    assert writer.save_factor_values(frame, 'factor') == 3
    pd.testing.assert_frame_equal(frame, original)
    with writer._get_connection(read_only=True) as conn:
        db = conn.execute('SELECT date,stock_code,factor_value FROM factor_values ORDER BY date,stock_code').fetchdf()
    files = writer._list_visible_parquet_files(tmp_path/'parquet'/'factor_values'/'factor')
    assert all(str(pq.ParquetFile(p).schema_arrow.field('factor_value').type) == 'double' for p in files)
    parquet = pd.concat([pq.ParquetFile(p).read().to_pandas() for p in files])
    parquet = parquet[['date','stock_code','factor_value']].sort_values(['date','stock_code']).reset_index(drop=True)
    parquet['date'] = pd.to_datetime(parquet['date']).astype(db['date'].dtype)
    pd.testing.assert_frame_equal(db, parquet)


def test_postcommit_export_failure_can_retry_without_rewriting_database(tmp_path, monkeypatch):
    import hashlib
    import utils.duckdb_results_writer as module
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), parquet_dir=str(tmp_path/'parquet'))
    frame = pd.DataFrame({'A': [1., 2.]}, index=pd.to_datetime(['2024-01-02','2024-01-03']))
    writer.save_factor_values(frame, 'factor')
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'parquet').rglob('*.parquet')}
    old_files = hashes()
    failure = OSError('injected export failure')
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise failure
        patch.setattr(module.pq, 'write_to_dataset', fail)
        with pytest.raises(module.CommittedParquetExportError) as caught:
            writer.save_factor_values(frame*10, 'factor')
    assert caught.value.database_committed is True
    assert caught.value.factor_name == 'factor'
    assert caught.value.__cause__ is failure
    assert hashes() == old_files
    with writer._get_connection(read_only=True) as conn:
        committed = conn.execute('SELECT * FROM factor_values ORDER BY ALL').fetchdf()
    assert committed.factor_value.sum() == 30.
    current = committed.pivot(index='date', columns='stock_code', values='factor_value')
    assert writer.export_factor_values_to_parquet(current, 'factor') == 2
    with writer._get_connection(read_only=True) as conn:
        pd.testing.assert_frame_equal(conn.execute('SELECT * FROM factor_values ORDER BY ALL').fetchdf(), committed)
    assert hashes() != old_files


def test_invalid_factor_snapshot_rejected_before_borrowed_transaction_changes(tmp_path):
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), export_parquet=False)
    frame = pd.DataFrame({'A':[1.]}, index=pd.to_datetime(['2024-01-02']))
    writer.save_factor_values(frame, 'factor')
    with writer.acquire_write_session() as session:
        conn = session.conn
        before = conn.execute('SELECT * FROM factor_values').fetchall()
        conn.execute('BEGIN TRANSACTION')
        with pytest.raises(ValueError):
            writer.save_factor_values(frame*np.inf, 'factor', duckdb_conn=conn)
        conn.execute('COMMIT')
        assert conn.execute('SELECT * FROM factor_values').fetchall() == before


@pytest.mark.parametrize('date_range', [(20240102, 20240103), ('2024-01-02 12:00', '2024-01-03'), ('2024-01-02', pd.NaT), '2024-01-02', ('2024-01-02',)])
def test_factor_invalid_date_range_rejected(tmp_path, date_range):
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), export_parquet=False)
    frame = pd.DataFrame({'A':[1.]}, index=pd.to_datetime(['2024-01-02']))
    with pytest.raises(ValueError):
        writer.save_factor_values(frame, 'factor', skip_parquet=True, date_range=date_range)


@pytest.mark.parametrize('second_operation', ['save', 'export'])
def test_factor_export_serializes_competing_writers(tmp_path, monkeypatch, second_operation):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import duckdb
    import pyarrow.parquet as pq
    path = tmp_path/'results.duckdb'
    first = DuckDBResultsWriter(str(path), parquet_dir=str(tmp_path/'parquet'))
    second = DuckDBResultsWriter(str(path), parquet_dir=str(tmp_path/'parquet'))
    # Use distinct thread locks to exercise the shared flock, as separate processes do.
    first._write_lock = threading.Lock()
    second._write_lock = threading.Lock()
    frame = pd.DataFrame({'A':[1.,2.]}, index=pd.to_datetime(['2024-01-02','2024-01-03']))
    first.save_factor_values(frame, 'factor')
    entered, release, attempted = threading.Event(), threading.Event(), threading.Event()
    original_export = first._export_factor_values_to_parquet
    def paused_export(*args):
        entered.set()
        assert release.wait(10)
        original_export(*args)
    monkeypatch.setattr(first, '_export_factor_values_to_parquet', paused_export)
    def competing():
        attempted.set()
        if second_operation == 'save':
            return second.save_factor_values(frame*20, 'factor')
        return second.export_factor_values_to_parquet(frame, 'factor')
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first.save_factor_values, frame*10, 'factor')
        try:
            assert entered.wait(10)
            b = pool.submit(competing)
            assert attempted.wait(10)
            assert not b.done()
            with duckdb.connect(str(path)) as conn:
                assert conn.execute('SELECT SUM(factor_value) FROM factor_values').fetchone()[0] == 30.
        finally:
            release.set()
        assert a.result(timeout=10) == 2
        if second_operation == 'save':
            assert b.result(timeout=10) == 2
            expected = 60.
        else:
            with pytest.raises(ValueError, match='differs from current'):
                b.result(timeout=10)
            expected = 30.
    with first._get_connection(read_only=True) as conn:
        assert conn.execute('SELECT SUM(factor_value) FROM factor_values').fetchone()[0] == expected
    files = first._list_visible_parquet_files(tmp_path/'parquet'/'factor_values'/'factor')
    assert pd.concat([pq.ParquetFile(p).read().to_pandas() for p in files]).factor_value.sum() == expected


@pytest.mark.parametrize('entry', ['save', 'export', 'borrowed'])
def test_factor_export_holds_database_lock_until_file_replacement(tmp_path, monkeypatch, entry):
    import fcntl
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), parquet_dir=str(tmp_path/'parquet'))
    frame = pd.DataFrame({'A':[1.]}, index=pd.to_datetime(['2024-01-02']))
    original_export = writer._export_factor_values_to_parquet
    def checked(*args):
        with (tmp_path/'results.duckdb.write.lock').open('a+') as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_export(*args)
    monkeypatch.setattr(writer, '_export_factor_values_to_parquet', checked)
    if entry == 'borrowed':
        with writer.acquire_write_session() as session:
            session.conn.execute('BEGIN')
            writer.save_factor_values(frame, 'factor', duckdb_conn=session.conn, skip_parquet=True)
            with pytest.raises(ValueError, match='Commit the DuckDB'):
                writer.export_factor_values_to_parquet(frame, 'factor', write_session=session)
            assert not (tmp_path/'parquet'/'factor_values'/'factor').exists()
            session.conn.execute('COMMIT')
            assert writer.export_factor_values_to_parquet(frame, 'factor', write_session=session) == 1
    elif entry == 'save':
        writer.save_factor_values(frame, 'factor')
    else:
        writer.export_factor_values_to_parquet(frame, 'factor')
    with (tmp_path/'results.duckdb.write.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_export_refuses_session_for_another_database(tmp_path):
    writer = DuckDBResultsWriter(str(tmp_path/'a.duckdb'), parquet_dir=str(tmp_path/'parquet'))
    other = DuckDBResultsWriter(str(tmp_path/'b.duckdb'), export_parquet=False)
    frame = pd.DataFrame({'A':[1.]}, index=pd.to_datetime(['2024-01-02']))
    with other.acquire_write_session() as session:
        with pytest.raises(ValueError, match='another database'):
            writer.export_factor_values_to_parquet(frame, 'factor', write_session=session)
    assert not (tmp_path/'parquet'/'factor_values').exists()


def test_export_rejects_stale_subset_and_accepts_reordered_current_values(tmp_path):
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), parquet_dir=str(tmp_path/'parquet'))
    frame = pd.DataFrame({'A':[1.,2.], 'B':[3.,4.]}, index=pd.to_datetime(['2024-01-02','2024-01-03']))
    writer.save_factor_values(frame, 'factor')
    for stale in (frame.iloc[:1], frame[['A']], frame*2):
        with pytest.raises(ValueError, match='differs from current'):
            writer.export_factor_values_to_parquet(stale, 'factor')
    assert writer.export_factor_values_to_parquet(frame.iloc[::-1,::-1], 'factor') == 4


def test_parquet_replacement_uses_stable_directory_lock(tmp_path, monkeypatch):
    import fcntl
    import utils.duckdb_results_writer as module
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), parquet_dir=str(tmp_path/'parquet'))
    frame = pd.DataFrame({'A':[1.]}, index=pd.to_datetime(['2024-01-02']))
    real_write = module.pq.write_to_dataset
    lock_path = tmp_path/'parquet'/'factor_values'/'.factor.export.lock'
    def checked(*args, **kwargs):
        with lock_path.open('a+') as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return real_write(*args, **kwargs)
    monkeypatch.setattr(module.pq, 'write_to_dataset', checked)
    writer.save_factor_values(frame, 'factor')
    inode = lock_path.stat().st_ino
    writer.save_factor_values(frame*2, 'factor')
    assert lock_path.stat().st_ino == inode
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize('interrupt_type', [KeyboardInterrupt, SystemExit])
def test_parquet_interruption_restores_previous_export(tmp_path, monkeypatch, interrupt_type):
    import fcntl
    import hashlib
    from pathlib import Path
    import duckdb

    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), parquet_dir=str(tmp_path/'parquet'))
    frame = pd.DataFrame({'A': [1., 2.]}, index=pd.to_datetime(['2024-01-02', '2024-02-02']))
    writer.export_factor_values_to_parquet(frame, 'factor')
    root = tmp_path/'parquet'/'factor_values'
    target = root/'factor'

    def snapshot():
        return {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in target.rglob('*.parquet')}

    before = snapshot()
    real_rename = Path.rename
    failure = interrupt_type('interrupted before publishing staging')

    def interrupted_rename(path, destination):
        if path.name.startswith('.factor.staging-') and destination == target:
            assert not target.exists()
            assert list(root.glob('.factor.backup-*'))
            raise failure
        return real_rename(path, destination)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, 'rename', interrupted_rename)
        with pytest.raises(interrupt_type) as caught:
            writer.export_factor_values_to_parquet(frame*3, 'factor')
    assert caught.value is failure
    assert target.is_dir()
    assert snapshot() == before
    assert not list(root.glob('.factor.staging-*'))
    assert not list(root.glob('.factor.backup-*'))
    with (root/'.factor.export.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert writer.export_factor_values_to_parquet(frame*3, 'factor') == 2
    with duckdb.connect(':memory:') as conn:
        assert conn.execute('SELECT sum(factor_value) FROM read_parquet(?)',
                            [str(target/'**/*.parquet')]).fetchone()[0] == 9.


def test_versioned_explicit_export_selects_snapshot_without_implicit_latest(tmp_path):
    import pyarrow.parquet as pq
    context = dict(run_id='one', pool_id='p', source_hash='s', config_hash='c',
                   input_versions={}, created_at='2024-01-01')
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), parquet_dir=str(tmp_path/'parquet'),
                                publication_metadata=context, data_bindings={})
    frame = pd.DataFrame({'A':[1.]}, index=pd.to_datetime(['2024-01-02']))
    writer.save_factor_values(frame, 'factor')
    writer.save_factor_values(frame*2, 'factor')
    assert writer.export_factor_values_to_parquet(frame, 'factor') == 1
    files = writer._list_visible_parquet_files(tmp_path/'parquet'/'factor_values'/'factor')
    assert pd.concat([pq.ParquetFile(p).read().to_pandas() for p in files]).factor_value.sum() == 1.


@pytest.mark.parametrize('failure_mode', ['borrowed_rollback', 'precommit_failure'])
def test_publication_failure_preserves_committed_catalog_and_can_retry(tmp_path, monkeypatch, failure_mode):
    import utils.data_revision as revision
    writer = DuckDBResultsWriter(str(tmp_path/'results.duckdb'), export_parquet=False,
        publication_metadata=dict(run_id='r',pool_id='p',source_hash='s',config_hash='c',
                                  input_versions={},created_at='2024-01-01'), data_bindings={})
    frame = pd.DataFrame({'A':[1.,2.]},index=pd.to_datetime(['2024-01-02','2024-01-03']))
    writer.save_factor_values(frame, 'factor')
    def catalog():
        with writer._get_connection(read_only=True) as conn:
            return conn.execute('SELECT * FROM meta.artifacts ORDER BY artifact_id').fetchall()
    before = catalog()
    if failure_mode == 'borrowed_rollback':
        with writer.acquire_write_session() as session:
            session.conn.execute('BEGIN')
            assert writer.save_factor_values(frame*2, 'factor', duckdb_conn=session.conn) == 2
            session.conn.execute('ROLLBACK')
    else:
        failure = RuntimeError('injected precommit failure')
        def fail(conn):
            raise failure
        with monkeypatch.context() as patch:
            patch.setattr(revision, 'advance_revision', fail)
            with pytest.raises(RuntimeError) as caught:
                writer.save_factor_values(frame*2, 'factor')
            assert caught.value is failure
    assert catalog() == before
    assert not hasattr(writer, 'published_refs')
    assert writer.save_factor_values(frame*2, 'factor') == 2
    after = catalog()
    assert len(after) == 2
    assert writer.save_factor_values(frame*2, 'factor') == 2
    assert catalog() == after
    with writer._get_connection(read_only=True) as conn:
        assert conn.execute('SELECT SUM(factor_value) FROM factor_values').fetchone()[0] == 9.


@pytest.mark.parametrize('zone', ['Asia/Shanghai', 'America/New_York', 'UTC'])
@pytest.mark.parametrize('values', [
    ['2024-03-10 06:59:59.1234+00', '2024-11-03 06:00:00+00', None],
    ['1960-01-02 00:00:00+00', '2024-01-03 01:00:00.000001+00'],
    ['-infinity', 'infinity', None],
    [None, None],
    [],
])
def test_audit_timezone_bounds_match_original_utc_without_changing_session(zone, values):
    import duckdb
    from utils.data_store import audit_relation
    with duckdb.connect() as conn:
        conn.execute('CREATE TABLE samples (event_time TIMESTAMPTZ, day DATE, local_time TIMESTAMP)')
        for value in values:
            conn.execute("INSERT INTO samples VALUES (?::TIMESTAMPTZ, DATE '2024-01-02', TIMESTAMP '2024-01-02 03:04:05')", [value])
        conn.execute("SET TimeZone='UTC'")
        original_bounds = conn.execute('SELECT CAST(min(event_time) AS VARCHAR), CAST(max(event_time) AS VARCHAR) FROM samples').fetchone() if values else (None, None)
        expected = audit_relation(conn, 'samples')
        assert expected['dates']['event_time'] == list(original_bounds)
        conn.execute('SET TimeZone=?', [zone])
        conn.execute('BEGIN')
        txid = conn.execute('SELECT current_transaction_id()').fetchone()[0]
        assert audit_relation(conn, 'samples') == expected
        assert conn.execute("SELECT current_setting('TimeZone')").fetchone()[0] == zone
        assert conn.execute('SELECT current_transaction_id()').fetchone()[0] == txid
        conn.execute('ROLLBACK')


@pytest.mark.parametrize('version', [123, True, ['version'], {'id':'version'}, '', ' \t\n', None])
def test_data_ref_rejects_nonstring_or_blank_version(version):
    from utils.data_store import DataRef
    with pytest.raises(ValueError, match='version'):
        DataRef('input', 'prices', version)


@pytest.mark.parametrize('version', ['v1', 'f'*64, ' legacy-id '])
def test_data_ref_preserves_exact_nonblank_string_version(version):
    from utils.data_store import DataRef
    assert DataRef('output', 'factor_values', version).version == version


def test_timezone_change_reuses_committed_version_and_existing_binding(tmp_path):
    from utils.data_store import write_store, publish_relation, read_reference
    path = tmp_path/'versions.duckdb'
    metadata = {'run_id':'timezone-test'}
    refs = []
    for zone in ['UTC', 'Asia/Shanghai', 'America/New_York']:
        with write_store(path) as conn:
            conn.execute('SET TimeZone=?', [zone])
            conn.execute("CREATE OR REPLACE TEMP VIEW source AS SELECT TIMESTAMPTZ '2024-01-02 00:00:00+00' AS known_at, 1.0 AS value")
            refs.append(publish_relation(conn, 'source', 'features.events', metadata=metadata))
    assert refs[0] == refs[1] == refs[2]
    assert len(read_reference(path, refs[0])) == 1


@pytest.mark.parametrize('key', [1, True, None, 1.5])
@pytest.mark.parametrize('location', ['root', 'dict', 'list', 'tuple', 'mixed'])
@pytest.mark.parametrize('function', ['canonical', 'digest'])
def test_canonical_rejects_nonstring_keys_at_every_depth(key, location, function):
    from utils import data_store
    value = {key: 'value'}
    if location == 'dict':
        value = {'outer': value}
    elif location == 'list':
        value = [value]
    elif location == 'tuple':
        value = ('ok', {'nested': [value]})
    elif location == 'mixed':
        value['valid'] = 'also present'
    with pytest.raises(TypeError):
        getattr(data_store, function)(value)


@pytest.mark.parametrize('name', [1, None, [], b'prices', True, 1.5, {'table': 'prices'}])
def test_table_name_rejects_nonstring_before_split(name):
    from utils.data_store import table_name
    with pytest.raises(ValueError):
        table_name(name)


def test_canonical_invalid_key_does_not_enter_document_legacy_fallback(tmp_path):
    from utils.data_store import publish_document
    source = tmp_path / 'legacy.json'
    source.write_text('{"1": NaN}')
    output = tmp_path / 'output.duckdb'
    with pytest.raises(TypeError):
        publish_document(output, {1: float('nan')},
                         metadata={'legacy_source': 'synthetic'}, original_file=str(source))
    assert not output.exists()


@pytest.mark.parametrize('container', ['dict', 'list'])
def test_canonical_circular_input_keeps_json_value_error(container):
    from utils.data_store import canonical
    value = {} if container == 'dict' else []
    if container == 'dict':
        value['self'] = value
    else:
        value.append(value)
    with pytest.raises(ValueError, match='Circular reference'):
        canonical(value)


def test_canonical_shared_subtree_and_string_subclass_preserve_json_bytes():
    import hashlib
    from utils.data_store import canonical, digest
    class Key(str):
        pass
    shared = {Key('中文'): ['é', None, True, -0.0]}
    value = {'b': shared, 'a': (shared,)}
    expected = '{"a":[{"中文":["é",null,true,-0.0]}],"b":{"中文":["é",null,true,-0.0]}}'
    assert canonical(value).encode('utf-8') == expected.encode('utf-8')
    assert digest(value) == hashlib.sha256(expected.encode('utf-8')).hexdigest()


@pytest.mark.parametrize('name', ['', 'schema.', '.table', 'a.b.c', 'nu\x00l'])
def test_table_name_invalid_string_rules_unchanged(name):
    from utils.data_store import table_name
    with pytest.raises(ValueError):
        table_name(name)


def test_table_name_quotes_identifiers_without_executing_embedded_sql():
    import duckdb
    from utils.data_store import table_name
    name = 'odd"; DROP TABLE sentinel;--'
    with duckdb.connect() as conn:
        conn.execute('CREATE TABLE sentinel AS SELECT 7 AS value')
        conn.execute(f'CREATE TABLE {table_name(name)} AS SELECT 11 AS value')
        assert conn.execute(f'SELECT value FROM {table_name(name)}').fetchone() == (11,)
        assert conn.execute('SELECT value FROM sentinel').fetchone() == (7,)


@pytest.mark.parametrize('legacy', [False, True])
def test_read_document_missing_body_is_explicit_value_error(tmp_path, legacy):
    import duckdb
    from utils.data_store import publish_document, read_document
    path = tmp_path / 'documents.duckdb'
    if legacy:
        source = tmp_path / 'legacy.json'
        source.write_text('{"value": NaN}')
        ref = publish_document(path, {'value': float('nan')},
                               metadata={'legacy_source': 'synthetic'}, original_file=str(source))
    else:
        ref = publish_document(path, {'value': 7}, metadata={})
    with duckdb.connect(str(path)) as conn:
        conn.execute('DELETE FROM meta.documents WHERE artifact_id=?', [ref.version])
        catalog = conn.execute('SELECT * FROM meta.artifacts').fetchall()
    with pytest.raises(ValueError, match='Document body is missing'):
        read_document(path, ref)
    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute('SELECT * FROM meta.artifacts').fetchall() == catalog
        assert conn.execute('SELECT count(*) FROM meta.documents').fetchone() == (0,)


@pytest.mark.parametrize('value', [None, {}, [], False, 0])
def test_read_document_present_falsy_body_is_not_missing(tmp_path, value):
    from utils.data_store import publish_document, read_document
    path = tmp_path / 'documents.duckdb'
    ref = publish_document(path, value, metadata={})
    result = read_document(path, ref)
    assert type(result) is type(value)
    assert result == value


def test_read_document_existing_reference_and_type_errors_unchanged(tmp_path):
    from utils.data_store import DataRef, publish_document, publish_artifact, read_document
    path = tmp_path / 'documents.duckdb'
    publish_document(path, {'value': 7}, metadata={})
    with pytest.raises(ValueError, match='reference does not exist'):
        read_document(path, DataRef('output', 'meta.documents', 'absent'))
    ref = publish_artifact(path, 'features.sample', frame=pd.DataFrame({'value': [7]}),
                           metadata={'run_id': 'synthetic'})
    with pytest.raises(TypeError, match='Expected a document artifact'):
        read_document(path, ref)
