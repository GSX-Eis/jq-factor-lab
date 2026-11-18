"""Synthetic, hand-computed Bar evaluations and publication boundaries."""
import json

import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.time_series import (
    BarCrossSectionalEvaluator, BarReturnLabeler, ContractValidationError,
    EXCLUSION_PRIORITY, ResultIdentity, TimeSeriesEvaluator, evaluate_bars,
    read_bar_csv, read_factor_csv, write_evaluation_artifacts,
)
from backtest_and_analysis.time_series.labels import LABEL_COLUMNS
from utils.session_calendar import SessionCalendar


T = pd.Timestamp('2026-12-01T14:30Z')


def labels(factors, returns=None, instrument='A', horizon=1):
    n = len(factors)
    times = pd.date_range(T, periods=n, freq='1min')
    returns = np.asarray(factors if returns is None else returns, dtype=float)
    return pd.DataFrame(dict(
        instrument_id=instrument, frequency='1min', entry_offset_bars=1,
        horizon_bars=horizon, signal_timestamp=times,
        signal_available_at=times + pd.Timedelta('1min'),
        entry_timestamp=times + pd.Timedelta('1min'),
        exit_bar_timestamp=times + pd.Timedelta(minutes=horizon),
        exit_timestamp=times + pd.Timedelta(minutes=horizon + 1),
        entry_price=100., exit_price=100. + returns, price_change_points=returns,
        return_pct=returns, status='valid', exclusion_reason=None,
        factor_value=np.asarray(factors, dtype=float)))


def ts(frame, **kwargs):
    return TimeSeriesEvaluator(analysis_mode='time_series', **kwargs).evaluate(frame)


def cs(frame, **kwargs):
    return BarCrossSectionalEvaluator(analysis_mode='cross_sectional', **kwargs).evaluate(frame)


def instrument_row(result):
    return result.ic_by_horizon.query("scope == 'per_instrument'").iloc[0]


@pytest.mark.parametrize('sign', [1, -1])
def test_monotonic_ic_and_nonoverlapping_sampling(sign):
    frame = labels(np.arange(1, 31), sign * np.arange(1, 31), horizon=5)
    result = ts(frame.sample(frac=1, random_state=7))
    row = instrument_row(result)
    assert row.spearman_ic == pytest.approx(sign)
    assert row.non_overlapping_ic == pytest.approx(sign)
    assert row.valid_count == 30 and row.non_overlapping_count == 6
    assert row.ic_status == row.non_overlapping_status == 'ok'
    assert result.quantile_stats.descriptive_only.all()
    assert result.quantile_stats.quantile_scope.eq('full_sample_descriptive_only').all()


def test_sampling_precedes_valid_filter_and_counts_all_reasons():
    frame = labels(np.arange(30), horizon=5)
    for i, reason in enumerate(EXCLUSION_PRIORITY):
        frame.loc[i, ['status', 'exclusion_reason']] = ['excluded', reason.value]
    row = instrument_row(ts(frame))
    assert row.non_overlapping_count == 5
    assert row.valid_count == 25 and row.excluded_count == 5
    assert list(row.exclusion_counts) == [r.value for r in EXCLUSION_PRIORITY]
    assert all(v == 1 for v in row.exclusion_counts.values())


def test_hand_computed_ties_and_multi_instrument_isolation():
    a = labels([1, 1, 3, 4], [1, 3, 2, 4])
    assert instrument_row(ts(a)).spearman_ic == pytest.approx(3 / np.sqrt(4.5 * 5))
    frame = pd.concat([labels([1, 2, 3, 4]), labels([10, 20, 30, 40], [4, 3, 2, 1], 'B')])
    result = ts(frame)
    rows = result.ic_by_horizon.set_index('instrument_id')
    assert rows.loc['A', 'spearman_ic'] == pytest.approx(1)
    assert rows.loc['B', 'spearman_ic'] == pytest.approx(-1)
    summary = rows.loc['__ALL__']
    assert summary.mean_spearman_ic == pytest.approx(0)
    assert summary.median_spearman_ic == pytest.approx(0)
    assert summary.valid_instrument_count == 2 and summary.valid_count == 8


@pytest.mark.parametrize('values,status,actual', [
    ([1], 'insufficient_samples', 0), ([1, 1, 1], 'constant_factor', 0),
    ([0] * 9 + [1], 'unable_to_form_quantiles', 1),
])
def test_quantile_edge_status(values, status, actual):
    result = ts(labels(values), n_quantiles=5)
    row = instrument_row(result)
    assert row.quantile_status == status
    assert row.actual_quantiles == actual
    if row.actual_quantiles < 2:
        assert result.quantile_stats.empty


def test_reduced_quantiles_do_not_split_ties():
    values = [0] * 4 + [1] * 4 + [2] * 4
    frame = labels(values, np.arange(12))
    result = ts(frame, n_quantiles=5)
    row = instrument_row(result)
    assert row.quantile_status == 'reduced_quantiles'
    # All four zeros/ones stay in the bottom bin, all four twos in the top bin.
    assert result.quantile_stats.sample_count.tolist() == [8, 4]
    assert result.quantile_stats.mean_return_pct.tolist() == [3.5, 9.5]
    assert result.quantile_stats['quantile'].tolist() == [1, 2]


@pytest.mark.parametrize('factors,returns,status', [
    ([1, 2], [2, 1], 'insufficient_samples'),
    ([1, 1, 1], [1, 2, 3], 'constant_factor'),
    ([1, 2, 3], [1, 1, 1], 'constant_return'),
])
def test_uncomputable_ic_is_nan(factors, returns, status):
    row = instrument_row(ts(labels(factors, returns)))
    assert row.ic_status == status and np.isnan(row.spearman_ic)


def test_cross_sectional_ic_quantiles_and_time_aggregation():
    frame = pd.concat([labels([i, i], [i, 4-i], str(i)) for i in range(1, 4)])
    result = cs(frame, n_quantiles=2)
    row = result.ic_by_horizon.iloc[0]
    assert result.cross_sections.spearman_ic.tolist() == pytest.approx([1, -1])
    assert row.mean_ic == pytest.approx(0) and row.median_ic == pytest.approx(0)
    assert row.std_ic == pytest.approx(np.sqrt(2)) and row.icir == pytest.approx(0)
    assert row.positive_ic_ratio == .5
    assert row.valid_cross_sections == 2 and row.excluded_cross_sections == 0
    assert row.mean_valid_instruments == row.median_valid_instruments == 3
    q = result.quantile_stats.set_index('quantile')
    assert q.loc[1, 'mean_return_pct'] == pytest.approx(2)
    assert q.loc[1, 'median_return_pct'] == pytest.approx(2)
    assert q.loc[1, 'total_sample_count'] == 4
    assert q.loc[2, 'cross_section_count'] == 2
    assert not q.descriptive_only.any()


def test_bad_cross_sections_do_not_pollute_good_ones():
    frame = pd.concat([labels([i, 1, i], instrument=str(i)) for i in range(1, 4)], ignore_index=True)
    frame.loc[(frame.signal_timestamp == T + pd.Timedelta('2min')) & (frame.instrument_id != '1'),
              ['status', 'exclusion_reason']] = ['excluded', 'missing_bar']
    result = cs(frame)
    assert result.cross_sections.cs_status.tolist() == ['ok', 'constant_factor', 'insufficient_cross_section']
    row = result.ic_by_horizon.iloc[0]
    assert row.valid_cross_sections == 1 and row.mean_ic == pytest.approx(1)
    assert row.constant_factor_cross_section_count == row.insufficient_cross_section_count == 1
    assert np.isnan(row.std_ic) and np.isnan(row.icir)
    assert row.exclusion_counts['missing_bar'] == 2
    assert result.exclusion_summary['valid_count'] == 7


@pytest.mark.parametrize('mode', [None, '', 'auto', 7, ['time_series']])
def test_explicit_mode_required(mode):
    with pytest.raises(ContractValidationError, match='analysis_mode'):
        evaluate_bars(pd.DataFrame(), analysis_mode=mode)
    with pytest.raises(ContractValidationError, match='analysis_mode'):
        TimeSeriesEvaluator(analysis_mode=mode)


def test_single_instrument_cs_never_infers_time_series():
    result = cs(labels([1, 2, 3]))
    assert result.analysis_mode == 'cross_sectional'
    assert result.cross_sections.cs_status.eq('insufficient_cross_section').all()
    assert result.ic_by_horizon.iloc[0].ic_status == 'insufficient_cross_section'


@pytest.mark.parametrize('factory,param', [(TimeSeriesEvaluator, 'min_ic_samples'),
    (TimeSeriesEvaluator, 'n_quantiles'), (BarCrossSectionalEvaluator, 'min_cross_section_size')])
@pytest.mark.parametrize('value', [True, np.bool_(True), 1, 2.5, '3', None])
def test_integer_parameter_validation(factory, param, value):
    mode = 'time_series' if factory is TimeSeriesEvaluator else 'cross_sectional'
    with pytest.raises(ContractValidationError, match=param):
        factory(analysis_mode=mode, **{param: value})


@pytest.mark.parametrize('column', ['timestamp_utc', 'instrument_id'])
def test_factor_csv_aliases_and_ambiguity(tmp_path, column):
    alias = {'timestamp_utc': 'timestamp', 'instrument_id': 'symbol'}
    path = tmp_path / 'f.csv'
    frame = pd.DataFrame(dict(timestamp=['2026-12-01T14:30Z'], symbol=['000001'], factor_value=[2.]))
    frame.to_csv(path, index=False)
    actual = read_factor_csv(path)
    assert actual.instrument_id.tolist() == ['000001']
    assert actual.timestamp_utc.tolist() == [T]
    frame[column] = frame[alias[column]]
    frame.to_csv(path, index=False)
    with pytest.raises(ContractValidationError, match='ambiguous'):
        read_factor_csv(path)


def test_empty_embedded_factor_dtype_and_alias_utc(tmp_path):
    path = tmp_path / 'empty.csv'
    pd.DataFrame(columns=['timestamp_utc', 'instrument_id', 'frequency', 'open', 'high', 'low',
        'close', 'volume', 'source', 'data_version', 'factor_value']).to_csv(path, index=False)
    assert read_bar_csv(path).factor_value.dtype == np.dtype('float64')
    pd.DataFrame(dict(timestamp=['2026-12-01T14:30'], symbol=['A'], factor_value=[1.])).to_csv(path, index=False)
    with pytest.raises(ContractValidationError, match='UTC'):
        read_factor_csv(path)


@pytest.fixture
def evaluated():
    times = pd.date_range(T, periods=10, freq='1min')
    bars = pd.DataFrame(dict(timestamp_utc=times, instrument_id='000001', frequency='1min',
        open=100., high=120., low=90., close=np.arange(101., 111.), volume=1.,
        source='SYNTHETIC', data_version='v1', factor_value=np.arange(10.)))
    calendar = SessionCalendar('NQ', '2026-12-01', '2026-12-01')
    result = evaluate_bars(bars, labeler=BarReturnLabeler(calendar, horizons=[1, 5]), analysis_mode='time_series')
    identity = ResultIdentity('a'*64, {'bars.csv': 'b'*64}, calendar.calendar_hash, 'test', 'bar_evaluation.v1')
    return bars, result, identity


def test_factor_join_on_labels_and_bars(evaluated):
    bars, result, _ = evaluated
    factors = bars[['timestamp_utc', 'instrument_id', 'factor_value']].sample(frac=1, random_state=3)
    joined = TimeSeriesEvaluator(analysis_mode='time_series').evaluate(
        result.per_bar_labels.drop(columns='factor_value'), factors)
    pd.testing.assert_frame_equal(joined.per_bar_labels, result.per_bar_labels)
    other = evaluate_bars(bars.drop(columns='factor_value'), factors,
        labeler=BarReturnLabeler(SessionCalendar('NQ', '2026-12-01', '2026-12-01'), horizons=[1, 5]),
        analysis_mode='time_series')
    pd.testing.assert_frame_equal(result.per_bar_labels, other.per_bar_labels)


def test_artifacts_roundtrip_and_identity(tmp_path, evaluated):
    bars, result, identity = evaluated
    paths = write_evaluation_artifacts(result, tmp_path, result_identity=identity, config={'analysis_mode': 'time_series'})
    assert set(paths) == {'per_bar_labels', 'ic_by_horizon', 'quantile_stats', 'summary'}
    raw = pd.read_csv(paths['per_bar_labels'], dtype={'instrument_id': str})
    assert set(raw.columns) == set(LABEL_COLUMNS) | {'factor_value'}
    for name in ['signal_timestamp', 'signal_available_at', 'entry_timestamp', 'exit_bar_timestamp', 'exit_timestamp']:
        assert raw[name].dropna().str.endswith('Z').all()
        pd.testing.assert_series_equal(pd.to_datetime(raw[name], utc=True), result.per_bar_labels[name], check_names=False)
    for name in ['factor_value', 'return_pct', 'entry_price', 'exit_price', 'price_change_points']:
        assert np.allclose(raw[name], result.per_bar_labels[name], equal_nan=True)
    summary_text = paths['summary'].read_text()
    summary = json.loads(summary_text)
    assert summary['data_identity']['instruments'] == ['000001']
    assert summary['data_identity']['raw_coverage_end_utc'] == bars.timestamp_utc.max().isoformat().replace('+00:00', 'Z')
    assert summary['calendar_identity']['calendar_hash'] == identity.calendar_hash
    assert summary['result_identity']['identity_hash'] == identity.identity_hash()
    assert summary['ic_by_horizon'] and summary['quantile_stats']
    assert not any(name in summary_text for name in ['p_value', 't_stat', 'is_significant', 'NaN', 'Infinity'])
    assert len(pd.read_csv(paths['ic_by_horizon'])) == len(result.ic_by_horizon)
    assert len(pd.read_csv(paths['quantile_stats'])) == len(result.quantile_stats)


def test_empty_quantile_artifact_and_json_null(tmp_path, evaluated):
    _, result, identity = evaluated
    result = ts(labels([1, 1, 1]))
    paths = write_evaluation_artifacts(result, tmp_path, result_identity=identity)
    assert pd.read_csv(paths['quantile_stats']).empty
    summary = json.loads(paths['summary'].read_text())
    assert summary['ic_by_horizon'][0]['spearman_ic'] is None


def test_failed_write_invalidates_stale_summary(tmp_path, evaluated, monkeypatch):
    _, result, identity = evaluated
    write_evaluation_artifacts(result, tmp_path, result_identity=identity)
    with pytest.raises(ContractValidationError):
        write_evaluation_artifacts(result, tmp_path, result_identity=identity, config={'bad': object()})
    assert not (tmp_path / 'summary.json').exists()
    write_evaluation_artifacts(result, tmp_path, result_identity=identity)
    def fail(*args, **kwargs):
        raise OSError('disk failure')
    monkeypatch.setattr(pd.DataFrame, 'to_csv', fail)
    with pytest.raises(OSError, match='disk failure'):
        write_evaluation_artifacts(result, tmp_path, result_identity=identity)
    assert not (tmp_path / 'summary.json').exists()


def test_empty_results_have_readable_artifacts(tmp_path, evaluated):
    _, result, identity = evaluated
    for evaluator in [ts, cs]:
        empty = evaluator(result.per_bar_labels.iloc[:0])
        paths = write_evaluation_artifacts(empty, tmp_path, result_identity=identity)
        assert all(pd.read_csv(paths[name]).empty for name in ['per_bar_labels', 'ic_by_horizon', 'quantile_stats'])


def test_empty_raw_bars(evaluated):
    bars, _, _ = evaluated
    empty = evaluate_bars(bars.iloc[:0], labeler=BarReturnLabeler(
        SessionCalendar('NQ', '2026-12-01', '2026-12-01')), analysis_mode='time_series')
    assert empty.per_bar_labels.empty and empty.ic_by_horizon.empty


def test_frequency_horizon_and_summary_isolate_uncomputable_assets():
    a = labels([1, 2, 3, 4, 5, 6], horizon=2)
    b = labels([1, 1, 1, 1, 1, 1], instrument='B', horizon=2)
    c = labels([6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6], horizon=3)
    d = a.assign(frequency='5min', signal_available_at=a.signal_timestamp + pd.Timedelta('5min'))
    result = ts(pd.concat([a, b, c, d]))
    summary = result.ic_by_horizon.query("scope == 'multi_instrument_summary'")
    assert len(summary) == 3
    h2 = summary.query("frequency == '1min' and horizon_bars == 2").iloc[0]
    assert h2.instrument_count == 2 and h2.valid_instrument_count == 1
    assert h2.mean_spearman_ic == pytest.approx(1)
    h3 = summary.query("frequency == '1min' and horizon_bars == 3").iloc[0]
    assert h3.mean_spearman_ic == pytest.approx(-1)
    assert h3.valid_non_overlapping_instrument_count == 0


def test_cross_sectional_constant_return_and_reduced_tied_groups():
    frame = pd.concat([labels([i, i], [1, 1], str(j))
                       for j, i in enumerate([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])])
    result = cs(frame)
    assert result.cross_sections.cs_status.eq('constant_return').all()
    assert result.ic_by_horizon.iloc[0].ic_status == 'constant_return'
    assert result.quantile_stats.quantile_status.eq('reduced_quantiles').all()
    assert result.quantile_stats.total_sample_count.tolist() == [16, 8]


@pytest.mark.parametrize('column,value', [('signal_timestamp', '2026-12-01'),
                                         ('factor_value', True), ('return_pct', np.inf)])
def test_malformed_label_values_raise_contract_error(column, value):
    frame = labels([1, 2, 3])
    frame[column] = value
    with pytest.raises(ContractValidationError):
        ts(frame)


def test_failure_during_rename_leaves_no_summary(tmp_path, evaluated, monkeypatch):
    import backtest_and_analysis.time_series.artifacts as artifacts
    _, result, identity = evaluated
    write_evaluation_artifacts(result, tmp_path, result_identity=identity)
    original = artifacts.os.replace
    def fail_summary(source, dest):
        if str(dest).endswith('summary.json'):
            raise OSError('rename failed')
        original(source, dest)
    monkeypatch.setattr(artifacts.os, 'replace', fail_summary)
    with pytest.raises(OSError, match='rename failed'):
        write_evaluation_artifacts(result, tmp_path, result_identity=identity)
    assert not (tmp_path / 'summary.json').exists()
    assert not list(tmp_path.glob('.bar-evaluation-*'))


def test_cross_sectional_artifacts_and_per_bar_text(tmp_path, evaluated):
    _, _, identity = evaluated
    frame = pd.concat([labels([i, i], [i, 4-i], str(i)) for i in range(1, 4)])
    result = cs(frame, n_quantiles=2)
    paths = write_evaluation_artifacts(result, tmp_path, result_identity=identity)
    summary = json.loads(paths['summary'].read_text())
    assert summary['analysis_mode'] == 'cross_sectional'
    assert [row['spearman_ic'] for row in summary['cross_sections']] == pytest.approx([1, -1])
    assert summary['ic_by_horizon'][0]['std_ic'] == pytest.approx(np.sqrt(2))
    assert summary['quantile_stats'][0]['quantile_scope'] == 'cross_sectional_time_aggregated'
    assert summary['quantile_stats'][0]['descriptive_only'] is False
    assert len(pd.read_csv(paths['quantile_stats'])) == 2
    # Real labeler exclusions must serialize as reason values, rather than enum reprs.
    _, real, identity = evaluated
    paths = write_evaluation_artifacts(real, tmp_path, result_identity=identity)
    reasons = pd.read_csv(paths['per_bar_labels']).exclusion_reason.dropna()
    assert reasons.eq('insufficient_history').all() and len(reasons) > 0


def test_identity_mismatch_invalidates_previous_publication(tmp_path, evaluated):
    _, result, identity = evaluated
    write_evaluation_artifacts(result, tmp_path, result_identity=identity)
    mismatch = ResultIdentity('a'*64, {}, 'c'*64, 'test', 'bar_evaluation.v1')
    with pytest.raises(ContractValidationError, match='calendar_hash'):
        write_evaluation_artifacts(result, tmp_path, result_identity=mismatch)
    assert not (tmp_path / 'summary.json').exists()


def test_join_rejects_wrong_label_key_type_as_contract_error(evaluated):
    bars, result, _ = evaluated
    frame = result.per_bar_labels.assign(instrument_id=1)
    factors = bars[['timestamp_utc', 'instrument_id', 'factor_value']]
    with pytest.raises(ContractValidationError, match='instrument_id'):
        TimeSeriesEvaluator(analysis_mode='time_series').evaluate(frame, factors)


@pytest.mark.parametrize('damage', ['missing_factor', 'infinite_factor', 'duplicate', 'bad_status', 'bad_horizon'])
def test_invalid_evaluation_inputs(damage):
    frame = labels([1, 2, 3])
    if damage == 'missing_factor':
        frame = frame.drop(columns='factor_value')
    elif damage == 'infinite_factor':
        frame.loc[0, 'factor_value'] = np.inf
    elif damage == 'duplicate':
        frame = pd.concat([frame, frame.iloc[:1]])
    elif damage == 'bad_status':
        frame.loc[0, 'status'] = 'unknown'
    else:
        frame.loc[0, 'horizon_bars'] = 0
    with pytest.raises(ContractValidationError):
        ts(frame)
