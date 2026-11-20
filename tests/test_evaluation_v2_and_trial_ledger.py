"""Bar v2, PIT split diagnostics and candidate-family evidence through the Bar entry points."""
import hashlib
import json
from math import erfc, sqrt
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.evaluation import EvaluationEngine, evaluate_interval_usability
from backtest_and_analysis.evaluation.calibration import Trial, TrialLedger
from backtest_and_analysis.evaluation.cross_section import bartlett_hac
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.evaluation.split import SplitConfig, align_released_features
from backtest_and_analysis.time_series import (
    BarReturnLabeler, BarTrialFamily, ResultIdentity, TimeSeriesEvaluator,
    evaluate_bars, write_evaluation_artifacts,
)
from backtest_and_analysis.time_series.evaluation_v2 import _cross_section_rows, _record_trials
from test_time_series_cli_end_to_end import base_config, read_outputs, run_cli
from utils.session_calendar import SessionCalendar


BOOTSTRAP = dict(block_length=4, n_boot=21, seed=73, min_valid_replicates=15, alpha=.1)
HORIZONS = (1, 3)
START = pd.Timestamp('2026-12-01T14:31Z')
TRAIN_END = START + pd.Timedelta(minutes=32)
OOS_START = START + pd.Timedelta(minutes=33)
SPLIT = SplitConfig(TRAIN_END, OOS_START)


def _bars(periods=64, assets=('NQZ26',)):
    rng = np.random.default_rng(511)
    frames = []
    for asset in assets:
        price = rng.uniform(99, 120, periods)
        frames.append(pd.DataFrame(dict(
            timestamp_utc=pd.date_range(START, periods=periods, freq='1min'),
            instrument_id=asset, frequency='1min', open=price, high=price + 2,
            low=price - 2, close=price + rng.uniform(-1, 1, periods), volume=1.,
            source='SYNTHETIC', data_version='NERE889', factor_value=rng.normal(size=periods),
        )))
    return pd.concat(frames, ignore_index=True)


def _labeler():
    return BarReturnLabeler(SessionCalendar('NQ', start_date='2026-12-01', end_date='2026-12-01'),
                            horizons=HORIZONS)


def _artifacts(payload):
    return {item['artifact_type']: Path(item['path']) for item in payload['artifacts']}


@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize('direction', [1, -1])
@pytest.mark.parametrize('mode', ['time_series', 'cross_sectional'])
@pytest.mark.parametrize('case', ['audit', 'mixed'])
def test_bar_v2_signed_integer_direction_counts(dtype, direction, mode, case):
    smallest, largest = np.iinfo(dtype).min, np.iinfo(dtype).max
    first = [smallest, -1, 0, 1, 2 if case == 'audit' else largest, 3]
    closes = [101] * 6 if case == 'audit' else [101, 101, 100, 101, 99, 101]
    signals = [first]
    if mode == 'cross_sectional':
        signals += [first, first] if case == 'audit' else [
            [-1, 0, -1, -1, smallest, -1],
            [1, 1, 1, 0, 0, 0],
        ]

    assets = ('A', 'B', 'C')[:len(signals)]
    bars = _bars(periods=6, assets=assets)
    bars['open'], bars['high'], bars['low'] = 100, 101, 99
    bars['close'] = np.tile(closes, len(assets))
    bars['factor_value'] = np.asarray(signals, dtype=dtype).ravel()
    labeler = BarReturnLabeler(
        SessionCalendar('NQ', start_date='2026-12-01', end_date='2026-12-01'),
        entry_offset_bars=1, horizons=(1,),
    )

    def directions(frame):
        result = evaluate_bars(
            frame, labeler=labeler, analysis_mode=mode, evaluation_v2=True,
            n_quantiles=2, direction=direction,
        )
        row, = result.evaluation_v2.splits['full']['horizons']
        buckets = ({q: bucket['direction'] for q, bucket in row['buckets'].items()}
                   if mode == 'time_series' else row['direction_by_bucket'])
        return row['direction'], buckets

    overall, buckets = directions(bars)
    float_overall, float_buckets = directions(bars.astype({'factor_value': 'float64'}))

    def counts(stats):
        return stats['hits'], stats['denominator'], stats['up_count']

    if case == 'audit':
        assert counts(overall) == (2 * len(assets), 4 * len(assets), 4 * len(assets))
    else:
        # Lag 1 gives the first five signals returns with signs +, 0, +, -, +.
        # A zero signal abstains. A zero return is an active miss for either direction.
        if mode == 'time_series':
            expected = {1: (0, 2, 1), 2: (1, 2, 1)} if direction == 1 else {
                1: (1, 2, 1), 2: (1, 2, 1),
            }
        else:
            expected = {1: (1, 6, 4), 2: (3, 5, 3)} if direction == 1 else {
                1: (4, 6, 4), 2: (1, 5, 3),
            }
        assert {q: counts(stats) for q, stats in buckets.items()} == expected
        assert counts(overall) == tuple(map(sum, zip(*expected.values(), strict=True)))

    assert overall == float_overall
    assert buckets == float_buckets


def test_preregister_split_family_and_use_expected_grid_and_scheduled_exits(monkeypatch):
    events = []
    original_init = TrialLedger.__init__
    original_evaluate = EvaluationEngine.evaluate

    def register(ledger, candidates):
        events.append(tuple(candidates))
        original_init(ledger, candidates)

    def evaluate(engine, *args, **options):
        events.append(options.get('split'))
        return original_evaluate(engine, *args, **options)

    monkeypatch.setattr(TrialLedger, '__init__', register)
    monkeypatch.setattr(EvaluationEngine, 'evaluate', evaluate)
    # The first scheduled OOS slot has no Bar. Both crossing labels lack exit prices.
    bars = _bars().drop(index=[32, 33, 34])
    labeler = _labeler()
    result = evaluate_bars(bars, labeler=labeler, analysis_mode='time_series', split=SPLIT,
                           bootstrap=BOOTSTRAP, factor_id='momentum', direction=-1, standardization='zscore')
    assert events[0] == (('momentum', 1), ('momentum', 3))
    assert events[1:].count(SPLIT) == 1
    assert not labeler.include_scheduled_exit
    assert 'scheduled_exit_timestamp' not in result.per_bar_labels
    advanced = result.evaluation_v2
    train, oos = (advanced.splits[name] for name in ('train', 'oos'))
    assert train['quantile_scheme'] == oos['quantile_scheme'] == 'fitted_on_train'
    assert train['fit_identity'] == oos['fit_identity']
    fit = train['fit_identity']['1min']['NQZ26']
    assert fit['direction'] == -1 and fit['standardization'] == 'zscore'
    assert train['split_purged_count'] == {1: 0, 3: 2}
    assert oos['split_purged_count'] == {1: 0, 3: 0}
    assert oos['horizons'][0]['grid_start_utc'] == OOS_START - pd.Timedelta(minutes=1)
    for row in (*train['horizons'], *oos['horizons']):
        assert row['intervals_verified']
        assert row['bootstrap']['grid_size'] == 32
        assert {k: row['bootstrap'][k] for k in BOOTSTRAP} == BOOTSTRAP
        assert row['bootstrap']['reason'] is None

    ledger = advanced.trial_ledger
    assert ledger['split'] == 'train' and ledger['attempt_count'] == 2
    assert ledger['method'] == 'holm' and ledger['alpha'] == .05
    assert ledger['missing_p_value_candidates'] == [('momentum', 1), ('momentum', 3)]
    assert not ledger['selection_complete'] and ledger['best'] is None
    for row in ledger['candidates']:
        assert row['p_value'] is row['adjusted_p_value'] is None
        assert row['reason'] == 'raw_p_value_unavailable'
        assert row['statistic'] is not None and row['n_effective'] > 0
        assert row['interval'] is not None

    # Changing all OOS factors must leave the training fit and candidate records unchanged.
    bars.loc[bars.timestamp_utc >= OOS_START, 'factor_value'] *= -1e6
    replay = evaluate_bars(bars, labeler=labeler, analysis_mode='time_series', split=SPLIT,
                           bootstrap=BOOTSTRAP, factor_id='momentum', direction=-1, standardization='zscore')
    assert replay.evaluation_v2.splits['train']['fit_identity'] == train['fit_identity']
    assert replay.evaluation_v2.trial_ledger == ledger


@pytest.mark.parametrize('constant', [False, True])
def test_v2_serializes_calibration_evidence_reason_and_usability(tmp_path, constant):
    bars = _bars(512)
    bars['timestamp_utc'] -= pd.Timedelta(hours=3)
    if constant:
        bars['factor_value'] = 1.
    result = evaluate_bars(
        bars, labeler=_labeler(), analysis_mode='time_series',
        bootstrap=dict(block_length=17, n_boot=11, seed=73, min_valid_replicates=6),
        calibration_scenario='ar1_null',
    )
    identity = ResultIdentity('a' * 64, {'bars': 'b' * 64}, result.calendar_identity['calendar_hash'],
                              'test-build', 'bar_evaluation.v1')
    paths = write_evaluation_artifacts(result, tmp_path, result_identity=identity)
    artifact = json.loads(paths['evaluation_v2'].read_text())
    assert artifact['schema_version'] == 'evaluation.v2'
    assert artifact['analysis_mode'] == 'time_series' and artifact['frequency'] == 'bar'
    assert artifact['return_unit'] == 'percent'
    summary = json.loads(paths['summary'].read_text())
    for key in ('data_identity', 'calendar_identity', 'result_identity'):
        assert artifact[key] == summary[key]
    assert artifact['result_identity']['identity_hash'] == identity.identity_hash()
    assert set(artifact['splits']) == {'full'}
    for row in artifact['splits']['full']['horizons']:
        assert row['bootstrap']['reason'] == ('constant_factor' if constant else None)
        usability = row['interval_usability']
        assert usability['reason'] == row['bootstrap']['reason']
        assert usability['calibration_status'] == 'calibrated_fail'
        assert usability['calibration_evidence']['observed_coverage'] == .914
        assert usability['descriptive_stats_usable'] is (not constant)
        assert usability['significance_interpretation_usable'] is False
        assert (row['ic']['spearman_ic'] is None) is constant
    if constant:
        assert {row['reason'] for row in artifact['trial_ledger']['candidates']} == {'constant_factor'}


@pytest.mark.parametrize('method', ['holm', 'bh'])
def test_hac_family_reports_raw_and_adjusted_p_values(method):
    bars = _bars(assets=('A', 'B', 'C', 'D'))
    result = evaluate_bars(bars, labeler=_labeler(), analysis_mode='cross_sectional', hac_lag=4,
                           trial_selection=dict(method=method, alpha=.1))
    advanced = result.evaluation_v2
    rows = advanced.splits['full']['horizons']
    assert all(row['hac']['lag'] == 4 for row in rows)
    raw = [erfc(abs(row['ic']['mean']) / sqrt(2 * row['hac']['variance'])) for row in rows]
    ledger = advanced.trial_ledger
    assert ledger['selection_complete'] and ledger['missing_p_value_candidates'] == []
    assert ledger['method'] == method and ledger['alpha'] == .1
    order = np.argsort(raw)
    low, high = np.sort(raw)
    expected = [min(1., 2 * low), min(1., max(2 * low, high))] if method == 'holm' else [
        min(2 * low, high), high,
    ]
    candidates = ledger['candidates']
    assert [row['p_value'] for row in candidates] == pytest.approx(raw)
    assert [candidates[i]['adjusted_p_value'] for i in order] == pytest.approx(expected)
    assert ledger['best'] is None
    assert all(not row['significant'] for row in candidates)
    assert [row['n_observations'] for row in candidates] == [row['ic']['n_valid'] for row in rows]
    assert all(row['n_dependence_adjusted'] is None for row in candidates)
    assert all(not row['interval_usability']['significance_interpretation_usable'] for row in rows)


def test_cross_section_split_counts_scheduled_exits_and_cross_section_scheme():
    bars = _bars(assets=('A', 'B', 'C', 'D'))
    bars = bars[~bars.timestamp_utc.isin(pd.date_range(START + pd.Timedelta(minutes=32), periods=3, freq='1min'))]
    result = evaluate_bars(bars, labeler=_labeler(), analysis_mode='cross_sectional',
                           split=SPLIT, hac_lag=4)
    train, oos = (result.evaluation_v2.splits[name] for name in ('train', 'oos'))
    assert train['quantile_scheme'] == oos['quantile_scheme'] == 'per_cross_section'
    assert train['fit_identity'] is oos['fit_identity'] is None
    assert train['split_purged_count'] == {1: 0, 3: 8}
    assert oos['split_purged_count'] == {1: 0, 3: 0}


@pytest.mark.parametrize('method', ['holm', 'bh'])
def test_shared_trial_family_keeps_every_factor_and_waits_for_raw_p_values(method):
    keys = [('alpha', 1), ('alpha', 3), ('beta', 1)]
    family = BarTrialFamily(keys, method=method)
    family.record(keys[0], Trial(.2, .031, 40))
    assert family.report()['pending_candidates'] == keys[1:]
    assert not family.report()['selection_complete']
    family.record(keys[1], Trial(.1, None, 25, (-.1, .3)), reason='test_unavailable')
    hac = bartlett_hac(np.linspace(-1., 1., 512) + .3, lag=0, calibration_scenario='independent_null')
    use = evaluate_interval_usability(hac, point_estimate=.3, horizon=1, cross_section_sizes=[32] * 512)
    family.record(keys[2], Trial(.3, .014, 512, (hac.lower, hac.upper), use))
    report = family.report()
    assert report['attempt_count'] == 3 and report['missing_p_value_candidates'] == [keys[1]]
    assert report['best'] is None and not report['selection_complete']
    assert [row['p_value'] for row in report['candidates']] == [.031, None, .014]
    assert [row['adjusted_p_value'] for row in report['candidates']] == [None] * 3
    assert [row['inference_adjusted_p_value'] for row in report['candidates']] == [None] * 3
    assert not any(row['significant'] for row in report['candidates'])
    assert report['candidates'][1]['reason'] == 'test_unavailable'
    assert report['candidates'][2]['interval_usability']['significance_interpretation_usable']


def test_cli_split_diagnostics_and_duckdb_keep_v1_compatible(tmp_path):
    bars_path = tmp_path / 'bars.csv'
    _bars().drop(index=[32, 33, 34]).to_csv(bars_path, index=False)
    config = base_config(bars_path)
    config['horizons'] = list(HORIZONS)
    baseline = run_cli(tmp_path, config, '--diagnostics-report', '--run-id', 'baseline')
    _, _, _, summary_before = read_outputs(baseline)
    db = tmp_path / 'results.duckdb'
    config.update(split=dict(train_end=TRAIN_END.isoformat(), oos_start=OOS_START.isoformat()),
                  bootstrap=BOOTSTRAP)
    payload = run_cli(tmp_path, config, '--diagnostics-report', '--results-db', str(db))
    _, ic, quantiles, summary = read_outputs(payload)
    artifacts, baseline_artifacts = _artifacts(payload), _artifacts(baseline)
    assert 'evaluation_v2' not in baseline_artifacts
    for kind in ('per_bar_labels', 'ic_by_horizon', 'quantile_stats'):
        assert artifacts[kind].read_bytes() == baseline_artifacts[kind].read_bytes()
    assert summary.keys() == summary_before.keys()
    for key in summary.keys() - {'config', 'result_identity'}:
        assert summary[key] == summary_before[key]
    artifact = json.loads(artifacts['evaluation_v2'].read_text())
    item = next(item for item in payload['artifacts'] if item['artifact_type'] == 'evaluation_v2')
    assert item['sha256'] == hashlib.sha256(artifacts['evaluation_v2'].read_bytes()).hexdigest()
    assert artifact['splits']['train']['split_purged_count'] == {'1': 0, '3': 2}
    diagnostic = artifact['diagnostics_report']
    assert payload['metrics']['diagnostics_report'] == diagnostic
    assert {row['split'] for row in diagnostic['horizons']} == {'train', 'oos'}
    csv = pd.read_csv(artifacts['diagnostics_report_csv'])
    assert set(csv['split']) == {'train', 'oos'}
    assert (csv.query("scope == 'overall' and split == 'train'").split_purged_count == [0, 2]).all()
    assert 'frozen training fit' in artifacts['diagnostics_report_markdown'].read_text()

    with duckdb.connect(str(db), read_only=True) as conn:
        assert {row[0] for row in conn.execute('SHOW TABLES').fetchall()} == {
            'bar_evaluation_runs', 'bar_evaluation_ic', 'bar_evaluation_quantiles', 'bar_evaluation_cross_sections',
        }
        for table, size in [('ic', len(ic)), ('quantiles', len(quantiles)), ('cross_sections', 0)]:
            assert conn.execute(f'SELECT count(*) FROM bar_evaluation_{table}').fetchone()[0] == size
        assert conn.execute('SELECT identity_hash FROM bar_evaluation_runs').fetchone()[0] == (
            artifact['result_identity']['identity_hash']
        )


@pytest.mark.parametrize('flag', ['--results-db', '--json-out'])
def test_v2_artifact_path_is_protected_before_worker(tmp_path, flag):
    bars_path = tmp_path / 'bars.csv'
    _bars().to_csv(bars_path, index=False)
    output = tmp_path / 'explicit'
    output.mkdir()
    collision = output / 'evaluation_v2.json'
    collision.write_bytes(b'original contents')
    result = run_cli(tmp_path, base_config(bars_path), '--output-dir', str(output),
                     '--evaluation-v2', flag, str(collision))
    assert result['exit_code'] == 2
    assert collision.read_bytes() == b'original contents'


def test_v2_factor_join_and_multi_instrument_family_withholds_unregistered_aggregation():
    bars = _bars(assets=('A', 'B'))
    factors = bars[['timestamp_utc', 'instrument_id', 'factor_value']].copy()
    factors = factors.drop(index=[10, 71])
    result = evaluate_bars(bars.drop(columns='factor_value'), factors, _labeler(),
                           analysis_mode='time_series', bootstrap=BOOTSTRAP)
    advanced = result.evaluation_v2
    assert {(row['instrument_id'], row['horizon_bars']) for row in advanced.splits['full']['horizons']} == {
        (asset, h) for asset in ('A', 'B') for h in HORIZONS
    }
    assert advanced.trial_ledger['missing_p_value_candidates'] == [('factor_value', h) for h in HORIZONS]
    assert {row['reason'] for row in advanced.trial_ledger['candidates']} == {'multiple_evaluation_streams'}
    assert result.exclusion_summary['exclusion_counts']['factor_missing'] == 4


def test_cli_explicit_v2_flag_with_default_configuration(tmp_path):
    bars_path = tmp_path / 'bars.csv'
    _bars().to_csv(bars_path, index=False)
    payload = run_cli(tmp_path, base_config(bars_path), '--evaluation-v2')
    assert payload['ok'], payload
    artifact = json.loads(_artifacts(payload)['evaluation_v2'].read_text())
    assert set(artifact['splits']) == {'full'}
    assert [row['horizon'] for row in artifact['trial_ledger']['candidates']] == [1, 2, 5]
    assert artifact['trial_ledger']['method'] == 'holm'


def test_evaluate_bars_records_only_current_factor_in_preregistered_family(monkeypatch):
    keys = [('alpha', 1), ('beta', 1)]
    family = BarTrialFamily(keys, method='holm')
    labeler = BarReturnLabeler(_labeler().calendar, horizons=[1])
    bars = _bars(assets=('A', 'B', 'C', 'D'))
    options = dict(labeler=labeler, analysis_mode='cross_sectional', trial_family=family)

    alpha = evaluate_bars(bars, factor_id='alpha', **options).evaluation_v2
    first = alpha.trial_ledger
    assert not first['selection_complete'] and first['best'] is None
    assert first['pending_candidates'] == [keys[1]]
    assert first['missing_p_value_candidates'] == []
    assert first['candidates'][1]['reason'] == 'not_evaluated'
    assert first['candidates'][1]['statistic'] is first['candidates'][1]['p_value'] is None
    assert first['candidates'][0]['p_value'] is not None
    assert all(row['adjusted_p_value'] is None for row in first['candidates'])

    beta_bars = bars.assign(factor_value=np.cos(np.arange(len(bars)) * .7))
    beta = evaluate_bars(beta_bars, factor_id='beta', **options).evaluation_v2
    second = beta.trial_ledger
    assert second['selection_complete'] and second['pending_candidates'] == []
    assert second['missing_p_value_candidates'] == []
    rows = [result.splits['full']['horizons'][0] for result in (alpha, beta)]
    means = [row['ic']['mean'] for row in rows]
    raw = [erfc(abs(mean) / sqrt(2 * row['hac']['variance'])) for mean, row in zip(means, rows, strict=True)]
    assert means[0] != pytest.approx(means[1])
    assert raw[0] != pytest.approx(raw[1])
    assert [row['statistic'] for row in second['candidates']] == pytest.approx(means)
    assert [row['p_value'] for row in second['candidates']] == pytest.approx(raw)
    order = np.argsort(raw)
    low, high = np.sort(raw)
    assert [second['candidates'][i]['adjusted_p_value'] for i in order] == pytest.approx(
        [min(1., 2 * low), min(1., max(2 * low, high))],
    )
    assert second['best'] is None
    assert first['pending_candidates'] == [keys[1]] and not first['selection_complete']

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail('The unregistered factor reached the evaluation engine')

    monkeypatch.setattr(EvaluationEngine, 'evaluate', unexpected_evaluation)
    with pytest.raises(ValueError, match='Unregistered candidate'):
        evaluate_bars(bars, factor_id='unregistered', **options)
    assert family.report() == second


def test_cross_market_split_aligns_different_opens_at_same_close(monkeypatch):
    from backtest_and_analysis.time_series.labels import _grid

    session = '2026-12-01'
    stamp = pd.Timestamp(f'{session}T15:00Z')
    close = pd.Timestamp(f'{session}T21:00Z')
    calendars, frames = {}, []
    for asset in ('NQ', 'NYSE'):
        calendar = SessionCalendar(asset, start_date=session, end_date=session)
        calendars[asset] = calendar
        starts, _, _ = _grid(calendar, '60min', stamp, stamp, 'ETH', None)
        frames.append(_bars(len(starts), (asset,)).assign(timestamp_utc=starts, frequency='60min'))

    captured = {}
    original = EvaluationEngine.evaluate

    def evaluate(engine, panel, **options):
        if options.get('split') is not None:
            captured.update(panel=panel, clock=options['signal_available_at'], expected=options['expected_grid'])
        return original(engine, panel, **options)

    monkeypatch.setattr(EvaluationEngine, 'evaluate', evaluate)
    result = evaluate_bars(pd.concat(frames, ignore_index=True), analysis_mode='cross_sectional',
                           labeler=BarReturnLabeler(calendars, horizons=[1, 2]),
                           split=SplitConfig(pd.Timestamp(f'{session}T20:30Z'), close))
    available = result.per_bar_labels.query('signal_available_at == @close and horizon_bars == 1')
    assert list(available.signal_timestamp.dt.strftime('%H:%M')) == ['20:00', '20:30']
    assert captured['clock'].is_unique and captured['clock'].is_monotonic_increasing
    row = captured['clock'].get_loc(close)
    assert captured['panel'].datetimes[row] == close
    assert list(captured['panel'].assets) == ['NQ', 'NYSE'] and captured['expected'][row].all()
    assert captured['panel'].factor[row] == pytest.approx(available.factor_value.to_numpy())
    train, oos = (result.evaluation_v2.splits[name] for name in ('train', 'oos'))
    assert train['horizons'][0]['grid_end_utc'] == pd.Timestamp(f'{session}T20:30Z')
    assert oos['horizons'][0]['grid_start_utc'] == close
    assert train['split_purged_count'] == {1: 0, 2: 1}
    assert oos['split_purged_count'] == {1: 0, 2: 0}


def test_time_series_split_reports_equal_weight_asset_ic_by_horizon_and_phase(tmp_path):
    bars = _bars(assets=('A', 'B'))
    i = bars.groupby('instrument_id').cumcount()
    bars = bars.assign(factor_value=i, open=100., high=102., low=98.,
                       close=100 + np.where(bars.instrument_id == 'A', i, -i) / 100)
    result = evaluate_bars(bars, labeler=_labeler(), analysis_mode='time_series', split=SPLIT,
                           bootstrap=BOOTSTRAP)
    advanced = result.evaluation_v2
    for name, sample_count in [('train', 32), ('oos', 31)]:
        split = advanced.splits[name]
        individual = [row for row in split['horizons'] if row['horizon_bars'] == 1]
        assert [row['ic']['sample_count'] for row in individual] == [sample_count] * 2
        assert [row['ic']['spearman_ic'] for row in individual] == pytest.approx([1., -1.])
        summaries = split['instrument_ic_summaries']
        assert [(row['frequency'], row['horizon_bars']) for row in summaries] == [('1min', h) for h in HORIZONS]
        for row in summaries:
            phases = row['non_overlapping_ic_summary']
            assert list(phases) == list(range(row['horizon_bars']))
            for summary in (row['ic_summary'], *phases.values()):
                assert summary == dict(n_instruments=2, n_valid=2, mean=0., median=0., reason=None)
    assert not advanced.trial_ledger['selection_complete']
    assert advanced.trial_ledger['missing_p_value_candidates'] == [('factor_value', h) for h in HORIZONS]
    identity = ResultIdentity('a' * 64, {'bars': 'b' * 64}, result.calendar_identity['calendar_hash'],
                              'test-build', 'bar_evaluation.v1')
    paths = write_evaluation_artifacts(result, tmp_path, result_identity=identity)
    persisted = json.loads(paths['evaluation_v2'].read_text())
    assert persisted['splits']['oos']['instrument_ic_summaries'][1]['non_overlapping_ic_summary']['2'] == (
        dict(n_instruments=2, n_valid=2, mean=0., median=0., reason=None)
    )


@pytest.mark.parametrize(('alpha_split', 'beta_split'), [
    (None, SPLIT),
    (SPLIT, None),
    (SPLIT, SplitConfig(TRAIN_END - pd.Timedelta(minutes=1), OOS_START)),
    (SPLIT, SplitConfig(TRAIN_END, OOS_START + pd.Timedelta(minutes=1))),
], ids=['full-train', 'train-full', 'different-train-end', 'different-oos-start'])
def test_shared_trial_family_rejects_a_different_selection_window(alpha_split, beta_split):
    bars = _bars(assets=('A', 'B', 'C', 'D'))
    labeler = BarReturnLabeler(_labeler().calendar, horizons=[1])
    family = BarTrialFamily([('alpha', 1), ('beta', 1)])
    options = dict(labeler=labeler, analysis_mode='cross_sectional', trial_family=family)
    evaluate_bars(bars, factor_id='alpha', split=alpha_split, **options)
    before = family.report()
    assert before['pending_candidates'] == [('beta', 1)]

    with pytest.raises(ValueError, match='selection window'):
        evaluate_bars(bars, factor_id='beta', split=beta_split, **options)
    assert family.report() == before

    accepted = evaluate_bars(bars, factor_id='beta', split=alpha_split, **options).evaluation_v2.trial_ledger
    assert accepted['split'] == ('full' if alpha_split is None else 'train')
    assert accepted['selection_complete'] and accepted['pending_candidates'] == []


def test_shared_train_selection_is_invariant_to_oos_factor_values_and_timezone_spelling():
    bars = _bars(assets=('A', 'B', 'C', 'D'))
    labeler = BarReturnLabeler(_labeler().calendar, horizons=[1])
    beta_bars = bars.assign(factor_value=np.cos(np.arange(len(bars)) * .2))
    equivalent_split = dict(
        train_end=TRAIN_END.tz_convert('America/New_York').isoformat(),
        oos_start=OOS_START.tz_convert('Asia/Shanghai').isoformat(),
    )
    changed_bars = bars.copy()
    oos = bars.timestamp_utc + pd.Timedelta(minutes=1) >= OOS_START
    next_bar = bars.groupby('instrument_id')[['close', 'open']].shift(-1)
    changed_bars.loc[oos, 'factor_value'] = (next_bar.close / next_bar.open - 1).loc[oos]

    def evaluate_family(alpha_bars):
        family = BarTrialFamily([('alpha', 1), ('beta', 1)])
        options = dict(labeler=labeler, analysis_mode='cross_sectional', trial_family=family)
        alpha = evaluate_bars(alpha_bars, factor_id='alpha', split=SPLIT, **options).evaluation_v2
        beta = evaluate_bars(beta_bars, factor_id='beta', split=equivalent_split, **options).evaluation_v2
        return alpha.splits['oos']['horizons'][0]['ic']['mean'], beta.trial_ledger

    before_oos, before = evaluate_family(bars)
    after_oos, after = evaluate_family(changed_bars)
    assert before_oos != pytest.approx(after_oos)
    assert before['split'] == 'train' and before['selection_complete']
    assert before['best'] is None
    assert [trial['n_effective'] for trial in before['candidates']] == [32, 32]
    assert after == before


def test_manual_trials_cannot_mix_with_evaluation_window_records():
    bars = _bars(assets=('A', 'B', 'C', 'D'))
    labeler = BarReturnLabeler(_labeler().calendar, horizons=[1])
    keys = [('alpha', 1), ('beta', 1)]
    manual = Trial(.1, .03, 32)
    family = BarTrialFamily(keys)
    family.record(keys[0], manual)
    options = dict(labeler=labeler, analysis_mode='cross_sectional', split=SPLIT)
    before = family.report()
    with pytest.raises(ValueError, match='selection window'):
        evaluate_bars(bars, factor_id='beta', trial_family=family, **options)
    assert family.report() == before

    family = BarTrialFamily(keys)
    evaluate_bars(bars, factor_id='alpha', trial_family=family, **options)
    before = family.report()
    with pytest.raises(ValueError, match='selection window'):
        family.record(keys[1], manual)
    assert family.report() == before


def test_incomplete_embedded_signal_cannot_change_training_fit_or_trials():
    bars = _bars().drop(index=12).assign(is_complete=True)
    bars.loc[5, ['is_complete', 'factor_value']] = [False, 0.]
    options = dict(labeler=_labeler(), analysis_mode='time_series', split=SPLIT,
                   bootstrap=BOOTSTRAP, n_quantiles=4, standardization='zscore')
    baseline = evaluate_bars(bars, **options)
    bars.loc[5, 'factor_value'] = 1e6
    replay = evaluate_bars(bars, **options)

    fit = baseline.evaluation_v2.splits['train']['fit_identity']
    assert replay.evaluation_v2.splits['train']['fit_identity'] == fit
    assert replay.evaluation_v2.splits['oos']['fit_identity'] == fit
    assert replay.evaluation_v2.trial_ledger == baseline.evaluation_v2.trial_ledger
    pd.testing.assert_frame_equal(replay.per_bar_labels, baseline.per_bar_labels)
    observed = fit['1min']['NQZ26']['instruments'][0]
    train_complete = (bars.timestamp_utc < TRAIN_END) & bars.is_complete
    training = bars.loc[train_complete, 'factor_value']
    assert observed['center'] == pytest.approx(training.mean())
    assert observed['training_edges'] == pytest.approx(np.quantile(training, np.linspace(0, 1, 5)))

    labels = replay.per_bar_labels.set_index(['signal_timestamp', 'horizon_bars'])
    incomplete = START + pd.Timedelta(minutes=5)
    assert labels.loc[incomplete, 'exclusion_reason'].tolist() == ['missing_bar', 'missing_bar']
    assert labels.loc[incomplete, 'entry_timestamp'].isna().all()
    before_gap = START + pd.Timedelta(minutes=10)
    assert labels.loc[(before_gap, 1), 'entry_timestamp'] == before_gap + pd.Timedelta(minutes=1)
    assert labels.loc[(before_gap, 3), 'exclusion_reason'] == 'missing_bar'


def test_incomplete_embedded_signal_cannot_change_cross_section_buckets():
    bars = _bars(assets=('A', 'B', 'C', 'D', 'E')).assign(is_complete=True)
    target = START + pd.Timedelta(minutes=5)
    section = bars.timestamp_utc.eq(target)
    bars.loc[section, 'factor_value'] = [-100., 1., 2., 3., 4.]
    incomplete = section & bars.instrument_id.eq('A')
    bars.loc[incomplete, 'is_complete'] = False
    options = dict(labeler=_labeler(), analysis_mode='cross_sectional', split=SPLIT, n_quantiles=2)
    baseline = evaluate_bars(bars, **options)
    bars.loc[incomplete, 'factor_value'] = 1e6
    replay = evaluate_bars(bars, **options)

    for before, after in zip(baseline.evaluation_v2.splits['train']['horizons'],
                             replay.evaluation_v2.splits['train']['horizons'], strict=True):
        assert after['buckets'] == before['buckets']
        assert after['direction_by_bucket'] == before['direction_by_bucket']
    assert replay.evaluation_v2.trial_ledger == baseline.evaluation_v2.trial_ledger


def test_external_signal_on_incomplete_bar_keeps_independent_availability():
    bars = _bars().assign(is_complete=True)
    bars.loc[5, 'is_complete'] = False
    factors = bars[['timestamp_utc', 'instrument_id', 'factor_value']]
    result = evaluate_bars(bars.drop(columns='factor_value'), factors, _labeler(),
                           analysis_mode='time_series', split=SPLIT, bootstrap=BOOTSTRAP)
    signal = bars.loc[5, 'timestamp_utc']
    labels = result.per_bar_labels.set_index(['signal_timestamp', 'horizon_bars'])
    assert labels.loc[signal, 'status'].tolist() == ['valid', 'valid']
    assert labels.loc[signal, 'entry_timestamp'].tolist() == [bars.loc[6, 'timestamp_utc']] * 2
    assert labels.loc[(bars.loc[4, 'timestamp_utc'], 1), 'exclusion_reason'] == 'missing_bar'


@pytest.mark.parametrize('column', ['signal_available_at', 'available_at_utc', 'published_at'])
def test_external_release_time_is_rejected_before_bar_evaluation(column):
    bars = _bars()
    factors = bars[['timestamp_utc', 'instrument_id', 'factor_value']].iloc[[0]].copy()
    factors[column] = START + pd.Timedelta(minutes=10)
    options = dict(labeler=_labeler(), analysis_mode='time_series', evaluation_v2=True,
                   bootstrap=BOOTSTRAP)
    family = BarTrialFamily(('factor_value', h) for h in HORIZONS)
    initial = family.report()
    for value in [1., 1e6]:
        factors['factor_value'] = value
        with pytest.raises(ValueError, match=rf'{column}.*align_released_features'):
            evaluate_bars(bars, factors, trial_family=family, **options)
        assert family.report() == initial



@pytest.mark.parametrize('mode', ['time_series', 'cross_sectional'])
def test_aligned_external_release_cannot_change_earlier_evaluation(mode):
    bars = _bars(assets=('A', 'B', 'C', 'D'))
    external = bars[['timestamp_utc', 'instrument_id', 'factor_value']]
    a = external[external.instrument_id.eq('A')]
    bar_ends = pd.DatetimeIndex(a.timestamp_utc + pd.Timedelta(minutes=1))
    # One observation is usable at the first close and its successor releases in OOS.
    release = OOS_START + pd.Timedelta(minutes=2)
    features = pd.DataFrame(dict(high_tf_bar_end=bar_ends[:2],
                                 signal_available_at=[bar_ends[0], release],
                                 factor_value=[1., 2.]))
    options = dict(labeler=_labeler(), analysis_mode=mode, split=SPLIT, n_quantiles=2)
    if mode == 'time_series':
        options.update(standardization='zscore', bootstrap=BOOTSTRAP)

    results = []
    for value in (2., 1e6):
        features.loc[1, 'factor_value'] = value
        aligned = align_released_features(features, bar_ends)
        a_factors = a.assign(factor_value=aligned.factor_value.to_numpy())
        factors = pd.concat([a_factors, external[~external.instrument_id.eq('A')]])
        results.append(evaluate_bars(bars.drop(columns='factor_value'), factors, **options))

    baseline, replay = results
    early = 'signal_available_at < @release'
    pd.testing.assert_frame_equal(baseline.per_bar_labels.query(early), replay.per_bar_labels.query(early))
    labels = replay.per_bar_labels.query("instrument_id == 'A' and horizon_bars == 1").set_index(
        'signal_available_at'
    )
    assert labels.loc[bar_ends[0], 'entry_timestamp'] == bar_ends[0]
    assert labels.loc[release, 'entry_timestamp'] == release
    assert labels.loc[release - pd.Timedelta(minutes=1), 'factor_value'] == 1.
    assert labels.loc[release, 'factor_value'] == 1e6
    train = baseline.evaluation_v2.splits['train']
    replay_train = replay.evaluation_v2.splits['train']
    assert replay_train['fit_identity'] == train['fit_identity']
    if mode == 'cross_sectional':
        assert [row['buckets'] for row in replay_train['horizons']] == [row['buckets'] for row in train['horizons']]
    else:
        fit = train['fit_identity']['1min']['A']['instruments'][0]
        assert fit['center'] == 1. and fit['training_edges'] == [1.]
    assert replay.evaluation_v2.trial_ledger == baseline.evaluation_v2.trial_ledger


def test_external_release_time_is_rejected_for_precomputed_labels_and_cli(tmp_path):
    bars = _bars()
    labeler = _labeler()
    labels = labeler.label(bars)
    factors = bars[['timestamp_utc', 'instrument_id', 'factor_value']].assign(
        signal_available_at=START + pd.Timedelta(minutes=10),
    )
    with pytest.raises(ValueError, match='signal_available_at.*align_released_features'):
        TimeSeriesEvaluator(analysis_mode='time_series').evaluate(labels, factors)

    bars_path, factors_path = tmp_path / 'bars.csv', tmp_path / 'factors.csv'
    bars.drop(columns='factor_value').to_csv(bars_path, index=False)
    factors.to_csv(factors_path, index=False)
    payload = run_cli(tmp_path, base_config(bars_path, factors_csv=str(factors_path)))
    assert payload['exit_code'] == 2
    assert 'align_released_features' in payload['error']['message']
    assert all(item['artifact_type'] != 'per_bar_labels' for item in payload['artifacts'])


@pytest.mark.parametrize(('method', 'audit', 'gated', 'claimed'), [
    ('holm', [.005, .008, .012, .02, .009], [.03, .08], [True, False]),
    ('bh', [.005, .005, .0075, .02, .005], [.03, .05], [True, True]),
])
def test_mixed_calibration_family_keeps_denominator_and_ignores_ineligible_evidence(method, audit, gated, claimed):
    keys = [(name, 1) for name in ('unknown', 'failed', 'best', 'other', 'wrong_width')]
    family = BarTrialFamily(keys, method=method)
    setups = [(None, 32), ('ar1_null', 32), ('independent_null', 32),
              ('independent_null', 32), ('independent_null', 31)]
    p_values = [.001, .002, .006, .02, .003]
    for key, (scenario, width), p in zip(keys, setups, p_values, strict=True):
        hac = bartlett_hac(np.linspace(-1., 1., 512) + .1, lag=0, calibration_scenario=scenario)
        use = evaluate_interval_usability(hac, point_estimate=.1, horizon=1, cross_section_sizes=[width] * 512)
        family.record(key, Trial(.1, p, 512, (hac.lower, hac.upper), use))

    report = family.report()
    rows = report['candidates']
    assert report['selection_complete'] and report['attempt_count'] == 5
    assert report['best'] == keys[2]
    assert [row['p_value'] for row in rows] == p_values
    assert [row['adjusted_p_value'] for row in rows] == pytest.approx(audit)
    assert [row['inference_adjusted_p_value'] for row in rows[2:4]] == pytest.approx(gated)
    assert [rows[i]['inference_adjusted_p_value'] for i in (0, 1, 4)] == [None] * 3
    assert [row['significant'] for row in rows] == [False, False, *claimed, False]
    assert [row['interval_usability']['calibration_status'] for row in rows] == [
        'uncalibrated', 'calibrated_fail', 'calibrated_pass', 'calibrated_pass', 'calibrated_pass',
    ]


@pytest.mark.parametrize(('width', 'horizon', 'missing', 'usable'), [
    (32, 1, False, True),
    (31, 1, False, False),
    (32, 2, False, False),
    (32, 1, True, False),
])
def test_dense_calibration_design_controls_candidate_selection(width, horizon, missing, usable):
    rng = np.random.default_rng(917)
    index = pd.MultiIndex.from_product([pd.date_range(START, periods=512, freq='min'), range(width)])
    factor = rng.standard_normal(len(index))
    returns = .3 * factor + rng.standard_normal(len(index))
    if missing:
        returns[:width] = np.nan
    panel = EvaluationPanel.from_long(pd.DataFrame({'factor': factor, horizon: returns}, index=index))
    scenario, lag = ('independent_null', 0) if horizon == 1 else ('overlap_null', 16)
    stats = EvaluationEngine('cross_sectional').evaluate(
        panel, calibration_scenario=scenario, hac_lag=lag,
    ).splits['full'].statistics
    rows = _cross_section_rows(stats, '1min')
    key = ('signal', horizon)
    family = BarTrialFamily([key])
    _record_trials(family, rows, keys=(key,), mode='cross_sectional', window=('full', None))

    report = family.report()
    row, = report['candidates']
    assert row['n_observations'] == row['n_effective'] == 512 - int(missing)
    assert row['n_dependence_adjusted'] is None
    assert row['statistic'] > 0.2 and row['adjusted_p_value'] == row['p_value'] < .05
    assert row['interval_usability']['calibration_status'] == 'calibrated_pass'
    assert row['interval_usability']['significance_interpretation_usable'] is usable
    assert row['significant'] is usable
    assert report['best'] == (key if usable else None)


def test_evaluate_bars_selects_significant_candidate_in_dense_train_with_exit_buffer():
    bars = _bars(525, assets=tuple(f'A{i:02}' for i in range(32)))
    bars['timestamp_utc'] -= pd.Timedelta(hours=3)
    start = bars.timestamp_utc.min()
    prior_signal = bars.groupby('instrument_id').factor_value.shift().fillna(0)
    bars['close'] += np.tanh(prior_signal)
    split = SplitConfig(start + pd.Timedelta(minutes=512), start + pd.Timedelta(minutes=516))
    labeler = BarReturnLabeler(_labeler().calendar, horizons=[1])

    result = evaluate_bars(
        bars, labeler=labeler, analysis_mode='cross_sectional', split=split,
        calibration_scenario='independent_null',
    ).evaluation_v2

    report = result.trial_ledger
    row, = report['candidates']
    assert report['split'] == 'train' and report['best'] == ('factor_value', 1)
    assert result.splits['train']['split_purged_count'] == {1: 0}
    assert row['n_observations'] == 512
    assert row['significant'] and row['interval_usability']['verified_horizon'] == 1
    assert row['inference_adjusted_p_value'] == row['adjusted_p_value'] == row['p_value'] < .05


@pytest.mark.parametrize('method,audit', [
    ('holm', [.003, .024, .04]),
    ('bh', [.003, .018, .04]),
])
@pytest.mark.parametrize('design', ['wrong_horizon', 'missing_width', 'missing_horizon'])
def test_public_ledgers_gate_candidates_without_matching_calibration_design(method, audit, design):
    keys = [('unsupported', 5 if design == 'wrong_horizon' else 1), ('one_step', 1), ('five_step', 5)]
    values = np.linspace(-1., 1., 512) + .1
    widths = [32] * 512
    one_step = bartlett_hac(values, lag=0, calibration_scenario='independent_null')
    five_step = bartlett_hac(values, lag=16, calibration_scenario='overlap_null')
    specs = [
        (one_step, .001, None if design == 'missing_horizon' else 1,
         None if design == 'missing_width' else widths),
        (one_step, .012, 1, widths),
        (five_step, .04, 5, widths),
    ]
    ledger = TrialLedger(keys)
    bar_family = BarTrialFamily(keys, method=method)
    for key, (interval, raw, horizon, sizes) in zip(keys, specs, strict=True):
        use = evaluate_interval_usability(interval, point_estimate=.1, horizon=horizon, cross_section_sizes=sizes)
        trial = Trial(.1, raw, 512, (interval.lower, interval.upper), use)
        ledger.record(key, trial)
        bar_family.record(key, trial)

    selection = ledger.select_best(method=method)
    report = bar_family.report()
    rows = report['candidates']
    assert selection.attempt_count == report['attempt_count'] == 3
    assert [row.trial.p_value for row in selection.trials] == [row['p_value'] for row in rows] == [.001, .012, .04]
    assert [row.adjusted_p_value for row in selection.trials] == pytest.approx(audit)
    assert [row['adjusted_p_value'] for row in rows] == pytest.approx(audit)
    assert keys[0] not in selection.inference_adjusted_p_values
    assert rows[0]['inference_adjusted_p_value'] is None
    assert selection.inference_adjusted_p_values[keys[1]] == pytest.approx(.036)
    assert keys[2] in selection.inference_adjusted_p_values
    assert [row.significant for row in selection.trials] == [row['significant'] for row in rows] == [False, True, False]
    assert selection.best == report['best'] == keys[1]
