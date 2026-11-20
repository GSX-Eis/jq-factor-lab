"""Bar v2, PIT split diagnostics and candidate-family evidence through the Bar entry points."""
import hashlib
import json
from math import erfc, sqrt
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.evaluation import EvaluationEngine
from backtest_and_analysis.evaluation.calibration import Trial, TrialLedger
from backtest_and_analysis.evaluation.split import SplitConfig
from backtest_and_analysis.time_series import (
    BarReturnLabeler, BarTrialFamily, ResultIdentity, evaluate_bars, write_evaluation_artifacts,
)
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
    assert ledger['best'] == ('factor_value', candidates[order[0]]['horizon'])
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


def test_shared_trial_family_keeps_every_factor_and_waits_for_raw_p_values():
    keys = [('alpha', 1), ('alpha', 3), ('beta', 1)]
    family = BarTrialFamily(keys, method='bh')
    family.record(keys[0], Trial(.2, .031, 40))
    assert family.report()['pending_candidates'] == keys[1:]
    assert not family.report()['selection_complete']
    family.record(keys[1], Trial(.1, None, 25, (-.1, .3)), reason='test_unavailable')
    family.record(keys[2], Trial(.3, .014, 40))
    report = family.report()
    assert report['attempt_count'] == 3 and report['missing_p_value_candidates'] == [keys[1]]
    assert report['best'] is None and not report['selection_complete']
    assert [row['p_value'] for row in report['candidates']] == [.031, None, .014]
    assert [row['adjusted_p_value'] for row in report['candidates']] == [None] * 3
    assert report['candidates'][1]['reason'] == 'test_unavailable'


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
    assert second['best'] == keys[order[0]]
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
    assert before['best'] is not None
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
