"""Synthetic CLI oracles: session boundaries, rank statistics and failed retries."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_cli(tmp_path, config=None, *extra, command='time-series-analysis'):
    path = tmp_path / 'config.json'
    if config is not None:
        path.write_text(json.dumps(config), encoding='utf-8')
    config_args = [] if command == 'read-bar-evaluations' else ['--config', str(path)]
    proc = subprocess.run([sys.executable, '-m', 'tools.agent_harness.cli', command,
        *config_args, '--output-root', str(tmp_path / 'harness'), *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=45)
    payload = json.loads(proc.stdout)
    assert proc.returncode == payload['exit_code'], proc.stderr
    return payload


def base_config(path, **overrides):
    return dict(bars_csv=str(path), analysis_mode='time_series', calendar='NQ',
                start_date='2026-12-01', end_date='2026-12-01', horizons=[1, 2, 5],
                n_quantiles=3, **overrides)


def nq_bars(tmp_path):
    times = pd.date_range('2026-12-01T20:56Z', periods=12, freq='1min')
    frame = pd.DataFrame(dict(timestamp_utc=times, instrument_id='NQZ26', frequency='1min',
        open=100. + np.arange(12), high=102. + np.arange(12), low=99. + np.arange(12),
        close=101. + np.arange(12), volume=1., source='SYNTHETIC', data_version='NERE711_v1',
        factor_value=np.arange(12) // 2))
    path = tmp_path / 'bars.csv'
    frame.to_csv(path, index=False)
    return path, frame


def read_outputs(payload):
    assert payload['ok'], payload
    items = {item['artifact_type']: Path(item['path']) for item in payload['artifacts']}
    for kind in ['per_bar_labels', 'ic_by_horizon', 'quantile_stats', 'evaluation_summary']:
        item = next(item for item in payload['artifacts'] if item['artifact_type'] == kind)
        assert items[kind].is_absolute()
        assert hashlib.sha256(items[kind].read_bytes()).hexdigest() == item['sha256']
    assert items['evaluation_summary'].name == 'summary.json'
    assert items['summary'] != items['evaluation_summary']
    assert json.loads(items['summary'].read_text()) == payload
    summary = json.loads(items['evaluation_summary'].read_text())
    assert summary['result_schema_version'] == 'bar_evaluation.v2'
    assert 'evaluation_v2' not in items
    assert not items['evaluation_summary'].with_name('evaluation_v2.json').exists()
    for key in ('result_schema_version', 'frequency', 'return_unit', 'splits',
                'interval_usability', 'trial_ledger', 'ic_by_horizon', 'quantile_stats', 'result_identity'):
        assert payload['metrics'][key] == summary[key]
    return (pd.read_csv(items['per_bar_labels']), pd.read_csv(items['ic_by_horizon']),
            pd.read_csv(items['quantile_stats']), summary)


def test_private_bar_results_db_config_still_uses_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.agent_harness import cli

    repo = tmp_path / 'repo'
    caller = tmp_path / 'caller'
    repo.mkdir()
    caller.mkdir()
    monkeypatch.setattr(cli, 'REPO_ROOT', repo)
    monkeypatch.chdir(caller)
    config_path = repo / 'bar.json'
    config_path.write_text(json.dumps({
        'output_dir': 'evaluation',
        'results_db': 'evaluation/summary.json',
    }))
    args = cli.build_parser().parse_args([
        'time-series-analysis', '--config', config_path.name,
        '--output-root', str(tmp_path / 'harness'),
    ])

    result = cli._handle_time_series_analysis(args)
    assert result.exit_code == 2
    assert result.error['code'] == 'bar_evaluation_store_precondition_failed'
    assert str(repo / 'evaluation/summary.json') in result.error['message']
    assert not (repo / 'evaluation').exists()
    assert not (caller / 'evaluation').exists()


@pytest.mark.parametrize('view', ['ETH', 'RTH'])
def test_nq_eth_rth_hand_computed_rows(tmp_path, view):
    path, bars = nq_bars(tmp_path)
    payload = run_cli(tmp_path, base_config(path, view=view))
    labels, ic, quantiles, summary = read_outputs(payload)
    candidates = 12 if view == 'ETH' else 4
    assert len(labels) == candidates * 3
    for row in labels.itertuples():
        i = int((pd.Timestamp(row.signal_timestamp) - bars.timestamp_utc.iloc[0]).total_seconds() / 60)
        h = row.horizon_bars
        reason = ('insufficient_history' if i + h >= 12 else
                  'session_boundary' if view == 'RTH' and i + h >= 4 else None)
        assert row.status == ('excluded' if reason else 'valid')
        if reason:
            assert row.exclusion_reason == reason
            assert np.isnan(row.return_pct)
        else:
            assert pd.isna(row.exclusion_reason)
            assert row.entry_price == 101 + i
            assert row.exit_price == 101 + i + h
            assert row.price_change_points == h
            assert row.return_pct == pytest.approx(100 * h / (101 + i))
            assert pd.Timestamp(row.entry_timestamp) == bars.timestamp_utc.iloc[i + 1]
            assert pd.Timestamp(row.exit_timestamp) == bars.timestamp_utc.iloc[i + h] + pd.Timedelta('1min')
    for h in [1, 2, 5]:
        detail = ic.query("scope == 'per_instrument' and horizon_bars == @h").iloc[0]
        valid_n = max(0, candidates - h)
        assert detail.total_count == candidates
        assert detail.valid_count == valid_n
        assert detail.excluded_count == candidates - valid_n
        reasons = json.loads(detail.exclusion_counts)
        assert reasons == dict(factor_missing=0, insufficient_history=h if view == 'ETH' else 0,
            session_boundary=min(h, 4) if view == 'RTH' else 0, missing_bar=0, invalid_price=0)
        values = pd.Series(np.arange(valid_n) // 2, dtype=float)
        returns = pd.Series(100 * h / (101 + np.arange(valid_n)))
        expected = values.corr(returns, method='spearman') if valid_n >= 3 else np.nan
        assert detail.spearman_ic == pytest.approx(expected, nan_ok=True)
        assert detail.non_overlapping_count == len(range(0, valid_n, h))
        sampled = values.iloc[::h].corr(returns.iloc[::h], method='spearman') if len(values.iloc[::h]) >= 3 else np.nan
        assert detail.non_overlapping_ic == pytest.approx(sampled, nan_ok=True)
        groups = quantiles.query('horizon_bars == @h')
        if valid_n >= 3:
            expected_groups = {11: [range(4), range(4, 8), range(8, 11)],
                10: [range(4), range(4, 6), range(6, 10)],
                7: [range(2), range(2, 4), range(4, 7)], 3: [range(2), range(2, 3)]}[valid_n]
            assert len(groups) == len(expected_groups)
            for group in groups.itertuples():
                r = returns.iloc[list(expected_groups[group.quantile - 1])]
                assert group.sample_count == len(r)
                assert group.mean_return_pct == pytest.approx(r.mean())
                assert group.median_return_pct == pytest.approx(r.median())
                assert group.positive_ratio == 1
        assert groups.descriptive_only.all()
        assert groups.quantile_scope.eq('full_sample_descriptive_only').all()
    assert summary['analysis_mode'] == 'time_series'
    identity = summary['result_identity']
    assert len(identity['config_hash']) == len(identity['calendar_hash']) == 64
    assert identity['input_file_hashes'][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity['calendar_hash'] == summary['calendar_identity']['calendar_hash']
    assert identity['code_version'] and identity['result_schema_version'] == summary['result_schema_version']
    assert payload['metrics']['exclusion_summary'] == summary['exclusion_summary']


@pytest.mark.parametrize('frequency,external', [('1min', False), ('5min', True)])
def test_cross_sectional_embedded_and_alias_factors(tmp_path, frequency, external):
    records = []
    for i in range(6):
        for j, timestamp in enumerate(pd.date_range('2026-12-01T14:30Z', periods=8, freq=frequency)):
            if i == 1 and j == 5:
                continue
            close = 100. + (6 - i if j == 2 else i + 1)
            if i == 0 and j == 5:
                close = 0.
            factor = 1. if j == 2 else float(i)
            if j == 3 and i >= 2:
                factor = np.nan
            records.append(dict(timestamp_utc=timestamp, instrument_id=f'A{i}', frequency=frequency,
                open=100., close=close, high=max(100., close) + 1, low=min(100., close) - 1,
                volume=1., source='SYNTHETIC', data_version='NERE711_v1', factor_value=factor))
    frame = pd.DataFrame(records)
    path = tmp_path / 'bars.csv'
    config = base_config(path)
    config.update(analysis_mode='cross_sectional', horizons=[1], n_quantiles=2)
    if external:
        factors = tmp_path / 'factors.csv'
        frame[['timestamp_utc', 'instrument_id', 'factor_value']].rename(columns={
            'timestamp_utc': 'timestamp', 'instrument_id': 'symbol'}).to_csv(factors, index=False)
        config['factors_csv'] = str(factors)
        frame = frame.drop(columns='factor_value')
    frame.to_csv(path, index=False)
    labels, ic, quantiles, summary = read_outputs(run_cli(tmp_path, config))
    assert len(labels) == 47
    for label in labels.itertuples():
        i = int(label.instrument_id[1:])
        j = int((pd.Timestamp(label.signal_timestamp) - pd.Timestamp('2026-12-01T14:30Z'))
                / pd.Timedelta(frequency))
        reason = ('factor_missing' if j == 3 and i >= 2 else
                  'insufficient_history' if j == 7 else
                  'missing_bar' if j == 4 and i == 1 else
                  'invalid_price' if j == 4 and i == 0 else None)
        assert label.status == ('excluded' if reason else 'valid')
        if reason:
            assert label.exclusion_reason == reason and np.isnan(label.return_pct)
        else:
            assert pd.isna(label.exclusion_reason)
            assert label.entry_price == 100
            assert label.return_pct == pytest.approx(6 - i if j == 1 else i + 1)
    assert labels.exclusion_reason.value_counts().to_dict() == dict(
        insufficient_history=6, factor_missing=4, missing_bar=1, invalid_price=1)
    sections = summary['cross_sections']
    assert [s['valid_instruments'] for s in sections] == [6, 6, 6, 2, 4, 5, 6, 0]
    assert [s['cs_status'] for s in sections] == ['ok', 'ok', 'constant_factor',
        'insufficient_cross_section', 'ok', 'ok', 'ok', 'insufficient_cross_section']
    row = ic.iloc[0]
    assert row.valid_count == 35 and row.excluded_count == 12
    assert row.valid_cross_sections == 5 and row.excluded_cross_sections == 3
    assert row.mean_ic == pytest.approx(.6)
    assert row.std_ic == pytest.approx(np.sqrt(.8))
    assert row.icir == pytest.approx(.6 / np.sqrt(.8))
    assert row.mean_valid_instruments == pytest.approx(35 / 8)
    assert not quantiles.descriptive_only.any()
    assert quantiles.cross_section_count.tolist() == [5, 5]
    assert quantiles.total_sample_count.tolist() == [14, 13]
    # The five computable sections have lower/upper means (2,5), (5,2),
    # (3.5,5.5), (8/3,5.5), (2,5), each expressed as percentage returns.
    for group, means in zip(quantiles.itertuples(), [[2, 5, 3.5, 8/3, 2], [5, 2, 5.5, 5.5, 5]]):
        assert group.mean_return_pct == pytest.approx(np.mean(means))
        assert group.median_return_pct == pytest.approx(np.median(means))
        assert group.positive_ratio == 1
    assert summary['analysis_mode'] == 'cross_sectional'
    assert len(summary['result_identity']['input_file_hashes']) == (2 if external else 1)


@pytest.mark.parametrize('failure,code', [
    ('missing_mode', 'missing_analysis_mode'), ('invalid_mode', 'invalid_analysis_mode'),
    ('missing_config', 'time_series_precondition_failed'), ('missing_csv', 'time_series_precondition_failed'),
    ('duplicate', 'time_series_precondition_failed'), ('non_utc', 'time_series_precondition_failed'),
    ('calendar_bounds', 'time_series_precondition_failed'),
])
def test_preconditions_remove_stale_completion_marker(tmp_path, failure, code):
    path, frame = nq_bars(tmp_path)
    config = base_config(path)
    if failure == 'missing_mode':
        del config['analysis_mode']
    elif failure == 'invalid_mode':
        config['analysis_mode'] = 'auto'
    elif failure == 'missing_config':
        config = None
    elif failure == 'missing_csv':
        config['bars_csv'] = str(tmp_path / 'absent.csv')
    elif failure == 'duplicate':
        pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    elif failure == 'non_utc':
        frame['timestamp_utc'] = frame.timestamp_utc.dt.tz_localize(None)
        frame.to_csv(path, index=False)
    else:
        config['start_date'] = config['end_date'] = '2026-12-02'
    destination = tmp_path / 'evaluation'
    destination.mkdir()
    (destination / 'summary.json').write_text('{"old_success": true}')
    payload = run_cli(tmp_path, config, '--output-dir', str(destination))
    assert payload['exit_code'] == 2
    assert payload['error']['code'] == code
    assert payload['error']['message'] and payload['error']['details']
    assert not (destination / 'summary.json').exists()
    assert all(item['artifact_type'] == 'summary' for item in payload['artifacts'])


@pytest.mark.parametrize('nested', [False, True])
def test_single_factor_dispatch_and_overrides(tmp_path, nested):
    path, _ = nq_bars(tmp_path)
    config = base_config(path)
    if nested:
        config = {'time_series_analysis': config}
    payload = run_cli(tmp_path, config, '--analysis-mode', 'cross_sectional', '--horizons', '1',
        '--view', 'RTH', '--n-quantiles', '2', '--entry-offset-bars', '2', '--calendar', 'NQ',
        command='single-factor-analysis')
    labels, _, _, summary = read_outputs(payload)
    assert payload['command']['name'] == 'single-factor-analysis'
    assert summary['analysis_mode'] == 'cross_sectional'
    assert summary['calendar_identity']['view'] == 'RTH'
    assert labels.entry_offset_bars.eq(2).all() and labels.horizon_bars.eq(1).all()


@pytest.mark.parametrize('command', ['time-series-analysis', 'single-factor-analysis'])
@pytest.mark.parametrize('nested', [False, True])
def test_bar_cli_rejects_retired_inference_config(tmp_path, command, nested):
    path, _ = nq_bars(tmp_path)
    config = base_config(path, evaluation_v2=True)
    if nested:
        config = {'time_series_analysis': config}
    destination = tmp_path / 'evaluation'
    payload = run_cli(tmp_path, config, '--output-dir', str(destination), command=command)
    assert payload['exit_code'] == 2
    assert payload['error']['code'] == 'time_series_precondition_failed'
    assert 'evaluation_v2' in payload['error']['message']
    assert '"trial_selection": {"method": "holm"}' in payload['error']['message']
    assert not (destination / 'summary.json').exists()


def test_bar_cli_rejects_s1_output_directory_and_succeeds_in_fresh_directory(tmp_path):
    path, _ = nq_bars(tmp_path)
    legacy = tmp_path / 's1'
    legacy.mkdir()
    old_identity = {'result_schema_version': 'bar_evaluation.v1', 'identity_hash': 'a' * 64}
    (legacy / 'summary.json').write_text(json.dumps(dict(
        result_schema_version='bar_evaluation.v1', result_identity=old_identity,
    )))
    (legacy / 'evaluation_v2.json').write_text(json.dumps(dict(
        schema_version='evaluation.v2', result_identity=old_identity, splits={'full': {}},
    )))
    (legacy / 'ic_by_horizon.csv').write_text('horizon_bars,spearman_ic\n1,1.0\n')
    original = {artifact.name: artifact.read_bytes() for artifact in legacy.iterdir()}
    config = base_config(path, trial_selection={'method': 'holm'})

    rejected = run_cli(tmp_path, config, '--output-dir', str(legacy))
    assert rejected['exit_code'] == 2
    assert rejected['error']['code'] == 'time_series_precondition_failed'
    assert 'evaluation_v2.json' in rejected['error']['message']
    assert 'new output directory' in rejected['error']['message']
    assert {artifact.name: artifact.read_bytes() for artifact in legacy.iterdir()} == original

    results_db = tmp_path / 'full.duckdb'
    fresh = run_cli(tmp_path, config, '--output-dir', str(tmp_path / 's2'), '--results-db', str(results_db))
    _, _, _, summary = read_outputs(fresh)
    assert set(summary['splits']) == {'full'}
    assert summary['trial_ledger']['method'] == 'holm'
    read = run_cli(tmp_path, None, '--results-db', str(results_db), '--split', 'full',
                   '--has-trial-ledger', command='read-bar-evaluations')
    assert read['ok'], read
    assert read['metrics']['runs'][0]['trial_ledger_json'] == summary['trial_ledger']
    assert read['metrics']['split_metrics'][0]['metrics_json'] == summary['splits']['full']['horizons'][0]


def test_read_bar_evaluations_pages_matching_factors(tmp_path):
    path, _ = nq_bars(tmp_path)
    db = tmp_path / 'pages.duckdb'
    common = ('--results-db', str(db))
    first = run_cli(tmp_path, base_config(path, factor_id='selected'), *common,
                    '--output-dir', str(tmp_path / 'first'))
    second = run_cli(tmp_path, base_config(path, factor_id='selected', view='RTH'), *common,
                     '--output-dir', str(tmp_path / 'second'))
    other = run_cli(tmp_path, base_config(path, factor_id='other'), *common,
                    '--output-dir', str(tmp_path / 'other'))
    assert all(run['ok'] for run in (first, second, other)), (first, second, other)

    for offset, run in enumerate((second, first)):
        page = run_cli(tmp_path, None, *common, '--factor-name', 'selected', '--limit', '1',
                       '--offset', str(offset), command='read-bar-evaluations')
        assert page['ok'] and page['metrics']['run_count'] == 1, page
        record, = page['metrics']['runs']
        assert record['factor_id'] == 'selected'
        assert record['identity_hash'] == run['metrics']['result_identity']['identity_hash']

    invalid = run_cli(tmp_path, None, *common, '--offset', '-1', command='read-bar-evaluations')
    assert invalid['exit_code'] == 2
    assert invalid['error']['message'] == 'offset must be a non-negative integer'


@pytest.mark.parametrize('flag', ['--json-out', '--results-db'])
def test_bar_cli_reserves_retired_sidecar_path(tmp_path, flag):
    path, _ = nq_bars(tmp_path)
    destination = tmp_path / 'evaluation'
    sidecar = destination / 'evaluation_v2.json'
    payload = run_cli(tmp_path, base_config(path), '--output-dir', str(destination), flag, str(sidecar))
    assert payload['exit_code'] == 2
    assert not sidecar.exists()
    assert not (destination / 'summary.json').exists()


def test_daily_dispatch_stays_on_original_runner(tmp_path):
    payload = run_cli(tmp_path, {'analysis_mode': 'cross_sectional'}, '--dry-run',
                      command='single-factor-analysis')
    assert any('run_analysis_main.py' in arg for arg in payload['command']['argv'])


@pytest.mark.parametrize('external', [False, True])
def test_calendar_mapping_and_standard_factor_keys(tmp_path, external):
    path, frame = nq_bars(tmp_path)
    config = base_config(path)
    config.pop('calendar')
    config['calendar_by_instrument'] = {'NQZ26': 'NQ'}
    destination = tmp_path / 'custom'
    if external:
        factors = tmp_path / 'factors.csv'
        frame[['timestamp_utc', 'instrument_id', 'factor_value']].to_csv(factors, index=False)
        frame.drop(columns='factor_value').to_csv(path, index=False)
        config['factors_csv'] = str(factors)
    payload = run_cli(tmp_path, config, '--output-dir', str(destination))
    _, _, _, summary = read_outputs(payload)
    assert summary['calendar_identity']['calendar_id'] == 'per_instrument'
    assert summary['calendar_identity']['calendars']['NQZ26']['calendar_hash']
    assert {p.name for p in destination.iterdir()} == {
        'per_bar_labels.csv', 'ic_by_horizon.csv', 'quantile_stats.csv', 'summary.json'}


@pytest.mark.parametrize('failure', ['malformed_config', 'invalid_horizons', 'duplicate_factors',
    'missing_factors', 'missing_calendar_mapping', 'non_utc_factors'])
def test_additional_config_and_factor_failures(tmp_path, failure):
    path, frame = nq_bars(tmp_path)
    config = base_config(path)
    extra = []
    if failure == 'malformed_config':
        config_path = tmp_path / 'config.json'
        config_path.write_text('[1, 2]')
        config = None
    elif failure == 'invalid_horizons':
        extra = ['--horizons', '1,nope']
    elif failure == 'missing_calendar_mapping':
        config['calendar_by_instrument'] = {'another': 'NQ'}
    else:
        factors = tmp_path / 'factors.csv'
        config['factors_csv'] = str(factors)
        values = frame[['timestamp_utc', 'instrument_id', 'factor_value']]
        if failure == 'duplicate_factors':
            pd.concat([values, values.iloc[:1]]).to_csv(factors, index=False)
        elif failure == 'non_utc_factors':
            values = values.assign(timestamp_utc=values.timestamp_utc.dt.tz_convert('America/New_York'))
            values.to_csv(factors, index=False)
    destination = tmp_path / 'evaluation'
    destination.mkdir()
    (destination / 'summary.json').write_text('{"old_success":true}')
    payload = run_cli(tmp_path, config, '--output-dir', str(destination), *extra)
    assert payload['exit_code'] == 2 and not payload['ok']
    assert not (destination / 'summary.json').exists()


@pytest.mark.parametrize('failure', ['timeout', 'broken_worker'])
def test_execution_failure_removes_completion_marker(tmp_path, monkeypatch, failure):
    from tools.agent_harness import cli
    path, _ = nq_bars(tmp_path)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(base_config(path)))
    destination = tmp_path / 'evaluation'
    destination.mkdir()
    (destination / 'summary.json').write_text('{"old_success":true}')
    args = cli.build_parser().parse_args(['time-series-analysis', '--config', str(config_path),
        '--output-root', str(tmp_path / 'harness'), '--output-dir', str(destination), '--timeout-sec', '1'])
    real_run = subprocess.run

    def fake_worker(command, **kwargs):
        if 'tools.agent_harness.time_series_analysis' not in command:
            return real_run(command, **kwargs)
        assert kwargs['timeout'] == 1
        (destination / 'summary.json').write_text('{"partial_success":true}')
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, 1)
        return subprocess.CompletedProcess(command, 1, '', 'worker crashed')

    monkeypatch.setattr(cli.subprocess, 'run', fake_worker)
    result = cli._handle_time_series_analysis(args)
    assert result.exit_code == 3
    assert result.error['code'] == 'time_series_execution_failed'
    if failure == 'broken_worker':
        assert 'worker crashed' in result.error['message']
    assert not (destination / 'summary.json').exists()


def test_harness_output_directory_cannot_overwrite_evaluation_summary(tmp_path, monkeypatch):
    from tools.agent_harness import cli
    path, _ = nq_bars(tmp_path)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(base_config(path)))
    output = tmp_path / 'attempt'
    monkeypatch.setattr(cli, '_output_dir', lambda *args: output)
    args = cli.build_parser().parse_args(['time-series-analysis', '--config', str(config_path),
        '--output-dir', str(output)])
    labels, _, _, summary = read_outputs(cli._handle_time_series_analysis(args).to_dict())
    assert len(labels) == 36
    assert summary['analysis_mode'] == 'time_series'
    assert (output / 'evaluation' / 'summary.json').is_file()


def test_json_mirror_cannot_overwrite_evaluation_marker(tmp_path):
    path, _ = nq_bars(tmp_path)
    destination = tmp_path / 'evaluation'
    payload = run_cli(tmp_path, base_config(path), '--output-dir', str(destination),
                      '--json-out', str(destination / 'summary.json'))
    assert payload['exit_code'] == 2
    assert not (destination / 'summary.json').exists()


@pytest.mark.parametrize('source', ['argument', 'config'])
@pytest.mark.parametrize('shape', ['file', 'summary_directory'])
def test_invalid_evaluation_destination_returns_json(tmp_path, source, shape):
    path, _ = nq_bars(tmp_path)
    destination = tmp_path / 'evaluation'
    if shape == 'file':
        destination.write_text('preserve this file')
    else:
        (destination / 'summary.json').mkdir(parents=True)
        (destination / 'summary.json' / 'keep.txt').write_text('preserve this directory')
    config = base_config(path)
    extra = ['--output-dir', str(destination)] if source == 'argument' else []
    if source == 'config':
        config['output_dir'] = str(destination)
    mirror = tmp_path / 'mirror.json'
    payload = run_cli(tmp_path, config, *extra, '--json-out', str(mirror))
    assert payload['exit_code'] == 2
    assert payload['error']['code'] == 'time_series_precondition_failed'
    assert payload['error']['message'] and payload['error']['details']
    assert json.loads(mirror.read_text()) == payload
    assert [a['artifact_type'] for a in payload['artifacts']] == ['summary']
    if shape == 'file':
        assert destination.read_text() == 'preserve this file'
    else:
        assert (destination / 'summary.json' / 'keep.txt').read_text() == 'preserve this directory'


@pytest.mark.parametrize('source', ['argument', 'config'])
def test_run_level_directory_cannot_overwrite_evaluation_summary(tmp_path, source):
    path, _ = nq_bars(tmp_path)
    run_dir = tmp_path / 'harness' / 'time-series-analysis' / 'myrun'
    config = base_config(path)
    extra = ['--output-dir', str(run_dir)] if source == 'argument' else []
    if source == 'config':
        config['output_dir'] = str(run_dir)
    payload = run_cli(tmp_path, config, '--run-id', 'myrun', *extra)
    _, _, _, summary = read_outputs(payload)
    assert summary['analysis_mode'] == 'time_series'
    assert json.loads((run_dir / 'summary.json').read_text()) == payload
    evaluation_path = Path(next(a['path'] for a in payload['artifacts']
                                if a['artifact_type'] == 'evaluation_summary'))
    assert evaluation_path == Path(payload['summary']['output_dir']) / 'evaluation' / 'summary.json'


@pytest.mark.parametrize('config,extra,code', [
    ({'analysis_mode': 'time_series', 'calendar': 'NQ'}, [], 'time_series_precondition_failed'),
    ({'time_series_analysis': {'analysis_mode': 'time_series'}}, [], 'time_series_precondition_failed'),
    ({'time_series_analysis': {'analysis_mode': 'cross_sectional'}}, [], 'time_series_precondition_failed'),
    ({'time_series_analysis': {'horizons': [1]}}, [], 'missing_analysis_mode'),
    ({'time_series_analysis': {}}, [], 'missing_analysis_mode'),
    ({'time_series_analysis': None}, [], 'time_series_precondition_failed'),
    ({}, ['--analysis-mode', 'time_series'], 'time_series_precondition_failed'),
    ({'analysis_mode': 'cross_sectional'}, ['--analysis-mode', 'time_series'], 'time_series_precondition_failed'),
])
def test_explicit_bar_intent_dispatches_before_missing_csv(tmp_path, config, extra, code):
    planned = run_cli(tmp_path, config, '--dry-run', *extra, command='single-factor-analysis')
    assert 'tools.agent_harness.time_series_analysis' in planned['command']['argv']
    # Check the dry-run route first so a regression cannot execute the daily producer.
    payload = run_cli(tmp_path, config, *extra, command='single-factor-analysis')
    assert payload['exit_code'] == 2
    assert payload['error']['code'] == code
    if code == 'time_series_precondition_failed' and config.get('time_series_analysis', {}) is not None:
        assert 'bars_csv' in payload['error']['message']


@pytest.mark.parametrize('mapping', [False, True])
def test_unknown_calendar_is_a_precondition_failure(tmp_path, mapping):
    path, _ = nq_bars(tmp_path)
    config = base_config(path)
    if mapping:
        config['calendar_by_instrument'] = {'NQZ26': 'UNKNOWN_CAL'}
    else:
        config['calendar'] = 'UNKNOWN_CAL'
    payload = run_cli(tmp_path, config)
    assert payload['exit_code'] == 2
    assert payload['error']['code'] == 'time_series_precondition_failed'
    assert 'UNKNOWN_CAL' in payload['error']['message']
    assert not (Path(payload['summary']['output_dir']) / 'evaluation' / 'summary.json').exists()


def test_config_output_directory_is_relative_to_repository(tmp_path, monkeypatch):
    from tools.agent_harness import cli
    path, _ = nq_bars(tmp_path)
    destination = tmp_path / 'evaluation'
    config = base_config(path)
    config['output_dir'] = os.path.relpath(destination, ROOT)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    other_cwd = tmp_path / 'caller'
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    args = cli.build_parser().parse_args(['time-series-analysis', '--config', str(config_path),
        '--output-root', str(tmp_path / 'harness')])
    payload = cli._handle_time_series_analysis(args).to_dict()
    read_outputs(payload)
    assert (destination / 'summary.json').is_file()


@pytest.mark.parametrize('nested_calendar', ['calendar', 'calendar_by_instrument'])
def test_nested_calendar_choice_overrides_top_level_alternative(tmp_path, nested_calendar):
    path, _ = nq_bars(tmp_path)
    config = base_config(path)
    if nested_calendar == 'calendar':
        config.pop('calendar')
        config['calendar_by_instrument'] = {'unrelated': 'NQ'}
        config['time_series_analysis'] = {'calendar': 'NQ'}
    else:
        config['calendar'] = 'UNKNOWN_CAL'
        config['time_series_analysis'] = {'calendar_by_instrument': {'NQZ26': 'NQ'}}
    _, _, _, summary = read_outputs(run_cli(tmp_path, config))
    alternative = 'calendar_by_instrument' if nested_calendar == 'calendar' else 'calendar'
    assert alternative not in summary['config']
    assert summary['calendar_identity']['calendar_id'] == (
        'CME Globex Equity' if nested_calendar == 'calendar' else 'per_instrument')


def _build_private_slot_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    from dataclasses import replace
    from utils.session_calendar import SessionCalendar
    from utils.slots import ExecutionProfile, ProviderConfig, load_slot_config
    from utils.time_axis import BarSchedule

    slot_dir = tmp_path / 'slot_fixture'
    slot_dir.mkdir(parents=True, exist_ok=True)
    example = load_slot_config(ROOT / 'examples' / 'declarative_crypto' / 'slot.yaml')
    days = ('2026-01-05', '2026-01-06')
    cal = SessionCalendar('crypto_utc', days[0], days[1], custom_sessions=example.calendar.options['custom_sessions'])
    sched = BarSchedule(cal, '60min', view='ETH', bucket_strategy='segment_anchored')
    rows = []
    for inst_idx, inst in enumerate(('BTC-USD', 'ETH-USD')):
        step = 0
        for day in cal.sessions():
            for ts in sched.for_session(day.strftime('%Y-%m-%d'))['start_utc']:
                step += 1
                base = 100.0 + 15.0 * inst_idx + 0.4 * step
                rows.append({
                    'time': ts.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'instrument_id': inst,
                    'open': base,
                    'high': base + 1.0,
                    'low': base - 1.0,
                    'close': base + 0.2,
                    'volume': 50.0 + step,
                })
    pd.DataFrame(rows).to_csv(slot_dir / 'bars.csv', index=False)
    universe_opts = dict(example.universe.options)
    universe_opts['instruments'] = {
        inst: {'asset_class': 'crypto', 'currency': 'USD', 'lot_size': 1, 'contract_multiplier': 1.0}
        for inst in ('BTC-USD', 'ETH-USD')
    }
    universe_opts['universes'] = {'spot': ['BTC-USD', 'ETH-USD']}
    bar_opts = {
        **example.bar_feed.options,
        'frequency': '60min',
        'timestamp_anchor': 'bar_start',
        'column_mapping': {
            'time': 'timestamp', 'instrument_id': 'instrument_id',
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume',
        },
        'paths': ['bars.csv'],
    }
    slot_cfg = replace(
        example,
        calendar=replace(example.calendar, options={**example.calendar.options, 'start': days[0], 'end': days[1]}),
        bar_feed=ProviderConfig('csv_files', bar_opts),
        universe=ProviderConfig('static', universe_opts),
        aux_features=ProviderConfig('none', {}),
        execution_profile=ExecutionProfile('open', 'close', 1, -1, 'decimal', 'endpoints_only', 'cross_session'),
    )
    slot_path = slot_dir / 'slot.json'
    slot_path.write_text(json.dumps(slot_cfg.as_dict(), indent=2), encoding='utf-8')
    factor_path = slot_dir / 'probe_factor.py'
    factor_path.write_text(
        'import pandas as pd\n'
        'from utils.factor_base import FactorBase\n\n'
        'class PrivateProbeFactor(FactorBase):\n'
        '    CAUSAL_VECTORIZED = True\n\n'
        '    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:\n'
        '        vals = self.session_rolling(data, "close", 3, min_periods=1, within_session=False)\n'
        '        return data[["timestamp_utc", "instrument_id"]].assign(factor_value=vals)\n',
        encoding='utf-8',
    )
    return slot_dir, slot_path, factor_path


@pytest.mark.parametrize('variant', ['cli_file_slot', 'inline_slot', 'inline_slot_no_output_dir'])
def test_private_slot_dry_run_and_live_execution_parity(tmp_path: Path, variant: str) -> None:
    slot_dir, slot_path, factor_path = _build_private_slot_fixture(tmp_path)
    config: dict = {
        'analysis_mode': 'time_series',
        'start_date': '2026-01-05',
        'end_date': '2026-01-06',
        'frequency': '60min',
        'horizons': [1, 2],
        'n_quantiles': 2,
        'min_ic_samples': 2,
        'min_cross_section_size': 2,
    }
    eval_dir = tmp_path / 'slot_eval'
    results_db = tmp_path / 'slot_results.duckdb'
    json_out = tmp_path / 'dry_plan.json'
    if variant == 'cli_file_slot':
        extra = [
            '--slot', str(slot_path),
            '--factor', str(factor_path),
            '--output-dir', str(eval_dir),
            '--json-out', str(json_out),
            '--results-db', str(results_db),
        ]
        expect_outputs = True
    elif variant == 'inline_slot':
        config.update(
            slot=json.loads(slot_path.read_text(encoding='utf-8')),
            slot_base_dir=str(slot_dir),
            factor_file=str(factor_path),
        )
        extra = [
            '--output-dir', str(eval_dir),
            '--json-out', str(json_out),
            '--results-db', str(results_db),
        ]
        expect_outputs = True
    else:
        config.update(
            slot=json.loads(slot_path.read_text(encoding='utf-8')),
            slot_base_dir=str(slot_dir),
            factor_file=str(factor_path),
        )
        extra = []
        expect_outputs = False

    dry_payload = run_cli(tmp_path, config, '--dry-run', *extra)
    assert dry_payload['ok'] is True
    assert dry_payload['exit_code'] == 0
    assert dry_payload['command']['mode'] == 'dry_run'
    assert dry_payload['summary']['reason'] == 'dry_run'
    assert not eval_dir.exists()
    assert not results_db.exists()
    if expect_outputs:
        assert json.loads(json_out.read_text(encoding='utf-8')) == dry_payload
        assert dry_payload['metrics']['planned_results_db'] == str(results_db.resolve())

    live_payload = run_cli(tmp_path, config, *extra)
    _, _, _, summary = read_outputs(live_payload)
    assert summary['analysis_mode'] == 'time_series'
    if expect_outputs:
        assert live_payload['metrics']['results_db']['status'] == 'inserted'
        assert (eval_dir / 'summary.json').is_file()
        assert results_db.is_file()


def test_private_market_db_slot_dry_run_and_live_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duckdb
    from utils.market_db import init_market_db, load_market_db_profile_yaml
    from utils.market_db.identity import market_db_identity
    from utils.market_db.research import open_research_slot
    from utils.market_factor_store import (
        MarketFactorValueStore, build_identity_record, compute_factor_identity_hash,
    )
    from utils.slots import ExecutionProfile, ProviderConfig, SlotConfig

    profile = load_market_db_profile_yaml(ROOT / 'config/market_databases/us_equity.yaml')
    db_path = tmp_path / 'market.duckdb'
    init_market_db(db_path, profile)
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TABLE bars AS
            SELECT DATE '2026-01-05' + i::INT + INTERVAL '16 hours' AS timestamp,
                   'AAPL.XNYS' AS instrument_id, 10.0+i AS open, 11.0+i AS high,
                   9.0+i AS low, 10.5+i AS close, 100 AS volume
            FROM range(5) data(i)
        """)
    monkeypatch.setenv('QUANT_DATA_BINDINGS', json.dumps({'market_dbs': {profile.db_id: str(db_path)}}))
    slot_cfg = SlotConfig(
        1, 'ny_private_fixture',
        ProviderConfig('pmc', {'calendar_id': 'XNYS', 'start': '2026-01-05', 'end': '2026-01-09'}),
        ProviderConfig('duckdb_table', {
            'table': 'bars', 'frequency': '1d', 'timestamp_anchor': 'bar_end',
            'source_timezone': 'America/New_York', 'volume_mode': 'interval', 'data_version': 'fixture',
            'column_mapping': {name: name for name in (
                'instrument_id', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
            )},
        }),
        ProviderConfig('static', {'instruments': {'AAPL.XNYS': {'asset_class': 'equity', 'currency': 'USD'}}}),
        ProviderConfig('none'), ProviderConfig('open'),
        ExecutionProfile('close', 'close', 0, 0, 'decimal', 'endpoints_only', 'cross_session'),
    )
    slot_path = tmp_path / 'slot.json'
    slot_path.write_text(json.dumps(slot_cfg.as_dict()), encoding='utf-8')
    store_root = tmp_path / 'factors'
    with open_research_slot(profile.db_id, slot_cfg, tmp_path) as (slot, _):
        identity_hash, payload = compute_factor_identity_hash(
            factor_id='alpha', factor_class='tests.PrivateMarketFactor', source_sha256='a' * 64, params={},
            frequency='1d', session_view='ETH', analysis_mode='cross_sectional', bucket_strategy='session',
            calendar_hash=slot.calendar.calendar_hash, code_version='test', slot_id=slot.slot_id,
            slot_config_hash=slot.slot_config_hash, adjustment_policy='none',
            slot_data_fingerprint=slot.slot_data_fingerprint,
        )
        record = build_identity_record(factor_identity_hash=identity_hash, payload=payload, code_version_source='test')
        grid = slot.calendar.bar_grid('1d')
        rows = grid[['timestamp_utc', 'session_id']].assign(
            factor_id='alpha', factor_identity_hash=identity_hash, frequency='1d', session_view='ETH',
            instrument_id='AAPL.XNYS', factor_value=np.arange(len(grid), dtype=float), source_build_id='test-build',
            calendar_hash=slot.calendar.calendar_hash, code_version='test',
            created_at=pd.Timestamp('2026-02-01T00:00:00Z'), batch_id=identity_hash[:24],
        )
        MarketFactorValueStore(store_root).append_factor_values(rows, identity_record=record, calendar=slot.calendar)

    eval_dir = tmp_path / 'market_eval'
    results_db = tmp_path / 'market_results.duckdb'
    json_out = tmp_path / 'market_dry.json'
    config = {
        'slot': str(slot_path),
        'slot_db': str(db_path),
        'factor_store_root': str(store_root),
        'factor_id': 'alpha',
        'analysis_mode': 'time_series',
        'frequency': '1d',
        'horizons': [1],
        'start_date': '2026-01-05',
        'end_date': '2026-01-09',
    }
    extra = [
        '--market-db', profile.db_id,
        '--output-dir', str(eval_dir),
        '--json-out', str(json_out),
        '--results-db', str(results_db),
    ]
    dry_payload = run_cli(tmp_path, config, '--dry-run', *extra)
    assert dry_payload['ok'] is True
    assert dry_payload['exit_code'] == 0
    assert dry_payload['command']['mode'] == 'dry_run'
    assert dry_payload['metrics']['market_db'] == market_db_identity(profile)
    assert dry_payload['metrics']['planned_results_db'] == str(results_db.resolve())
    assert json.loads(json_out.read_text(encoding='utf-8')) == dry_payload
    assert not eval_dir.exists()
    assert not results_db.exists()

    live_payload = run_cli(tmp_path, config, *extra)
    read_outputs(live_payload)
    assert live_payload['metrics']['results_db']['status'] == 'inserted'
    assert (eval_dir / 'summary.json').is_file()
    assert results_db.is_file()
