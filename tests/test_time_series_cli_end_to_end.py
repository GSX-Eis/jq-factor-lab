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
    proc = subprocess.run([sys.executable, '-m', 'tools.agent_harness.cli', command,
        '--config', str(path), '--output-root', str(tmp_path / 'harness'), *extra],
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
    return (pd.read_csv(items['per_bar_labels']), pd.read_csv(items['ic_by_horizon']),
            pd.read_csv(items['quantile_stats']), json.loads(items['evaluation_summary'].read_text()))


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
