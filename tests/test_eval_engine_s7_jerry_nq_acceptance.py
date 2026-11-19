"""Frozen S7 acceptance for the Bar CLI and unified time-series statistics."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import runpy
from shutil import copytree

import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.evaluation import EvaluationEngine
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.time_series.artifacts import _json_value
from test_time_series_cli_end_to_end import run_cli


FIXTURES = Path(__file__).parent / 'fixtures' / 'eval_engine' / 'jerry_contract'
FROZEN = json.loads((FIXTURES / 'expected.json').read_text(encoding='utf-8'))
STATISTICS = dict(block_length=5, seed=791, n_boot=9, min_valid_replicates=2)


def _jerry_config(path):
    return dict(
        bars_csv=str(path), analysis_mode='time_series',
        start_date='2026-12-01', end_date='2026-12-01',
        calendar='CME Globex Equity', view='RTH', horizons=[1, 2, 5], n_quantiles=2,
    )


def test_jerry_harness_json_and_csv_contract(tmp_path):
    expected = FROZEN['harness']
    path = FIXTURES / 'bars.csv'
    config = _jerry_config(path)
    payload = run_cli(tmp_path, config)
    assert payload['ok'], payload
    artifacts = {item['artifact_type']: item for item in payload['artifacts']}
    assert set(artifacts) == {'summary', *expected['artifacts']}
    for key, basename in expected['artifacts'].items():
        artifact = artifacts[key]
        output = Path(artifact['path'])
        assert output.is_absolute() and output.name == basename
        assert sha256(output.read_bytes()).hexdigest() == artifact['sha256']

    summary_path = Path(artifacts['evaluation_summary']['path'])
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert set(summary) == set(expected['summary_fields'])
    assert payload['metrics'] == {key: summary[key] for key in expected['metric_fields']}
    assert summary['result_schema_version'] == expected['evaluation_schema']
    assert summary['analysis_mode'] == 'time_series' and summary['cross_sections'] == []
    assert summary['config'] == config
    assert set(summary['result_identity']) == set(expected['identity_fields'])
    identity = summary['result_identity']
    assert identity['input_file_hashes'] == {str(path.resolve()): sha256(path.read_bytes()).hexdigest()}
    assert identity['calendar_hash'] == summary['calendar_identity']['calendar_hash']
    assert summary['calendar_identity']['calendar_id'] == 'CME Globex Equity'
    assert summary['data_identity']['sources'] == ['SYNTHETIC']
    assert summary['data_identity']['data_versions'] == ['SYNTHETIC_NERE828_JERRY_v1']

    tables = {}
    for name, columns in expected['csv_columns'].items():
        with Path(artifacts[name]['path']).open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == columns.split(',')
            tables[name] = list(reader)
    for name in ('ic_by_horizon', 'quantile_stats'):
        for row, json_row in zip(tables[name], summary[name], strict=True):
            for key, value in json_row.items():
                if isinstance(value, dict):
                    assert json.loads(row[key]) == value
                elif value is None:
                    assert row[key] == ''
                elif isinstance(value, float):
                    assert float(row[key]) == pytest.approx(value, abs=1e-12)
                else:
                    assert row[key] == str(value)

    t0 = pd.Timestamp('2026-12-01T14:30Z')
    minute = pd.Timedelta(minutes=1)
    closes = [100, 96, 98, 100, 102, 104, 98, 106, 108]
    for (i, h), row in zip(
        [(i, h) for h in config['horizons'] for i in range(9)], tables['per_bar_labels'], strict=True,
    ):
        assert row['instrument_id'] == 'NQZ26' and row['frequency'] == '1min'
        assert int(row['horizon_bars']) == h and int(row['entry_offset_bars']) == 1
        assert pd.Timestamp(row['signal_timestamp']) == t0 + i * minute
        assert pd.Timestamp(row['signal_available_at']) == t0 + (i + 1) * minute
        reason = 'factor_missing' if i in (4, 8) else 'insufficient_history' if i + h >= 9 else ''
        assert row['exclusion_reason'] == reason
        assert row['status'] == ('excluded' if reason else 'valid')
        assert row['signal_timestamp'].endswith('Z')
        if reason:
            assert all(row[key] == '' for key in (
                'entry_timestamp', 'exit_bar_timestamp', 'exit_timestamp', 'entry_price',
                'exit_price', 'price_change_points', 'return_pct',
            ))
        else:
            assert pd.Timestamp(row['entry_timestamp']) == t0 + (i + 1) * minute
            assert pd.Timestamp(row['exit_bar_timestamp']) == t0 + (i + h) * minute
            assert pd.Timestamp(row['exit_timestamp']) == t0 + (i + h + 1) * minute
            assert (float(row['entry_price']), float(row['exit_price'])) == (100, closes[i + h])
            assert float(row['price_change_points']) == closes[i + h] - 100
            assert float(row['return_pct']) == pytest.approx(closes[i + h] - 100, abs=1e-12)

    counts = summary['exclusion_summary']
    assert (counts['total_count'], counts['valid_count'], counts['excluded_count']) == (27, 17, 10)
    assert counts['exclusion_counts'] == expected['exclusion_counts']
    assert counts['nonzero_exclusion_counts'] == {'factor_missing': 6, 'insufficient_history': 4}
    for h, oracle in expected['horizons'].items():
        row, = [row for row in summary['ic_by_horizon']
                if row['scope'] == 'per_instrument' and str(row['horizon_bars']) == h]
        assert row['total_count'] == 9
        assert row['valid_count'] == oracle['valid']
        assert row['valid_count'] + sum(row['exclusion_counts'].values()) == row['total_count']
        assert row['spearman_ic'] == pytest.approx(oracle['ic'], abs=1e-12)
        assert row['non_overlapping_count'] == oracle['phase_zero_count']
        phase_ic = oracle['phase_zero_ic']
        assert row['non_overlapping_ic'] == (None if phase_ic is None else pytest.approx(phase_ic, abs=1e-12))
        buckets = [row for row in summary['quantile_stats'] if str(row['horizon_bars']) == h]
        for bucket, values in zip(buckets, oracle['quantiles'], strict=True):
            assert bucket['quantile_scope'] == 'full_sample_descriptive_only'
            assert bucket['descriptive_only'] is True
            columns = ['sample_count', 'mean_return_pct', 'median_return_pct', 'positive_ratio']
            assert [bucket[key] for key in columns] == pytest.approx(values, abs=1e-12)


def _check_values(actual, expected):
    for value, oracle in zip(actual, expected, strict=True):
        assert value == (pytest.approx(oracle, rel=0, abs=1e-12) if isinstance(oracle, (int, float)) else oracle)


def _check_direction_and_distribution(row, oracle):
    direction, dist = row['direction'], row['return_distribution']
    hits, denominator, ups = oracle['direction']
    assert [direction[k] for k in ('hits', 'denominator', 'up_count')] == [hits, denominator, ups]
    if denominator:
        _check_values([direction[k] for k in ('accuracy', 'up_baseline', 'lift')],
                      [hits / denominator, ups / denominator, (hits - ups) / denominator])
    else:
        assert direction['accuracy'] is None and direction['reason'] == 'no_directional_samples'
    assert dist['n_samples'] == oracle['samples']
    if not dist['n_samples']:
        assert dist['reason'] == 'empty_bucket'
    _check_values([dist[k] for k in ('p5', 'p25', 'p50', 'p75', 'p95')], oracle['percentiles'])
    _check_values([dist[k] for k in ('mean_le_p5', 'mean_le_p25', 'mean_ge_p75', 'mean_ge_p95')], oracle['tails'])


def test_jerry_opt_in_cli_publishes_frozen_json_csv_and_markdown(tmp_path):
    expected = FROZEN['jerry_cli']
    payload = run_cli(tmp_path, _jerry_config(FIXTURES / 'bars.csv'), '--jerry-report')
    assert payload['ok'], payload
    artifacts = {item['artifact_type']: item for item in payload['artifacts']}
    expected_files = FROZEN['harness']['artifacts'] | expected['artifacts']
    assert set(artifacts) == {'summary', *expected_files}
    paths = {name: Path(artifacts[name]['path']) for name in expected_files}
    for name, path in paths.items():
        assert path.is_absolute() and path.name == expected_files[name]
        assert sha256(path.read_bytes()).hexdigest() == artifacts[name]['sha256']

    labels = pd.read_csv(paths['per_bar_labels']).query('horizon_bars == 2')
    offset_one = labels.iloc[1::2].query("status == 'valid'")
    assert len(offset_one) == 3
    assert offset_one.factor_value.corr(offset_one.return_pct, method='spearman') == 1
    summary = json.loads(paths['evaluation_summary'].read_text(encoding='utf-8'))
    assert summary['result_schema_version'] == FROZEN['harness']['evaluation_schema']
    report = summary['jerry_report']
    assert report == payload['metrics']['jerry_report']
    assert report['schema'] == expected['schema']
    assert report['return_unit'] == 'percent'
    assert report['quantile_scope'] == 'full_sample_descriptive_only'

    with paths['jerry_report_csv'].open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == expected['csv_columns'].split(',')
        csv_rows = {(r['horizon_bars'], r['scope'], r['index']): r for r in reader}
    for row, h in zip(report['horizons'], expected['horizons'], strict=True):
        assert row['horizon_bars'] == int(h)
        assert (row['instrument_id'], row['frequency']) == ('NQZ26', '1min')
        assert (row['grid_start_utc'], row['grid_end_utc']) == ('2026-12-01T14:30:00Z', '2026-12-01T14:38:00Z')
        oracle = expected['horizons'][h]
        ci, counts = row['bootstrap'], row['counts']
        assert [ci[k] for k in ('grid_size', 'block_length', 'seed', 'n_boot', 'min_valid_replicates', 'alpha')] == [
            9, 5, 0, 999, 900, 0.05,
        ]
        _check_values([ci[k] for k in ('valid_replicates', 'lower', 'upper', 'reason')], oracle['bootstrap'])
        assert counts['total_count'] == counts['valid_count'] + sum(counts['exclusion_counts'].values()) == 9
        assert counts['excluded_count'] == sum(counts['exclusion_counts'].values())
        assert counts['valid_count'] == row['ic']['sample_count'] == oracle['overall']['samples']
        assert row['ic']['spearman_ic'] == pytest.approx(FROZEN['harness']['horizons'][h]['ic'], abs=1e-12)
        _check_direction_and_distribution(row, oracle['overall'])
        assert list(row['buckets']) == list(oracle['buckets'])
        for key, bucket in row['buckets'].items():
            _check_direction_and_distribution(bucket, oracle['buckets'][key])
        assert list(row['non_overlapping_ic']) == [str(i) for i in range(int(h))]
        for phase, phase_oracle in zip(row['non_overlapping_ic'].values(), oracle['offsets'], strict=True):
            _check_values([phase[k] for k in ('sample_count', 'spearman_ic', 'reason')], phase_oracle)

        scopes = [
            ('overall', '', {k: counts[k] for k in ('total_count', 'valid_count', 'excluded_count')},
             [('excluded', counts['exclusion_counts']), ('ic', row['ic']), ('bootstrap', ci),
              ('direction', row['direction']), ('return', row['return_distribution'])]),
            *(('bucket', key, {}, [('direction', bucket['direction']), ('return', bucket['return_distribution'])])
              for key, bucket in row['buckets'].items()),
            *(('offset', key, {}, [('ic', phase)]) for key, phase in row['non_overlapping_ic'].items()),
        ]
        for scope, index, values, groups in scopes:
            csv_row = csv_rows.pop((h, scope, index))
            assert (csv_row['instrument_id'], csv_row['frequency']) == ('NQZ26', '1min')
            for prefix, fields in groups:
                values.update({f'{prefix}_{k}': v for k, v in fields.items()})
            for key, value in values.items():
                actual = csv_row[key]
                if value is None:
                    assert actual == ''
                elif isinstance(value, float):
                    assert float(actual) == pytest.approx(value, rel=0, abs=1e-12)
                else:
                    assert actual == str(value)
    assert not csv_rows
    markdown = paths['jerry_report_markdown'].read_text(encoding='utf-8').splitlines()
    assert [line for line in markdown if line.startswith('#')] == expected['markdown_headings']
    assert all(row in markdown for row in expected['markdown_rows'])


def test_jerry_cli_keeps_missing_minutes_in_the_phase_grid(tmp_path):
    rows = (FIXTURES / 'bars.csv').read_text(encoding='utf-8').splitlines()
    bars = tmp_path / 'gapped.csv'
    bars.write_text('\n'.join(row for row in rows if '14:35:00Z' not in row) + '\n', encoding='utf-8')
    config = _jerry_config(bars) | {'horizons': [2], 'jerry_report': True}
    payload = run_cli(tmp_path, config)
    assert payload['ok'], payload
    row, = payload['metrics']['jerry_report']['horizons']
    assert row['bootstrap']['grid_size'] == 9 and row['counts']['total_count'] == 8
    assert {k: v['sample_count'] for k, v in row['non_overlapping_ic'].items()} == {'0': 3, '1': 1}
    assert row['non_overlapping_ic']['0']['spearman_ic'] == 1.0
    assert row['counts']['exclusion_counts']['missing_bar'] == 1


@pytest.fixture(scope='module')
def published_jerry_report(tmp_path_factory):
    root = tmp_path_factory.mktemp('published_jerry')
    destination = root / 'evaluation'
    payload = run_cli(root, _jerry_config(FIXTURES / 'bars.csv'), '--jerry-report',
                      '--output-dir', str(destination))
    assert payload['ok'], payload
    assert {path.name for path in destination.iterdir()} == {
        *FROZEN['harness']['artifacts'].values(), *FROZEN['jerry_cli']['artifacts'].values(),
    }
    return destination


@pytest.mark.parametrize('filename', ['jerry_report.csv', 'jerry_report.md'])
@pytest.mark.parametrize('mode, extra', [
    ('dry_run', ('--dry-run',)),
    ('fake', ('--fake-result', 'success')),
    ('invalid_horizons', ('--horizons', '1,nope')),
    ('live', ()),
    ('dry_run_without_jerry', ('--dry-run',)),
])
def test_jerry_cli_rejects_json_mirror_collision(tmp_path, published_jerry_report, filename, mode, extra):
    destination = tmp_path / 'evaluation'
    copytree(published_jerry_report, destination)
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    config = _jerry_config(FIXTURES / 'bars.csv')
    if mode in ('fake', 'invalid_horizons'):
        config = {'time_series_analysis': config | {'output_dir': str(destination), 'jerry_report': True}}
    else:
        extra += ('--output-dir', str(destination))
        if mode != 'dry_run_without_jerry':
            extra += ('--jerry-report',)
    payload = run_cli(tmp_path, config, *extra, '--json-out', str(destination / filename))
    after = {path.name: path.read_bytes() for path in destination.iterdir()}
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    assert (payload['exit_code'], changed) == (2, [])
    assert not payload['ok']
    assert '--json-out must differ from evaluation artifact paths' in payload['error']['message']


def _panel(factor, outcomes, times):
    index = pd.MultiIndex.from_product([times, ['NQZ26']])
    return EvaluationPanel.from_long(pd.DataFrame({'factor': factor, **outcomes}, index=index))


def _intervals(times, horizons):
    entry = times.to_numpy(dtype='datetime64[ns]')[:, None] + np.timedelta64(1, 'm')
    return dict(entry_timestamps={h: entry for h in horizons},
                exit_timestamps={h: entry + np.timedelta64(h, 'm') for h in horizons})


def test_jerry_direction_and_conditional_return_distribution():
    fixture = FROZEN['direction_and_returns']
    times = pd.date_range('2025-12-31T14:30Z', periods=10, freq='min')
    panel = _panel(fixture['factor'], {1: fixture['return_pct']}, times)
    valid = np.ones((len(times), 1), dtype=bool)
    valid[fixture['masked_row']] = False
    result = EvaluationEngine('time_series', 'bar').evaluate(
        panel, signal_valid=valid, bucket_labels=np.array(fixture['bucket'])[:, None], n_quantiles=4,
        **_intervals(times, [1]), **STATISTICS,
    )
    row = result.splits['full'].statistics.horizons[1].instruments['NQZ26']
    for key, expected in fixture['expected'].items():
        direction = row.direction if key == 'all' else row.direction_by_bucket[int(key)]
        dist = row.returns if key == 'all' else row.returns_by_bucket[int(key)]
        assert [direction.hits, direction.denominator, direction.up_count] == expected['counts']
        assert [direction.accuracy, direction.up_baseline, direction.lift] == pytest.approx(expected['accuracy'])
        assert [dist.p5, dist.p25, dist.p50, dist.p75, dist.p95] == pytest.approx(expected['percentiles'])
        assert [dist.mean_le_p5, dist.mean_le_p25, dist.mean_ge_p75, dist.mean_ge_p95] == pytest.approx(
            expected['tails'],
        )
    for bucket in (3, 4):
        assert row.returns_by_bucket[bucket].n_samples == 0
        assert _json_value(row.returns_by_bucket[bucket].p50) is None
        assert row.direction_by_bucket[bucket].reason == 'no_directional_samples'


def test_jerry_all_expected_grid_offsets_are_non_overlapping():
    fixture = FROZEN['offsets']
    first = pd.date_range('2025-12-30T14:30Z', periods=6, freq='min')
    times = first.append(first + pd.Timedelta(days=1))
    panel = _panel(fixture['factor'], {2: fixture['return_pct']}, times)
    intervals = _intervals(times, [2])
    engine = EvaluationEngine('time_series', 'bar')
    options = dict(expected_grid=np.ones_like(panel.valid[2]), **intervals, **STATISTICS)
    statistics = engine.evaluate(panel, **options).splits['full'].statistics
    row = statistics.horizons[2].instruments['NQZ26']
    assert statistics.intervals_verified
    assert list(row.non_overlapping_ic) == [0, 1]
    for offset, keys in enumerate(fixture['sample_keys']):
        assert np.flatnonzero(panel.valid[2][:, 0] & (np.arange(12) % 2 == offset)).tolist() == keys
        ic = row.non_overlapping_ic[offset]
        assert ic.counts[0] == len(keys)
        assert ic.values[0] == pytest.approx(fixture['ic'][offset])
    intervals['exit_timestamps'][2][1, 0] += np.timedelta64(1, 'ns')
    with pytest.raises(AssertionError, match='horizon 2, offset 1'):
        engine.evaluate(panel, **options)


def test_jerry_moving_block_bootstrap_reranks_each_replicate():
    fixture = FROZEN['bootstrap']
    times = pd.date_range('2025-12-31T14:30Z', periods=9, freq='min')
    panel = _panel(fixture['factor'], {2: fixture['return_pct']}, times)
    valid = np.ones((len(times), 1), dtype=bool)
    valid[fixture['masked_row']] = False
    result = EvaluationEngine('time_series', 'bar').evaluate(
        panel, signal_valid=valid, **_intervals(times, [2]), **fixture['settings'],
    )
    row = result.splits['full'].statistics.horizons[2].instruments['NQZ26']
    interval = row.bootstrap
    assert interval.grid_size == 9 and row.ic.counts[0] == 7
    assert interval.valid_replicates == fixture['valid_replicates'] and interval.reason is None
    assert [interval.lower, interval.upper] == pytest.approx(fixture['bounds'], abs=1e-12)


def test_jerry_split_purges_late_exit_and_keeps_touching_exit():
    times = pd.date_range('2025-12-31T23:50Z', periods=20, freq='min')
    signal = times + pd.Timedelta(minutes=1)
    intervals = _intervals(times, [2, 5])
    panel = _panel(np.arange(20), {h: np.arange(20) for h in (2, 5)}, times)
    result = EvaluationEngine('time_series', 'bar').evaluate(
        panel, split={'train_end': '2025-12-31T23:59:59Z', 'oos_start': '2026-01-01T00:00Z'},
        signal_available_at=signal, quantile_scheme='fitted_on_train', standardization='zscore',
        **intervals, **STATISTICS,
    )
    train, oos = result.splits['train'], result.splits['oos']
    assert train.split_purged_count == {2: 1, 5: 4}
    for h in (2, 5):
        assert train.valid[h][:, 0].tolist() == [True] * (10 - h) + [False] * (h - 1)
    assert oos.split_purged_count == {2: 0, 5: 0}
    assert train.fit_identity == oos.fit_identity
    assert train.fit_identity['instruments'][0]['training_edges'][::5] == [0, 8]


def test_nq_acceptance_report_contract_on_year_boundary():
    fixture = runpy.run_path(str(FIXTURES / 'nq_1min.py'))
    calendar = fixture['SessionCalendar']('CME Globex Equity', '2025-12-31', '2026-01-02',
                                         rth_reference_calendar='NYSE')
    bars = fixture['synthetic_bars'](calendar)
    identity = fixture['input_identity'](bars)
    assert identity['source'] == 'SYNTHETIC' and identity['real_market_data_used'] is False
    assert identity['rows'] == 2760
    assert identity['sha256'] == 'eff380e43f100973f7306d7c815eeb78c21702f2563d5f08ea55b8c1c1c5d76f'
    views = {}
    for view, expected_rows in [('ETH', 1380), ('RTH', 390)]:
        result = fixture['evaluate_view'](bars, calendar, view)
        views[view] = result
        assert result['train']['fit_identity'] == result['oos']['fit_identity']
        assert result['train']['fit_identity']['direction'] == 1
        for name, year in [('train', 2025), ('oos', 2026)]:
            split = result[name]
            assert split['calendar_year_counts'] == {year: expected_rows}
            assert split['grid_rows'] == expected_rows
            assert split['quantile_scheme'] == 'fitted_on_train'
            assert split['intervals_verified']
            for h, row in split['horizons'].items():
                counts = row['counts']
                assert counts['valid'] == expected_rows - h
                assert counts['split_purged_count'] == 0
                assert sum(counts['label_exclusions'].values()) == h
                assert list(row['non_overlapping_ic']) == list(range(h))
                assert sum(phase['sample_count'] for phase in row['non_overlapping_ic'].values()) == counts['valid']
                assert row['bootstrap']['valid_replicates'] == 99
                assert len(row['direction_by_bucket']) == len(row['return_distribution_by_bucket']) == 5
    markdown = fixture['render_markdown'](_json_value(dict(data_identity=identity, views=views)))
    expected = FROZEN['markdown']
    assert [line for line in markdown.splitlines() if line.startswith('#')] == expected['headings']
    assert markdown.count(f"| {expected['columns']} |") == 4
    assert markdown.count('**SYNTHETIC**') == 1
    assert identity['sha256'] in markdown
    assert all(result['train']['fit_identity']['sha256'] in markdown for result in views.values())
