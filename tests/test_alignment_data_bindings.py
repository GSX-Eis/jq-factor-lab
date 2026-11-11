"""Frozen alignment stages replay without source files and cannot mix runs."""

import json
import shutil
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from factor_synthesis.vectorbt_adapter.alignment_data_writer import AlignmentDataWriter
from scripts.alignment.parse_local_backtest_log import parse_from_alignment_dir


def test_alignment_publication_replay_and_version_isolation(tmp_path, monkeypatch):
    monkeypatch.delenv('QUANT_DATA_BINDINGS', raising=False)
    monkeypatch.delenv('QUANT_OUTPUT_CONTEXT', raising=False)
    db = tmp_path / 'output.duckdb'
    binding = {'output_db': str(db)}
    context = dict(build_id='one', pool_id='pool', source_hash='source', config_hash='config',
                   input_versions={}, created_at='2026-09-15T00:00:00')

    def writer(name, value, factor='factor'):
        result = AlignmentDataWriter(str(tmp_path / 'sources'), name, data_bindings=binding,
                                     publication_metadata={**context, 'build_id': name})
        raw = pd.DataFrame({factor: [value, value + 1]}, index=['stock1', 'stock2'])
        scores = pd.Series([0.1, 0.2], index=raw.index)
        result.add_stage2_raw_factors('2019-01-03', '2019-01-02', raw, {factor: 0.7})
        result.add_stage3_preprocessed('2019-01-03', '2019-01-02', raw * 0.1)
        result.add_stage4_composite('2019-01-03', '2019-01-02', scores)
        result.add_rebalance_snapshot('2019-01-03', '2019-01-02', list(raw.index),
                                      raw_factors=raw, processed_factors=raw * 0.1,
                                      composite_scores=scores, target_holdings=[])
        result.save()
        return result

    one = writer('one', 1.)
    publication = json.loads((one.output_dir / 'publication.json').read_text())
    refs = publication['data_refs']
    baseline = parse_from_alignment_dir(one.output_dir, data_bindings={})
    assert baseline['2019-01-03']['weights'] == {'factor': 0.7}
    with duckdb.connect(str(db)) as conn:
        count = conn.execute('SELECT count(*) FROM meta.artifacts').fetchone()[0]
    one.save()
    with duckdb.connect(str(db)) as conn:
        assert conn.execute('SELECT count(*) FROM meta.artifacts').fetchone()[0] == count
    assert json.loads((one.output_dir / 'publication.json').read_text()) == publication
    two = writer('two', 10., 'another_factor')
    newer = json.loads((two.output_dir / 'publication.json').read_text())['data_refs']
    frozen = {**binding, 'documents': {'meta.alignment_summary': refs['meta.alignment_summary']}}
    shutil.rmtree(tmp_path / 'sources')
    assert parse_from_alignment_dir(one.output_dir, data_bindings=frozen) == baseline
    contract_path = tmp_path / 'bindings.json'
    contract_path.write_text(json.dumps(frozen))
    report = tmp_path / 'parsed.json'
    cli = Path(__file__).resolve().parents[1] / 'scripts/alignment/parse_local_backtest_log.py'
    subprocess.run([sys.executable, str(cli), '--data-bindings', str(contract_path), '--output', str(report)],
                   cwd=tmp_path, env={k: v for k, v in os.environ.items() if k != 'PYTHONPATH'},
                   capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(report.read_text()) == json.loads(json.dumps(baseline))
    mixed = deepcopy(frozen)
    mixed['outputs'] = {'alignment.stage2_raw_factors': newer['alignment.stage2_raw_factors']}
    with pytest.raises(ValueError, match='differs from frozen summary'):
        parse_from_alignment_dir(one.output_dir, data_bindings=mixed)
    with pytest.raises(ValueError, match='frozen summary'):
        parse_from_alignment_dir(one.output_dir, data_bindings=binding)
    with pytest.raises(ValueError, match='published weights'):
        parse_from_alignment_dir(one.output_dir, bundle_path=tmp_path / 'bundle.json', data_bindings=frozen)

    # Interrupted publication exposes no successful summary; a retry reuses committed stages.
    import utils.data_store as store
    original = store.publish_artifact
    empty = AlignmentDataWriter(str(tmp_path / 'sources'), 'empty', data_bindings=binding,
                                publication_metadata={**context, 'build_id': 'empty'})
    empty.add_stage4_composite('2019-01-03', '2019-01-02', pd.Series([1.], index=['stock']))

    def fail_holdings(path, target, **kwargs):
        if target == 'alignment.holdings_snapshot':
            raise RuntimeError('interrupted')
        return original(path, target, **kwargs)

    monkeypatch.setattr(store, 'publish_artifact', fail_holdings)
    with pytest.raises(RuntimeError, match='interrupted'):
        empty.save()
    assert not (empty.output_dir / 'summary.json').exists()
    assert not (empty.output_dir / 'publication.json').exists()
    monkeypatch.setattr(store, 'publish_artifact', original)
    empty.save()
    empty_refs = json.loads((empty.output_dir / 'publication.json').read_text())['data_refs']
    replay = parse_from_alignment_dir(tmp_path / 'absent', data_bindings={**binding, 'documents': {
        'meta.alignment_summary': empty_refs['meta.alignment_summary']}})
    assert list(replay) == ['2019-01-03']
    assert replay['2019-01-03']['stage4_composite']['stock']['score'] == 1.
