"""Skill readers cannot silently select another pool or artifact version."""
from dataclasses import asdict

import pandas as pd
import pytest

from scripts.alignment.compare_factors import load_local_data
from scripts.alignment.find_max_diff_date import find_max_diff_date
from utils.data_store import publish_artifact


def test_skill_reader_freezes_version_and_rejects_ambiguous_pool(tmp_path):
    output = tmp_path / 'output.duckdb'
    frame = pd.DataFrame({
        'date': ['2019-01-02'] * 6, 'stock_code': list('abcdef'),
        'factor_name': ['F_small_cap_9_rules'] * 6,
        'factor_value': list(range(6)),
    })
    first = publish_artifact(output, 'main.factor_values', frame=frame, metadata={'run_id': 'first'})
    publish_artifact(output, 'main.factor_values', frame=frame.assign(factor_value=999), metadata={'run_id': 'second'})
    binding = {'output_db': str(output), 'outputs': {'factor_values': asdict(first)}}
    assert load_local_data(None, 'F', bindings=binding).value.tolist() == list(range(6))
    with pytest.raises(ValueError, match='frozen data bindings'):
        load_local_data(str(output), 'F', bindings={})
    with pytest.raises(ValueError, match='must match'):
        load_local_data(str(tmp_path / 'missing.duckdb'), 'F', bindings=binding)
    bad = {**binding, 'outputs': {'factor_values': {**asdict(first), 'version': 'missing'}}}
    with pytest.raises((ValueError, KeyError)):
        load_local_data(None, 'F', bindings=bad)

    both = pd.concat([frame, frame.assign(factor_name='F_small_cap_percentile_10', factor_value=42)])
    ref = publish_artifact(output, 'main.factor_values', frame=both, metadata={'run_id': 'both'})
    binding['outputs']['factor_values'] = asdict(ref)
    with pytest.raises(ValueError, match='Multiple pool factors'):
        load_local_data(None, 'F', bindings=binding)
    assert load_local_data(None, 'F', bindings=binding, local_factor_name='F_small_cap_percentile_10').value.tolist() == [42] * 6

    local = frame.rename(columns={'stock_code': 'code', 'factor_value': 'value'})[['date', 'code', 'value']]
    csv = tmp_path / 'local.csv'; local.to_csv(csv, index=False)
    cloud = tmp_path / 'cloud.csv'; local.assign(factor='F').to_csv(cloud, index=False)
    assert find_max_diff_date('F', cloud, local_csv=csv) == '2019-01-02'
    local.assign(date='2019-01-03').to_csv(csv, index=False)
    assert find_max_diff_date('F', cloud, local_csv=csv) is None
