"""Execute the documented offline workflow through import, synthesis and build."""
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
import pandas as pd
import pytest

from examples.local_factor_workflow import run_workflow
from utils.market_db import read_market_db_profile

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_workflow_uses_price_labels_and_matching_deployment_formula(tmp_path):
    # Ensure real utils.factor_base is loaded in fresh workers before snapshotting
    # aliases so this assertion verifies bundle replay restores it rather than testing cold import.
    import utils.factor_base  # noqa: F401

    aliases = ('factor_base', 'utils.factor_base', 'jq_adapter', 'utils.jq_adapter')
    missing = object()
    before = {name: sys.modules.get(name, missing) for name in aliases}
    result = run_workflow(tmp_path/'workflow')
    assert all(sys.modules.get(name, missing) is value for name, value in before.items())
    assert result['imported_rows'] == 80*12
    assert result['score_rows'] > result['evaluation_rows'] > 0
    labels = pd.read_csv(tmp_path/'workflow/labels.csv')
    source = pd.read_csv(tmp_path/'workflow/csv/daily/DEMO000.csv')
    first = labels[labels.stock_code == 'DEMO000'].iloc[0]
    assert first['1D'] == pytest.approx(source.close.iloc[1]/source.close.iloc[0]-1)
    assert first['5D'] == pytest.approx(source.close.iloc[5]/source.close.iloc[0]-1)
    assert (pd.Timestamp(result['bundle_previous_date']) + pd.offsets.BDay(1)
            == pd.Timestamp(result['bundle_execution_date']))
    assert json.loads((tmp_path/'workflow/summary.json').read_text()) == result
    with pytest.raises(FileExistsError):
        run_workflow(tmp_path/'workflow')


def test_local_factor_workflow_cli_end_to_end(tmp_path):
    caller_cwd = tmp_path / 'empty_cwd'
    caller_cwd.mkdir()
    output_dir = tmp_path / 'cli_workflow'

    env = os.environ.copy()
    existing_pythonpath = env.get('PYTHONPATH')
    env['PYTHONPATH'] = (
        f'{REPO_ROOT}{os.pathsep}{existing_pythonpath}' if existing_pythonpath else str(REPO_ROOT)
    )

    completed = subprocess.run(
        [sys.executable, '-m', 'examples.local_factor_workflow', '--output-dir', str(output_dir)],
        cwd=str(caller_cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    stdout_summary = json.loads(completed.stdout)
    file_summary = json.loads((output_dir / 'summary.json').read_text())
    assert stdout_summary == file_summary
    assert stdout_summary['imported_rows'] == 80 * 12
    assert stdout_summary['score_rows'] > stdout_summary['evaluation_rows'] > 0

    for idx in range(12):
        assert (output_dir / 'csv' / 'daily' / f'DEMO{idx:03d}.csv').is_file()
    for relative_file in (
        'market.duckdb',
        'user_factors.py',
        'requirements.json',
        'features.csv',
        'scores.csv',
        'labels.csv',
        'evaluation_rows.csv',
        'joinquant_strategy.py',
        'summary.json',
    ):
        assert (output_dir / relative_file).is_file(), f'Missing expected artifact: {relative_file}'
    assert (output_dir / 'market_factor_values').is_dir()

    with duckdb.connect(str(output_dir / 'market.duckdb'), read_only=True) as conn:
        profile, source = read_market_db_profile(conn)
    assert source == 'table'
    assert profile.market_domain == 'cn_equity'
    assert 'jq_ashare_local' in profile.allowed_slot_providers
    assert profile.calendar_id == 'XSHG'
    assert profile.timezone == 'Asia/Shanghai'
