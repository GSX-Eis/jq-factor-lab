"""Nested static configuration must not silently execute workstation defaults."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_nested_static_config_uses_explicit_database_before_any_result_write(tmp_path):
    missing = tmp_path / 'explicit-missing.duckdb'
    destination = tmp_path / 'results.duckdb'
    config = tmp_path / 'analysis.json'
    config.write_text(json.dumps({
        'database': {'db_backend': 'duckdb', 'duckdb_path': str(missing)},
        'results_database': {'db_backend': 'duckdb', 'duckdb_path': str(destination)},
        'analysis_period': {'start_date': '2022-01-04', 'end_date': '2022-01-31'},
        'analysis_settings': {'code_list': ['A'], 'max_workers': 1},
    }))
    env = dict(os.environ, QUANTATION_OUTPUTS_DIR=str(tmp_path / 'outputs'),
               QUANT_DATA_BINDINGS='{}')
    env.pop('QUANT_EXECUTION_DIR', None)
    runner = Path(__file__).resolve().parents[1] / 'run_analysis_main.py'
    result = subprocess.run([sys.executable, str(runner), '--config', str(config)],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert str(missing) in result.stderr
    assert not destination.exists()
