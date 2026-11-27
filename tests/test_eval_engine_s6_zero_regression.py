"""Offline S6 contracts for daily Harness output and the JoinQuant bundle."""

import ast
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import duckdb

from market_db_samples import declare_market_db
import pandas as pd
import pytest

from calendar_test_support import seed_calendar
from scripts import build_jq_strategy as builder
from scripts.build import assembler

ROOT = Path(__file__).resolve().parents[1]


def test_daily_harness_preserves_execution_and_persisted_evaluation(tmp_path, monkeypatch):
    input_db, result_db = (tmp_path / name for name in ("input.duckdb", "result.duckdb"))
    with duckdb.connect(str(input_db)) as connection:
        declare_market_db(connection, 'cn_equity')
        seed_calendar(connection, "2024-01-01", "2024-01-31")
        connection.execute("""
            CREATE TABLE price_data_daily AS
            WITH prices AS (
                SELECT date, stock_code, 100 * pow(1 + rate, row_number() OVER (
                    PARTITION BY stock_code ORDER BY date
                )) AS close
                FROM trading_calendar
                CROSS JOIN (VALUES ('A', 0.01), ('B', 0.02), ('C', 0.03), ('D', 0.04))
                    AS assets(stock_code, rate)
                WHERE is_open
            )
            SELECT *, close AS open, close AS high, close AS low, 1000 AS volume,
                1000 * close AS money, 1.0 AS factor FROM prices
        """)
    source = tmp_path / "daily_factor.py"
    source.write_text(
        "import pandas as pd\n"
        "from utils.factor_base import FactorBase\n"
        "class DailyRegressionFactor(FactorBase):\n"
        "    def get_factor_values(self):\n"
        "        return pd.DataFrame({asset: rank for rank, asset in enumerate('ABCD')},\n"
        "                            index=pd.bdate_range('2024-01-02', '2024-01-22'))\n",
        encoding="utf-8",
    )
    config = tmp_path / "analysis.json"
    config.write_text(json.dumps({
        "duckdb_path": str(input_db), "results_duckdb_path": str(result_db),
        "start_date": "2024-01-02", "end_date": "2024-01-22", "code_list": list("ABCD"),
        "ic_periods": [1], "n_quantiles": 2, "max_workers": 1, "results_export_parquet": False,
        "cache_dir": str(tmp_path / "cache"), "use_duckdb_cache": False,
    }), encoding="utf-8")
    monkeypatch.setenv("QUANT_DATA_BINDINGS", "{}")
    monkeypatch.setenv("QUANTATION_OUTPUTS_DIR", str(tmp_path / "outputs"))
    mirror = tmp_path / "mirror.json"
    run = subprocess.run([
        sys.executable, "-m", "tools.agent_harness.cli", "single-factor-analysis",
        "--config", str(config), "--factor", str(source), "--run-id", "s6-daily",
        "--output-root", str(tmp_path / "harness"), "--json-out", str(mirror),
    ], cwd=ROOT, capture_output=True, text=True, timeout=60)
    summary = json.loads(run.stdout)
    assert run.returncode == 0, summary.get("stderr_tail", run.stderr)
    assert summary == json.loads(mirror.read_text()) and summary["ok"] is True
    assert summary["command"]["name"] == "single-factor-analysis"
    artifacts = {item["artifact_type"]: Path(item["path"]) for item in summary["artifacts"]}
    assert json.loads(artifacts["summary"].read_text()) == summary
    result_path = artifacts["execution_result"]
    result = json.loads(result_path.read_text())
    assert result["run_id"] == "s6-daily"
    assert result["execution_id"] == result_path.parent.name
    assert result["config_sha256"] == sha256(config.read_bytes()).hexdigest()
    assert result["execution"] == {"expected": 1, "completed": 1, "failed": 0, "errors": [], "status": "succeeded"}

    with duckdb.connect(str(result_db), read_only=True) as connection:
        rows = connection.execute("""
            SELECT date, period, quantile, ic, ir, return FROM factor_analysis_results
            WHERE factor_name = 'DailyRegressionFactor' ORDER BY date, quantile NULLS FIRST
        """).fetchall()
    dates = pd.bdate_range("2024-01-02", "2024-01-22").date
    assert [(day, period, bucket) for day, period, bucket, *_ in rows] == [
        (day, 1, bucket) for day in dates for bucket in (None, 1, 2)
    ]
    assert [row[3] for row in rows[::3]] == pytest.approx([1.0] * len(dates), abs=1e-12, rel=0)
    assert all(row[4] is None for row in rows)
    assert [row[5] for row in rows] == pytest.approx([0.0, -.01, .01] * len(dates), abs=1e-12, rel=0)


def test_joinquant_bundle_matches_stage6_sha256(tmp_path, monkeypatch):
    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2024, 1, 2, 3, 4, 5, tzinfo=tz)

    monkeypatch.setattr(assembler, "datetime", FixedClock)
    monkeypatch.delenv("FP_KEEP_LOGS", raising=False)
    source = tmp_path / "s6_factor.py"
    source.write_text(
        "from utils.factor_base import FactorBase\n"
        "class S6Return(FactorBase):\n"
        "    def calculate_factor(self, data):\n"
        "        return data.pivot(index='date', columns='code', values='close').pct_change()\n",
        encoding="utf-8",
    )
    requirements = tmp_path / "requirements.json"
    requirements.write_text(json.dumps({"factors": {"S6Return": {
        "requirements": {"daily": {"min_days": 3, "fields": ["close"]}},
    }}}), encoding="utf-8")
    bundle = tmp_path / "strategy.py"
    assert builder.build_jq_strategy_main([
        "--public", "--factor-files", str(source), "--factors", "S6Return",
        "--requirements", str(requirements), "--single-file", "--out", str(bundle),
    ]) == 0
    artifact = bundle.read_bytes()
    tree = ast.parse(artifact, filename=str(bundle), feature_version=(3, 6))
    allowed_imports = sys.stdlib_module_names | {"duckdb", "jqdata", "jqfactor", "numba", "numpy", "pandas", "scipy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in allowed_imports for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                assert node.level == 1 and node.module in {"duckdb_adapter", "dynamic_stock_pool"}
            else:
                assert (node.module or "").split(".")[0] in allowed_imports

    # Guard against PEP 585 / PEP 604 runtime-evaluated annotations that break Python 3.6 import.
    pep585_builtins = {"tuple", "list", "dict", "set", "frozenset", "type"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations = [node.returns] + [
                arg.annotation for arg in (*node.args.args, *node.args.kwonlyargs)
            ]
            for annotation in annotations:
                if annotation is None:
                    continue
                for sub in ast.walk(annotation):
                    if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                        assert sub.value.id not in pep585_builtins, (
                            f"PEP 585 annotation at line {node.lineno}: {ast.dump(sub)}"
                        )
                    if isinstance(sub, ast.BinOp):
                        assert not isinstance(sub.op, ast.BitOr), (
                            f"PEP 604 union annotation at line {node.lineno}: {ast.dump(sub)}"
                        )

    # NERE-1227 changes the S6a digest (6915c894...) solely through inlined factor_runtime.py bar support.
    # NERE-1399 changes the digest (40d4fa8e...) solely through utils/pit.py (idempotent price_adjusted
    # attrs and Real check in apply_pit_adjustment; local factor scaling moved to utils/price_adjustment.py).
    # NERE-1401 changes the digest (2c9789cc...) solely through inlined utils/strategy_kernel.py
    # (added rank preprocessing, explicit "none" handling, and shared preprocessing helpers).
    # NERE-1429 changes the digest (3f380aa2...) solely through inlined utils/strategy_kernel.py
    # (shared factor validity mask, degenerate cross-section NaN preservation, and causal time-series standardization).
    # NERE-1471 changes the digest (c31905c2...) solely through inlined utils/strategy_kernel.py
    # (added model intercept validation and addition in linear_signal).
    assert sha256(artifact).hexdigest() == "c31905c2a946a555dfedb6b10dcd9ed067c4508fffc4d3f69bb5fb9e7e9a9f11"


def test_daily_adapter_has_no_alphalens_import():
    runner = ROOT / "backtest_and_analysis" / "alphalens_runner.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "alphalens" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "alphalens"
