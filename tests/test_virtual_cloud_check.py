from __future__ import annotations

import json
from pathlib import Path

from scripts.alignment import alignment_cycle as CYCLE
from scripts.alignment import virtual_cloud_check as MODULE


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_cycle_resolves_conda_env_python_without_active_venv_path(monkeypatch) -> None:
    class Result:
        stdout = json.dumps({"envs": ["/opt/conda/envs/qlib_m3"]})

    monkeypatch.setattr(CYCLE.subprocess, "run", lambda *_args, **_kwargs: Result())

    assert CYCLE._conda_env_python("qlib_m3") == Path(
        "/opt/conda/envs/qlib_m3/bin/python"
    )


def test_virtual_cloud_check_accepts_a_self_contained_py36_bundle(
    tmp_path: Path,
) -> None:
    factor = _write(tmp_path / "factor.py", "class DemoFactor(object):\n    pass\n")
    bundle = _write(tmp_path / "bundle.py", "class DemoFactor(object):\n    pass\n")
    requirements = _write(
        tmp_path / "requirements.json",
        json.dumps(
            {"factors": {"DemoFactor": {"requirements": {"daily": {"min_days": 5}}}}}
        ),
    )

    result = MODULE.run_check(
        factor=factor,
        factor_class="DemoFactor",
        bundle=bundle,
        requirements=requirements,
    )

    assert result["status"] == "PASS"


def test_virtual_cloud_check_blocks_declared_unavailable_event_loader(
    tmp_path: Path,
) -> None:
    factor = _write(
        tmp_path / "factor.py",
        'META = {"cloud_limitation": "requires injected event_loader because event data is local data"}\nclass DemoFactor(object):\n    pass\n',
    )
    bundle = _write(tmp_path / "bundle.py", "class DemoFactor(object):\n    pass\n")
    requirements = _write(
        tmp_path / "requirements.json",
        json.dumps(
            {"factors": {"DemoFactor": {"requirements": {"daily": {"min_days": 5}}}}}
        ),
    )

    result = MODULE.run_check(
        factor=factor,
        factor_class="DemoFactor",
        bundle=bundle,
        requirements=requirements,
    )

    assert result["status"] == "FAIL"
    assert result["error_code"] == "cloud_event_loader_required"


def test_virtual_cloud_check_blocks_legacy_pandas_and_invalid_jq_history_fields(
    tmp_path: Path,
) -> None:
    factor = _write(tmp_path / "factor.py", "class DemoFactor(object):\n    pass\n")
    bundle = _write(
        tmp_path / "bundle.py",
        """import pandas as pd
class DemoFactor(object):
    def calculate_factor(self, data):
        columns = ['left', 'right']
        data[columns] = data.groupby('code')[columns].ffill()
        return pd.MultiIndex.from_frame(data[['date', 'code']])
""",
    )
    requirements = _write(
        tmp_path / "requirements.json",
        json.dumps(
            {
                "factors": {
                    "DemoFactor": {
                        "requirements": {
                            "5m": {
                                "min_bars": 6000,
                                "fields": ["close", "low_limit", "factor"],
                            }
                        }
                    }
                }
            }
        ),
    )

    result = MODULE.run_check(
        factor=factor,
        factor_class="DemoFactor",
        bundle=bundle,
        requirements=requirements,
    )
    checks = {item["name"]: item for item in result["checks"]}

    assert result["status"] == "FAIL"
    assert checks["legacy_pandas_api"]["passed"] is False
    assert checks["jq_api_contract"]["passed"] is False
    assert result["warnings"] == ["cloud_runtime_budget_risk:5m:bars=6000"]
    report = _write(tmp_path / "virtual_cloud.json", json.dumps(result))
    stage = {"name": "virtual_cloud", "ok": False, "returncode": 2}
    CYCLE._attach_virtual_warnings(stage, report)
    assert stage["issues"][0]["message"] == "cloud_runtime_budget_risk:5m:bars=6000"


def test_cycle_ignores_static_warn_and_global_requirement_drift(tmp_path: Path) -> None:
    report = _write(
        tmp_path / "static.json",
        json.dumps(
            {
                "issues": [
                    {"check_id": "CHECK-7", "status": "WARN"},
                    {"check_id": "CHECK-11", "status": "ISSUE"},
                ]
            }
        ),
    )
    stage = {"name": "static_review", "ok": False, "returncode": 1}

    CYCLE._classify_static_review(stage, report)

    assert stage["ok"] is True
    assert len(stage["ignored_global_issues"]) == 1


def test_cycle_requires_strict_spearman_and_overlap_for_cloud_comparison(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "comparison"
    report_dir.mkdir()
    _write(
        report_dir / "alignment_results.json",
        json.dumps(
            [
                {
                    "cloud_rows": 100,
                    "local_rows": 100,
                    "merged_rows": 95,
                    "cloud_rows_in_local_range": 100,
                    "stock_pool_overlap": 0.95,
                    "summary": {
                        "avg_spearman": 0.99,
                        "min_spearman": 0.96,
                        "max_max_abs_diff": 0.0,
                    },
                }
            ]
        ),
    )
    stage = {"name": "compare", "ok": True, "returncode": 0}

    CYCLE._apply_comparison_verdict(stage, report_dir)

    assert stage["ok"] is False
    assert stage["reason"] == "alignment_metrics_below_threshold"
