from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from tools.agent_harness.d9_capacity_audit import (
    D9CapacityAuditError,
    build_capacity_audit,
    load_capacity_audit_context,
    preflight_capacity_audit,
    write_capacity_audit,
)


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, count: int = 3) -> tuple[Path, dict]:
    factors = [f"Factor{index}" for index in range(count)]
    protocol = _write(
        tmp_path / "protocol.json",
        {
            "experiment_id": "replay",
            "evaluation_folds": [{"evaluation": ["2020-01-01", "2023-12-31"]}],
        },
    )
    snapshot = _write(
        tmp_path / "snapshot.json", {"factor_count": count, "factor_ids": factors}
    )
    output = tmp_path / "output"
    config = {
        "analysis_period": {"start_date": "2018-01-01", "end_date": "2023-12-31"},
        "factor_groups": {"audit": {"factors": factors}},
        "synthesis_methods": {
            "audit": {
                "method": "lasso",
                "model_type": "elasticnet",
                "positive": True,
                "factor_group": "audit",
                "factor_count_grid": [count],
            }
        },
        "output_settings": {
            "weights_output_dir": str(output / "weights"),
            "diagnostics_output_dir": str(output / "diagnostics"),
        },
        "d9_capacity_audit": {
            "enabled": True,
            "frozen_protocol_path": str(protocol),
            "factor_snapshot_path": str(snapshot),
            "expected_snapshot_sha256": hashlib.sha256(
                snapshot.read_bytes()
            ).hexdigest(),
            "expected_factor_count": count,
            "anchor_factor_id": "Anchor",
            "max_allowed_date": "2023-12-31",
            "persist_window_audit": True,
        },
    }
    return _write(tmp_path / "config.json", config), config


def _model_evidence(context: dict, tmp_path: Path) -> dict:
    models = []
    audits = []
    for index in range(12):
        year = 2018 + index // 2
        month = 1 if index % 2 == 0 else 7
        start = f"{year:04d}-{month:02d}-01"
        end = f"{year:04d}-{month:02d}-28"
        gamma = 0.25 if index % 2 else 0.0
        models.append(
            {
                "weights": {"Factor0": 0.5, "Factor1": -0.2} if gamma else {},
                "selection": {
                    "train_start": "2016-01-01",
                    "train_end": start,
                    "validation_start": start,
                    "validation_end": end,
                    "selected_residual_gamma": gamma,
                    "cv_fallback_reason": None if gamma else "pure_size_won_inner_cv",
                    "residual_sleeve_n_selected": 2,
                    "residual_sleeve_coefficients": {"Factor0": 0.5, "Factor1": -0.2},
                },
            }
        )
        audits.append(
            {
                "train_start": "2016-01-01",
                "train_end": start,
                "status": "passed",
                "configured_factor_count": 3,
                "surviving_factor_count": 3,
            }
        )
    _write(Path(context["model_bundle_path"]), {"index": {"models": models}})
    _write(Path(context["training_audit_path"]), {"windows": audits})
    returns = tmp_path / "returns.csv"
    returns.write_text("date,net_return\n2020-01-01,0.1\n", encoding="utf-8")
    metadata = _write(
        tmp_path / "returns.metadata.json",
        {
            "start_date": "2020-01-01",
            "end_date": "2023-12-29",
            "sha256": hashlib.sha256(returns.read_bytes()).hexdigest(),
        },
    )
    artifact = {
        "artifact_type": "algorithm_return_series",
        "format": "csv",
        "path": str(returns),
        "metadata_path": str(metadata),
    }
    return {"backtest_summary": {"return_series_artifact": artifact}}


def test_preflight_is_count_driven_and_rejects_reorder(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    context = load_capacity_audit_context(config_path)
    assert context is not None
    assert preflight_capacity_audit(context)["factor_count"] == 3
    config["factor_groups"]["audit"]["factors"].reverse()
    _write(config_path, config)
    with pytest.raises(D9CapacityAuditError, match="preflight failed"):
        preflight_capacity_audit(load_capacity_audit_context(config_path))  # type: ignore[arg-type]


@pytest.mark.parametrize("mutation", ["hash", "count", "anchor", "date"])
def test_preflight_rejects_frozen_contract_mutations(
    tmp_path: Path, mutation: str
) -> None:
    config_path, config = _fixture(tmp_path)
    if mutation == "hash":
        config["d9_capacity_audit"]["expected_snapshot_sha256"] = "0" * 64
    elif mutation == "count":
        config["d9_capacity_audit"]["expected_factor_count"] = 4
    elif mutation == "anchor":
        config["d9_capacity_audit"]["anchor_factor_id"] = "Factor0"
    else:
        config["analysis_period"]["end_date"] = "2024-01-01"
    _write(config_path, config)
    with pytest.raises(D9CapacityAuditError) as raised:
        preflight_capacity_audit(load_capacity_audit_context(config_path))  # type: ignore[arg-type]
    assert raised.value.code == "d9_audit_invalid_config"


def test_wrapper_preflight_failure_does_not_run_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.agent_harness import cli

    config_path, config = _fixture(tmp_path)
    config["factor_groups"]["audit"]["factors"].reverse()
    _write(config_path, config)
    called = False

    def fail_if_called(**_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "_run_subprocess", fail_if_called)
    result = cli._handle_factor_synthesis(
        argparse.Namespace(
            config=str(config_path),
            output_root=str(tmp_path / "harness"),
            run_id="reordered",
            fake_result="",
            dry_run=False,
            json_out="",
            timeout_sec=30,
            return_series_output_dir="",
        )
    )
    assert result.exit_code == cli.EXIT_PRECONDITION
    assert result.reason == "d9_audit_invalid_config"
    assert called is False


def test_wrapper_runs_once_and_attaches_capacity_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.agent_harness import cli

    config_path, _ = _fixture(tmp_path)
    context = load_capacity_audit_context(config_path)
    assert context is not None
    metrics = _model_evidence(context, tmp_path)
    calls = 0

    def successful_run(**kwargs: object) -> cli.HarnessResult:
        nonlocal calls
        calls += 1
        return cli.HarnessResult(
            action="factor-synthesis",
            ok=True,
            exit_code=0,
            status="succeeded",
            reason="ok",
            mode="subprocess",
            output_dir=Path(kwargs["output_dir"]),
            command=list(kwargs["command"]),
            metrics=metrics,
            artifacts=[],
            warnings=[],
            run_id="success",
        )

    monkeypatch.setattr(cli, "_run_subprocess", successful_run)
    result = cli._handle_factor_synthesis(
        argparse.Namespace(
            config=str(config_path),
            output_root=str(tmp_path / "harness"),
            run_id="success",
            fake_result="",
            dry_run=False,
            json_out="",
            timeout_sec=30,
            return_series_output_dir="",
        )
    )
    assert calls == 1
    assert result.ok is True
    assert result.metrics["d9_capacity_audit"]["aggregate"]["window_count"] == 12
    assert result.artifacts[-1]["artifact_type"] == "d9_capacity_audit"
    assert Path(result.artifacts[-1]["path"]).is_file()


def test_build_normalizes_twelve_windows_and_writes_deterministically(
    tmp_path: Path,
) -> None:
    config_path, _ = _fixture(tmp_path)
    context = load_capacity_audit_context(config_path)
    assert context is not None
    preflight = preflight_capacity_audit(context)
    payload = build_capacity_audit(
        context, preflight, _model_evidence(context, tmp_path)
    )
    assert payload["aggregate"]["positive_gamma_window_count"] == 6
    assert payload["aggregate"]["median_active_nonzero_count"] == 2
    assert payload["aggregate"]["factor_selection_frequency"]["Factor0"] == 1
    assert payload["aggregate"]["coefficient_sign_stability"]["Factor1"] == 1
    assert payload["aggregate"]["fallback_counts"] == {
        "none": 6,
        "pure_size_won_inner_cv": 6,
    }
    assert payload["aggregate"]["pure_anchor_count"] == 6
    assert payload["windows"][0]["fitted_residual_nonzero_count"] == 2
    assert payload["windows"][0]["active_residual_nonzero_count"] == 0
    first = write_capacity_audit(tmp_path / "one.json", payload).read_bytes()
    second = write_capacity_audit(tmp_path / "two.json", payload).read_bytes()
    assert first == second
