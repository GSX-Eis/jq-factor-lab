"""Exercise the bundle compiler and virtual JoinQuant runtime with offline inputs."""
from __future__ import annotations

import ast
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import duckdb

from market_db_samples import declare_market_db
import pandas as pd
import pytest

from test_deploy_target_contracts_and_signal_export import _make_bundle
from factor_synthesis import StrategyFactorSpec, write_strategy_bundle
from factor_synthesis.strategy_bundle import sha256_file
from factor_synthesis.vectorbt_adapter.slot_provider import SlotDataProvider
from scripts.alignment.virtual_cloud_check import execute_bundle_candidate
from slots.jq_ashare_local import build_jq_ashare_local
from utils.time_axis import BarSchedule, Frequency
from targets import (
    TARGET_ALIGNMENT_MISMATCH,
    TARGET_ARTIFACT_INTEGRITY_FAILED,
    TARGET_AUX_FEATURE_UNSUPPORTED,
    TARGET_BUNDLE_INVALID,
    TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
    TARGET_SIGNAL_SOURCE_MISSING,
    TARGET_SLOT_INCOMPATIBLE,
    DeployTarget,
    DeployArtifactManifest,
    DeployTargetError,
    build_default_target_registry,
    verify_deploy_manifest,
)
from targets.jq_cloud import JqCloudTarget
from targets.jq_cloud.compiler import DUMPER_FILENAME, REQUIREMENTS_FILENAME, STRATEGY_FILENAME

ROOT = Path(__file__).resolve().parents[1]
STOCKS = ["000001.XSHE", "000002.XSHE", "600001.XSHG"]
FREEZE_PATHS = (
    "scripts/build/cloud_runtime/factor_base.py.txt",
    "scripts/build/cloud_runtime/jq_factor_adapter.py.txt",
)


@pytest.fixture
def cloud_bundle(tmp_path):
    return _build_cloud_bundle(tmp_path)


def _build_cloud_bundle(tmp_path, frequency="1d", *, aux_features=("close",)):
    bundle, sources = _make_bundle(
        tmp_path, slot_id="jq_ashare_local", frequency=frequency, allow_short=False,
        benchmark_instrument="000300.XSHG",
        rebalance_config={"selection_mode": "top_k", "top_k": 1},
    )
    factors = (
        replace(bundle.factors[0], source_module="fixture.alpha.Alpha", aux_features_required=aux_features),
        StrategyFactorSpec("beta", "4" * 64, frequency, source_module="fixture.beta.Beta"),
    )
    bundle = replace(bundle, factors=factors, bundle_hash="")
    model_path = sources["models/weights.json"]
    payload = json.loads(model_path.read_text())
    payload["index"]["models"][0]["standardize"] = False
    model_path.write_text(json.dumps(payload), encoding="utf-8")
    bundle = replace(
        bundle,
        model_artifacts=(replace(bundle.model_artifacts[0], sha256=sha256_file(model_path), size_bytes=model_path.stat().st_size),),
        bundle_hash="",
    )
    bundle_dir = write_strategy_bundle(bundle, tmp_path / "bundle", source_files=sources)
    files = {}
    for factor_id, name, field, operation in (
        ("alpha", "Alpha", "close", "* 2"),
        ("beta", "Beta", "open", "+ 1"),
    ):
        path = tmp_path / (name + ".py")
        path.write_text(
            "from utils.factor_base import FactorBase\n"
            f"class {name}(FactorBase):\n"
            f"    required_fields = [{field!r}]\n"
            "    def calculate_factor(self, frame):\n"
            f"        return frame.pivot(index='date', columns='code', values={field!r}) {operation}\n",
            encoding="utf-8",
        )
        files[factor_id] = path
    requirements = tmp_path / "requirements.json"
    requirements.write_text(json.dumps({"factors": {
        "Alpha": {"requirements": {"daily": {"min_days": 1, "fields": ["close"]}}},
        "Beta": {"requirements": {"daily": {"min_days": 1, "fields": ["open"]}}},
        "Unused": {"requirements": {"daily": {"min_days": 999, "fields": ["volume"]}}},
    }}), encoding="utf-8")
    frames = pd.DataFrame({
        "date": pd.Timestamp("2023-03-01"), "code": STOCKS,
        "close": [1.5, 1.0, float("nan")], "open": [0.0, 4.0, 6.0],
    })
    return bundle, bundle_dir, files, requirements, frames


@pytest.mark.parametrize("slot_id", ["cn_futures", "global_futures", "declarative_custom"])
def test_incompatible_market_slots(tmp_path, slot_id):
    bundle, _ = _make_bundle(
        tmp_path, slot_id=slot_id, frequency="1d", allow_short=False, benchmark_instrument=None,
    )
    report = JqCloudTarget().check_compatibility(bundle)
    assert not report.compatible
    assert report.issues[0].code == TARGET_SLOT_INCOMPATIBLE


def test_execution_and_auxiliary_feature_gates(tmp_path, cloud_bundle):
    target = JqCloudTarget()
    short_bundle, _ = _make_bundle(
        tmp_path, slot_id="jq_ashare_local", frequency="1d", allow_short=True, benchmark_instrument=None,
    )
    weekly_bundle = replace(short_bundle, allow_short=False,
                            execution_profile=dict(short_bundle.execution_profile, allow_short=False),
                            frequency="1w", bundle_hash="")
    bundle, _, files, _, _ = cloud_bundle
    unsupported_model = replace(bundle, synthesis_method="lightgbm", bundle_hash="")
    for candidate in (short_bundle, weekly_bundle, unsupported_model):
        assert target.check_compatibility(candidate).issues[0].code == TARGET_EXECUTION_CONTRACT_UNSUPPORTED

    alpha, beta = bundle.factors
    restricted = replace(bundle, factors=(
        replace(alpha, aux_features_required=("unmapped_volume", "unknown_alpha")),
        replace(beta, aux_features_required=("unknown_beta",)),
    ), bundle_hash="")
    report = target.check_compatibility(restricted, factor_files=files)
    assert not report.compatible
    assert {issue.code for issue in report.issues} == {TARGET_AUX_FEATURE_UNSUPPORTED}
    assert all(field in report.issues[0].message for field in ("alpha", "unknown_alpha", "unmapped_volume"))
    assert "beta" in report.issues[1].message
    assert "unknown_beta" in report.issues[1].message
    assert target.check_compatibility(bundle).issues[0].code == TARGET_BUNDLE_INVALID


def _export(target, fixture, output, **kwargs):
    bundle, bundle_dir, files, requirements, _ = fixture
    return target.export(
        bundle, output, bundle_dir=bundle_dir, factor_files=files,
        requirements_path=requirements, **kwargs,
    )




def _verify(target, fixture, artifact, **kwargs):
    bundle, bundle_dir, _, _, frames = fixture
    inputs = {
        "bundle_dir": bundle_dir,
        "trusted_manifest_hash": artifact.manifest_hash if isinstance(artifact, DeployArtifactManifest) else None,
        "virtual_cloud_inputs": {
            "stocks": STOCKS, "asof": "2023-03-01", "bars_by_frequency": {"daily": frames},
        },
        "local_factor_values": pd.DataFrame({
            "beta": [7.0, 5.0, 1.0], "alpha": [float("nan"), 2.0, 3.0],
        }, index=STOCKS[::-1]),
        "local_positions": {STOCKS[1]: 1.0},
    }
    inputs.update(kwargs)
    return target.verify_alignment(bundle, artifact, **inputs)


def test_compile_probe_and_virtual_alignment(tmp_path, cloud_bundle):
    target = build_default_target_registry().get("jq_cloud")
    assert isinstance(target, DeployTarget)
    bundle, bundle_dir, files, _, _ = cloud_bundle
    assert target.check_compatibility(bundle_dir, factor_files=list(files.values())).compatible
    output = tmp_path / "candidate"
    output.mkdir()
    manifest = _export(target, cloud_bundle, output)
    assert manifest.bundle_hash == bundle.bundle_hash
    assert manifest.slot_id == bundle.slot_id
    assert manifest.metadata["auto_deploy_blocked"] is True
    assert manifest.metadata["human_approval_required"] is True
    assert len(manifest.metadata["frozen_cloud_runtime_hashes"]) == 2
    assert set(path.name for path in output.iterdir()) == {
        STRATEGY_FILENAME, DUMPER_FILENAME, REQUIREMENTS_FILENAME, "deploy_manifest.json",
    }
    assert verify_deploy_manifest(output) == manifest
    for filename in (STRATEGY_FILENAME, DUMPER_FILENAME):
        code = (output / filename).read_text()
        ast.parse(code, feature_version=(3, 6))
        assert "STRATEGY_BUNDLE_METADATA" in code
        assert all(f"class {name}" in code for name in ("Alpha", "Beta"))
    assert "def dump_factor_values(" in (output / DUMPER_FILENAME).read_text()
    assert set(json.loads((output / REQUIREMENTS_FILENAME).read_text())["factors"]) == {"Alpha", "Beta"}

    modules = {name: sys.modules.get(name) for name in ("jqdata", "jqfactor", "factor_base", "jq_adapter")}
    report = _verify(target, cloud_bundle, manifest, write_report=True)
    assert report.passed, report.to_dict()
    assert report.max_abs_diff <= 1e-12
    assert report.metrics["factor_values_count"] == 6
    assert report.metrics["target_positions_count"] == 1
    assert {check.name for check in report.checks} >= {
        "python36_syntax", "bundle_imports", "legacy_pandas_api",
        "cloud_runtime_budget:Alpha", "cloud_runtime_budget:Beta",
        "virtual_cloud_load", "factor_values", "target_positions",
    }
    assert all(sys.modules.get(name) is old for name, old in modules.items())
    assert json.loads((output / "alignment_report.json").read_text()) == report.to_dict()
    with pytest.raises(DeployTargetError, match="empty output directory"):
        _export(target, cloud_bundle, output)


def test_virtual_price_api_uses_frozen_adapter(tmp_path, cloud_bundle):
    target = JqCloudTarget()
    manifest = _export(target, cloud_bundle, tmp_path / "candidate")
    frames = cloud_bundle[-1].assign(volume=100.0, factor=1.0)
    calls = []

    def get_price(**kwargs):
        calls.append(kwargs)
        codes = kwargs["security"]
        if isinstance(codes, str):
            codes = [codes]
        fields = kwargs.get("fields", ["open", "close", "volume", "factor"])
        return frames.loc[frames.code.isin(codes), ["date", "code", *fields]]

    report = _verify(
        target, cloud_bundle, manifest,
        virtual_cloud_inputs={"stocks": STOCKS, "asof": "2023-03-01", "api": {"get_price": get_price}},
    )
    assert report.passed, report.to_dict()
    assert any(call.get("fq") == "pre" and set(call["fields"]) >= {"open", "close"} for call in calls)


def _minute_requirements(fixture, frequency, bars):
    path = fixture[3]
    config = json.loads(path.read_text())
    for name, field in (("Alpha", "close"), ("Beta", "open")):
        config["factors"][name]["requirements"] = {
            frequency: {"min_bars": bars, "fields": [field]},
        }
    path.write_text(json.dumps(config))


def _minute_session(day):
    return pd.date_range(day + " 09:45", periods=8, freq="15min").append(
        pd.date_range(day + " 13:15", periods=8, freq="15min")
    )


def _price_api(frames):
    def get_price(*, security, end_date, frequency, fields=None, count=None, **kwargs):
        frame = frames[frequency]
        codes = [security] if isinstance(security, str) else security
        data = frame.loc[frame.code.isin(codes) & (frame.date <= pd.Timestamp(end_date))]
        columns = ["date", "code", *fields] if fields is not None else list(data.columns)
        return data.sort_values("date").groupby("code").tail(count)[columns]

    return get_price


def test_incomplete_minute_session_rejected_by_both_price_inputs(tmp_path, cloud_bundle):
    target = JqCloudTarget()
    _minute_requirements(cloud_bundle, "5m", 1)
    manifest = _export(target, cloud_bundle, tmp_path / "candidate")
    frames = {"5m": cloud_bundle[-1].assign(date=pd.Timestamp("2023-03-01 09:45"))}
    for provider in ({"bars_by_frequency": frames}, {"api": {"get_price": _price_api(frames)}}):
        report = _verify(target, cloud_bundle, manifest, virtual_cloud_inputs={
            "stocks": STOCKS, "asof": "2023-03-01", **provider,
        })
        assert not report.passed
        assert "factor_input_empty" in report.checks[-1].reason


def test_full_15_minute_session_initializes_and_runs_daily_factor(tmp_path, cloud_bundle):
    target = JqCloudTarget()
    _minute_requirements(cloud_bundle, "15m", 16)
    manifest = _export(target, cloud_bundle, tmp_path / "candidate")
    bars = pd.concat([cloud_bundle[-1].assign(date=ts) for ts in _minute_session("2023-03-01")])
    report = _verify(target, cloud_bundle, manifest, virtual_cloud_inputs={
        "stocks": STOCKS, "asof": "2023-03-01", "bars_by_frequency": {"15m": bars},
    })
    assert report.passed, report.to_dict()


def test_minute_factor_frequency_plan_mismatch_rejected_before_export(tmp_path):
    fixture = _build_cloud_bundle(tmp_path, "15min")
    _minute_requirements(fixture, "5m", 48)
    bundle, _, files, requirements, _ = fixture
    target = JqCloudTarget()
    report = target.check_compatibility(bundle, factor_files=files, requirements_path=requirements)
    assert not report.compatible
    assert report.issues[0].code == TARGET_EXECUTION_CONTRACT_UNSUPPORTED
    with pytest.raises(DeployTargetError) as error:
        _export(target, fixture, tmp_path / "candidate")
    assert error.value.code == TARGET_EXECUTION_CONTRACT_UNSUPPORTED
    assert not (tmp_path / "candidate").exists()


def test_minute_signal_and_callback_use_available_slot_bar(tmp_path):
    frequency = Frequency.parse("5min")
    fixture = _build_cloud_bundle(tmp_path, frequency.canonical)
    _minute_requirements(fixture, frequency.joinquant, 48)
    target = JqCloudTarget()
    output = tmp_path / "candidate"
    manifest = _export(target, fixture, output)
    previous_day, day = "2023-02-28", "2023-03-01"

    with duckdb.connect(":memory:") as conn:
        declare_market_db(conn, 'cn_equity')
        slot = build_jq_ashare_local(conn, start=previous_day, end=day)
        grid = slot.calendar.bar_grid(frequency, previous_day, day)
        schedule = BarSchedule(slot.calendar, frequency)
        available = pd.DatetimeIndex([schedule.available_at(key) for key in grid.timestamp_utc])
        local_ends = available.tz_convert("Asia/Shanghai").tz_localize(None)
        winners = {"09:35": 0, "09:55": 0, "10:00": 1, "10:05": 2, "11:30": 1}
        bars = pd.concat([
            fixture[-1].assign(
                date=end, close=[10.0 if i == winners.get(end.strftime("%H:%M"), 2) else 1.0 for i in range(3)],
                open=1.0, high=10.0, low=1.0, volume=1.0, factor=1.0,
            )
            for end in local_ends
        ], ignore_index=True)
        conn.register("fixture_prices", bars)
        conn.execute("CREATE TABLE price_data_5m AS SELECT code AS stock_code, * EXCLUDE (code) FROM fixture_prices")
        frames = {frequency.joinquant: bars}
        utc_frames = {frequency.joinquant: bars.assign(date=bars.date.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC"))}
        namespace = execute_bundle_candidate(
            (output / STRATEGY_FILENAME).read_text(), STRATEGY_FILENAME, {"get_price": _price_api(frames)},
        )
        context = SimpleNamespace(
            previous_date=pd.Timestamp(previous_day).date(),
            strategy_universe=STOCKS, strategy_rebalance_due=True,
        )
        namespace["initialize"](context)

        for decision, expected_open, winner in (
            ("09:30", "2023-02-28T06:55:00+00:00", STOCKS[2]),
            ("09:35", "2023-03-01T01:30:00+00:00", STOCKS[0]),
            ("10:00", "2023-03-01T01:55:00+00:00", STOCKS[1]),
            ("10:05", "2023-03-01T02:00:00+00:00", STOCKS[2]),
            ("13:00", "2023-03-01T03:25:00+00:00", STOCKS[1]),
        ):
            local_decision = pd.Timestamp(day + " " + decision, tz="Asia/Shanghai")
            decision_utc = local_decision.tz_convert("UTC")
            signal_open = grid.timestamp_utc[available <= decision_utc].iloc[-1]
            key = SlotDataProvider.bar_key(signal_open, daily_naive=False)
            assert key == expected_open
            local = slot.bar_feed.read_signal_bars(
                STOCKS, frequency, start_utc=signal_open, end_utc=signal_open, as_of_utc=decision_utc,
            ).set_index("instrument_id")
            factors = pd.DataFrame({"alpha": local.close * 2, "beta": local.open + 1})
            scores = factors.alpha * 0.6 + factors.beta * 0.4
            assert scores.idxmax() == winner
            expected = {winner: 1.0}
            report = _verify(target, fixture, manifest, virtual_cloud_inputs={
                "stocks": STOCKS, "asof": key, "bars_by_frequency": utc_frames,
            }, local_factor_values=factors, local_positions=expected)
            assert report.passed, report.to_dict()
            context.current_dt = local_decision.tz_localize(None)
            namespace["handle_data"](context, None)
            assert context.strategy_target_weights == expected

    missing = (bars.date == pd.Timestamp(day + " 09:35")) & (bars.code == STOCKS[0])
    report = _verify(target, fixture, manifest, virtual_cloud_inputs={
        "stocks": STOCKS, "asof": day + "T01:55:00Z",
        "bars_by_frequency": {frequency.joinquant: bars.loc[~missing]},
    })
    assert not report.passed
    assert "factor_input_incomplete_current_session" in report.checks[-1].reason


def test_preflight_rejects_aux_unmapped_at_actual_input_frequency(tmp_path):
    fixture = _build_cloud_bundle(tmp_path, aux_features=("close", "style_factor_exposure.beta"))
    _minute_requirements(fixture, "15m", 16)
    bundle, _, files, requirements, _ = fixture
    target = JqCloudTarget()
    report = target.check_compatibility(bundle, factor_files=files, requirements_path=requirements)
    assert not report.compatible
    issue = report.issues[0]
    assert issue.code == TARGET_AUX_FEATURE_UNSUPPORTED
    assert "style_factor_exposure.beta" in issue.message
    assert "15m" in issue.message
    with pytest.raises(DeployTargetError) as error:
        _export(target, fixture, tmp_path / "candidate")
    assert error.value.code == issue.code
    assert str(error.value) == f"[{issue.code}] {issue.message}"


def test_qualified_aux_field_reaches_slice_and_virtual_api(tmp_path):
    fixture = _build_cloud_bundle(tmp_path, aux_features=("close", "style_factor_exposure.beta"))
    bundle, _, files, _, bars = fixture
    target = JqCloudTarget()
    assert target.check_compatibility(bundle, factor_files=files).compatible
    source = files["alpha"]
    source.write_text(source.read_text().replace(
        "        return frame.pivot", "        assert 'FACTOR_CNE5_BETA' in frame.columns\n        return frame.pivot",
    ))
    output = tmp_path / "candidate"
    manifest = _export(target, fixture, output)
    requirements = json.loads((output / REQUIREMENTS_FILENAME).read_text())
    assert requirements["factors"]["Alpha"]["requirements"]["daily"]["fields"] == ["FACTOR_CNE5_BETA", "close"]
    calls = []

    def get_factor_values(*, securities, factors, start_date, end_date):
        calls.append(factors)
        return {"beta": pd.DataFrame(1.0, index=[pd.Timestamp("2023-03-01")], columns=securities)}

    report = _verify(target, fixture, manifest, virtual_cloud_inputs={
        "stocks": STOCKS, "asof": "2023-03-01", "bars_by_frequency": {"daily": bars},
        "api": {"get_factor_values": get_factor_values},
    })
    assert report.passed, report.to_dict()
    assert calls == [["beta"]]


@pytest.mark.parametrize("failure", ["factor", "position", "sha256", "keys", "missing", "symlink", "binding", "asof"])
def test_alignment_fails_closed(tmp_path, cloud_bundle, failure):
    target = JqCloudTarget()
    output = tmp_path / "candidate"
    manifest = _export(target, cloud_bundle, output, generate_dumper=False)
    expected_code = TARGET_ALIGNMENT_MISMATCH
    kwargs = {"trusted_manifest_hash": manifest.manifest_hash}
    if failure == "factor":
        kwargs["local_factor_values"] = pd.DataFrame({
            "alpha": [3 + 2e-12, 2.0, float("nan")], "beta": [1.0, 5.0, 7.0],
        }, index=STOCKS)
    elif failure == "position":
        kwargs["local_positions"] = {STOCKS[1]: 0.99}
    elif failure == "keys":
        kwargs["local_positions"] = {STOCKS[0]: 1.0}
    elif failure == "missing":
        kwargs["local_positions"] = None
        expected_code = TARGET_SIGNAL_SOURCE_MISSING
    elif failure == "asof":
        expected_code = TARGET_SIGNAL_SOURCE_MISSING
        kwargs["virtual_cloud_inputs"] = {
            "stocks": STOCKS, "asof": "today", "bars_by_frequency": {"daily": cloud_bundle[-1]},
        }
    else:
        expected_code = TARGET_ARTIFACT_INTEGRITY_FAILED
        path = output / STRATEGY_FILENAME
        if failure == "sha256":
            path.write_text("# changed\n")
        elif failure == "symlink":
            outside = tmp_path / "outside.py"
            path.replace(outside)
            path.symlink_to(outside)
        else:
            manifest_path = output / "deploy_manifest.json"
            manifest_path.write_text(replace(manifest, bundle_hash="a" * 64, manifest_hash="").to_json())
    report = _verify(target, cloud_bundle, output, **kwargs)
    assert not report.passed
    assert report.error_code == expected_code, report.to_dict()
    if failure in {"factor", "position"}:
        assert report.max_abs_diff > 1e-12
    if failure == "asof":
        assert "ISO date" in report.checks[-1].reason


def test_alignment_requires_trusted_export_and_references_before_loading(tmp_path, cloud_bundle, monkeypatch):
    target = JqCloudTarget()
    output = tmp_path / "candidate"
    manifest = _export(target, cloud_bundle, output, generate_dumper=False)

    def unexpected_load(*args, **kwargs):
        pytest.fail("Candidate reached the virtual runtime without trusted provenance and inputs")

    monkeypatch.setattr("targets.jq_cloud.verifier.execute_bundle_candidate", unexpected_load)
    missing = _verify(target, cloud_bundle, manifest, local_positions=None)
    assert missing.error_code == TARGET_SIGNAL_SOURCE_MISSING
    untrusted = _verify(target, cloud_bundle, output)
    assert untrusted.checks[-1].name == "trusted_compilation"
    assert untrusted.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED

    strategy = output / STRATEGY_FILENAME
    strategy.write_text(strategy.read_text() + "\n# Alternate candidate\n")
    entries = tuple(
        replace(item, sha256=sha256_file(strategy), size_bytes=strategy.stat().st_size)
        if item.path == STRATEGY_FILENAME else item
        for item in manifest.artifacts
    )
    rebuilt = replace(manifest, artifacts=entries, manifest_hash="")
    (output / "deploy_manifest.json").write_text(rebuilt.to_json())
    rejected = _verify(target, cloud_bundle, output, trusted_manifest_hash=manifest.manifest_hash)
    assert rejected.checks[-1].name == "trusted_compilation"
    assert rejected.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED


def test_budget_preflight_uses_runtime_window(tmp_path, cloud_bundle):
    target = JqCloudTarget()
    bundle, bundle_dir, files, requirements, _ = cloud_bundle
    config = json.loads(requirements.read_text())
    config["factors"]["Alpha"] = {
        "requirements": {"5m": {"min_bars": 100, "fields": ["close"]}},
        "runtime": {"window": 5001},
    }
    requirements.write_text(json.dumps(config))
    manifest = _export(target, cloud_bundle, tmp_path / "candidate")
    report = _verify(target, cloud_bundle, manifest)
    assert report.error_code == TARGET_EXECUTION_CONTRACT_UNSUPPORTED, report.to_dict()
    assert any("cloud_runtime_budget_risk:5m:bars=5040" in warning for warning in report.warnings)
    assert not any(check.name == "virtual_cloud_load" for check in report.checks)


@pytest.mark.parametrize(("window", "request_count"), [(4992, 5040), (4993, 5088)])
def test_budget_preflight_counts_partial_session_and_rounding(tmp_path, window, request_count):
    fixture = _build_cloud_bundle(tmp_path, "5min")
    _minute_requirements(fixture, "5m", window)
    target = JqCloudTarget()
    output = tmp_path / "candidate"
    manifest = _export(target, fixture, output)
    bars = fixture[-1].assign(date=pd.Timestamp("2023-03-01 09:35"))
    get_price = _price_api({"5m": bars})
    calls = []

    def record_price(**kwargs):
        calls.append(kwargs["count"])
        return get_price(**kwargs)

    report = _verify(target, fixture, manifest, virtual_cloud_inputs={
        "stocks": STOCKS, "asof": "2023-03-01 09:35", "api": {"get_price": record_price},
    })
    assert report.error_code == TARGET_EXECUTION_CONTRACT_UNSUPPORTED, (report.to_dict(), calls)
    assert f"Alpha:cloud_runtime_budget_risk:5m:bars={request_count}" in report.warnings
    candidate = execute_bundle_candidate(
        (output / STRATEGY_FILENAME).read_text(), STRATEGY_FILENAME, {"get_price": record_price},
    )
    with pytest.raises(ValueError, match=f"cloud_runtime_budget_risk:5m:bars={request_count}"):
        candidate["compute_bundle_signal"](STOCKS, "2023-03-01 09:35")
    assert not calls


def test_target_preserves_frozen_files_and_has_no_upload_imports(tmp_path, cloud_bundle):
    contents = {path: (ROOT / path).read_bytes() for path in FREEZE_PATHS}
    for protected_dir in (ROOT / "scripts/build/cloud_runtime", ROOT / "backtest_and_analysis/evaluation/calibration_data"):
        with pytest.raises(DeployTargetError, match="protected input"):
            _export(JqCloudTarget(), cloud_bundle, protected_dir)
    _export(JqCloudTarget(), cloud_bundle, tmp_path / "candidate")
    assert contents == {path: (ROOT / path).read_bytes() for path in FREEZE_PATHS}
    for path in (ROOT / "targets/jq_cloud").glob("*.py"):
        source = path.read_text()
        assert "jq_api_upload_run" not in source
        assert "manual_cloud_deploy" not in source
