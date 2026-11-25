"""Tests for automatic causal verification gate and aux_features revision checks."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from slots import build_slot_from_config
from slots.cn_futures import build_cn_futures_slot
from test_slot_futures import _populate
from test_slot_pit_replay_fast_path import _build_minute_declarative_slot
from tools.agent_harness import cli as harness_cli
from tools.agent_harness.causal_gate import (
    CAUSAL_GATE_SCHEMA_VERSION,
    validate_causal_gate_evidence,
    verify_factor_causality,
)
from tools.agent_harness.run_result import execution_result, read_execution_result, validate_execution_result_payload
from utils.bars.futures_roll import build_roll_schedule
from utils.factor_base import FactorBase
from utils.factor_core.slot_execution_mode import (
    IncrementalRollingSpec,
    fast_path_request_status,
    resolve_slot_execution_mode,
    slot_has_historical_revision,
)
from utils.slots import ExecutionProfile, ProviderConfig, load_slot_config
from utils.time_axis import BarSchedule
from utils.session_calendar import SessionCalendar

import duckdb


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "declarative_crypto"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_dense_declarative_slot(
    tmp_path: Path,
    *,
    days: tuple[str, str] = ("2026-01-05", "2026-01-06"),
    frequency: str = "60min",
    instruments: tuple[str, ...] = ("BTC-USD", "ETH-USD"),
    with_aux: bool = False,
) -> tuple[object, Path]:
    config = load_slot_config(EXAMPLE / "slot.yaml")
    cal = SessionCalendar(
        "crypto_utc",
        days[0],
        days[1],
        custom_sessions=config.calendar.options["custom_sessions"],
    )
    sched = BarSchedule(cal, frequency, view="ETH", bucket_strategy="segment_anchored")
    rows = []
    for inst_idx, inst in enumerate(instruments):
        step = 0
        for day in cal.sessions():
            slots = sched.for_session(day.strftime("%Y-%m-%d"))
            for ts in slots["start_utc"]:
                step += 1
                base = 100.0 + 15.0 * inst_idx + 0.4 * step + np.sin(step * 0.25)
                rows.append(
                    {
                        "time": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "instrument_id": inst,
                        "open": base,
                        "high": base + 1.0,
                        "low": base - 1.0,
                        "close": base + 0.2,
                        "volume": 50.0 + (step % 7) * 5.0,
                    }
                )
    data_dir = tmp_path / "slot_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(data_dir / "bars.csv", index=False)
    if with_aux:
        pd.read_csv(EXAMPLE / "funding.csv").to_csv(data_dir / "funding.csv", index=False)

    universe_opts = dict(config.universe.options)
    universe_opts["instruments"] = {
        inst: {
            "asset_class": "crypto",
            "currency": "USD",
            "lot_size": 1,
            "contract_multiplier": 1.0,
        }
        for inst in instruments
    }
    universe_opts["universes"] = {"spot": list(instruments)}
    bar_opts = {
        **config.bar_feed.options,
        "frequency": frequency,
        "timestamp_anchor": "bar_start",
        "column_mapping": {
            "time": "timestamp",
            "instrument_id": "instrument_id",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        },
        "paths": ["bars.csv"],
    }
    bar_opts.pop("instrument_id", None)
    aux_cfg = (
        config.aux_features
        if with_aux
        else ProviderConfig("none", {})
    )
    new_cfg = replace(
        config,
        calendar=replace(
            config.calendar,
            options={**config.calendar.options, "start": days[0], "end": days[1]},
        ),
        bar_feed=ProviderConfig("csv_files", bar_opts),
        universe=ProviderConfig("static", universe_opts),
        aux_features=aux_cfg,
        execution_profile=ExecutionProfile(
            "open", "close", 1, -1, "decimal", "endpoints_only", "cross_session"
        ),
    )
    slot_yaml = data_dir / "slot.json"
    slot_yaml.write_text(json.dumps(new_cfg.as_dict(), indent=2), encoding="utf-8")
    slot = build_slot_from_config(new_cfg, base_dir=data_dir)
    slot = replace(slot, slot_data_fingerprint="dense-fp-v1")
    return slot, slot_yaml


class _CausalVecFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        vals = self.session_rolling(data, "close", 4, min_periods=2, within_session=False)
        return data[["timestamp_utc", "instrument_id"]].assign(factor_value=vals)

    def _get_cache_inputs(self) -> dict:
        return {"kind": "causal_rolling_4"}


class _CausalBuiltinIncFactor(FactorBase):
    CAUSAL_VECTORIZED = False
    INCREMENTAL_ENGINE = IncrementalRollingSpec(
        column="close", window=4, min_periods=2, within_session=False
    )

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        vals = self.session_rolling(data, "close", 4, min_periods=2, within_session=False)
        return data[["timestamp_utc", "instrument_id"]].assign(factor_value=vals)

    def _get_cache_inputs(self) -> dict:
        return {"kind": "causal_inc_4"}


class _ShiftFutureLookaheadFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        ordered = data.sort_values(["instrument_id", "timestamp_utc"])
        vals = ordered.groupby("instrument_id", sort=False)["close"].shift(-1)
        return ordered[["timestamp_utc", "instrument_id"]].assign(
            factor_value=vals.reindex(data.index)
        )


class _CenterRollingLookaheadFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        ordered = data.sort_values(["instrument_id", "timestamp_utc"])
        vals = ordered.groupby("instrument_id", sort=False)["close"].transform(
            lambda s: s.rolling(5, min_periods=1, center=True).mean()
        )
        return ordered[["timestamp_utc", "instrument_id"]].assign(
            factor_value=vals.reindex(data.index)
        )


class _BfillLookaheadFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        ordered = data.sort_values(["instrument_id", "timestamp_utc"])
        vals = ordered.groupby("instrument_id", sort=False)["close"].transform(
            lambda s: s.bfill()
        )
        return ordered[["timestamp_utc", "instrument_id"]].assign(
            factor_value=vals.reindex(data.index)
        )


class _FullSampleZScoreLookaheadFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        ordered = data.sort_values(["instrument_id", "timestamp_utc"])
        vals = ordered.groupby("instrument_id", sort=False)["close"].transform(
            lambda s: (s - s.mean()) / s.std()
        )
        return ordered[["timestamp_utc", "instrument_id"]].assign(
            factor_value=vals.reindex(data.index)
        )


def test_aux_features_historical_revision_detection(tmp_path: Path) -> None:
    # 1. Built-in 'none' aux_features and DeclarativeAuxFeatures must NOT be falsely downgraded.
    slot_none, _ = _build_dense_declarative_slot(tmp_path / "none", with_aux=False)
    slot_decl_aux, _ = _build_dense_declarative_slot(tmp_path / "decl", with_aux=True)

    for label, slot in [("none", slot_none), ("declarative_aux", slot_decl_aux)]:
        f = _CausalVecFactor(
            None, f"vec_{label}", "2026-01-05", "2026-01-06",
            stock_pool=["BTC-USD", "ETH-USD"], slot=slot,
            data_frequency="60min", analysis_mode="time_series",
        )
        has_rev, reason = slot_has_historical_revision(f)
        assert has_rev is False, f"{label} unexpectedly flagged: {reason}"
        mode, mode_reason = resolve_slot_execution_mode(f, FactorBase)
        assert mode == "vectorized"
        assert mode_reason == "auto_causal_vectorized"

    # 2. Custom aux_features without has_historical_revision=False must fall back to pit_replay.
    class _UnverifiedCustomAux:
        def read_aux_features(self, grid: pd.DataFrame, *, features):
            return grid[["instrument_id", "timestamp_utc", "session_id"]].assign(
                available_at_utc=grid["timestamp_utc"],
                **{feat: 1.0 for feat in features},
            )

    slot_custom_unverified = replace(slot_none, aux_features=_UnverifiedCustomAux())
    for req_mode in ("auto", "vectorized"):
        f_unv = _CausalVecFactor(
            None, "unv_aux", "2026-01-05", "2026-01-06",
            stock_pool=["BTC-USD", "ETH-USD"], slot=slot_custom_unverified,
            data_frequency="60min", analysis_mode="time_series",
            slot_execution_mode=req_mode,
        )
        has_rev, rev_reason = slot_has_historical_revision(f_unv)
        assert has_rev is True
        assert rev_reason == "unverified_aux_features"
        mode, mode_reason = resolve_slot_execution_mode(f_unv, FactorBase)
        assert mode == "pit_replay"
        assert mode_reason == "fallback_pit_replay:unverified_aux_features"
        f_unv.get_factor_values()
        assert f_unv.last_slot_execution_mode == "pit_replay"

    f_inc_unv = _CausalBuiltinIncFactor(
        None, "inc_unv_aux", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"], slot=slot_custom_unverified,
        data_frequency="60min", analysis_mode="time_series",
        slot_execution_mode="incremental",
    )
    assert resolve_slot_execution_mode(f_inc_unv, FactorBase) == (
        "pit_replay",
        "fallback_pit_replay:unverified_aux_features",
    )

    # 3. aux_features with explicit has_historical_revision=True must fall back to pit_replay.
    class _RevisingAux(_UnverifiedCustomAux):
        has_historical_revision = True

    slot_revising = replace(slot_none, aux_features=_RevisingAux())
    f_rev = _CausalVecFactor(
        None, "rev_aux", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"], slot=slot_revising,
        data_frequency="60min", analysis_mode="time_series",
        slot_execution_mode="vectorized",
    )
    assert slot_has_historical_revision(f_rev) == (True, "aux_features_has_historical_revision")
    assert resolve_slot_execution_mode(f_rev, FactorBase) == (
        "pit_replay",
        "fallback_pit_replay:aux_features_has_historical_revision",
    )
    gate_rev = verify_factor_causality(f_rev)
    assert gate_rev.ok is True
    assert gate_rev.causal_passed is None
    assert gate_rev.status == "blocked_historical_revision"
    assert gate_rev.resolved_mode == "pit_replay"
    assert gate_rev.fast_path_allowed is False
    validate_causal_gate_evidence(gate_rev.to_dict())

    # 4. Custom aux_features with explicit has_historical_revision=False is allowed.
    class _StaticCustomAux(_UnverifiedCustomAux):
        has_historical_revision = False

    slot_static_aux = replace(slot_none, aux_features=_StaticCustomAux())
    f_static = _CausalVecFactor(
        None, "static_aux", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"], slot=slot_static_aux,
        data_frequency="60min", analysis_mode="time_series",
        slot_execution_mode="vectorized",
    )
    assert slot_has_historical_revision(f_static) == (False, "immutable_no_revision")
    assert resolve_slot_execution_mode(f_static, FactorBase) == ("vectorized", "explicit_vectorized")

    # 5. Native futures slot (adjustment='none') uses _FuturesData as aux_features and must NOT regress.
    with duckdb.connect(config={"threads": 1}) as conn:
        _populate(conn, root="CU", exchange="SHFE", start="2026-01-05", end="2026-01-09", expiries=["2026-02", "2026-03"])
        build_roll_schedule(conn, product_roots=["CU"], start="2026-01-06", end="2026-01-09")
        fut_slot = build_cn_futures_slot(conn, product_root="CU", start="2026-01-05", end="2026-01-09", adjustment="none")
        f_fut = _CausalVecFactor(
            None, "fut_static", "2026-01-06", "2026-01-09",
            stock_pool=["CU"], slot=fut_slot,
            data_frequency="1d", analysis_mode="time_series",
        )
        assert slot_has_historical_revision(f_fut) == (False, "immutable_no_revision")


def test_causal_vectorized_repositioned_as_request_and_gate_enforced(tmp_path: Path) -> None:
    slot, _ = _build_dense_declarative_slot(tmp_path)
    factor = _CausalVecFactor(
        None, "req_only", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"], slot=slot,
        data_frequency="60min", analysis_mode="time_series",
    )
    assert fast_path_request_status(factor, FactorBase) == (True, "vectorized", "auto_vectorized_requested")
    # Strict verification requirement blocks unverified CAUSAL_VECTORIZED=True declarations.
    assert resolve_slot_execution_mode(
        factor, FactorBase, require_causal_verification=True
    ) == ("pit_replay", "fallback_pit_replay:causal_verification_required")
    # Failed gate blocks fast path even without require_causal_verification.
    assert resolve_slot_execution_mode(
        factor, FactorBase, causal_verified=False
    ) == ("pit_replay", "fallback_pit_replay:causal_gate_failed")

    gate = verify_factor_causality(factor, n_truncations=3, n_pit_samples=4, seed=7)
    assert gate.ok is True
    assert gate.causal_passed is True
    assert gate.status == "passed"
    assert gate.fast_path_requested is True
    assert gate.fast_path_allowed is True
    assert gate.resolved_mode == "vectorized"
    assert gate.nan_mismatch_count == 0
    assert gate.failure_reason is None
    assert len(gate.truncation_cutoffs) == 3
    assert len(gate.pit_sample_cutoffs) == 4
    validate_causal_gate_evidence(gate.to_dict())

    factor._require_causal_verification = True
    assert resolve_slot_execution_mode(factor, FactorBase) == ("vectorized", "auto_causal_vectorized")
    vec_df = factor.get_factor_values()
    assert factor.last_slot_execution_mode == "vectorized"
    pit_factor = _CausalVecFactor(
        None, "pit_ref", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"], slot=slot,
        data_frequency="60min", analysis_mode="time_series",
        slot_execution_mode="pit_replay",
    )
    pd.testing.assert_frame_equal(vec_df, pit_factor.get_factor_values(), check_exact=False, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("factor_cls", "dense"),
    [
        (_ShiftFutureLookaheadFactor, False),
        (_CenterRollingLookaheadFactor, False),
        (_BfillLookaheadFactor, False),
        (_BfillLookaheadFactor, True),
        (_FullSampleZScoreLookaheadFactor, False),
    ],
    ids=[
        "shift_minus_1",
        "rolling_center_true",
        "bfill_with_nans",
        "bfill_dense_nan_probe",
        "full_sample_zscore",
    ],
)
def test_lookahead_factors_with_wrong_causal_declaration_are_blocked(
    tmp_path: Path, factor_cls: type[FactorBase], dense: bool
) -> None:
    if dense:
        slot, _ = _build_dense_declarative_slot(tmp_path)
    else:
        slot = _build_minute_declarative_slot(
            tmp_path, days=("2026-01-05", "2026-01-06"), frequency="60min", with_gaps_and_nans=True
        )

    # Reject policy (default)
    bad_reject = factor_cls(
        None, "bad_reject", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"], slot=slot,
        data_frequency="60min", analysis_mode="time_series",
        slot_execution_mode="vectorized",
    )
    gate_reject = verify_factor_causality(bad_reject, n_truncations=3, n_pit_samples=4, seed=11, on_failure="reject")
    assert gate_reject.ok is False
    assert gate_reject.causal_passed is False
    assert gate_reject.status == "failed_rejected"
    assert gate_reject.fast_path_requested is True
    assert gate_reject.fast_path_allowed is False
    assert gate_reject.resolved_mode == "pit_replay"
    assert gate_reject.failure_reason is not None
    assert len(gate_reject.truncation_cutoffs) >= 1
    assert len(gate_reject.mismatched_samples) >= 1
    assert (gate_reject.nan_mismatch_count > 0) or (
        gate_reject.max_abs_diff is not None and gate_reject.max_abs_diff > 1e-12
    )
    validate_causal_gate_evidence(gate_reject.to_dict())
    # Applying gate result permanently blocks fast path on that factor instance.
    assert resolve_slot_execution_mode(bad_reject, FactorBase) == (
        "pit_replay",
        "fallback_pit_replay:causal_gate_failed",
    )

    # Fallback policy downgrades safely to pit_replay.
    bad_fallback = factor_cls(
        None, "bad_fallback", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"], slot=slot,
        data_frequency="60min", analysis_mode="time_series",
        slot_execution_mode="auto",
    )
    gate_fb = verify_factor_causality(bad_fallback, n_truncations=3, n_pit_samples=4, seed=11, on_failure="fallback")
    assert gate_fb.ok is True
    assert gate_fb.causal_passed is False
    assert gate_fb.status == "failed_fallback"
    assert gate_fb.fast_path_allowed is False
    assert gate_fb.resolved_mode == "pit_replay"
    assert bad_fallback.slot_execution_mode == "pit_replay"
    bad_fallback.get_factor_values()
    assert bad_fallback.last_slot_execution_mode == "pit_replay"


def test_incremental_and_panel_slot_fast_paths_verified_by_gate(tmp_path: Path) -> None:
    slot = _build_minute_declarative_slot(
        tmp_path, days=("2026-01-05", "2026-01-06"), frequency="60min", with_gaps_and_nans=True
    )
    inc_factor = _CausalBuiltinIncFactor(
        None, "inc_ok", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"], slot=slot,
        data_frequency="60min", analysis_mode="time_series",
    )
    inc_gate = verify_factor_causality(inc_factor, n_truncations=3, n_pit_samples=3, seed=5)
    assert inc_gate.ok is True
    assert inc_gate.causal_passed is True
    assert inc_gate.resolved_mode == "incremental"
    assert inc_gate.nan_mismatch_count == 0

    class _CausalPanelFactor(FactorBase):
        CAUSAL_VECTORIZED = True

        def _get_raw_data_for_factor(self):
            return self._fetch_raw_data(["close", "volume"], table_name="price_data")

        def calculate_factor(self, data):
            return data.assign(val=data["close"] * data["volume"]).pivot(
                index="date", columns="code", values="val"
            )

    class _LookaheadPanelFactor(_CausalPanelFactor):
        def calculate_factor(self, data):
            wide = super().calculate_factor(data)
            return (wide - wide.mean(axis=0)) / wide.std(axis=0)

    good_panel = _CausalPanelFactor(
        None, "panel_ok", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD"], slot=slot, data_frequency="60min", analysis_mode=None,
    )
    bad_panel = _LookaheadPanelFactor(
        None, "panel_bad", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD"], slot=slot, data_frequency="60min", analysis_mode=None,
    )
    assert verify_factor_causality(good_panel, seed=3).causal_passed is True
    bad_panel_gate = verify_factor_causality(bad_panel, seed=3)
    assert bad_panel_gate.ok is False
    assert bad_panel_gate.causal_passed is False
    assert bad_panel_gate.resolved_mode == "pit_replay"


def _write_factor_module(path: Path, body: str) -> Path:
    path.write_text(
        "import numpy as np\nimport pandas as pd\nfrom utils.factor_base import FactorBase\n\n" + body,
        encoding="utf-8",
    )
    return path


def test_cli_verify_factor_causality_and_time_series_gate_integration(tmp_path: Path) -> None:
    _, slot_json = _build_dense_declarative_slot(tmp_path / "slot_env")
    good_py = _write_factor_module(
        tmp_path / "good_factor.py",
        """
class GoodRollingFactor(FactorBase):
    CAUSAL_VECTORIZED = True
    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        vals = self.session_rolling(data, "close", 3, min_periods=1, within_session=False)
        return data[["timestamp_utc", "instrument_id"]].assign(factor_value=vals)
""",
    )
    bad_py = _write_factor_module(
        tmp_path / "bad_factor.py",
        """
class BadShiftFactor(FactorBase):
    CAUSAL_VECTORIZED = True
    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        ordered = data.sort_values(["instrument_id", "timestamp_utc"])
        vals = ordered.groupby("instrument_id", sort=False)["close"].shift(-1)
        return ordered[["timestamp_utc", "instrument_id"]].assign(factor_value=vals.reindex(data.index))
""",
    )

    # 1. CLI dry-run
    dry_proc = subprocess.run(
        [
            sys.executable, "-m", "tools.agent_harness.cli",
            "verify-factor-causality", "--dry-run", "--factor", str(good_py), "--slot", str(slot_json),
            "--output-root", str(tmp_path / "out_dry"),
        ],
        cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=60, check=False,
    )
    assert dry_proc.returncode == 0, dry_proc.stderr
    assert json.loads(dry_proc.stdout)["summary"]["reason"] == "dry_run"

    # 2. CLI verify-factor-causality passes causal factor
    ok_proc = subprocess.run(
        [
            sys.executable, "-m", "tools.agent_harness.cli",
            "verify-factor-causality", "--factor", str(good_py), "--slot", str(slot_json),
            "--frequency", "60min", "--start-date", "2026-01-05", "--end-date", "2026-01-06",
            "--output-root", str(tmp_path / "out_ok"),
        ],
        cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=60, check=False,
    )
    assert ok_proc.returncode == 0, ok_proc.stderr
    ok_payload = json.loads(ok_proc.stdout)
    assert ok_payload["ok"] is True
    assert ok_payload["metrics"]["causal_gate"]["status"] == "passed"
    assert ok_payload["metrics"]["causal_gate"]["resolved_mode"] == "vectorized"

    # 3. CLI verify-factor-causality rejects shift(-1) with EXIT_QUALITY_GATE (4)
    rej_proc = subprocess.run(
        [
            sys.executable, "-m", "tools.agent_harness.cli",
            "verify-factor-causality", "--factor", str(bad_py), "--slot", str(slot_json),
            "--frequency", "60min", "--start-date", "2026-01-05", "--end-date", "2026-01-06",
            "--output-root", str(tmp_path / "out_rej"),
        ],
        cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=60, check=False,
    )
    assert rej_proc.returncode == harness_cli.EXIT_QUALITY_GATE, rej_proc.stderr
    rej_payload = json.loads(rej_proc.stdout)
    assert rej_payload["ok"] is False
    assert rej_payload["summary"]["reason"] == "causal_fast_path_gate_failed"
    assert rej_payload["metrics"]["causal_gate"]["status"] == "failed_rejected"
    assert rej_payload["metrics"]["causal_gate"]["failure_reason"] is not None

    # 4. CLI verify-factor-causality with --on-failure fallback succeeds safely in pit_replay
    fb_proc = subprocess.run(
        [
            sys.executable, "-m", "tools.agent_harness.cli",
            "verify-factor-causality", "--factor", str(bad_py), "--slot", str(slot_json),
            "--frequency", "60min", "--start-date", "2026-01-05", "--end-date", "2026-01-06",
            "--on-failure", "fallback", "--output-root", str(tmp_path / "out_fb"),
        ],
        cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=60, check=False,
    )
    assert fb_proc.returncode == 0, fb_proc.stderr
    fb_payload = json.loads(fb_proc.stdout)
    assert fb_payload["ok"] is True
    assert fb_payload["metrics"]["causal_gate"]["status"] == "failed_fallback"
    assert fb_payload["metrics"]["causal_gate"]["resolved_mode"] == "pit_replay"
    assert fb_payload["warnings"]

    # 5. single-factor-analysis -> time-series-analysis with --slot & --factor enforces gate end-to-end
    ts_cfg = tmp_path / "ts_cfg.json"
    ts_cfg.write_text(
        json.dumps(
            {
                "start_date": "2026-01-05",
                "end_date": "2026-01-06",
                "frequency": "60min",
                "horizons": [1, 2],
                "n_quantiles": 2,
                "min_ic_samples": 2,
                "min_cross_section_size": 2,
            }
        ),
        encoding="utf-8",
    )
    ts_bad = subprocess.run(
        [
            sys.executable, "-m", "tools.agent_harness.cli",
            "single-factor-analysis",
            "--slot", str(slot_json),
            "--factor", str(bad_py),
            "--analysis-mode", "time_series",
            "--config", str(ts_cfg),
            "--output-root", str(tmp_path / "out_ts_bad"),
        ],
        cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=120, check=False,
    )
    assert ts_bad.returncode == harness_cli.EXIT_QUALITY_GATE, ts_bad.stderr
    ts_bad_payload = json.loads(ts_bad.stdout)
    assert ts_bad_payload["ok"] is False
    assert ts_bad_payload["summary"]["reason"] == "causal_fast_path_gate_failed"
    assert ts_bad_payload["metrics"]["causal_gate"]["causal_passed"] is False

    ts_good = subprocess.run(
        [
            sys.executable, "-m", "tools.agent_harness.cli",
            "single-factor-analysis",
            "--slot", str(slot_json),
            "--factor", str(good_py),
            "--analysis-mode", "time_series",
            "--config", str(ts_cfg),
            "--output-root", str(tmp_path / "out_ts_good"),
        ],
        cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=120, check=False,
    )
    assert ts_good.returncode == 0, f"stderr={ts_good.stderr}\nstdout={ts_good.stdout}"
    ts_good_payload = json.loads(ts_good.stdout)
    assert ts_good_payload["ok"] is True
    assert ts_good_payload["metrics"]["causal_gate"]["status"] == "passed"
    assert ts_good_payload["metrics"]["causal_gate"]["resolved_mode"] == "vectorized"

    ts_fallback = subprocess.run(
        [
            sys.executable, "-m", "tools.agent_harness.cli",
            "single-factor-analysis",
            "--slot", str(slot_json),
            "--factor", str(bad_py),
            "--analysis-mode", "time_series",
            "--causal-gate-policy", "fallback",
            "--config", str(ts_cfg),
            "--output-root", str(tmp_path / "out_ts_fallback"),
        ],
        cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=120, check=False,
    )
    assert ts_fallback.returncode == 0, f"stderr={ts_fallback.stderr}\nstdout={ts_fallback.stdout}"
    ts_fb_payload = json.loads(ts_fallback.stdout)
    assert ts_fb_payload["ok"] is True
    assert ts_fb_payload["metrics"]["causal_gate"]["status"] == "failed_fallback"
    assert ts_fb_payload["metrics"]["causal_gate"]["resolved_mode"] == "pit_replay"
    assert ts_fb_payload["warnings"]


def test_run_result_and_run_analysis_main_causal_gate_contract(tmp_path: Path) -> None:
    import run_analysis_main

    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    res_path = tmp_path / "attempt" / "result.json"
    with execution_result(
        res_path, command="single-factor-analysis", run_id="r1", execution_id="attempt", config_path=cfg
    ) as payload:
        payload["execution"].update(expected=1, completed=1)
        payload["metrics"]["causal_gate"] = {
            "schema_version": CAUSAL_GATE_SCHEMA_VERSION,
            "ok": True,
            "causal_passed": True,
            "status": "passed",
            "fast_path_requested": True,
            "fast_path_allowed": True,
            "resolved_mode": "vectorized",
            "truncation_cutoffs": ["2026-01-05T12:00:00+00:00"],
            "pit_sample_cutoffs": ["2026-01-05T12:00:00+00:00"],
            "max_abs_diff": 0.0,
            "nan_mismatch_count": 0,
            "failure_reason": None,
        }
    loaded = read_execution_result(
        res_path, command="single-factor-analysis", run_id="r1", execution_id="attempt",
        source_sha256=harness_cli.config_sha256(cfg),
    )
    assert loaded["metrics"]["causal_gate"]["status"] == "passed"

    # Malformed causal_gate metadata is rejected by run_result contract.
    bad_payload = dict(loaded, metrics={"causal_gate": {"schema_version": "wrong"}})
    with pytest.raises(ValueError, match="causal_gate"):
        validate_execution_result_payload(bad_payload, require_identity=False)

    # _run_causal_gate_for_factor intercepts lookahead A-share factor declaring CAUSAL_VECTORIZED=True.
    raw_panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]),
            "code": ["000001.XSHE"] * 4,
            "close": [10.0, 11.0, 12.0, 13.0],
        }
    )

    class _AShareLookahead(FactorBase):
        CAUSAL_VECTORIZED = True

        def _get_raw_data_for_factor(self):
            return raw_panel.copy()

        def calculate_factor(self, data):
            wide = data.pivot(index="date", columns="code", values="close")
            return wide.shift(-1)

    ashare_bad = _AShareLookahead(
        None, "ashare_bad", "2026-01-05", "2026-01-08",
        stock_pool=["000001.XSHE"], data_bindings={}, pit_enabled=False,
    )
    run_cfg: dict = {"causal_gate_policy": "reject"}
    with pytest.raises(Exception, match="Causal fast-path gate rejected"):
        run_analysis_main._run_causal_gate_for_factor(ashare_bad, run_cfg)
    assert run_cfg["_causal_gate_failure"]["causal_passed"] is False
    assert run_cfg["_causal_gate_failure"]["failure_reason"] == "lookahead_truncation_mismatch"
    assert run_cfg["_causal_gate_failure"]["resolved_mode"] == "rejected"

    # A-share panel factors have no pit_replay execution path, so fallback policy MUST still reject.
    ashare_fb = _AShareLookahead(
        None, "ashare_fb", "2026-01-05", "2026-01-08",
        stock_pool=["000001.XSHE"], data_bindings={}, pit_enabled=False,
    )
    gate_ashare_fb = verify_factor_causality(ashare_fb, on_failure="fallback", apply_to_factor=True)
    assert gate_ashare_fb.ok is False
    assert gate_ashare_fb.causal_passed is False
    assert gate_ashare_fb.status == "failed_rejected"
    assert gate_ashare_fb.fast_path_allowed is False
    assert gate_ashare_fb.resolved_mode == "rejected"
    assert gate_ashare_fb.resolved_reason.startswith("rejected_no_pit_replay:")
    assert ashare_fb.slot_execution_mode != "pit_replay"
    validate_causal_gate_evidence(gate_ashare_fb.to_dict())
    with pytest.raises(RuntimeError, match="no pit_replay fallback"):
        ashare_fb.get_factor_values()

    fb_cfg: dict = {"causal_gate_policy": "fallback"}
    with pytest.raises(Exception, match="Causal fast-path gate rejected"):
        run_analysis_main._run_causal_gate_for_factor(ashare_fb, fb_cfg)
    assert fb_cfg["_causal_gate_failure"]["status"] == "failed_rejected"
    assert fb_cfg["_causal_gate_failure"]["resolved_mode"] == "rejected"

    # End-to-end single-factor-analysis CLI with --causal-gate-policy fallback must reject A-share lookahead.
    ashare_py = _write_factor_module(
        tmp_path / "ashare_lookahead_factor.py",
        """
class AShareLookaheadCliFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def _get_raw_data_for_factor(self):
        return pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]),
            "code": ["000001.XSHE"] * 4,
            "close": [10.0, 11.0, 12.0, 13.0],
            "factor": [1.0, 1.0, 1.0, 1.0],
        })

    def calculate_factor(self, data):
        wide = data.pivot(index="date", columns="code", values="close")
        return wide.shift(-1)
""",
    )
    ashare_cfg = tmp_path / "ashare_cfg.json"
    ashare_db = tmp_path / "ashare_dummy.duckdb"
    with duckdb.connect(str(ashare_db)) as conn:
        conn.execute(
            "CREATE TABLE price_data_daily (code VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, factor DOUBLE)"
        )
    ashare_cfg.write_text(
        json.dumps(
            {
                "duckdb_path": str(ashare_db),
                "start_date": "2026-01-05",
                "end_date": "2026-01-08",
                "code_list": ["000001.XSHE"],
                "use_dynamic_pools": False,
                "use_precomputed_returns": False,
                "max_workers": 1,
            }
        ),
        encoding="utf-8",
    )
    ashare_cli_proc = subprocess.run(
        [
            sys.executable, "-m", "tools.agent_harness.cli",
            "single-factor-analysis",
            "--config", str(ashare_cfg),
            "--factor", str(ashare_py),
            "--causal-gate-policy", "fallback",
            "--output-root", str(tmp_path / "out_ashare_fb"),
        ],
        cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=120, check=False,
    )
    assert ashare_cli_proc.returncode == harness_cli.EXIT_QUALITY_GATE, (
        f"stderr={ashare_cli_proc.stderr}\nstdout={ashare_cli_proc.stdout}"
    )
    ashare_cli_payload = json.loads(ashare_cli_proc.stdout)
    assert ashare_cli_payload["ok"] is False
    assert ashare_cli_payload["summary"]["reason"] == "causal_fast_path_gate_failed"


def test_causal_gate_non_blocking_diagnostics_and_time_series_helpers(tmp_path: Path) -> None:
    from tools.agent_harness.time_series_analysis import _default_slot_instruments, _evaluate_slot

    slot, slot_json = _build_dense_declarative_slot(tmp_path)

    # 1. Skipped statuses report causal_passed=None and pass evidence validation.
    class _UnmarkedFactor(FactorBase):
        CAUSAL_VECTORIZED = False

        def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
            return data[["timestamp_utc", "instrument_id"]].assign(factor_value=data["close"])

    unmarked = _UnmarkedFactor(
        None, "unmarked", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"], slot=slot,
        data_frequency="60min", analysis_mode="time_series",
    )
    skipped_gate = verify_factor_causality(unmarked)
    assert skipped_gate.status == "skipped_not_requested"
    assert skipped_gate.causal_passed is None
    validate_causal_gate_evidence(skipped_gate.to_dict())

    pit_req_gate = verify_factor_causality(unmarked, requested_mode="pit_replay")
    assert pit_req_gate.status == "skipped_pit_replay"
    assert pit_req_gate.causal_passed is None
    validate_causal_gate_evidence(pit_req_gate.to_dict())

    # 2. bfill_nan_probe records explicit skip reason when calculate_factor raises on NaN injection.
    class _NanProbeErrorFactor(FactorBase):
        CAUSAL_VECTORIZED = True

        def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
            if data["close"].isna().any():
                raise ValueError("probe saw NaN")
            return data[["timestamp_utc", "instrument_id"]].assign(factor_value=data["close"])

    probe_err_factor = _NanProbeErrorFactor(
        None, "nan_err", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"], slot=slot,
        data_frequency="60min", analysis_mode="time_series",
    )
    probe_gate = verify_factor_causality(probe_err_factor, n_truncations=2, n_pit_samples=2, seed=3)
    assert probe_gate.ok is True
    assert "bfill_nan_probe_skipped:ValueError" in probe_gate.checks_performed

    # 3. _default_slot_instruments inference across static, specs, root, and missing universe.
    from types import SimpleNamespace

    assert _default_slot_instruments(slot) == ["BTC-USD", "ETH-USD"]
    no_static_slot = SimpleNamespace(
        universe=SimpleNamespace(static=None, specs={"CL": {}, "NG": {}}),
        bar_feed=SimpleNamespace(root=None),
    )
    assert _default_slot_instruments(no_static_slot) == ["CL", "NG"]
    root_slot = SimpleNamespace(
        universe=SimpleNamespace(static=None, specs=None),
        bar_feed=SimpleNamespace(root="CU"),
    )
    assert _default_slot_instruments(root_slot) == ["CU"]
    empty_slot = SimpleNamespace(
        universe=SimpleNamespace(static=None, specs=None),
        bar_feed=SimpleNamespace(root=None),
    )
    with pytest.raises(ValueError, match="instruments must be provided"):
        _default_slot_instruments(empty_slot)

    # 4. time_series_analysis._evaluate_slot rejects unmapped session dates cleanly instead of producing NaT.
    bad_session_py = _write_factor_module(
        tmp_path / "bad_session_factor.py",
        """
class UnmappedSessionFactor(FactorBase):
    CAUSAL_VECTORIZED = False

    def __init__(self, *args, **kwargs):
        kwargs["output_frequency"] = "1d"
        super().__init__(*args, **kwargs)

    def get_factor_values(self, force_recalculate=False):
        return pd.DataFrame(
            {"BTC-USD": [1.0, 2.0]},
            index=pd.to_datetime(["2026-01-05", "2099-01-01"]),
        )
""",
    )
    with pytest.raises(ValueError, match="could not be mapped to Slot session timestamps"):
        _evaluate_slot(
            {
                "slot": str(slot_json),
                "factor_file": str(bad_session_py),
                "start_date": "2026-01-05",
                "end_date": "2026-01-06",
                "frequency": "60min",
                "instruments": ["BTC-USD"],
            },
            {"analysis_mode": "cross_sectional"},
        )


def test_ashare_causal_gate_fails_closed_for_unverified_and_lookahead_factors() -> None:
    from utils.factor_core.slot_execution_mode import _ACTIVE_ASHARE_PROBE_IDS, require_ashare_causal_gate

    raw_panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"] * 2
            ),
            "code": ["000001.XSHE"] * 5 + ["000002.XSHE"] * 5,
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        }
    )

    class _CausalAShareFactor(FactorBase):
        CAUSAL_VECTORIZED = True

        def _get_raw_data_for_factor(self):
            return raw_panel.copy()

        def calculate_factor(self, data):
            wide = data.pivot(index="date", columns="code", values="close").sort_index()
            return wide.pct_change(fill_method=None)

    class _LegacyAShareFactor(FactorBase):
        def _get_raw_data_for_factor(self):
            return raw_panel.copy()

        def calculate_factor(self, data):
            return data.pivot(index="date", columns="code", values="close").sort_index()

    class _FutureReturnAShareFactor(FactorBase):
        CAUSAL_VECTORIZED = True

        def _get_raw_data_for_factor(self):
            return raw_panel.copy()

        def calculate_factor(self, data):
            prices = data.pivot(index="date", columns="code", values="close").sort_index()
            return prices.shift(-2) / prices.shift(-1) - 1.0

    # 1. Unverified CAUSAL_VECTORIZED=True A-share factor MUST fail closed on all direct evaluation entry points.
    unverified = _CausalAShareFactor(
        None, "ashare_unverified", "2026-01-05", "2026-01-09",
        stock_pool=["000001.XSHE", "000002.XSHE"], data_bindings={}, pit_enabled=False,
    )
    for method_name in ("get_factor_values", "get_factor_values_unfiltered", "_get_market_factor_values"):
        with pytest.raises(RuntimeError, match="verify_factor_causality"):
            getattr(unverified, method_name)()

    # apply_to_factor=False must leave the factor unverified and still blocked.
    preview_gate = verify_factor_causality(unverified, apply_to_factor=False)
    assert preview_gate.ok is True
    assert preview_gate.causal_passed is True
    assert getattr(unverified, "_causal_gate_passed", None) is None
    with pytest.raises(RuntimeError, match="verify_factor_causality"):
        unverified.get_factor_values()

    # Requesting slot_execution_mode="pit_replay" on A-share cannot bypass CAUSAL_VECTORIZED=True gate.
    pit_mode_ashare = _CausalAShareFactor(
        None, "ashare_pit_mode", "2026-01-05", "2026-01-09",
        stock_pool=["000001.XSHE", "000002.XSHE"], data_bindings={}, pit_enabled=False,
        slot_execution_mode="pit_replay",
    )
    skipped_pit = verify_factor_causality(pit_mode_ashare, apply_to_factor=True)
    assert skipped_pit.status == "skipped_pit_replay"
    with pytest.raises(RuntimeError, match="verify_factor_causality"):
        pit_mode_ashare.get_factor_values()

    # 2. Legacy A-share factor without CAUSAL_VECTORIZED=True remains unaffected.
    legacy = _LegacyAShareFactor(
        None, "ashare_legacy", "2026-01-05", "2026-01-09",
        stock_pool=["000001.XSHE", "000002.XSHE"], data_bindings={}, pit_enabled=False,
    )
    assert not legacy.get_factor_values().empty
    assert not legacy.get_factor_values_unfiltered().empty

    # 3. Valid causal A-share factor passes verify_factor_causality(apply_to_factor=True) and evaluates normally.
    ok_gate = verify_factor_causality(unverified, apply_to_factor=True)
    assert ok_gate.ok is True
    assert ok_gate.causal_passed is True
    assert getattr(unverified, "_causal_gate_passed", None) is True
    df_filtered = unverified.get_factor_values()
    df_unfiltered = unverified.get_factor_values_unfiltered()
    assert not df_filtered.empty
    pd.testing.assert_frame_equal(df_filtered, df_unfiltered)

    # 4. Internal probe context allows re-entrant gate checks only during verify_factor_causality and cleans up on error.
    class _ProbeCheckAndErrorFactor(_CausalAShareFactor):
        def _get_raw_data_for_factor(self):
            require_ashare_causal_gate(self)
            raise RuntimeError("boom_in_raw_fetch")

    probe_err = _ProbeCheckAndErrorFactor(
        None, "ashare_probe_err", "2026-01-05", "2026-01-09",
        stock_pool=["000001.XSHE"], data_bindings={}, pit_enabled=False,
    )
    with pytest.raises(RuntimeError, match="boom_in_raw_fetch"):
        verify_factor_causality(probe_err, apply_to_factor=True)
    assert id(probe_err) not in _ACTIVE_ASHARE_PROBE_IDS.get()
    with pytest.raises(RuntimeError, match="verify_factor_causality"):
        probe_err.get_factor_values()

    # 5. Future-return factor (shift(-2)/shift(-1) - 1) cannot produce values with or without calling the gate.
    unverified_future = _FutureReturnAShareFactor(
        None, "ashare_future_unverified", "2026-01-05", "2026-01-09",
        stock_pool=["000001.XSHE", "000002.XSHE"], data_bindings={}, pit_enabled=False,
    )
    with pytest.raises(RuntimeError, match="verify_factor_causality"):
        unverified_future.get_factor_values()

    for policy in ("reject", "fallback"):
        checked_future = _FutureReturnAShareFactor(
            None, f"ashare_future_{policy}", "2026-01-05", "2026-01-09",
            stock_pool=["000001.XSHE", "000002.XSHE"], data_bindings={}, pit_enabled=False,
        )
        gate_res = verify_factor_causality(checked_future, on_failure=policy, apply_to_factor=True)
        assert gate_res.ok is False
        assert gate_res.causal_passed is False
        assert gate_res.status == "failed_rejected"
        assert gate_res.resolved_mode == "rejected"
        assert gate_res.failure_reason == "lookahead_truncation_mismatch"
        with pytest.raises(RuntimeError, match="no pit_replay fallback"):
            checked_future.get_factor_values()
