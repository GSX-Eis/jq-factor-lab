from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from backtest_and_analysis.evaluation.engine import EvaluationEngine
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.time_series.factor_bridge import read_slot_factor_values
from factor_synthesis.core.factor_ops import neutralize_market_features
from factor_synthesis.core.feature_preprocessor import cross_sectional_winsorize_zscore, cross_sectional_zscore
from factor_synthesis.factor_reader import FactorReader, plan_market_factor_values
from factor_synthesis.strategy_bundle import StrategyBundle
from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner
from factor_synthesis.vectorbt_adapter.factor_loader import FactorLoader
from factor_synthesis.vectorbt_adapter.slot_provider import SlotDataProvider
from tools.agent_harness import cli
from utils.bar_evaluation_store import query_bar_evaluation_detail
from utils.factor_base import FactorBase
from utils.market_db import adopt_market_db_sidecar, init_market_db, load_market_db_profile_yaml
from utils.market_db.identity import market_db_identity
from utils.market_db.research import open_research_slot
from utils.market_factor_store import MarketFactorValueStore, build_identity_record, compute_factor_identity_hash
from utils.slots import ExecutionProfile, ProviderConfig, SlotConfig

ROOT = Path(__file__).resolve().parents[1]
START, END = "2026-01-05", "2026-01-09"
SYMBOL = "AAPL.XNYS"


@pytest.fixture
def us_market(tmp_path, monkeypatch):
    monkeypatch.delenv("QUANT_OUTPUT_CONTEXT", raising=False)
    profile = load_market_db_profile_yaml(ROOT / "config/market_databases/us_equity.yaml")
    db = tmp_path / "ny.duckdb"
    init_market_db(db, profile)
    with duckdb.connect(str(db)) as connection:
        connection.execute("""
            CREATE TABLE bars AS
            SELECT DATE '2026-01-05' + i::INT + INTERVAL '16 hours' AS timestamp,
                   'AAPL.XNYS' AS instrument_id, 10.0+i AS open, 11.0+i AS high,
                   9.0+i AS low, 10.5+i AS close, 100 AS volume
            FROM range(5) data(i)
        """)
    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps({"market_dbs": {profile.db_id: str(db)}}))
    config = SlotConfig(
        1, "ny_fixture",
        ProviderConfig("pmc", {"calendar_id": "XNYS", "start": START, "end": END}),
        ProviderConfig("duckdb_table", {
            "table": "bars", "frequency": "1d", "timestamp_anchor": "bar_end",
            "source_timezone": "America/New_York", "volume_mode": "interval", "data_version": "fixture",
            "column_mapping": {name: name for name in (
                "instrument_id", "timestamp", "open", "high", "low", "close", "volume",
            )},
        }),
        ProviderConfig("static", {"instruments": {SYMBOL: {"asset_class": "equity", "currency": "USD"}}}),
        ProviderConfig("none"), ProviderConfig("open"),
        ExecutionProfile("close", "close", 0, 0, "decimal", "endpoints_only", "cross_session"),
    )
    return profile, db, config


def _with_fingerprint(slot, fingerprint):
    if getattr(slot, "market_db_profile", None) is not None:
        return replace(slot, market_db_profile=None, slot_data_fingerprint=fingerprint)
    return replace(slot, slot_data_fingerprint=fingerprint)


def _save(store, slot, value, *, factor_id="alpha", instrument=SYMBOL):
    fingerprint = slot.slot_data_fingerprint
    hash_, payload = compute_factor_identity_hash(
        factor_id=factor_id, factor_class="tests.ProfileFactor", source_sha256="a" * 64, params={},
        frequency="1d", session_view="ETH", analysis_mode="cross_sectional", bucket_strategy="session",
        calendar_hash=slot.calendar.calendar_hash, code_version="test", slot_id=slot.slot_id,
        slot_config_hash=slot.slot_config_hash, adjustment_policy="none", slot_data_fingerprint=fingerprint,
    )
    record = build_identity_record(factor_identity_hash=hash_, payload=payload, code_version_source="test")
    grid = slot.calendar.bar_grid("1d")
    rows = grid[["timestamp_utc", "session_id"]].assign(
        factor_id=factor_id, factor_identity_hash=hash_, frequency="1d", session_view="ETH",
        instrument_id=instrument, factor_value=value, source_build_id="test-build",
        calendar_hash=slot.calendar.calendar_hash, code_version="test",
        created_at=pd.Timestamp("2026-02-01T00:00:00Z"), batch_id=hash_[:24],
    )
    store.append_factor_values(rows, identity_record=record, calendar=slot.calendar)
    return hash_, payload


def test_scheme_b_isolates_identical_keys_by_database_and_profile(us_market, tmp_path):
    profile, _, config = us_market
    store = MarketFactorValueStore(tmp_path / "factors")
    with open_research_slot(profile.db_id, config, tmp_path) as (slot, _):
        identity = market_db_identity(profile)
        other_db = _with_fingerprint(slot, {**identity, "db_id": "ny_other"})
        other_revision = _with_fingerprint(slot, {**identity, "profile_hash": "f" * 64})
        entries = [(item, *_save(store, item, value, instrument="000001"))
                   for value, item in enumerate((slot, other_db, other_revision))]
        assert len({hash_ for _, hash_, _ in entries}) == 3
        for value, (item, hash_, payload) in enumerate(entries):
            assert market_db_identity(payload) == market_db_identity(item.slot_data_fingerprint)
            found, row = read_slot_factor_values(
                store.root, item, factor_id="alpha", frequency="1d", start_date=START, end_date=END,
            )
            assert row.factor_identity_hash == hash_
            assert found.factor_value.tolist() == [value] * 5
        with pytest.raises(ValueError, match="market database"):
            read_slot_factor_values(
                store.root, slot, factor_id="alpha", frequency="1d", start_date=START, end_date=END,
                factor_identity_hash=entries[1][1],
            )
        combined = store.read_factor_values(factor_id="alpha")
        with pytest.raises(ValueError, match="multiple market databases"):
            cross_sectional_zscore(combined)
        selected, _ = plan_market_factor_values(store, ["alpha"], {}, START, END, None, slot=slot)
        assert selected == {"alpha": entries[0][1]}
        loader = FactorLoader(factor_values_store=store, slot_id=slot.slot_id, market_db=identity)
        assert loader._resolve_market_identities(["alpha"]) == selected
        assert loader.cache_revision() != FactorLoader(
            factor_values_store=store, slot_id=slot.slot_id, market_db=market_db_identity(other_db.slot_data_fingerprint),
        ).cache_revision()
        _save(store, other_db, 4, factor_id="other")
        reader = FactorReader(str(tmp_path / "legacy.duckdb"), factor_values_store=store, slot=slot)
        assert reader.get_available_factors() == ["alpha"]
        with pytest.raises(ValueError, match="multiple market databases"):
            plan_market_factor_values(
                store, ["alpha", "other"], {"alpha": entries[0][1]}, START, END, None, slot_id=slot.slot_id,
            )


def test_factor_disk_cache_binds_database_and_profile(us_market, tmp_path):
    class IdentityFactor(FactorBase):
        def _get_cache_inputs(self):
            return {}

        def calculate_factor(self, data):
            return data

    profile, _, config = us_market
    with open_research_slot(profile.db_id, config, tmp_path) as (slot, _):
        identity = market_db_identity(profile)
        raw = pd.DataFrame({"date": [pd.Timestamp(START)], "code": [SYMBOL], "close": [2.0]})

        def cache_key(fingerprint):
            current = _with_fingerprint(slot, fingerprint)
            factor = IdentityFactor(None, "db_cache", START, END, stock_pool=[SYMBOL], slot=current, data_frequency="1d")
            return factor._get_cache_key(raw)

        key = cache_key(identity)
        assert key is not None and cache_key(dict(identity)) == key
        assert cache_key({**identity, "db_id": "ny_other"}) != key
        assert cache_key({**identity, "profile_hash": "b" * 64}) != key



def test_profile_only_slot_keeps_exact_panel_fingerprint(us_market, tmp_path):
    from test_factor_base_slot_single_track import TurnoverFactor

    profile, _, config = us_market
    store = MarketFactorValueStore(tmp_path / "factors")
    with open_research_slot(profile.db_id, config, tmp_path) as (slot, _):
        factor = TurnoverFactor(None, "turnover", START, END, stock_pool=[SYMBOL], slot=slot, data_frequency="1d")
        panel = pd.DataFrame({SYMBOL: [1., 2., 3., 4., 5.]}, index=pd.bdate_range(START, END))
        first = factor.save_factor_values(store, factor_values=panel, factor_params={})
        second = factor.save_factor_values(store, factor_values=panel * 2, factor_params={})
    assert first.factor_identity_hash != second.factor_identity_hash
    for fingerprint in store.list_identities().slot_data_fingerprint:
        assert market_db_identity(fingerprint) == market_db_identity(profile)
        assert "realized_panel_sha256" in fingerprint

@pytest.mark.parametrize("command", ("single-factor-analysis", "time-series-analysis", "factor-synthesis"))
def test_research_cli_market_db_preflight(command, us_market, tmp_path, capsys):
    profile, _, config = us_market
    request = tmp_path / "run.json"
    request.write_text(json.dumps({"factor_values_store": str(tmp_path / "factors")}))
    slot_file = tmp_path / "slot.json"
    slot_file.write_text(json.dumps(config.as_dict()))

    def run(*args, slot_path=slot_file):
        code = cli.main([command, "--config", str(request), "--slot-config", str(slot_path), "--dry-run",
                         "--output-root", str(tmp_path / "out"), *args])
        return code, json.loads(capsys.readouterr().out)

    code, plan = run("--market-db", profile.db_id)
    assert code == 0 and plan["metrics"]["market_db"] == market_db_identity(profile)
    code, failure = run("--market-db", "missing")
    assert code == 2 and "No market_dbs binding" in failure["error"]["message"]
    code, failure = run("--market-db", profile.db_id, slot_path="jq_ashare_local")
    assert code == 2 and "conflicts" in failure["error"]["message"]
    bad_path = tmp_path / "wrong-slot.json"
    wrong_slot = replace(config, bar_feed=replace(config.bar_feed, options={
        **config.bar_feed.options, "path": "not-the-bound-db.duckdb",
    }))
    bad_path.write_text(json.dumps(wrong_slot.as_dict()))
    code, failure = run("--market-db", profile.db_id, slot_path=bad_path)
    assert code == 2 and "bar_feed path conflicts" in failure["error"]["message"]
    code, failure = run("--market-db", str(tmp_path / "ny.duckdb"))
    assert code == 2 and "requires a db_id" in failure["error"]["message"]


def test_bound_evaluation_manifest_and_backtest_metadata(us_market, tmp_path, capsys):
    profile, _, config = us_market
    identity = market_db_identity(profile)
    root = tmp_path / "factors"
    with open_research_slot(profile.db_id, config, tmp_path) as (slot, _):
        _save(MarketFactorValueStore(root), slot, 0.5)
        backtest = VectorBTBacktestRunner(
            {}, slot=slot, return_series_output_dir=str(tmp_path / "returns"), return_series_trial_id="ny",
        )
        portfolio = SimpleNamespace(returns=lambda: pd.Series([0.01], index=[pd.Timestamp(START)]))
        assert backtest._write_return_series_artifact(portfolio) is not None
    metadata = json.loads((tmp_path / "returns/ny.metadata.json").read_text())
    assert {key: metadata[key] for key in identity} == identity
    assert "XSHG" not in json.dumps(metadata)

    request = tmp_path / "evaluation.json"
    request.write_text(json.dumps({
        "slot_config": config.as_dict(), "factor_store_root": str(root), "factor_id": "alpha",
        "analysis_mode": "time_series", "frequency": "1d", "horizons": [1],
        "start_date": START, "end_date": END,
    }))
    destination = tmp_path / "evaluation"
    result = cli.main([
        "single-factor-analysis", "--config", str(request), "--market-db", profile.db_id,
        "--output-root", str(tmp_path / "out"), "--output-dir", str(destination),
        "--results-db", str(tmp_path / "evaluations.duckdb"),
    ])
    payload = json.loads(capsys.readouterr().out)
    assert result == 0, payload
    summary = json.loads((destination / "summary.json").read_text())
    assert {key: summary["data_identity"][key] for key in identity} == identity
    assert market_db_identity(summary["data_identity"]["slot_data_fingerprint"]) == identity
    assert "XSHG" not in json.dumps(summary)
    assert summary["config"]["market_db_identity"] == identity
    receipt = payload["metrics"]["results_db"]
    stored = query_bar_evaluation_detail(tmp_path / "evaluations.duckdb", receipt["identity_hash"])
    assert market_db_identity(stored["run"]["data_identity"]) == identity



def test_synthesis_loads_only_the_selected_database(us_market, tmp_path, monkeypatch):
    from run_factor_synthesis import FactorSynthesizer

    profile, _, config = us_market
    root = tmp_path / "factors"
    with open_research_slot(profile.db_id, config, tmp_path) as (slot, _):
        store = MarketFactorValueStore(root)
        expected, _ = _save(store, slot, 0.5)
        other = _with_fingerprint(slot, {**market_db_identity(profile), "db_id": "another_db"})
        _save(store, other, 9.0)
    request = tmp_path / "synthesis.json"
    request.write_text(json.dumps({
        "slot_config": config.as_dict(), "factor_values_store": str(root),
        "analysis_period": {"start_date": START, "end_date": END},
        "factor_groups": {"alpha_group": {"factors": ["alpha"]}},
        "synthesis_methods": {"simple": {"factor_group": "alpha_group"}},
        "cache_settings": {"use_parquet_cache": False, "cache_dir": str(tmp_path / "cache")},
    }))
    monkeypatch.setenv("QUANT_EXECUTION_DIR", str(tmp_path / "exec"))
    synth = FactorSynthesizer(str(request), market_db=profile.db_id)
    assert synth._get_slot().slot_data_fingerprint == market_db_identity(profile)
    panel = synth.load_factor_data(["alpha"], START, END)
    assert not panel.empty and panel.alpha.eq(0.5).all()
    assert panel.attrs["market_factor_inputs"]["identities"] == {"alpha": expected}

def test_strategy_bundle_market_metadata_is_hash_bound(us_market, tmp_path):
    from strategy_bundle_test_support import _make_sample_strategy_bundle

    profile, _, _ = us_market
    identity = market_db_identity(profile)
    bundle, _ = _make_sample_strategy_bundle(tmp_path)
    updated = replace(bundle, slot_data_fingerprint=identity, bundle_hash="")
    assert updated.bundle_hash != bundle.bundle_hash
    assert updated.to_dict()["metadata"] == identity
    assert StrategyBundle.from_mapping(updated.to_dict()).metadata == identity
    with pytest.raises(ValueError, match="metadata must match"):
        StrategyBundle.from_mapping({**updated.to_dict(), "metadata": {**identity, "db_id": "wrong"}})


@pytest.mark.parametrize("tags", [{"db_ids": ["first", "second"]}, {"market_domains": ["cn_equity", "us_equity"]}])
def test_cross_sectional_preprocessing_and_evaluation_reject_mixed_sources(tags):
    index = pd.MultiIndex.from_product([[pd.Timestamp(START)], ["A", "B"]], names=["date", "stock_code"])
    frame = pd.DataFrame({"factor": [1.0, 2.0], 1: [0.01, 0.02]}, index=index)
    frame.attrs.update(tags)
    for preprocess in (cross_sectional_winsorize_zscore, cross_sectional_zscore):
        with pytest.raises(ValueError, match="multiple market databases"):
            preprocess(frame)
    with pytest.raises(ValueError, match="multiple market databases"):
        neutralize_market_features(frame, {}, slot=None, start_date=START, end_date=END)
    with pytest.raises(ValueError, match="multiple market databases"):
        EvaluationEngine("cross_sectional", "daily").evaluate(EvaluationPanel.from_long(frame))


def test_single_database_cross_section_matches_untagged_input():
    index = pd.MultiIndex.from_product([[pd.Timestamp(START)], ["A", "B"]], names=["date", "stock_code"])
    plain = pd.DataFrame({"alpha": [1.0, 2.0]}, index=index)
    tagged = plain.copy()
    tagged.attrs["market_databases"] = [
        {"db_id": "first", "market_domain": "us_equity", "profile_hash": digest * 64}
        for digest in ("a", "b")
    ]
    pd.testing.assert_frame_equal(cross_sectional_zscore(tagged), cross_sectional_zscore(plain))


@pytest.mark.parametrize("collision", (
    "summary", "fake_summary", "attempt_bindings", "return_tree", "return_alias", "configured_weights",
))
def test_synthesis_harness_rejects_market_db_output_collisions(collision, us_market, tmp_path, monkeypatch, capsys):
    profile, db, config = us_market
    request = tmp_path / "synthesis.json"
    raw = {"slot_config": config.as_dict(), "factor_values_store": "factors"}
    output_root = tmp_path / "harness"
    run = output_root / "factor-synthesis" / "review"
    flags = ["--dry-run"]
    if collision in ("summary", "fake_summary"):
        source = run / "summary.json"
        if collision == "fake_summary":
            flags = ["--fake-result", "success"]
    elif collision == "attempt_bindings":
        monkeypatch.setattr("tools.agent_harness.handlers.synthesis.uuid.uuid4", lambda: SimpleNamespace(hex="attempt"))
        source = run / "attempts" / "attempt" / "output_bindings.json"
    elif collision == "configured_weights":
        source = tmp_path / "weights" / "market.duckdb"
        raw["output_settings"] = {"weights_output_dir": str(source.parent)}
    else:
        returns = tmp_path / "returns"
        source = returns / "market.duckdb"
        flags += ["--return-series-output-dir", str(returns)]
        if collision == "return_alias":
            returns.mkdir()
            (returns / "trial.metadata.json").hardlink_to(db)
            source = db
    if source != db:
        source.parent.mkdir(parents=True)
        db.rename(source)
    request.write_text(json.dumps(raw))
    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps({"market_dbs": {profile.db_id: str(source)}}))
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    code = cli.main(["factor-synthesis", "--config", str(request), "--market-db", profile.db_id,
                     "--output-root", str(output_root), "--run-id", "review", *flags])

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    payload = json.loads(capsys.readouterr().out)
    assert code == 2 and payload["error"]["code"] == "market_db_precondition_failed"
    assert "selected market database" in payload["error"]["message"]
    assert payload["artifacts"] == []


@pytest.mark.parametrize("collision", ("result", "sidecar_from_config", "output_bindings"))
def test_direct_synthesis_preflights_market_db_before_failure_artifacts(collision, us_market, tmp_path):
    profile, db, _ = us_market
    sidecar = adopt_market_db_sidecar(db, profile)
    result = tmp_path / "run" / "result.json"
    request = tmp_path / "invalid-synthesis.json"
    raw = {"database": None} if collision == "result" else {}
    if collision == "sidecar_from_config":
        raw["market_db"] = profile.db_id
    request.write_text(json.dumps(raw))
    args = [] if collision == "sidecar_from_config" else ["--market-db", profile.db_id]
    if collision == "result":
        result = db
    elif collision == "sidecar_from_config":
        result = sidecar
    else:
        result.parent.mkdir()
        (result.parent / "output_bindings.json").symlink_to(db)
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (db, sidecar)]
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run_factor_synthesis.py"), "--config", str(request),
         "--result-out", str(result), *args],
        env={**os.environ, "TASK_MANAGER_ROOT": str(tmp_path)}, capture_output=True, text=True, check=False,
    )

    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in (db, sidecar)] == before
    assert proc.returncode == 2 and "selected market database" in proc.stderr
    if collision == "output_bindings":
        assert not result.exists()
        assert (result.parent / "output_bindings.json").is_symlink()


@pytest.mark.parametrize("collision", ("database", "database_hardlink", "sidecar_symlink"))
def test_direct_synthesis_failure_preserves_files_at_old_result_temp_path(collision, us_market, tmp_path, monkeypatch):
    profile, db, _ = us_market
    result = tmp_path / "result.json"
    old_temp = result.with_suffix(".tmp")
    if collision == "database":
        db = db.rename(old_temp)
        monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps({"market_dbs": {profile.db_id: str(db)}}))
    sidecar = adopt_market_db_sidecar(db, profile)
    if collision == "database_hardlink":
        old_temp.hardlink_to(db)
    elif collision == "sidecar_symlink":
        old_temp.symlink_to(sidecar)
    request = tmp_path / "invalid-synthesis.json"
    request.write_text("{}", encoding="utf-8")

    protected = {db, sidecar, old_temp}
    digests = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    links = {path: (path.lstat().st_ino, path.lstat().st_nlink) for path in protected}
    link_targets = {path: path.readlink() for path in protected if path.is_symlink()}

    proc = subprocess.run(
        [sys.executable, str(ROOT / "run_factor_synthesis.py"), "--config", str(request),
         "--result-out", str(result), "--market-db", profile.db_id],
        env={**os.environ, "TASK_MANAGER_ROOT": str(tmp_path)}, capture_output=True, text=True, check=False,
    )

    assert proc.returncode == 1 and "factor_groups" in proc.stderr, proc.stderr
    assert json.loads(result.read_text(encoding="utf-8"))["execution"]["status"] == "failed"
    assert all(path.exists() for path in protected)
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected} == digests
    assert {path: (path.lstat().st_ino, path.lstat().st_nlink) for path in protected} == links
    assert {path: path.readlink() for path in protected if path.is_symlink()} == link_targets


def test_auto_vectorbt_slot_rejects_factor_identity_from_another_market_db(us_market, tmp_path, monkeypatch):
    profile, db, config = us_market
    store = MarketFactorValueStore(tmp_path / "factors")
    with open_research_slot(profile.db_id, config, tmp_path) as (price_slot, _):
        other = _with_fingerprint(price_slot, {**market_db_identity(profile), "db_id": "another_db"})
        foreign_hash, _ = _save(store, other, 9.0)
        # The S7-2 native builder supplies a bound Slot. Exercise the production auto-resolution and loader.
        monkeypatch.setattr(
            "factor_synthesis.vectorbt_adapter.slot_provider.build_jq_ashare_local",
            lambda conn, **kwargs: price_slot,
        )
        provider = SlotDataProvider(factor_values_store=store)
        try:
            loader = provider.create_factor_loader(
                factor_db_path=None, factor_db_conn=None, shared_cache={},
                price_db_path=str(db),
            )
            assert loader is not None and loader.slot_id == price_slot.slot_id
            assert loader.calendar is price_slot.calendar
            assert loader.load_factors(START, [SYMBOL], ["alpha", foreign_hash]).isna().all().all()
            assert loader.market_db == provider.market_db_identity == market_db_identity(profile)
        finally:
            provider.close()
