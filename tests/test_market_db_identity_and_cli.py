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
from factor_synthesis.factor_reader import FactorReader, plan_market_factor_values, read_market_factor_values
from factor_synthesis.strategy_bundle import StrategyBundle
from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner
from factor_synthesis.vectorbt_adapter.factor_loader import FactorLoader
from factor_synthesis.vectorbt_adapter.slot_provider import SlotDataProvider
from market_db_samples import declare_market_db, market_profile
from slots.jq_ashare_local import build_jq_ashare_local
from tools.agent_harness import cli
from utils.bar_evaluation_store import query_bar_evaluation_detail
from utils.factor_base import FactorBase
from utils.market_db import MarketDbProfileError, adopt_market_db_sidecar, init_market_db, load_market_db_profile_yaml
from utils.market_db.identity import market_db_identity, slot_market_db_identity
from utils.market_db.research import open_research_slot
from utils.market_factor import save_bar_factor_values as public_save_bar_factor_values
from utils.market_factor import save_factor_values as public_save_factor_values
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


class _AshareCloseFactor(FactorBase):
    def _get_raw_data_for_factor(self):
        return self._fetch_raw_data(["close"], table_name="price_data")

    def calculate_factor(self, data):
        return data.pivot(index="date", columns="code", values="close").sort_index()


def _create_ashare_price_db(
    db_path: Path,
    *,
    db_id: str = "cn_ashare_fixture",
    calendar_coverage: str = "2026-01-01..2026-01-31",
    sidecar: bool = False,
):
    start, end = "2026-01-05", "2026-01-07"
    codes = ["000001.XSHE", "600000.XSHG"]
    profile = market_profile("cn_equity", db_id=db_id, calendar_coverage=calendar_coverage)
    with duckdb.connect(str(db_path)) as conn:
        if not sidecar:
            declare_market_db(conn, "cn_equity", db_id=db_id, calendar_coverage=calendar_coverage)
        rows = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"] * 2),
            "stock_code": ["000001.XSHE"] * 3 + ["600000.XSHG"] * 3,
            "close": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            "factor": [1.0] * 6,
            "volume": [100.0] * 6,
        })
        conn.from_df(rows).create("price_data_daily")
    if sidecar:
        adopt_market_db_sidecar(db_path, profile)
    return profile, start, end, codes


@pytest.mark.parametrize("binding_mode", [
    "duckdb_path",
    "duckdb_path_bar_grid",
    "duckdb_path_sidecar",
    "db_connection",
    "slot",
])
def test_ashare_factor_bindings_preserve_market_db_identity_and_synthesis_handoff(tmp_path, binding_mode):
    sidecar = binding_mode == "duckdb_path_sidecar"
    db_path = tmp_path / "market.duckdb"
    profile, start, end, codes = _create_ashare_price_db(db_path, sidecar=sidecar)
    expected_identity = market_db_identity(profile)
    store = MarketFactorValueStore(tmp_path / "factors")

    native_conn = duckdb.connect(str(db_path), read_only=True)
    try:
        native_slot = build_jq_ashare_local(native_conn, start=start, end=end)
        kwargs = {}
        conn_arg = None
        if binding_mode == "duckdb_path":
            kwargs["duckdb_path"] = str(db_path)
            kwargs["data_bindings"] = {}
        elif binding_mode == "duckdb_path_bar_grid":
            kwargs["duckdb_path"] = str(db_path)
            kwargs["data_bindings"] = {}
            kwargs["data_frequency"] = "1d"
            kwargs["analysis_mode"] = "cross_sectional"
        elif binding_mode == "duckdb_path_sidecar":
            kwargs["duckdb_path"] = str(db_path)
            kwargs["data_bindings"] = {}
        elif binding_mode == "db_connection":
            conn_arg = native_conn
        else:
            kwargs["slot"] = native_slot

        factor = _AshareCloseFactor(
            conn_arg, "close_alpha", start, end, stock_pool=codes, pit_enabled=False, **kwargs,
        )
        try:
            if binding_mode in ("duckdb_path", "duckdb_path_sidecar"):
                assert factor._duckdb_adapter._conn is None
            assert factor.slot.market_db_profile == profile
            assert factor.slot.bar_feed.profile == profile
            assert slot_market_db_identity(factor.slot) == expected_identity

            values = factor.get_factor_values(force_recalculate=True)
            receipt = factor.save_factor_values(store, factor_values=values)
        finally:
            factor.close()

        identities = store.list_identities(factor_identity_hash=receipt.factor_identity_hash)
        assert len(identities) == 1
        payload = json.loads(identities.iloc[0].identity_payload_json)
        assert market_db_identity(payload) == expected_identity

        # Hand off to plan_market_factor_values (both pinned and unpinned) and FactorReader synthesis readback.
        for pinned in ({}, {"close_alpha": receipt.factor_identity_hash}):
            selected, plan = plan_market_factor_values(
                store, ["close_alpha"], pinned, start, end, codes, slot=native_slot,
            )
            assert selected == {"close_alpha": receipt.factor_identity_hash}
            assert plan is not None
            planned_frame = read_market_factor_values(plan, "1d")
            pivoted = planned_frame.pivot(index="date", columns="stock_code", values="factor_value").sort_index()
            expected_panel = values.copy()
            if expected_panel.index.tz is not None:
                expected_panel.index = expected_panel.index.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
            expected_panel.index.name = "date"
            expected_panel.columns.name = "stock_code"
            pd.testing.assert_frame_equal(pivoted, expected_panel, check_exact=False, rtol=0, atol=1e-12)

        reader = FactorReader(str(tmp_path / "results.duckdb"), factor_values_store=store, slot=native_slot)
        assert reader.get_available_factors() == ["close_alpha"]
        read_back = reader.get_factor_values("close_alpha", start, end)
        assert read_back["000001.XSHE"].tolist() == [10.0, 11.0, 12.0]
        assert read_back["600000.XSHG"].tolist() == [20.0, 21.0, 22.0]
        loader = FactorLoader(factor_values_store=store, slot_id=native_slot.slot_id, market_db=expected_identity)
        assert loader._resolve_market_identities(["close_alpha"]) == {"close_alpha": receipt.factor_identity_hash}
    finally:
        native_conn.close()


def test_ashare_duckdb_path_cross_db_and_slot_mismatch_rejections(tmp_path):
    db_a = tmp_path / "cn_a.duckdb"
    db_b = tmp_path / "cn_b.duckdb"
    profile_a, start, end, codes = _create_ashare_price_db(db_a, db_id="cn_a")
    profile_b, _, _, _ = _create_ashare_price_db(db_b, db_id="cn_b")
    store = MarketFactorValueStore(tmp_path / "factors")

    factor_a = _AshareCloseFactor(
        None, "alpha_a", start, end, stock_pool=codes, duckdb_path=str(db_a), data_bindings={}, pit_enabled=False,
    )
    try:
        receipt_a = factor_a.save_factor_values(store, factor_values=factor_a.get_factor_values(force_recalculate=True))
    finally:
        factor_a.close()

    with duckdb.connect(str(db_b), read_only=True) as conn_b:
        slot_b = build_jq_ashare_local(conn_b, start=start, end=end)
        with pytest.raises(ValueError, match=r"Matching hashes: \[\]"):
            plan_market_factor_values(
                store, ["alpha_a"], {"alpha_a": receipt_a.factor_identity_hash}, start, end, codes, slot=slot_b,
            )
        assert FactorReader(str(tmp_path / "r.duckdb"), factor_values_store=store, slot=slot_b).get_available_factors() == []

        factor_b = _AshareCloseFactor(
            None, "alpha_b", start, end, stock_pool=codes, duckdb_path=str(db_b), data_bindings={}, pit_enabled=False,
        )
        try:
            receipt_b = factor_b.save_factor_values(store, factor_values=factor_b.get_factor_values(force_recalculate=True))
        finally:
            factor_b.close()

        with pytest.raises(ValueError, match="multiple market databases"):
            plan_market_factor_values(
                store,
                ["alpha_a", "alpha_b"],
                {"alpha_a": receipt_a.factor_identity_hash, "alpha_b": receipt_b.factor_identity_hash},
                start,
                end,
                codes,
                slot_id="jq_ashare_local",
            )

        with pytest.raises(MarketDbProfileError, match="MARKET_DB_PROFILE_HASH_MISMATCH"):
            _AshareCloseFactor(
                None, "mismatch", start, end, stock_pool=codes, slot=slot_b, duckdb_path=str(db_a), data_bindings={},
            )
        with pytest.raises(MarketDbProfileError, match="MARKET_DB_PROFILE_HASH_MISMATCH"):
            _AshareCloseFactor(
                None, "mismatch_factor_slot", start, end, stock_pool=codes, slot=factor_a.slot,
                duckdb_path=str(db_b), data_bindings={},
            )
        with pytest.raises(MarketDbProfileError, match="MARKET_DB_PROFILE_HASH_MISMATCH"):
            _AshareCloseFactor(
                conn_b, "mismatch_conn", start, end, stock_pool=codes, duckdb_path=str(db_a), data_bindings={},
            )

    # Same db_id but mismatched profile_hash (different calendar_coverage) is also rejected.
    db_same_id_diff_hash = tmp_path / "cn_a_other_hash.duckdb"
    _create_ashare_price_db(db_same_id_diff_hash, db_id="cn_a", calendar_coverage="2026-01-01..2026-02-28")
    with duckdb.connect(str(db_same_id_diff_hash), read_only=True) as conn_other_hash:
        slot_other_hash = build_jq_ashare_local(conn_other_hash, start=start, end=end)
        with pytest.raises(ValueError, match=r"Matching hashes: \[\]"):
            plan_market_factor_values(
                store, ["alpha_a"], {"alpha_a": receipt_a.factor_identity_hash}, start, end, codes,
                slot=slot_other_hash,
            )
        with pytest.raises(MarketDbProfileError, match="MARKET_DB_PROFILE_HASH_MISMATCH"):
            _AshareCloseFactor(
                None, "mismatch_profile_hash", start, end, stock_pool=codes, slot=slot_other_hash,
                duckdb_path=str(db_a), data_bindings={},
            )


def test_ashare_slot_binding_keeps_legacy_and_nonfile_compat(tmp_path, monkeypatch):
    start, end, codes = "2026-01-05", "2026-01-07", ["000001.XSHE"]
    store = MarketFactorValueStore(tmp_path / "factors")

    # 1. Unprofiled in-memory DB remains compatible at init, compute, and save.
    with duckdb.connect(":memory:") as mem:
        mem.execute("""
            CREATE TABLE price_data_daily AS
            SELECT DATE '2026-01-05' + i::INT AS date, '000001.XSHE' AS stock_code, 10.0 + i AS close
            FROM range(3) t(i)
        """)
        mem_factor = _AshareCloseFactor(mem, "mem_alpha", start, end, stock_pool=codes, pit_enabled=False)
        assert mem_factor.slot.market_db_profile is None
        mem_receipt = mem_factor.save_factor_values(store, factor_values=mem_factor.get_factor_values(force_recalculate=True))
        assert mem_receipt.inserted == 3

    # 2. Unprofiled duckdb_path stays compatible at init and is still rejected on panel read.
    unprofiled = tmp_path / "unprofiled.duckdb"
    with duckdb.connect(str(unprofiled)) as conn:
        conn.execute("CREATE TABLE price_data_daily AS SELECT DATE '2026-01-05' AS date, '000001.XSHE' AS stock_code, 10.0 AS close")
    lazy_factor = _AshareCloseFactor(
        None, "unprofiled", start, end, stock_pool=codes, duckdb_path=str(unprofiled), data_bindings={}, pit_enabled=False,
    )
    try:
        assert lazy_factor._duckdb_adapter._conn is None
        assert lazy_factor.slot.market_db_profile is None
        with pytest.raises(MarketDbProfileError, match="MARKET_DB_PROFILE_MISSING"):
            lazy_factor.get_price_data()
    finally:
        lazy_factor.close()

    # 3. Same-file RW connection + slot + duckdb_path does not trigger RW/RO lock failure.
    rw_db = tmp_path / "rw_same.duckdb"
    _, _, _, _ = _create_ashare_price_db(rw_db, db_id="cn_rw")
    with duckdb.connect(str(rw_db)) as rw_conn:
        rw_slot = build_jq_ashare_local(rw_conn, start=start, end=end)
        same_factor = _AshareCloseFactor(
            None, "same_rw", start, end, stock_pool=codes, slot=rw_slot, duckdb_path=str(rw_db),
            data_bindings={}, pit_enabled=False,
        )
        assert same_factor.slot.market_db_profile == rw_slot.market_db_profile
        same_factor.close()

    # 4. Parquet-only and cloud-mode constructions succeed with no local profile probing.
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    pq_factor = _AshareCloseFactor(
        None, "pq_only", start, end, stock_pool=codes, parquet_dir=str(parquet_dir), data_bindings={}, pit_enabled=False,
    )
    try:
        assert pq_factor.slot.market_db_profile is None
    finally:
        pq_factor.close()

    monkeypatch.setenv("JQ_RUNNING", "1")
    cloud_factor = _AshareCloseFactor(
        None, "cloud", start, end, stock_pool=codes, duckdb_path=str(tmp_path / "missing_cloud.duckdb"),
        data_bindings={}, pit_enabled=False,
    )
    assert cloud_factor.slot.market_db_profile is None


@pytest.mark.parametrize("borrow_timing", ["rw_closed_before_borrow", "rw_closed_after_borrow_owner_reads_first"])
def test_deferred_borrowed_slot_preserves_identity_and_synthesis_handoff(tmp_path, borrow_timing):
    db_a = tmp_path / "cn_deferred_a.duckdb"
    profile_a, start, end, _ = _create_ashare_price_db(db_a, db_id="cn_deferred_a")
    codes = ["000001.XSHE"]
    expected_identity = market_db_identity(profile_a)
    store = MarketFactorValueStore(tmp_path / "factors")

    rw_conn = duckdb.connect(str(db_a))
    try:
        owner = _AshareCloseFactor(
            None, "owner", start, end, stock_pool=codes, duckdb_path=str(db_a),
            data_bindings={}, pit_enabled=False,
        )
        assert owner.slot.market_db_profile is None
        if borrow_timing == "rw_closed_after_borrow_owner_reads_first":
            borrower = _AshareCloseFactor(
                None, "borrower", start, end, stock_pool=codes, slot=owner.slot,
                data_bindings={}, pit_enabled=False,
            )
    finally:
        rw_conn.close()

    try:
        if borrow_timing == "rw_closed_before_borrow":
            borrower = _AshareCloseFactor(
                None, "borrower", start, end, stock_pool=codes, slot=owner.slot,
                data_bindings={}, pit_enabled=False,
            )
            values = borrower.get_factor_values(force_recalculate=True)
            assert values.to_numpy().tolist() == [[10.0], [11.0], [12.0]]
            receipt = borrower.save_factor_values(store, factor_values=values)
        else:
            owner_values = owner.get_factor_values(force_recalculate=True)
            assert owner_values.to_numpy().tolist() == [[10.0], [11.0], [12.0]]
            precomputed = owner_values.copy()
            receipt = borrower.save_factor_values(store, factor_values=precomputed)

        assert owner.slot.market_db_profile == profile_a
        assert borrower.slot.market_db_profile == profile_a
        assert borrower.slot.bar_feed.profile == profile_a
        assert slot_market_db_identity(borrower.slot) == expected_identity

        identities = store.list_identities(factor_identity_hash=receipt.factor_identity_hash)
        assert len(identities) == 1
        payload = json.loads(identities.iloc[0].identity_payload_json)
        assert market_db_identity(payload) == expected_identity

        borrower.close()
        owner_read = owner.get_factor_values(force_recalculate=True)
        assert owner_read.to_numpy().tolist() == [[10.0], [11.0], [12.0]]
        assert owner._duckdb_adapter._conn is not None
        assert owner._duckdb_adapter._conn.execute("SELECT 1").fetchone() == (1,)
    finally:
        owner.close()

    with duckdb.connect(str(db_a), read_only=True) as conn_a:
        formal_slot = build_jq_ashare_local(conn_a, start=start, end=end)
        selected, plan = plan_market_factor_values(
            store, ["borrower"], {"borrower": receipt.factor_identity_hash}, start, end, codes,
            slot=formal_slot,
        )
        assert selected == {"borrower": receipt.factor_identity_hash}
        assert plan is not None
        read_back = FactorReader(str(tmp_path / "results.duckdb"), factor_values_store=store, slot=formal_slot).get_factor_values(
            "borrower", start, end,
        )
        assert read_back["000001.XSHE"].tolist() == [10.0, 11.0, 12.0]


def test_deferred_borrowed_slot_rejects_cross_db_without_polluting_owner(tmp_path):
    db_a = tmp_path / "cn_owner_a.duckdb"
    db_b = tmp_path / "cn_borrower_b.duckdb"
    profile_a, start, end, _ = _create_ashare_price_db(db_a, db_id="cn_owner_a")
    _create_ashare_price_db(db_b, db_id="cn_borrower_b")
    codes = ["000001.XSHE"]
    store = MarketFactorValueStore(tmp_path / "factors")

    rw_conn = duckdb.connect(str(db_a))
    try:
        owner = _AshareCloseFactor(
            None, "owner_a", start, end, stock_pool=codes, duckdb_path=str(db_a),
            data_bindings={}, pit_enabled=False,
        )
        assert owner.slot.market_db_profile is None
        with pytest.raises((MarketDbProfileError, duckdb.ConnectionException)):
            _AshareCloseFactor(
                None, "borrower_while_locked", start, end, stock_pool=codes, slot=owner.slot,
                duckdb_path=str(db_b), data_bindings={}, pit_enabled=False,
            )
        assert owner.slot.market_db_profile is None
        assert owner.slot.bar_feed.profile is None
    finally:
        rw_conn.close()

    try:
        with pytest.raises(MarketDbProfileError, match="MARKET_DB_PROFILE_HASH_MISMATCH"):
            _AshareCloseFactor(
                None, "borrower_b", start, end, stock_pool=codes, slot=owner.slot,
                duckdb_path=str(db_b), data_bindings={}, pit_enabled=False,
            )
        assert owner.slot.market_db_profile is None
        assert owner.slot.bar_feed.profile is None

        owner_values = owner.get_factor_values(force_recalculate=True)
        assert owner_values.to_numpy().tolist() == [[10.0], [11.0], [12.0]]
        assert owner.slot.market_db_profile == profile_a
        receipt = owner.save_factor_values(store, factor_values=owner_values)
        payload = json.loads(
            store.list_identities(factor_identity_hash=receipt.factor_identity_hash).iloc[0].identity_payload_json,
        )
        assert market_db_identity(payload) == market_db_identity(profile_a)
    finally:
        owner.close()


@pytest.mark.parametrize("save_fn", [public_save_factor_values, public_save_bar_factor_values])
@pytest.mark.parametrize("pass_precomputed", [False, True])
def test_public_save_factor_values_resolves_deferred_duckdb_path_identity(tmp_path, save_fn, pass_precomputed):
    db_a = tmp_path / "cn_public_save.duckdb"
    profile_a, start, end, _ = _create_ashare_price_db(db_a, db_id="cn_public_save")
    codes = ["000001.XSHE"]
    expected_identity = market_db_identity(profile_a)
    store = MarketFactorValueStore(tmp_path / "factors")

    rw_conn = duckdb.connect(str(db_a))
    try:
        factor = _AshareCloseFactor(
            None, "public_alpha", start, end, stock_pool=codes, duckdb_path=str(db_a),
            data_bindings={}, pit_enabled=False,
        )
        assert factor.slot.market_db_profile is None
    finally:
        rw_conn.close()

    try:
        if pass_precomputed:
            precomputed = pd.DataFrame(
                {"000001.XSHE": [10.0, 11.0, 12.0]},
                index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
            )
            receipt = save_fn(factor, store, factor_values=precomputed)
        else:
            receipt = save_fn(factor, store)

        assert factor.slot.market_db_profile == profile_a
        identities = store.list_identities(factor_identity_hash=receipt.factor_identity_hash)
        assert len(identities) == 1
        payload = json.loads(identities.iloc[0].identity_payload_json)
        assert market_db_identity(payload) == expected_identity
    finally:
        factor.close()

    with duckdb.connect(str(db_a), read_only=True) as conn_a:
        formal_slot = build_jq_ashare_local(conn_a, start=start, end=end)
        selected, _ = plan_market_factor_values(
            store, ["public_alpha"], {"public_alpha": receipt.factor_identity_hash}, start, end, codes,
            slot=formal_slot,
        )
        assert selected == {"public_alpha": receipt.factor_identity_hash}
        read_back = FactorReader(str(tmp_path / "results.duckdb"), factor_values_store=store, slot=formal_slot).get_factor_values(
            "public_alpha", start, end,
        )
        assert read_back["000001.XSHE"].tolist() == [10.0, 11.0, 12.0]

    # Unprofiled in-memory DB stays compatible via public save_factor_values / save_bar_factor_values.
    mem_store = MarketFactorValueStore(tmp_path / "mem_factors")
    with duckdb.connect(":memory:") as mem:
        mem.execute("""
            CREATE TABLE price_data_daily AS
            SELECT DATE '2026-01-05' + i::INT AS date, '000001.XSHE' AS stock_code, 10.0 + i AS close
            FROM range(3) t(i)
        """)
        mem_factor = _AshareCloseFactor(mem, "mem_public", start, end, stock_pool=codes, pit_enabled=False)
        mem_receipt = save_fn(mem_factor, mem_store)
        mem_payload = json.loads(
            mem_store.list_identities(factor_identity_hash=mem_receipt.factor_identity_hash).iloc[0].identity_payload_json,
        )
        assert market_db_identity(mem_payload) == {}


def test_parquet_only_slot_rejects_foreign_duckdb_borrower_without_polluting_owner(tmp_path):
    parquet_dir = tmp_path / "pq_root"
    part_dir = parquet_dir / "price_data_daily" / "year=2026" / "month=1"
    part_dir.mkdir(parents=True)
    pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        "stock_code": ["000001.XSHE"] * 3,
        "close": [90.0, 91.0, 92.0],
        "factor": [1.0, 1.0, 1.0],
        "volume": [100.0, 100.0, 100.0],
    }).to_parquet(part_dir / "prices.parquet", index=False)

    db_b = tmp_path / "probe_b.duckdb"
    _, start, end, _ = _create_ashare_price_db(db_b, db_id="probe_b")
    codes = ["000001.XSHE"]
    store = MarketFactorValueStore(tmp_path / "factors")

    pq_owner = _AshareCloseFactor(
        None, "pq_owner", start, end, stock_pool=codes, parquet_dir=str(parquet_dir),
        data_bindings={}, pit_enabled=False,
    )
    pq_borrower = None
    try:
        owner_vals = pq_owner.get_factor_values(force_recalculate=True)
        assert owner_vals.to_numpy().tolist() == [[90.0], [91.0], [92.0]]
        assert pq_owner.slot.market_db_profile is None
        assert pq_owner.slot.bar_feed.profile is None

        # Borrowing without a foreign DB source keeps working on the same Parquet feed.
        pq_borrower = _AshareCloseFactor(
            None, "pq_borrower", start, end, stock_pool=codes, slot=pq_owner.slot,
            data_bindings={}, pit_enabled=False,
        )
        assert pq_borrower.get_factor_values(force_recalculate=True).to_numpy().tolist() == [[90.0], [91.0], [92.0]]
        assert pq_borrower.slot.market_db_profile is None

        with pytest.raises(MarketDbProfileError):
            _AshareCloseFactor(
                None, "pq_cross_b", start, end, stock_pool=codes, slot=pq_owner.slot,
                duckdb_path=str(db_b), data_bindings={}, pit_enabled=False,
            )

        assert pq_owner.slot.market_db_profile is None
        assert pq_owner.slot.bar_feed.profile is None

        recomputed = pq_owner.get_factor_values(force_recalculate=True)
        assert recomputed.to_numpy().tolist() == [[90.0], [91.0], [92.0]]
        receipt = pq_owner.save_factor_values(store, factor_values=recomputed)
        payload = json.loads(
            store.list_identities(factor_identity_hash=receipt.factor_identity_hash).iloc[0].identity_payload_json,
        )
        assert market_db_identity(payload) == {}

        with duckdb.connect(str(db_b), read_only=True) as conn_b:
            slot_b = build_jq_ashare_local(conn_b, start=start, end=end)
            with pytest.raises(ValueError, match=r"Matching hashes: \[\]"):
                plan_market_factor_values(
                    store, ["pq_owner"], {"pq_owner": receipt.factor_identity_hash}, start, end, codes,
                    slot=slot_b,
                )
    finally:
        if pq_borrower is not None:
            pq_borrower.close()
        pq_owner.close()


def test_uncreated_duckdb_path_slot_rejects_foreign_duckdb_borrower_without_polluting_owner(tmp_path):
    db_a = tmp_path / "future_a.duckdb"
    db_b = tmp_path / "probe_b.duckdb"
    _create_ashare_price_db(db_b, db_id="probe_b")
    start, end = "2026-01-05", "2026-01-07"
    codes = ["000001.XSHE"]

    future_owner = _AshareCloseFactor(
        None, "future_owner", start, end, stock_pool=codes, duckdb_path=str(db_a),
        data_frequency="60min", analysis_mode="cross_sectional", data_bindings={}, pit_enabled=False,
    )
    assert future_owner.slot.market_db_profile is None
    assert future_owner.slot.bar_feed.profile is None
    assert not db_a.exists()

    # Legitimate borrowing without attaching a foreign source still defers cleanly.
    future_borrower = _AshareCloseFactor(
        None, "future_borrower", start, end, stock_pool=codes, slot=future_owner.slot,
        data_frequency="60min", analysis_mode="cross_sectional", data_bindings={}, pit_enabled=False,
    )
    assert future_borrower.slot.market_db_profile is None

    with pytest.raises(MarketDbProfileError):
        _AshareCloseFactor(
            None, "borrower_b", start, end, stock_pool=codes, slot=future_owner.slot,
            duckdb_path=str(db_b), data_bindings={}, pit_enabled=False,
        )

    assert future_owner.slot.market_db_profile is None
    assert future_owner.slot.bar_feed.profile is None
    assert future_borrower.slot.market_db_profile is None

    profile_a, _, _, _ = _create_ashare_price_db(db_a, db_id="future_a")
    with future_owner.slot.bar_feed._reader() as reader:
        assert reader.market_db_profile == profile_a
    assert future_owner.slot.market_db_profile == profile_a
    assert future_borrower.slot.market_db_profile == profile_a
    assert future_owner.slot.bar_feed.profile == profile_a


def test_dual_duckdb_and_parquet_slot_allows_same_binding_and_rejects_mismatched_parquet(tmp_path):
    db_a = tmp_path / "cn_dual_a.duckdb"
    profile_a, start, end, codes = _create_ashare_price_db(db_a, db_id="cn_dual_a")
    pq_a = tmp_path / "pq_a"
    pq_other = tmp_path / "pq_other"
    pq_a.mkdir()
    pq_other.mkdir()

    options = dict(
        duckdb_path=str(db_a),
        parquet_dir=str(pq_a),
        source_policy="duckdb_then_parquet_on_empty",
        data_bindings={},
        stock_pool=codes,
        pit_enabled=False,
    )
    owner = _AshareCloseFactor(None, "dual_owner", start, end, **options)
    borrower = None
    duckdb_only_owner = None
    try:
        borrower = _AshareCloseFactor(None, "dual_borrower", start, end, slot=owner.slot, **options)
        assert owner.slot.market_db_profile == profile_a
        assert borrower.slot.market_db_profile == profile_a
        expected = [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]
        assert owner.get_factor_values(force_recalculate=True).to_numpy().tolist() == expected
        assert borrower.get_factor_values(force_recalculate=True).to_numpy().tolist() == expected

        with pytest.raises(MarketDbProfileError):
            _AshareCloseFactor(
                None, "dual_wrong_pq", start, end, slot=owner.slot,
                **{**options, "parquet_dir": str(pq_other)},
            )

        duckdb_only_owner = _AshareCloseFactor(
            None, "duckdb_only", start, end, stock_pool=codes, duckdb_path=str(db_a),
            data_bindings={}, pit_enabled=False,
        )
        with pytest.raises(MarketDbProfileError):
            _AshareCloseFactor(None, "add_pq_to_duckdb", start, end, slot=duckdb_only_owner.slot, **options)
    finally:
        if duckdb_only_owner is not None:
            duckdb_only_owner.close()
        if borrower is not None:
            borrower.close()
        owner.close()


def test_unprofiled_same_file_dual_readonly_connections_preserve_init_compat_and_reject_distinct_sources(tmp_path):
    start, end = "2026-01-05", "2026-01-07"
    codes = ["000001.XSHE"]
    unprof_a = tmp_path / "unprof_a.duckdb"
    unprof_b = tmp_path / "unprof_b.duckdb"
    for path in (unprof_a, unprof_b):
        with duckdb.connect(str(path)) as conn:
            conn.execute(
                "CREATE TABLE price_data_daily AS "
                "SELECT DATE '2026-01-05' + i::INT AS date, '000001.XSHE' AS stock_code, 10.0 + i AS close "
                "FROM range(3) t(i)"
            )

    with (
        duckdb.connect(str(unprof_a), read_only=True) as conn1,
        duckdb.connect(str(unprof_a), read_only=True) as conn2,
        duckdb.connect(str(unprof_b), read_only=True) as conn_b,
        duckdb.connect(":memory:") as mem1,
        duckdb.connect(":memory:") as mem2,
    ):
        owner = _AshareCloseFactor(conn1, "ro_owner", start, end, stock_pool=codes, data_bindings={}, pit_enabled=False)
        borrower = _AshareCloseFactor(
            conn2, "ro_borrower", start, end, stock_pool=codes, slot=owner.slot, data_bindings={}, pit_enabled=False,
        )
        assert owner.slot.market_db_profile is None
        assert borrower.slot.market_db_profile is None
        assert borrower.get_price_data()["000001.XSHE"].tolist() == [10.0, 11.0, 12.0]
        with pytest.raises(MarketDbProfileError, match="MARKET_DB_PROFILE_MISSING"):
            borrower.slot.bar_feed.instrument_metadata(codes)

        with pytest.raises(MarketDbProfileError):
            _AshareCloseFactor(
                conn_b, "ro_diff_file", start, end, stock_pool=codes, slot=owner.slot, data_bindings={}, pit_enabled=False,
            )

        mem1.execute(
            "CREATE TABLE price_data_daily AS "
            "SELECT DATE '2026-01-05' + i::INT AS date, '000001.XSHE' AS stock_code, 10.0 + i AS close "
            "FROM range(3) t(i)"
        )
        mem2.execute(
            "CREATE TABLE price_data_daily AS "
            "SELECT DATE '2026-01-05' + i::INT AS date, '000001.XSHE' AS stock_code, 20.0 + i AS close "
            "FROM range(3) t(i)"
        )
        mem_owner = _AshareCloseFactor(mem1, "mem_owner", start, end, stock_pool=codes, pit_enabled=False)
        with pytest.raises(MarketDbProfileError):
            _AshareCloseFactor(mem2, "mem_other", start, end, stock_pool=codes, slot=mem_owner.slot, pit_enabled=False)
