"""Exercise the XNYS profile from Harness import through bound factor evaluation."""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from test_factor_base_slot_single_track import TurnoverFactor
def _json(path, data):
    path.write_text(json.dumps(data, sort_keys=True, allow_nan=False), encoding='utf-8')
    return path
from tools.agent_harness.cli import main
from utils.bar_evaluation_store import query_bar_evaluation_detail
from utils.market_db import load_market_db_profile_yaml
from utils.market_db.identity import market_db_identity
from utils.market_db.research import open_research_slot
from utils.market_factor import MarketFactorValueStore
from utils.session_calendar import SessionCalendar
from utils.slots import ExecutionProfile, ProviderConfig, SlotConfig
from utils.time_axis import BarSchedule, Frequency

ROOT = Path(__file__).resolve().parents[1]
START, END = "2026-03-05", "2026-03-13"
SYMBOLS = ["AAPL.XNAS", "MSFT.XNAS", "JPM.XNYS"]


def _assert_no_ashare_rules(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"st_filter", "price_limit"}:
                assert child is False
            _assert_no_ashare_rules(key)
            _assert_no_ashare_rules(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_ashare_rules(child)
    elif isinstance(value, str):
        assert "XSHG" not in value
        assert "LOCAL_A_SHARE_XSHG_CROSSCHECKED" not in value


def test_stage7_us_equity_import_compute_persist_and_evaluate(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("QUANT_OUTPUT_CONTEXT", raising=False)
    template = ROOT / "config/market_databases/us_equity.yaml"
    profile = load_market_db_profile_yaml(template)
    identity = market_db_identity(profile)
    database = tmp_path / "ny.duckdb"
    out = tmp_path / "runs"
    def run(*args, run_id="stage7-us"):
        code = main([*map(str, args), "--output-root", str(out), "--run-id", run_id])
        payload = json.loads(capsys.readouterr().out)
        assert code == payload["exit_code"] == 0, payload
        return payload

    initialized = run("market-db", "init", "--db", database, "--profile", template)
    assert initialized["metrics"]["profile_hash"] == profile.profile_hash

    calendar = SessionCalendar("XNYS", START, END)
    schedule = BarSchedule(calendar, Frequency.parse("1d"), bucket_strategy="session")
    rows = []
    for day, session in enumerate(calendar.sessions()):
        timestamp = schedule.for_session(session.strftime("%Y-%m-%d")).start_utc.item()
        for index, symbol in enumerate(SYMBOLS):
            close = 50.0 + 20 * index + day * (index + 1) + (-1) ** (index + day) * 0.5
            rows.append({
                "session_id": session.strftime("%Y-%m-%d"), "timestamp_utc": timestamp,
                "instrument_id": symbol, "frequency": "1d", "open": close - 0.25,
                "high": close + 1, "low": close - 1, "close": close,
                "volume": float(100 + 5 * day + index), "source": "synthetic", "data_version": "stage7",
            })
    bars = pd.DataFrame(rows)
    csv = tmp_path / "bars.csv"
    bars.drop(columns="session_id").to_csv(csv, index=False)
    metadata = _json(tmp_path / "instruments.json", [
        {"instrument_id": symbol, "exchange": symbol.split(".")[1], "asset_class": "equity",
         "calendar_id": "XNYS", "rth_reference_calendar": "XNYS", "timezone": "America/New_York",
         "price_adjustment": "none"}
        for symbol in SYMBOLS
    ])
    imported = run("import-bars", "--db", database, "--bars", csv, "--metadata", metadata)
    assert imported["metrics"]["imported_bars"] == len(bars)
    # The declarative adapter accepts a source projection of the canonical bars.
    with duckdb.connect(str(database)) as conn:
        conn.execute("""
            CREATE VIEW us_daily_inputs AS
            SELECT instrument_id, timestamp_utc, open, high, low, close, volume FROM market_bars_1d
        """)
    market_bytes = hashlib.sha256(database.read_bytes()).hexdigest()

    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps({"market_dbs": {profile.db_id: str(database)}}))
    slot_config = SlotConfig(
        1, "stage7_us", ProviderConfig("pmc", {"calendar_id": "XNYS", "start": START, "end": END}),
        ProviderConfig("duckdb_table", {
            "table": "us_daily_inputs", "frequency": "1d", "timestamp_anchor": "bar_start",
            "source_timezone": "America/New_York", "volume_mode": "interval", "data_version": "stage7",
            "column_mapping": {"timestamp_utc": "timestamp", **{name: name for name in (
                "instrument_id", "open", "high", "low", "close", "volume",
            )}},
        }),
        ProviderConfig("static", {"instruments": {
            symbol: {"asset_class": "equity", "currency": "USD"} for symbol in SYMBOLS
        }}),
        ProviderConfig("none"), ProviderConfig("open"),
        ExecutionProfile("close", "close", 0, 0, "decimal", "endpoints_only", "cross_session"),
        market_domain="us_equity",
    )

    store = MarketFactorValueStore(tmp_path / "factors")
    with open_research_slot(profile.db_id, slot_config, tmp_path) as (slot, path):
        assert path == database
        assert slot.calendar.calendar_id == "XNYS"
        assert slot.calendar.timezone == "America/New_York"
        assert {slot.universe.instrument_spec(symbol).currency for symbol in SYMBOLS} == {"USD"}
        clocks = slot.calendar.bar_grid("1d").set_index("session_id").timestamp_utc
        assert clocks.loc["2026-03-06"] == pd.Timestamp("2026-03-06 21:00Z")
        assert clocks.loc["2026-03-09"] == pd.Timestamp("2026-03-09 20:00Z")
        factor = TurnoverFactor(None, "turnover", START, END, stock_pool=SYMBOLS, slot=slot,
                                data_frequency="1d", cache_dir=str(tmp_path / "cache"))
        values = factor.get_factor_values(force_recalculate=True)
        assert values.notna().sum().sum() == len(bars)
        receipt = factor.save_bar_factor_values(store)

    factor_id = receipt.factor_identity_hash
    assert store.verify_store()["ok"]
    identities = store.list_identities(factor_identity_hash=factor_id)
    assert len(identities) == 1
    assert market_db_identity(identities.iloc[0].to_dict()) == identity
    factor_identity = json.loads(identities.iloc[0].identity_payload_json)
    assert market_db_identity(factor_identity) == identity
    _assert_no_ashare_rules(factor_identity)
    keys = ["session_id", "instrument_id"]
    saved = store.read_factor_values(factor_identity_hash=factor_id).set_index(keys).sort_index()
    expected = bars.assign(turnover=bars.close * bars.volume).set_index(keys).sort_index()
    assert saved.index.tolist() == expected.index.tolist()
    np.testing.assert_allclose(saved.factor_value, expected.turnover, rtol=0, atol=1e-12)

    request = _json(tmp_path / "evaluation.json", {
        "slot_config": slot_config.as_dict(), "factor_store_root": str(store.root), "factor_id": "turnover",
        "analysis_mode": "cross_sectional", "frequency": "1d", "horizons": [1], "n_quantiles": 3,
        "start_date": START, "end_date": END,
    })
    for command in ("single-factor-analysis", "time-series-analysis"):
        destination = tmp_path / command
        results_db = tmp_path / f"{command}.duckdb"
        result = run(command, "--config", request, "--market-db", profile.db_id,
                     "--output-dir", destination, "--results-db", results_db, run_id=command)
        manifest_path = Path(result["summary"]["summary_path"])
        assert manifest_path.is_relative_to(out)
        manifest = json.loads(manifest_path.read_text())
        assert manifest == result
        summary = json.loads((destination / "summary.json").read_text())
        stored = query_bar_evaluation_detail(results_db, result["metrics"]["results_db"]["identity_hash"])
        assert market_db_identity(manifest["metrics"]["data_identity"]) == identity
        assert market_db_identity(summary["data_identity"]) == identity
        assert market_db_identity(stored["run"]["data_identity"]) == identity
        assert summary["calendar_identity"]["calendar_id"] == "XNYS"
        assert summary["config"]["factor_identity_hash"] == factor_id
        assert summary["ic_by_horizon"]
        for document in (manifest, summary, stored["run"]):
            _assert_no_ashare_rules(document)
        labels = pd.read_csv(destination / "per_bar_labels.csv")
        assert set(labels.instrument_id) == set(SYMBOLS)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == market_bytes
