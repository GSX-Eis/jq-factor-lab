"""Stage 7 fail-closed acceptance at the storage, Slot and Scheme B boundaries."""
import hashlib
import json

import duckdb
import pandas as pd
import pytest

from backtest_and_analysis.time_series.contracts import ContractValidationError
from data_pipeline.loaders.market_bar_importer import import_market_bars
from factor_synthesis.core.feature_preprocessor import cross_sectional_zscore
from market_db_samples import market_profile
from slots import build_slot_from_config
from slots.cn_futures import build_cn_futures_slot
from slots.global_futures import build_global_futures_slot
from slots.jq_ashare_local import daily_slot_config
from test_market_db_identity_and_cli import _save
from test_market_db_import_slot_guard import bar_inputs, us_declaration
from tools.agent_harness.cli import main
from utils.market_db import (
    MarketDbProfileError, adopt_market_db_sidecar, init_market_db, market_db_sidecar_path,
)
from utils.market_db.errors import (
    MARKET_DB_PROFILE_HASH_MISMATCH, MARKET_DB_PROFILE_INVALID, MARKET_DB_PROFILE_MISSING,
    MARKET_DB_PROVIDER_MISMATCH, MARKET_DB_SIDECAR_FINGERPRINT_MISMATCH,
)
from utils.market_factor import MarketFactorValueStore


@pytest.mark.parametrize(("domain", "symbol", "time", "metadata", "message"), [
    ("cn_equity", "AAPL.XNAS", "2026-01-05 14:30Z", {
        "exchange": "XNAS", "calendar_id": "XNYS", "timezone": "America/New_York",
        "rth_reference_calendar": "XNYS",
    }, "instrument_id"),
    ("cn_equity", "000001", "2026-01-05 01:30Z", {}, "instrument_id"),
    ("us_equity", "AAPL", "2026-01-05 14:30Z", {
        "exchange": "XNAS", "calendar_id": "XNYS", "timezone": "America/New_York",
        "rth_reference_calendar": "XNYS",
    }, "instrument_id"),
    ("cn_equity", "000001.XSHG", "2026-01-05 04:00Z", {}, "open Session"),
    ("global_futures", "NQM26.CME", "2026-01-05 14:30Z", {
        "asset_class": "futures", "exchange": "CME", "timezone": "America/Chicago",
        "calendar_id": None, "rth_reference_calendar": "NYSE", "point_value": 20.,
        "contract_expiry": "2026-06",
    }, "Explicit calendar_id"),
], ids=["us-bar-in-cn", "bare-cn", "bare-us", "outside-session", "cme-without-calendar"])
def test_stage7_invalid_import_has_no_writes(tmp_path, capsys, domain, symbol, time, metadata, message):
    path = tmp_path / "market.duckdb"
    init_market_db(path, market_profile(domain))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    frame, instruments = bar_inputs(symbol, time, **metadata)
    if domain == "us_equity":
        frame["frequency"] = "1d"
    code = "bar_storage_precondition_failed" if message == "open Session" else MARKET_DB_PROFILE_INVALID

    with duckdb.connect(str(path)) as conn:
        error_type = ContractValidationError if message == "open Session" else MarketDbProfileError
        with pytest.raises(error_type, match=message) as exc:
            import_market_bars(conn, frame, instruments, legacy_compatibility=False)
        assert type(exc.value) is error_type
        if isinstance(exc.value, MarketDbProfileError):
            assert exc.value.code == code

    bars, meta = tmp_path / "bars.csv", tmp_path / "instruments.json"
    frame.to_csv(bars, index=False)
    instruments.to_json(meta, orient="records")
    assert main(["import-bars", "--db", str(path), "--bars", str(bars), "--metadata", str(meta),
                 "--output-root", str(tmp_path / "runs")]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["reason"] == code
    assert message in result["metrics"]["message"]
    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute("SHOW TABLES").fetchall() == [("market_db_profile",)]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(("scenario", "code"), [
    ("us_database", MARKET_DB_PROVIDER_MISMATCH),
    ("unprofiled", MARKET_DB_PROFILE_MISSING),
    ("tampered_table", MARKET_DB_PROFILE_HASH_MISMATCH),
    ("stale_sidecar", MARKET_DB_SIDECAR_FINGERPRINT_MISMATCH),
])
def test_stage7_invalid_slot_open_has_no_writes(tmp_path, capsys, scenario, code):
    path = tmp_path / "market.duckdb"
    if scenario in {"us_database", "tampered_table"}:
        init_market_db(path, market_profile("us_equity" if scenario == "us_database" else "cn_equity"))
    else:
        with duckdb.connect(str(path)) as conn:
            conn.execute("CREATE TABLE legacy (i INT)")

    if scenario == "tampered_table":
        with duckdb.connect(str(path)) as conn:
            conn.execute("UPDATE market_db_profile SET profile_hash = ?", ["0" * 64])
    if scenario == "stale_sidecar":
        adopt_market_db_sidecar(path, market_profile("cn_equity"))
        with duckdb.connect(str(path)) as conn:
            conn.execute("INSERT INTO legacy VALUES (7)")

    sidecar = market_db_sidecar_path(path)
    database_bytes, sidecar_bytes = path.read_bytes(), sidecar.read_bytes() if sidecar.exists() else None
    with duckdb.connect(str(path), read_only=True) as conn:
        with pytest.raises(MarketDbProfileError) as exc:
            build_slot_from_config(daily_slot_config(), connection=conn)
        assert type(exc.value) is MarketDbProfileError
        assert exc.value.code == code

    if scenario != "us_database":
        bars, metadata = bar_inputs()
        csv, meta = tmp_path / "bars.csv", tmp_path / "instruments.json"
        bars.to_csv(csv, index=False)
        metadata.to_json(meta, orient="records")
        assert main(["import-bars", "--db", str(path), "--bars", str(csv), "--metadata", str(meta),
                     "--output-root", str(tmp_path / "runs")]) == 2
        assert json.loads(capsys.readouterr().out)["summary"]["reason"] == code
    assert path.read_bytes() == database_bytes
    assert (sidecar.read_bytes() if sidecar.exists() else None) == sidecar_bytes


@pytest.mark.parametrize(("field", "claimed", "implemented"), [
    ("st_filter", False, True),
    ("price_limit", False, True),
    ("t_plus", 0, 1),
])
def test_stage7_native_ashare_rejects_unsupported_market_rule(tmp_path, field, claimed, implemented):
    path = tmp_path / "market.duckdb"
    profile = market_profile("cn_equity")
    init_market_db(path, market_profile("cn_equity", market_rules={**profile.market_rules, field: claimed}))
    with duckdb.connect(str(path), read_only=True) as conn:
        with pytest.raises(MarketDbProfileError) as exc:
            build_slot_from_config(
                daily_slot_config(start="2026-01-05", end="2026-01-09"), connection=conn,
            )
    assert exc.value.code == MARKET_DB_PROFILE_INVALID
    assert f"market_rules.{field}: profile={claimed!r}, implemented={implemented!r}" in str(exc.value)


def test_stage7_declarative_us_rejects_unsupported_market_rule(tmp_path):
    path = tmp_path / "us.duckdb"
    profile = market_profile("us_equity")
    init_market_db(path, market_profile("us_equity", market_rules={**profile.market_rules, "t_plus": 1}))
    with pytest.raises(MarketDbProfileError) as exc:
        build_slot_from_config(us_declaration(path))
    assert exc.value.code == MARKET_DB_PROFILE_INVALID
    assert "market_rules.t_plus: profile=1, implemented=0" in str(exc.value)


@pytest.mark.parametrize(("domain", "builder", "product"), [
    ("cn_futures", build_cn_futures_slot, "RB"),
    ("global_futures", build_global_futures_slot, "NQ"),
])
def test_stage7_futures_rejects_unsupported_market_rule(tmp_path, domain, builder, product):
    path = tmp_path / f"{domain}.duckdb"
    profile = market_profile(domain)
    init_market_db(path, market_profile(domain, market_rules={**profile.market_rules, "price_limit": True}))
    with duckdb.connect(str(path), read_only=True) as conn:
        with pytest.raises(MarketDbProfileError) as exc:
            builder(conn, product_root=product, start="2026-01-05", end="2026-01-09")
    assert exc.value.code == MARKET_DB_PROFILE_INVALID
    assert "market_rules.price_limit: profile=True, implemented=False" in str(exc.value)


def test_stage7_same_symbol_in_two_databases_has_distinct_identity(tmp_path):
    store = MarketFactorValueStore(tmp_path / "scheme-b")
    symbol = "000001.XSHE"
    hashes = []
    for ordinal, db_id in enumerate(("cn_first", "cn_second"), start=1):
        path = tmp_path / f"{db_id}.duckdb"
        profile = market_profile("cn_equity", db_id=db_id)
        init_market_db(path, profile)
        with duckdb.connect(str(path), read_only=True) as conn:
            slot = build_slot_from_config(
                daily_slot_config(start="2026-01-05", end="2026-01-09"), connection=conn,
            )
            digest, payload = _save(store, slot, float(ordinal), instrument=symbol)
        assert payload["db_id"] == db_id
        assert payload["profile_hash"] == profile.profile_hash
        hashes.append(digest)

    assert hashes[0] != hashes[1]
    identities = store.list_identities().set_index("factor_identity_hash")
    assert set(identities.index) == set(hashes)
    for ordinal, digest in enumerate(hashes, start=1):
        rows = store.read_factor_values(factor_identity_hash=digest)
        assert rows.instrument_id.unique().tolist() == [symbol]
        assert rows.factor_value.unique().tolist() == [float(ordinal)]

    before = store.verify_store()
    mixed = store.read_factor_values(factor_id="alpha")
    snapshot = mixed.copy(deep=True)
    with pytest.raises(ValueError, match=r"multiple market databases or domains \(db_id\)") as exc:
        cross_sectional_zscore(mixed)
    assert type(exc.value) is ValueError
    pd.testing.assert_frame_equal(mixed, snapshot)
    assert before["ok"] and store.verify_store() == before
