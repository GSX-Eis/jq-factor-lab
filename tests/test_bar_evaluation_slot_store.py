import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.time_series import (
    RESULT_SCHEMA_VERSION, BarReturnLabeler, TimeSeriesEvaluator, evaluate_bars, write_evaluation_artifacts,
)
from utils.bar_evaluation import grid_validation
from utils.bar_evaluation_store import (
    BarEvaluationConflictError, BarEvaluationIntegrityError, ResultIdentity,
    get_bar_evaluation_filter_options, ingest_evaluation, migrate_bar_evaluation_db,
    query_bar_evaluation_detail, query_bar_evaluation_runs, query_bar_evaluations, sha256_file,
)
from utils.session_calendar import SessionCalendar
from utils.slots import slot_config_hash
from utils.time_axis import BarSchedule


@pytest.fixture
def custom_calendar():
    template = {
        "timezone": "UTC", "weekdays": [0, 1, 2, 3, 4],
        "segments": [{"open": "09:00", "close": "16:00"}],
        "overrides": {"2025-01-08": [{"open": "09:00", "close": "13:00"}], "2025-01-15": None},
    }
    calendar = SessionCalendar("custom", "2025-01-06", "2025-01-24", custom_sessions=template)
    return calendar, template


@pytest.fixture
def publish(tmp_path, custom_calendar):
    calendar, template = custom_calendar

    def make(frequency="1d", unit="decimal", *, slot="synthetic", config_calendar=False,
             frozen_entry=None, override=None, analysis_mode="time_series", anchor="close"):
        session_calendar = (calendar if anchor == "close" else SessionCalendar(
            calendar.calendar_id, "2025-01-06", "2025-01-24", custom_sessions=template, daily_anchor=anchor,
        ))
        schedule = BarSchedule(session_calendar, frequency)
        slots = pd.concat([schedule.for_session(day.date().isoformat()) for day in session_calendar.sessions()],
                          ignore_index=True)
        # The split grid includes an unobserved slot.
        slots = slots.iloc[np.arange(len(slots)) != 4]
        i = np.arange(len(slots))
        price = 100.0 + i
        frame = pd.DataFrame(dict(
            timestamp_utc=slots.start_utc, instrument_id="A", frequency=frequency, open=price,
            high=price + 4, low=price - 1, close=price + 1 + (i % 3), factor_value=i,
            volume=1, source="synthetic", data_version="v1",
        ))
        if analysis_mode == "cross_sectional":
            frame = pd.concat([
                frame.assign(instrument_id="A"),
                frame.iloc[1:].assign(instrument_id="B", close=frame.close + 1, high=frame.high + 1,
                                      factor_value=frame.factor_value + 1),
                frame.assign(instrument_id="C", close=frame.close + 2, high=frame.high + 2,
                             factor_value=frame.factor_value + 2),
            ], ignore_index=True)
        effective_entry = (override or {}).get("entry_price_field", frozen_entry or "open")

        class FixtureLabeler(BarReturnLabeler):
            def label(self, bars, *args, **kwargs):
                labels = super().label(bars, *args, **kwargs)
                valid = labels["status"].eq("valid")
                if effective_entry == "close":
                    close = bars.set_index(["instrument_id", "timestamp_utc"])["close"]
                    keys = pd.MultiIndex.from_frame(labels.loc[valid, ["instrument_id", "signal_timestamp"]])
                    labels.loc[valid, "entry_price"] = close.reindex(keys).to_numpy()
                    labels.loc[valid, "entry_timestamp"] = labels.loc[valid, "signal_available_at"]
                change = labels.loc[valid, "exit_price"] - labels.loc[valid, "entry_price"]
                labels.loc[valid, "price_change_points"] = change
                returns = labels.loc[valid, "exit_price"] / labels.loc[valid, "entry_price"] - 1
                labels.loc[valid, "return_pct"] = returns * (100 if unit == "percent" else 1)
                return labels

        labeler = FixtureLabeler(session_calendar, horizons=[1])
        # Generate synthetic cross-session output with the existing Bar test producer.
        labeler.boundary_policy = "cross_session"
        result = evaluate_bars(
            frame, labeler=labeler,
            analysis_mode=analysis_mode, n_quantiles=2,
            split={"train_end": slots.iloc[6].end_utc.isoformat(), "oos_start": slots.iloc[8].end_utc.isoformat()},
            **({"bootstrap": {"n_boot": 9, "min_valid_replicates": 2}} if analysis_mode == "time_series" else {}),
        )
        if slot is not None:
            result.data_identity.update(slot_id=slot, slot_config_hash="a" * 64,
                                        slot_data_fingerprint={"bars": {"build_id": "bars-v1"}},
                                        build_id="bars-v1")
        config = {}
        if config_calendar or frozen_entry is not None:
            config["slot_config"] = {"calendar": {"provider": "custom_sessions", "options": {
                "calendar_id": "custom", "start": "2025-01-06", "end": "2025-01-24", "custom_sessions": template,
                "daily_anchor": anchor,
            }}}
        else:
            result.calendar_identity["schedule"]["custom_sessions"] = template
        if frozen_entry is not None:
            config["slot_config"].update(
                schema_version=1, slot_id=slot,
                execution_profile=dict(entry_price_field=frozen_entry, exit_price_field="close",
                                       min_entry_offset_bars=int(frozen_entry == "open"),
                                       exit_adjustment=int(frozen_entry == "close") - 1, return_unit=unit,
                                       gap_rule="endpoints_only", boundary_policy="cross_session"),
                **{field: {"provider": "fixture"} for field in ("bar_feed", "universe", "aux_features", "microstructure")},
            )
            result.data_identity["slot_config_hash"] = slot_config_hash(config["slot_config"])
        if override is not None:
            config["execution_profile"] = override
        identity = ResultIdentity("b" * 64, {"bars": "c" * 64}, session_calendar.calendar_hash,
                                  "slot-test", RESULT_SCHEMA_VERSION)
        paths = write_evaluation_artifacts(result, tmp_path / "artifacts", result_identity=identity, config=config)
        if effective_entry == "close":
            labels = pd.read_csv(paths["per_bar_labels"], float_precision="round_trip")
            labels["entry_offset_bars"] = 0
            labels.to_csv(paths["per_bar_labels"], index=False)
        summary = json.loads(paths["summary"].read_text())
        summary.update(frequency="daily" if frequency == "1d" else frequency, return_unit=unit)
        paths["summary"].write_text(json.dumps(summary))
        return paths, summary

    return make


def ingest(db, paths):
    return ingest_evaluation(db, paths["summary"].parent,
                             expected_hashes={name: sha256_file(path) for name, path in paths.items()})


@pytest.mark.parametrize(("frequency", "unit", "config_calendar"), [
    ("1d", "decimal", False), ("2h", "percent", True), ("4h", "decimal", False),
])
def test_slot_full_frequency_ingest(tmp_path, publish, frequency, unit, config_calendar):
    db = tmp_path / "evaluation.duckdb"
    paths, summary = publish(frequency, unit, config_calendar=config_calendar,
                             anchor="open" if config_calendar else "close")
    receipt = ingest(db, paths)
    assert receipt.status == "inserted" and receipt.split_rows == 2
    assert ingest(db, paths).status == "noop"
    detail = query_bar_evaluation_detail(db, receipt.identity_hash)
    run = detail["run"]
    assert run["frequency"] == frequency and run["return_unit"] == unit
    assert run["slot_id"] == "synthetic" and run["slot_config_hash"] == "a" * 64
    for key, value in summary["data_identity"].items():
        assert run["data_identity"][key] == value
    assert query_bar_evaluation_runs(db, slot_id="synthetic", frequency=summary["frequency"]) == [run]
    assert query_bar_evaluation_runs(db, slot_id="other") == []
    assert query_bar_evaluations(db, slot_id="other")["run_count"] == 0
    assert get_bar_evaluation_filter_options(db)["slot_ids"] == ["synthetic"]
    for row in detail["split_metrics"]:
        expected, = summary["splits"][row["split"]]["horizons"]
        assert row["metrics_json"]["bootstrap"]["grid_size"] == expected["bootstrap"]["grid_size"]
        assert row["metrics_json"]["return_distribution"] == expected["return_distribution"]
        assert row["metrics_json"]["buckets"] == expected["buckets"]
    labels = pd.read_csv(paths["per_bar_labels"], float_precision="round_trip")
    oos = summary["splits"]["oos"]["oos_start"]
    valid_oos = labels.loc[labels["status"].eq("valid") & labels["signal_available_at"].ge(oos)]
    oos_row = next(row for row in detail["split_metrics"] if row["split"] == "oos")
    assert oos_row["metrics_json"]["return_distribution"]["mean"] == pytest.approx(valid_oos["return_pct"].mean())


@pytest.mark.parametrize("statistic", ["return_distribution", "split_bucket", "quantile_stats"])
def test_slot_rejects_mixed_unit_statistics(tmp_path, publish, statistic):
    paths, summary = publish()
    oos, = summary["splits"]["oos"]["horizons"]
    if statistic == "return_distribution":
        oos["return_distribution"]["mean"] *= 100
    elif statistic == "split_bucket":
        distribution = next(bucket["return_distribution"] for bucket in oos["buckets"].values()
                            if bucket["return_distribution"]["n_samples"])
        distribution["p50"] *= 100
    else:
        summary["quantile_stats"][0]["mean_return_pct"] *= 100
        quantiles = pd.read_csv(paths["quantile_stats"], float_precision="round_trip")
        quantiles.loc[0, "mean_return_pct"] *= 100
        quantiles.to_csv(paths["quantile_stats"], index=False)
    paths["summary"].write_text(json.dumps(summary))
    db = tmp_path / "mixed.duckdb"
    with pytest.raises(BarEvaluationIntegrityError, match=statistic.replace("split_bucket", "buckets")):
        ingest(db, paths)
    assert not db.exists()


def test_slot_cross_sectional_decimal_returns(tmp_path, publish):
    paths, summary = publish(analysis_mode="cross_sectional")
    db = tmp_path / "cross_sectional.duckdb"
    receipt = ingest(db, paths)
    detail = query_bar_evaluation_detail(db, receipt.identity_hash)
    assert detail["run"]["return_unit"] == "decimal"
    for split in detail["split_metrics"]:
        expected, = summary["splits"][split["split"]]["horizons"]
        assert split["metrics_json"]["buckets"] == expected["buckets"]
        assert split["metrics_json"]["spread"] == expected["spread"]
    oos, = summary["splits"]["oos"]["horizons"]
    oos["spread"]["mean"] *= 100
    paths["summary"].write_text(json.dumps(summary))
    with pytest.raises(BarEvaluationIntegrityError, match="spread.mean"):
        ingest(tmp_path / "mixed.duckdb", paths)


@pytest.mark.parametrize(("field", "value"), [("slot_config_hash", "a" * 64), ("slot_id", "other")])
def test_slot_rejects_identity_disagreeing_with_frozen_config(tmp_path, publish, field, value):
    paths, summary = publish(frozen_entry="open")
    # Keep the older inline schedule reconstructible so identity is the only defect.
    summary["calendar_identity"]["schedule"]["custom_sessions"] = (
        summary["config"]["slot_config"]["calendar"]["options"]["custom_sessions"]
    )
    summary["data_identity"][field] = value
    paths["summary"].write_text(json.dumps(summary))
    with pytest.raises(BarEvaluationIntegrityError, match=field):
        ingest(tmp_path / "forged.duckdb", paths)


@pytest.mark.parametrize("schedule_present", [False, True])
def test_slot_frozen_config_keeps_omitted_optional_identity_fields(tmp_path, publish, schedule_present):
    paths, summary = publish(unit="percent", frozen_entry="close")
    summary["frequency"] = "bar"
    if not schedule_present:
        del summary["calendar_identity"]["schedule"]
    for field in ("slot_id", "slot_config_hash"):
        del summary["data_identity"][field]
    paths["summary"].write_text(json.dumps(summary))
    db = tmp_path / "untagged.duckdb"
    assert ingest(db, paths).status == "inserted"
    run, = query_bar_evaluation_runs(db)
    assert run["slot_id"] is None and run["slot_config_hash"] is None


@pytest.mark.parametrize("override", [None, "decimal", "percent"])
def test_slot_enforces_effective_return_unit(tmp_path, publish, override):
    paths, summary = publish(frozen_entry="open")
    slot = summary["config"]["slot_config"]
    slot["execution_profile"]["return_unit"] = "percent"
    summary["data_identity"]["slot_config_hash"] = slot_config_hash(slot)
    if override is not None:
        summary["config"]["execution_profile"] = {"return_unit": override}
    paths["summary"].write_text(json.dumps(summary))
    db = tmp_path / "unit.duckdb"
    if override == "decimal":
        assert ingest(db, paths).status == "inserted"
        assert query_bar_evaluation_runs(db)[0]["return_unit"] == "decimal"
    else:
        with pytest.raises(BarEvaluationIntegrityError, match="return_unit"):
            ingest(db, paths)
        assert not db.exists()


@pytest.mark.parametrize(("conflict", "published_hash"), [
    (None, True), (None, False), ("end_date", True), ("calendar_hash", True),
    ("calendar_hash", False), ("calendar_id", True),
])
def test_slot_checks_frozen_calendar_without_published_optional_ids(tmp_path, publish, conflict, published_hash):
    paths, summary = publish(frozen_entry="open")
    for field in ("slot_id", "slot_config_hash"):
        del summary["data_identity"][field]
    calendar = summary["calendar_identity"]
    del calendar["calendar_id"]
    if not published_hash:
        del calendar["calendar_hash"]
    options = summary["config"]["slot_config"]["calendar"]["options"]
    if conflict == "end_date":
        options["end"] = "2025-01-20"
    elif conflict == "calendar_hash":
        # Both declarations agree with one another but no longer reconstruct the sealed hash.
        options["end"] = calendar["schedule"]["end_date"] = "2025-01-20"
    elif conflict == "calendar_id":
        del options["calendar_id"]
    paths["summary"].write_text(json.dumps(summary))
    db = tmp_path / "calendar.duckdb"
    if conflict is None:
        assert ingest(db, paths).status == "inserted"
        run, = query_bar_evaluation_runs(db)
        assert run["slot_id"] is None and run["slot_config_hash"] is None
    else:
        with pytest.raises(BarEvaluationIntegrityError, match=conflict):
            ingest(db, paths)
        assert not db.exists()


@pytest.mark.parametrize(("start_suffix", "end_suffix", "valid"), [
    ("T00:00:00", "T00:00:00", True),
    ("T12:00:00", "", False),
    ("", "T00:00:00+00:00", False),
])
def test_slot_compares_calendar_session_dates(tmp_path, publish, start_suffix, end_suffix, valid):
    paths, summary = publish(frozen_entry="open")
    slot = summary["config"]["slot_config"]
    slot["calendar"]["options"]["start"] += start_suffix
    slot["calendar"]["options"]["end"] += end_suffix
    summary["data_identity"]["slot_config_hash"] = slot_config_hash(slot)
    paths["summary"].write_text(json.dumps(summary))
    db = tmp_path / "dates.duckdb"
    if valid:
        assert ingest(db, paths).status == "inserted"
    else:
        with pytest.raises(BarEvaluationIntegrityError, match="schedule.(start|end)_date"):
            ingest(db, paths)
        assert not db.exists()


@pytest.mark.parametrize("field", ["custom_sessions", "daily_anchor", "end_date", "calendar_id"])
def test_slot_rejects_conflicting_calendar_declarations(tmp_path, publish, field):
    paths, summary = publish(config_calendar=True)
    identity = summary["calendar_identity"]
    options = summary["config"]["slot_config"]["calendar"]["options"]
    identity["schedule"]["custom_sessions"] = options["custom_sessions"]
    if field == "custom_sessions":
        identity["schedule"][field] = {**options[field], "weekdays": [0, 1, 2, 3]}
    elif field == "daily_anchor":
        identity["schedule"][field], options[field] = "close", "open"
    elif field == "end_date":
        options["end"] = "2025-01-25"
    else:
        options["calendar_id"] = "other"
    paths["summary"].write_text(json.dumps(summary))
    with pytest.raises(BarEvaluationIntegrityError, match=field):
        ingest(tmp_path / "conflicting.duckdb", paths)


@pytest.mark.parametrize(("entry", "override", "forged"), [
    ("close", None, False),
    ("open", {"entry_price_field": "close", "min_entry_offset_bars": 0}, False),
    ("open", None, True),
    ("close", {"min_entry_offset_bars": 1}, True),
])
def test_slot_enforces_effective_execution_offset(tmp_path, publish, entry, override, forged):
    paths, summary = publish(frozen_entry=entry, override=override)
    summary["calendar_identity"]["schedule"]["custom_sessions"] = (
        summary["config"]["slot_config"]["calendar"]["options"]["custom_sessions"]
    )
    paths["summary"].write_text(json.dumps(summary))
    if forged:
        labels = pd.read_csv(paths["per_bar_labels"], float_precision="round_trip")
        labels["entry_offset_bars"] = 0
        labels.to_csv(paths["per_bar_labels"], index=False)
        with pytest.raises(BarEvaluationIntegrityError, match="entry_offset_bars"):
            ingest(tmp_path / "forged.duckdb", paths)
    else:
        assert ingest(tmp_path / "close.duckdb", paths).status == "inserted"


@pytest.mark.parametrize("offset", [0, 1, 2])
def test_slot_enforces_explicit_entry_offset(tmp_path, publish, offset):
    paths, summary = publish(frozen_entry="open")
    summary["config"]["entry_offset_bars"] = offset
    paths["summary"].write_text(json.dumps(summary))
    db = tmp_path / "offset.duckdb"
    if offset == 1:
        assert ingest(db, paths).status == "inserted"
    else:
        with pytest.raises(BarEvaluationIntegrityError, match="entry_offset_bars"):
            ingest(db, paths)


@pytest.mark.parametrize(("location", "field", "value"), [
    ("data_identity", "slot_id", " "),
    ("data_identity", "slot_id", None),
    ("data_identity", "slot_config_hash", "A" * 64),
    ("data_identity", "slot_data_fingerprint", {}),
    ("data_identity", "data_versions", [1]),
    ("data_identity", "sources", "synthetic"),
    ("data_identity", "frequencies", ["2h"]),
    (None, "frequency", "2h"),
    (None, "return_unit", "basis_points"),
    (None, "return_unit", "percent"),
    ("calendar_identity.schedule", "end_date", "2025-01-25"),
])
def test_slot_inconsistent_rehashed_artifact_is_rejected(tmp_path, publish, location, field, value):
    paths, summary = publish()
    target = summary
    for key in location.split(".") if location else ():
        target = target[key]
    target[field] = value
    paths["summary"].write_text(json.dumps(summary))
    db = tmp_path / "evaluation.duckdb"
    diagnostic = "calendar_hash" if field == "end_date" else field
    with pytest.raises(BarEvaluationIntegrityError, match=diagnostic):
        ingest(db, paths)
    assert not db.exists()


@pytest.mark.parametrize("slot", [None, "synthetic"])
def test_slot_conflict_and_read_only_fallback_before_migration(tmp_path, publish, slot):
    db = tmp_path / "evaluation.duckdb"
    paths, summary = publish(slot=slot)
    receipt = ingest(db, paths)
    with duckdb.connect(str(db)) as conn:
        conn.execute("ALTER TABLE bar_evaluation_runs DROP COLUMN slot_id")
        conn.execute("ALTER TABLE bar_evaluation_runs DROP COLUMN slot_config_hash")
    before = db.read_bytes()
    run, = query_bar_evaluation_runs(db, slot_id="any")
    assert run["slot_id"] is None and run["slot_config_hash"] is None
    assert get_bar_evaluation_filter_options(db)["slot_ids"] == []
    assert db.read_bytes() == before
    migrate_bar_evaluation_db(db)
    migrate_bar_evaluation_db(db)
    assert ingest(db, paths).status == "noop"
    assert query_bar_evaluation_runs(db, slot_id=slot) == [
        run | {"slot_id": slot, "slot_config_hash": "a" * 64 if slot else None},
    ]
    migrated, = query_bar_evaluation_runs(db)

    summary["data_identity"]["slot_id"] = "other"
    paths["summary"].write_text(json.dumps(summary))
    with pytest.raises(BarEvaluationConflictError, match="slot_id"):
        ingest(db, paths)
    assert query_bar_evaluation_detail(db, receipt.identity_hash)["run"] == migrated


@pytest.mark.parametrize("frequency", ["1d", "1w"])
@pytest.mark.parametrize("clock", ["opening", "anchor"])
def test_session_split_grid_uses_verified_open_or_anchor(custom_calendar, frequency, clock):
    calendar, template = custom_calendar
    schedule = BarSchedule(calendar, frequency)
    session_grid = calendar.bar_grid(frequency)
    expected_ends = pd.DatetimeIndex([schedule.available_at(t) for t in session_grid.timestamp_utc])
    if clock == "opening":
        expected_starts = pd.DatetimeIndex([
            schedule.for_session(day).start_utc.iloc[0] for day in session_grid.session_id
        ])
    else:
        expected_starts = pd.DatetimeIndex(session_grid.timestamp_utc)
    keep = np.arange(len(session_grid)) != 1
    labels = pd.DataFrame(dict(instrument_id="A", frequency=frequency,
                               signal_timestamp=expected_starts[keep], signal_available_at=expected_ends[keep]))
    summary = {"calendar_identity": {"view": "ETH", "calendar_id": "custom",
                                   "calendar_hash": calendar.calendar_hash, "schedule": {
                                       "start_date": "2025-01-06", "end_date": "2025-01-24",
                                       "rth_reference_calendar": "custom", "custom_sessions": template,
                                   }}, "data_identity": {"slot_id": "custom"}}
    calendars = grid_validation.matching_calendars(summary)
    grid = grid_validation.build_label_grid(summary, labels, calendars, bucket_strategy=None)
    actual = grid.check(time_series=True, split="full", start=expected_starts[0], end=expected_starts[-1],
                        train_end=None, oos_start=None, grid_size=len(expected_starts),
                        observed=expected_starts[keep], field="splits.full")
    assert actual.equals(expected_starts)
    with pytest.raises(BarEvaluationIntegrityError, match="grid_size"):
        grid.check(time_series=True, split="full", start=expected_starts[0], end=expected_starts[-1],
                   train_end=None, oos_start=None, grid_size=len(expected_starts) + 1,
                   observed=expected_starts[keep], field="splits.full")


def test_weekly_descriptive_ingestion_checks_published_anchor(tmp_path, custom_calendar):
    calendar, template = custom_calendar
    weekly = calendar.bar_grid("1w")
    anchors = weekly.timestamp_utc
    next_anchor = anchors.shift(-1)
    schedule = BarSchedule(calendar, "1w")
    next_open = pd.Series([schedule.for_session(day).start_utc.iloc[0] for day in weekly.session_id]).shift(-1)
    labels = pd.DataFrame(dict(
        instrument_id="A", frequency="1w", entry_offset_bars=1, horizon_bars=1, signal_timestamp=anchors,
        signal_available_at=anchors, entry_timestamp=next_open, exit_bar_timestamp=next_open,
        exit_timestamp=next_anchor, entry_price=[100., 100., np.nan], exit_price=[101., 102., np.nan],
        price_change_points=[1., 2., np.nan], return_pct=[1., 2., np.nan],
        status=["valid", "valid", "excluded"], exclusion_reason=[None, None, "insufficient_history"],
        factor_value=[1., 2., 3.],
    ))
    result = TimeSeriesEvaluator(analysis_mode="time_series", min_ic_samples=2).evaluate(labels)
    result.data_identity.update(slot_id="weekly", sources=["synthetic"], data_versions=["v1"])
    result.calendar_identity = dict(
        calendar_id="custom", calendar_hash=calendar.calendar_hash, view="ETH",
        schedule=dict(start_date="2025-01-06", end_date="2025-01-24", rth_reference_calendar="custom",
                      custom_sessions=template),
    )
    identity = ResultIdentity("a" * 64, {}, calendar.calendar_hash, "weekly-test", RESULT_SCHEMA_VERSION)
    paths = write_evaluation_artifacts(result, tmp_path / "weekly", result_identity=identity)
    db = tmp_path / "weekly.duckdb"
    assert ingest(db, paths).status == "inserted"
    run, = query_bar_evaluation_runs(db, frequency="1w", slot_id="weekly")
    assert run["frequencies"] == ["1w"]

    summary = json.loads(paths["summary"].read_text())
    summary["calendar_identity"]["schedule"]["daily_anchor"] = "open"
    paths["summary"].write_text(json.dumps(summary))
    with pytest.raises(BarEvaluationIntegrityError, match="calendar_hash"):
        ingest(tmp_path / "forged_weekly.duckdb", paths)


@pytest.mark.parametrize(("complete_options", "starts_on_sunday", "error"), [
    (False, True, "Cannot reconstruct calendar_id='custom' with the published calendar_hash"),
    (False, False, "Cannot reconstruct calendar_id='custom' with the published calendar_hash"),
    (True, True, "clocks disagree with the published Bar schedule"),
    (True, False, None),
], ids=["empty-options-sunday", "empty-options-monday", "complete-options-sunday", "complete-options-monday"])
def test_slot_daily_descriptive_ingestion_requires_reconstructible_calendar(
    tmp_path, custom_calendar, complete_options, starts_on_sunday, error,
):
    _, template = custom_calendar
    start, end = "2025-01-06", "2025-01-10"
    calendar = SessionCalendar("custom", start, end, custom_sessions=template)
    sessions = calendar.bar_grid("1d").iloc[:3]
    closes = sessions.timestamp_utc.copy()
    opens = pd.to_datetime(sessions.session_id + "T09:00:00Z")
    if starts_on_sunday:
        opens.iloc[0] -= pd.Timedelta(days=1)
        closes.iloc[0] -= pd.Timedelta(days=1)
    labels = pd.DataFrame(dict(
        instrument_id="A", frequency="1d", entry_offset_bars=1, horizon_bars=1, signal_timestamp=opens,
        signal_available_at=closes, entry_timestamp=opens.shift(-1), exit_bar_timestamp=opens.shift(-1),
        exit_timestamp=closes.shift(-1), entry_price=[100., 100., np.nan], exit_price=[101., 102., np.nan],
        price_change_points=[1., 2., np.nan], return_pct=[1., 2., np.nan],
        status=["valid", "valid", "excluded"], exclusion_reason=[None, None, "insufficient_history"],
        factor_value=[1., 2., 3.],
    ))
    result = TimeSeriesEvaluator(analysis_mode="time_series", min_ic_samples=2).evaluate(labels)
    result.calendar_identity = dict(calendar_id="custom", calendar_hash=calendar.calendar_hash, view="ETH")
    options = dict(calendar_id="custom", start=start, end=end, custom_sessions=template) if complete_options else {}
    slot = dict(
        schema_version=1, slot_id="synthetic",
        calendar=dict(provider="custom_sessions", options=options),
        execution_profile=dict(entry_price_field="open", exit_price_field="close", min_entry_offset_bars=1,
                               exit_adjustment=-1, return_unit="percent",
                               gap_rule="endpoints_only", boundary_policy="cross_session"),
        **{field: {"provider": "fixture"} for field in ("bar_feed", "universe", "aux_features", "microstructure")},
    )
    result.data_identity.update(slot_id="synthetic", slot_config_hash=slot_config_hash(slot))
    identity = ResultIdentity("a" * 64, {}, calendar.calendar_hash, "daily-test", RESULT_SCHEMA_VERSION)
    paths = write_evaluation_artifacts(result, tmp_path / "daily", result_identity=identity,
                                       config={"slot_config": slot})
    db = tmp_path / "daily.duckdb"
    if error is not None:
        with pytest.raises(BarEvaluationIntegrityError, match=error):
            ingest(db, paths)
        assert not db.exists()
    else:
        receipt = ingest(db, paths)
        assert receipt.status == "inserted"
        assert receipt.split_rows == 0
