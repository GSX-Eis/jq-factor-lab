# Declarative crypto market

This fixture contains two hourly bars and two funding releases. Open a disposable DuckDB destination to try the same slot declaration used by Python callers.

```sh
python3 -m tools.agent_harness.cli import-bars \
  --manifest examples/declarative_crypto/slot.yaml \
  --db /tmp/crypto-slot-demo.duckdb --derive 2h
```

The import emits quality counts by frequency and status. Both frequencies report a partial session because the fixture covers two hours. The destination retains the custom calendar for subsequent reads, builds and audits.

```python
from slots import build_slot_from_config

slot = build_slot_from_config("examples/declarative_crypto/slot.yaml")
grid = slot.calendar.bar_grid("1h", "2026-01-03", "2026-01-03")
funding = slot.read_aux_features(
    grid.assign(instrument_id="BTC-USD"), features=["funding_rate"],
)
```

Paths are relative to the manifest directory. Use `csv_files` with a `paths` list, `parquet_glob` with a `path` pattern, or `duckdb_table` with a database `path` and a `table` name. DuckDB inputs open read-only. The bar feed expresses DuckDB `TIMESTAMPTZ` values in the declared `source_timezone` before normalizing bar keys to UTC. Text timestamps with explicit offsets must match that timezone. Supply offset-aware timestamps for UTC. Column mappings run from source names to the declared field names. The slot requires an instrument specification for every bar instrument.

Use `calendar.provider: pmc` with an explicit PMC `calendar_id` and coverage dates for exchanges such as NYSE and XHKG. A weekday-only UTC session template uses `weekdays: [0, 1, 2, 3, 4]`. Session labels own the declared date even when a segment opens the previous evening or closes the following morning.

The feed serves its declared frequency through both views. Intraday keys use the UTC bar open and daily keys use the scheduled close. Signal reads expose a bar after its scheduled end. Set `volume_mode: cumulative` for session counters that reset at session boundaries. The source must include the first bar of each session so interval volume can be reconstructed.

To add time-varying universes, select any data driver for `universe` and supply complete membership snapshots keyed by `(universe_id, instrument_id, available_at_utc)`. The latest snapshot published by the query time wins. Auxiliary feature tables and optional microstructure events use `(instrument_id, available_at_utc)`. Auxiliary features use the latest release per instrument and omit grid rows before the first release. Microstructure events use a boolean `tradable` column and carry forward the most recent known state within open sessions.

Native slots use the same `build_slot_from_config` entry point with their provider name in every component and a caller-owned DuckDB connection. Declare their coverage under `calendar.options` and native feed settings, such as futures `product_root`, under `bar_feed.options`. The execution profile must match the native convention after applying the declared fees.
