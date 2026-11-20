"""Public append-only market factor store."""
from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from utils.session_calendar import SessionCalendar

from . import ledger, reader, verifier, writer
from .schema import (
    FactorReadPlan,
    FactorWriteReceipt,
)


class MarketFactorValueStore:
    """Append-only Hive-partitioned Parquet + flock-protected JSONL ledger + DuckDB view."""

    def __init__(self, root: str | Path) -> None:
        if root is None or (isinstance(root, str) and not root.strip()):
            raise ValueError("MarketFactorValueStore requires an explicit root directory")
        self.root = Path(root).expanduser().resolve()
        self.parquet_root = self.root / "parquet_data"
        self.identities_path = self.root / "identities.jsonl"
        self.batches_path = self.root / "batches.jsonl"
        self.lock_path = self.root / ".ledger.lock"
        self._last_read_plan: FactorReadPlan | None = None

    @property
    def last_read_plan(self) -> FactorReadPlan | None:
        return self._last_read_plan

    def _write_lock(self) -> AbstractContextManager[None]:
        return ledger._write_lock(self.root, self.parquet_root, self.lock_path)

    def _load_committed_state(self) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        return ledger._load_committed_state(self.identities_path, self.batches_path)

    def append_factor_values(
        self,
        df: pd.DataFrame,
        *,
        identity_record: Mapping[str, Any] | None = None,
        calendar: SessionCalendar | Mapping[str, SessionCalendar] | None = None,
        strict_source_build_id: bool = False,
    ) -> FactorWriteReceipt:
        """Validate and idempotently append a single-identity batch; never updates or deletes."""
        return writer.append_factor_values(
            df,
            root=self.root,
            parquet_root=self.parquet_root,
            identities_path=self.identities_path,
            batches_path=self.batches_path,
            write_lock=self._write_lock,
            load_committed_state=self._load_committed_state,
            identity_record=identity_record,
            calendar=calendar,
            strict_source_build_id=strict_source_build_id,
        )

    def list_identities(
        self,
        factor_id: str | None = None,
        frequency: str | None = None,
    ) -> pd.DataFrame:
        """List registered factor_value_identities from the append-only ledger (lock-free)."""
        return reader.list_identities(self.identities_path, factor_id, frequency)

    def _committed_shards(self) -> list[dict[str, Any]]:
        return reader._committed_shards(self._load_committed_state)

    def plan_read(
        self,
        *,
        factor_id: str | Sequence[str] | None = None,
        factor_identity_hash: str | Sequence[str] | None = None,
        frequency: str | Sequence[str] | None = None,
        session_view: str | None = None,
        instruments: Sequence[str] | None = None,
        start_utc: dt.datetime | pd.Timestamp | str | None = None,
        end_utc: dt.datetime | pd.Timestamp | str | None = None,
        session_start: str | None = None,
        session_end: str | None = None,
    ) -> FactorReadPlan:
        """Resolve target partitions and committed Parquet shard paths from the ledger."""
        return reader.plan_read(
            self.parquet_root, self._load_committed_state, self._committed_shards,
            factor_id=factor_id,
            factor_identity_hash=factor_identity_hash,
            frequency=frequency,
            session_view=session_view,
            instruments=instruments,
            start_utc=start_utc,
            end_utc=end_utc,
            session_start=session_start,
            session_end=session_end,
        )

    def view_sql(self, view_name: str = "market_factor_values") -> str:
        """Build CREATE OR REPLACE VIEW SQL bound strictly to committed shards."""
        return reader.view_sql(self.parquet_root, self._committed_shards, view_name)

    def create_view(self, connection: duckdb.DuckDBPyConnection, view_name: str = "market_factor_values") -> None:
        """Create or refresh a DuckDB view over committed shards only."""
        reader.create_view(connection, view_name, self.view_sql)

    def explain_read(self, **kwargs: Any) -> str:
        """Return DuckDB EXPLAIN output for a planned read query."""
        return reader.explain_read(self.plan_read(**kwargs))

    def read_factor_values(
        self,
        *,
        factor_id: str | Sequence[str] | None = None,
        factor_identity_hash: str | Sequence[str] | None = None,
        frequency: str | Sequence[str] | None = None,
        session_view: str | None = None,
        instruments: Sequence[str] | None = None,
        start_utc: dt.datetime | pd.Timestamp | str | None = None,
        end_utc: dt.datetime | pd.Timestamp | str | None = None,
        session_start: str | None = None,
        session_end: str | None = None,
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> pd.DataFrame:
        """Read committed factor values via DuckDB read_parquet (lock-free)."""
        plan = self.plan_read(
            factor_id=factor_id,
            factor_identity_hash=factor_identity_hash,
            frequency=frequency,
            session_view=session_view,
            instruments=instruments,
            start_utc=start_utc,
            end_utc=end_utc,
            session_start=session_start,
            session_end=session_end,
        )
        self._last_read_plan = plan
        return reader.read_factor_values(plan, connection)

    def verify_store(self) -> dict[str, Any]:
        """Read-only consistency check reconciling disk shards and staging with JSONL ledger."""
        return verifier.verify_store(self.root, self.parquet_root, self.identities_path, self.batches_path)
