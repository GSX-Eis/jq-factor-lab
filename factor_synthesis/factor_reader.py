"""Synthesis inputs and composite outputs for Scheme B and frozen DuckDB bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.market_db.guards import require_single_market_db
from utils.market_db.identity import market_db_identity, slot_market_db_identity
from utils.duckdb_results_writer import DuckDBResultsWriter, connect_read_only_with_retry
from utils.analysis_results_guard import _duckdb_tables
from utils.market_factor import MarketFactorValueStore, save_factor_values, validate_sha256_hex
from utils.market_factor.reader import read_factor_values as read_market_plan
from utils.market_factor.schema import FactorReadPlan
from utils.slots import MarketSlot
from utils.time_axis import Frequency

if TYPE_CHECKING:
    from utils.factor_base import FactorBase


DEFAULT_SLOT_ID = "jq_ashare_local"


def resolve_market_input(
    slot: MarketSlot | None, slot_id: str | None, frequency: str,
) -> tuple[str, str]:
    if slot is not None and not isinstance(slot, MarketSlot):
        raise TypeError("slot must be a MarketSlot")
    if slot_id is not None and (not isinstance(slot_id, str) or not slot_id.strip()):
        raise ValueError("slot_id must be a nonempty string")
    if slot is not None and slot_id is not None and slot.slot_id != slot_id:
        raise ValueError(f"slot_id {slot_id!r} does not match MarketSlot {slot.slot_id!r}")
    parsed = Frequency.parse(frequency)
    if parsed.canonical is None:
        raise ValueError(f"Invalid synthesis frequency: {frequency!r}")
    return slot.slot_id if slot is not None else slot_id or DEFAULT_SLOT_ID, parsed.canonical


def _committed_identities(
    store: MarketFactorValueStore,
    slot_id: str,
    frequency: str,
    market_db: Mapping[str, str],
) -> dict[str, list[str]]:
    """Resolve committed candidate identity hashes from committed batches only."""
    identities, batches = store._load_committed_state()
    by_factor: dict[str, set[str]] = {}
    for batch in batches:
        fhash = str(batch["factor_identity_hash"])
        if str(batch["frequency"]) != frequency:
            continue
        record = identities.get(fhash, {})
        if record.get("slot_id") != slot_id:
            continue
        if market_db and market_db_identity(record) != market_db:
            continue
        if not batch.get("shards"):
            continue
        by_factor.setdefault(str(batch["factor_id"]), set()).add(fhash)
    return {name: sorted(hashes) for name, hashes in by_factor.items()}


def plan_market_factor_values(
    store: MarketFactorValueStore,
    factor_names: Sequence[str],
    factor_identities: Mapping[str, str],
    start_date: str,
    end_date: str,
    instruments: list[str] | None,
    *,
    slot_id: str | None = None,
    frequency: str = "1d",
    slot: MarketSlot | None = None,
) -> tuple[dict[str, str], FactorReadPlan | None]:
    """Pin committed shards for one slot and output frequency."""
    if not isinstance(factor_identities, Mapping):
        raise TypeError("factor_identities must map factor names to full identity hashes")
    slot_id, frequency = resolve_market_input(slot, slot_id, frequency)
    committed = _committed_identities(store, slot_id, frequency, slot_market_db_identity(slot))
    selected: dict[str, str] = {}
    for name in dict.fromkeys(factor_names):
        pinned = factor_identities.get(name)
        hashes = committed.get(name, [])
        if not hashes and pinned is None:
            continue
        if pinned is not None:
            pinned_hash = validate_sha256_hex(pinned, f"factor_identities[{name!r}]")
            hashes = [item for item in hashes if item == pinned_hash]
        if len(hashes) != 1:
            raise ValueError(
                f"Select one Scheme B identity for {name!r} in {slot_id}/{frequency} "
                f"with factor_identities. Matching hashes: {hashes}"
            )
        selected[name] = hashes[0]
    plan = store.plan_read(
        factor_identity_hash=list(selected.values()), frequency=frequency, slot_id=slot_id,
        session_start=start_date, session_end=end_date, instruments=instruments,
    ) if selected else None
    if plan is not None:
        require_single_market_db(list(plan.market_databases.values()))
    return selected, plan


def plan_daily_market_values(
    store: MarketFactorValueStore,
    factor_names: Sequence[str],
    factor_identities: Mapping[str, str],
    start_date: str,
    end_date: str,
    instruments: list[str] | None,
    *,
    slot_id: str = DEFAULT_SLOT_ID,
) -> tuple[dict[str, str], FactorReadPlan | None]:
    return plan_market_factor_values(
        store, factor_names, factor_identities, start_date, end_date, instruments, slot_id=slot_id,
    )


def read_market_factor_values(plan: FactorReadPlan, frequency: str) -> pd.DataFrame:
    rows = read_market_plan(plan, None)
    dates = pd.to_datetime(rows.session_id) if frequency == "1d" else rows.timestamp_utc
    frame = rows.assign(date=dates, stock_code=rows.instrument_id, factor_name=rows.factor_id)
    keys = ["date", "stock_code", "factor_name"]
    if frame.duplicated(keys).any():
        raise ValueError("Scheme B factor rows contain duplicate synthesis keys")
    return frame[[*keys, "factor_value"]].sort_values(keys)


def read_daily_market_values(plan: FactorReadPlan) -> pd.DataFrame:
    return read_market_factor_values(plan, "1d")


def market_factor_inputs(
    identities: Mapping[str, str], plan: FactorReadPlan, slot_id: str, frequency: str,
) -> dict:
    return {
        "identities": dict(identities),
        "shard_paths": [str(path) for path in plan.shard_paths],
        "factors": [
            {"factor_id": name, "factor_identity_hash": identity, "slot_id": slot_id, "frequency": frequency,
             **plan.market_databases.get(identity, {})}
            for name, identity in identities.items()
        ],
    }


class FactorReader:
    """Resolve each Scheme B input by its unique or pinned identity."""

    def __init__(
        self,
        database_path: str,
        results_database_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        data_bindings: Optional[dict] = None,
        publication_metadata: Optional[dict] = None,
        *,
        factor_values_store: MarketFactorValueStore | str | Path | None = None,
        slot: MarketSlot | None = None,
        slot_id: str | None = None,
        frequency: str = "1d",
        factor_identities: Mapping[str, str] | None = None,
    ) -> None:
        from utils.data_store import binding_contract, output_context, resolve_connection_binding

        self.data_bindings = binding_contract(data_bindings)
        if self.data_bindings and not self.data_bindings.get("output_db"):
            raise ValueError("Bound factor reads require output_db")
        self.publication_metadata = output_context(publication_metadata)
        self.logger = logger or logging.getLogger(__name__)
        self.database_path = database_path
        self.results_database_path = results_database_path or database_path
        self._read_bindings: dict[bool, dict] = {}
        for results, path in ((False, self.database_path), (True, self.results_database_path)):
            selected = self.data_bindings
            if selected:
                requested = str(Path(path).expanduser().resolve())
                path, selected = resolve_connection_binding(requested, selected, allow_labels=True)
                if Path(path).resolve() != Path(self.data_bindings["output_db"]).resolve():
                    raise ValueError("FactorReader source must select the output store")
            if results:
                self.results_database_path = path
            else:
                self.database_path = path
            self._read_bindings[results] = selected

        # Reads must be side-effect free.  Create a writer only at an explicit
        # save operation, after the caller has chosen an output store.
        self._writer: Optional[DuckDBResultsWriter] = None
        if self.data_bindings and (factor_values_store is not None or factor_identities):
            raise ValueError("Frozen factor bindings already select their input artifacts")
        self.slot = slot
        self.slot_id, self.frequency = resolve_market_input(slot, slot_id, frequency)
        if self.data_bindings and (self.slot_id != DEFAULT_SLOT_ID or self.frequency != "1d"):
            raise ValueError("Frozen DuckDB factor bindings use the jq_ashare_local daily contract")
        self._factor_identities = {
            name: validate_sha256_hex(identity, f"factor_identities[{name!r}]")
            for name, identity in (factor_identities or {}).items()
        }
        self._slot = slot
        self._market_store = None
        if not self.data_bindings:
            if isinstance(factor_values_store, MarketFactorValueStore):
                self._market_store = factor_values_store
            elif factor_values_store is not None:
                self._market_store = MarketFactorValueStore(factor_values_store)
            else:
                self._market_store = MarketFactorValueStore.beside_results_database(self.database_path)
        self._market_output = (
            self._market_store if factor_values_store is not None or self.results_database_path == self.database_path
            else MarketFactorValueStore.beside_results_database(self.results_database_path)
        )
        if self.data_bindings:
            with self.get_connection() as conn:
                if "factor_values" not in _duckdb_tables(conn):
                    raise ValueError("Frozen DuckDB factor store has no bound factor_values table")

    def _market_values(
        self, factor_names: Sequence[str], start_date: str, end_date: str, stock_pool: list[str] | None,
    ) -> pd.DataFrame:
        selected, plan = plan_market_factor_values(
            self._market_store, factor_names, self._factor_identities, start_date, end_date, stock_pool or None,
            slot_id=self.slot_id, frequency=self.frequency, slot=self._slot,
        )
        if plan is None:
            return pd.DataFrame(columns=["date", "stock_code", "factor_name", "factor_value"])
        frame = read_market_factor_values(plan, self.frequency)
        frame.attrs["market_factor_inputs"] = market_factor_inputs(selected, plan, self.slot_id, self.frequency)
        return frame

    def with_market_input(
        self, *, slot: MarketSlot | None, slot_id: str | None, frequency: str | None,
        factor_values_store: MarketFactorValueStore | str | Path | None,
        factor_identities: Mapping[str, str] | None,
    ) -> FactorReader:
        selected_slot = slot if slot is not None else self.slot
        return FactorReader(
            self.database_path, self.results_database_path, self.logger, self.data_bindings, self.publication_metadata,
            slot=selected_slot,
            slot_id=slot_id if slot_id is not None else self.slot_id if selected_slot is None else None,
            frequency=self.frequency if frequency is None else frequency,
            factor_values_store=self._market_store if factor_values_store is None else factor_values_store,
            factor_identities=self._factor_identities if factor_identities is None else factor_identities,
        )

    @contextmanager
    def get_connection(self, results_db: bool = False):
        path = self.results_database_path if results_db else self.database_path
        bindings = self._read_bindings[results_db]
        if bindings:
            from utils.data_store import connect_output

            with connect_output(path, bindings=bindings, allow_labels=True) as conn:
                yield conn
            return
        conn = connect_read_only_with_retry(path, bindings=bindings)
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _fetchdf(conn: Any, query: str, params: list[Any]) -> pd.DataFrame:
        return conn.execute(query, params).fetchdf()

    def get_available_factors(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[str]:
        if self._market_store is not None:
            committed = _committed_identities(
                self._market_store, self.slot_id, self.frequency, slot_market_db_identity(self._slot),
            )
            if not committed:
                return []
            if start_date or end_date:
                hashes = [item for group in committed.values() for item in group]
                rows = self._market_store.read_factor_values(
                    factor_identity_hash=hashes, slot_id=self.slot_id,
                    frequency=self.frequency, session_start=start_date, session_end=end_date,
                )
                return sorted(rows.factor_id.unique())
            return sorted(committed)
        clauses, params = ["1=1"], []
        if start_date:
            clauses.append("date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("date <= ?")
            params.append(end_date)
        with self.get_connection() as conn:
            query = "SELECT DISTINCT factor_name FROM factor_values WHERE " + " AND ".join(clauses)
            return sorted(self._fetchdf(conn, query, params)["factor_name"])

    def get_factor_values(
        self,
        factor_name: str,
        start_date: str,
        end_date: str,
        stock_pool: Optional[List[str]] = None,
        pivot_format: bool = True,
    ) -> pd.DataFrame:
        if self._market_store is not None:
            frame = self._market_values([factor_name], start_date, end_date, stock_pool).drop(columns="factor_name")
            if pivot_format:
                wide = frame.pivot(index="date", columns="stock_code", values="factor_value")
                wide.attrs = frame.attrs.copy()
                return wide
            return frame
        query = """
            SELECT date, stock_code, factor_value FROM factor_values
            WHERE factor_name = ? AND date >= ? AND date <= ?
        """
        params: list[Any] = [factor_name, start_date, end_date]
        if stock_pool:
            query += " AND stock_code = ANY(?)"; params.append(stock_pool)
        query += " ORDER BY date, stock_code"
        with self.get_connection() as conn:
            frame = self._fetchdf(conn, query, params)
        if frame.empty:
            return pd.DataFrame()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.pivot(index="date", columns="stock_code", values="factor_value") if pivot_format else frame

    def get_multiple_factors(
        self,
        factor_names: List[str],
        start_date: str,
        end_date: str,
        stock_pool: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        if not factor_names:
            return {}
        if self._market_store is not None:
            frame = self._market_values(factor_names, start_date, end_date, stock_pool)
            panels = {}
            metadata = frame.attrs.get("market_factor_inputs")
            for name in factor_names:
                rows = frame.loc[frame.factor_name.eq(name)]
                panel = rows.pivot(index="date", columns="stock_code", values="factor_value")
                if metadata and name in metadata["identities"]:
                    panel.attrs["market_factor_inputs"] = {
                        "identities": {name: metadata["identities"][name]},
                        "shard_paths": metadata["shard_paths"],
                        "factors": [factor for factor in metadata["factors"] if factor["factor_id"] == name],
                    }
                panels[name] = panel
            return panels
        query = """
            SELECT date, stock_code, factor_name, factor_value FROM factor_values
            WHERE factor_name = ANY(?) AND date >= ? AND date <= ?
        """
        params: list[Any] = [factor_names, start_date, end_date]
        if stock_pool:
            query += " AND stock_code = ANY(?)"; params.append(stock_pool)
        query += " ORDER BY factor_name, date, stock_code"
        with self.get_connection() as conn:
            frame = self._fetchdf(conn, query, params)
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"])
        return {
            name: (frame.loc[frame["factor_name"] == name].pivot(index="date", columns="stock_code", values="factor_value")
                   if not frame.empty and (frame["factor_name"] == name).any() else pd.DataFrame())
            for name in factor_names
        }

    def save_composite_factor(
        self,
        composite_factor_df: pd.DataFrame,
        factor_name: str,
        overwrite: bool = True,
        batch_size: int = 1000,
        *,
        factor: FactorBase | None = None,
    ) -> bool:
        del batch_size
        if composite_factor_df.empty:
            return False
        if not overwrite and self.data_bindings:
            with self.get_connection(results_db=True) as conn:
                exists = "factor_values" in _duckdb_tables(conn) and conn.execute(
                    "SELECT 1 FROM factor_values WHERE factor_name = ? LIMIT 1", [factor_name],
                ).fetchone()
            if exists:
                raise ValueError(f"Factor already exists and overwrite=False: {factor_name}")
        if not self.publication_metadata:
            if factor is None:
                raise ValueError("Supply the composite FactorBase or use the harness write-factor-values command")
            store = self._market_output or MarketFactorValueStore.beside_results_database(self.results_database_path)
            if not overwrite and factor_name in _committed_identities(
                store, self.slot_id, self.frequency, slot_market_db_identity(factor.slot),
            ):
                raise ValueError(f"Factor already exists and overwrite=False: {factor_name}")
            receipt = save_factor_values(factor, store, factor_id=factor_name, factor_values=composite_factor_df)
            return receipt.inserted + receipt.skipped > 0
        if self._writer is None:
            self._writer = DuckDBResultsWriter(
                self.results_database_path,
                publication_metadata=self.publication_metadata,
                data_bindings=self.data_bindings,
            )
        return self._writer.publish_factor_values(composite_factor_df, factor_name) > 0

    def check_data_quality(self, factor_name: str, start_date: str, end_date: str, stock_pool: Optional[List[str]] = None) -> Dict[str, Any]:
        frame = self.get_factor_values(factor_name, start_date, end_date, stock_pool)
        if frame.empty:
            return {"error": "No data available"}
        values = frame.to_numpy().ravel(); valid = values[~pd.isna(values)]
        total = frame.size; non_null = int(frame.count().sum())
        mean, std = (float(np.mean(valid)), float(np.std(valid))) if len(valid) else (0.0, 0.0)
        outlier_ratio = float(np.mean(np.abs(valid - mean) > 3 * std)) if len(valid) and std else 0.0
        return {
            "data_shape": frame.shape, "total_cells": total, "non_null_cells": non_null,
            "null_cells": total - non_null, "coverage_ratio": non_null / total if total else 0,
            "outlier_ratio": outlier_ratio,
            "value_stats": {"mean": mean, "std": std, "min": float(np.min(valid)) if len(valid) else None, "max": float(np.max(valid)) if len(valid) else None},
        }
