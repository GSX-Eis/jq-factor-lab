"""
Factor Loader for VectorBT Backtest

Reads factor panels from Scheme B ``MarketFactorValueStore`` by
``(slot_id, factor_identity_hash / factor_id, frequency)``, while keeping an
explicit read-only legacy DuckDB bridge and cross-sectional normalization helpers.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd

from utils.market_db.guards import require_single_market_db
from utils.market_db.identity import market_db_identity
from utils.data_revision import connection_revision, revision_cached
from utils.duckdb_results_writer import connect_read_only_with_retry
from utils.market_factor import MarketFactorValueStore, validate_sha256_hex
from utils.strategy_kernel import bin_series, preprocess_series
from utils.time_axis import Frequency

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SESSION_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def cross_sectional_raw_zscore(series):
    return preprocess_series(series, "raw_zscore")


def cross_sectional_zscore(series):
    return preprocess_series(series, "zscore")


def cross_sectional_rank_zscore(series):
    return preprocess_series(series, "rank_zscore")


def cross_sectional_binning(series, n_bins=5):
    return bin_series(series, n_bins)


def cross_sectional_winsorize_zscore(series, percentile=0.025):
    return preprocess_series(series, "winsorize_zscore", percentile=percentile)


class FactorLoader:
    """Loads factor panels from Scheme B ``MarketFactorValueStore`` or legacy DuckDB."""

    def __init__(
        self,
        duckdb_path: Optional[str] = None,
        factor_table: str = "factor_values",
        logger: Optional[logging.Logger] = None,
        existing_conn: Optional["duckdb.DuckDBPyConnection"] = None,
        shared_cache: Optional[Dict[tuple, pd.DataFrame]] = None,
        *,
        factor_values_store: Optional[Union[str, Path, MarketFactorValueStore]] = None,
        slot_id: Optional[str] = None,
        frequency: str = "1d",
        factor_identities: Optional[Mapping[str, str]] = None,
        session_view: Optional[str] = None,
        calendar: Optional[Any] = None,
        market_db: Mapping[str, str] | None = None,
    ):
        """Initialize factor loader with Scheme B store or explicit legacy DuckDB bridge."""
        parsed_freq = Frequency.parse(frequency)
        if not parsed_freq.is_valid or parsed_freq.canonical is None:
            raise ValueError(f"Invalid frequency: {frequency!r}")

        self.duckdb_path = duckdb_path
        self.factor_table = factor_table
        self.logger = logger or logging.getLogger(__name__)
        self.slot_id = slot_id
        self.frequency = parsed_freq.canonical
        self._freq_kind = parsed_freq.kind
        self.session_view = session_view
        self.calendar = calendar
        self.market_db = market_db_identity(market_db)
        self._factor_identities: Dict[str, str] = {
            str(name): validate_sha256_hex(identity, f"factor_identities[{name!r}]")
            for name, identity in (factor_identities or {}).items()
        }

        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._owns_conn = True
        self._market_store: Optional[MarketFactorValueStore] = None
        self._load_cache: Optional[Dict[tuple, pd.DataFrame]] = shared_cache
        self._prefetched: Dict[str, pd.DataFrame] = {}
        self._prefetch_dates: Set[str] = set()
        self._prefetch_stocks: Set[str] = set()
        self._prefetch_factors: Set[str] = set()

        if factor_values_store is not None:
            self._market_store = (
                factor_values_store
                if isinstance(factor_values_store, MarketFactorValueStore)
                else MarketFactorValueStore(factor_values_store)
            )
            self._owns_conn = False
        elif existing_conn is not None:
            self._conn = existing_conn
            self._owns_conn = False
        elif duckdb_path is not None:
            from utils.data_store import binding_contract

            beside = MarketFactorValueStore.beside_results_database(duckdb_path)
            if beside.identities_path.is_file() and not binding_contract():
                self._market_store = beside
                self._owns_conn = False
        else:
            raise ValueError(
                "Either factor_values_store, duckdb_path, or existing_conn must be provided"
            )

    def _get_conn(self) -> "duckdb.DuckDBPyConnection":
        """Get or create DuckDB connection (read-only)."""
        if self._conn is None:
            if not DUCKDB_AVAILABLE:
                raise ImportError("duckdb is required. Install with: pip install duckdb")
            if self.duckdb_path is None:
                raise ValueError("Legacy DuckDB connection requested without duckdb_path")
            self._conn = connect_read_only_with_retry(self.duckdb_path)
            self._owns_conn = True
        return self._conn

    @staticmethod
    def _path_stamp(path: Path) -> Optional[tuple[int, int]]:
        try:
            stat = path.stat()
            return int(stat.st_mtime_ns), int(stat.st_size)
        except OSError:
            return None

    def cache_revision(self):
        if self._market_store is not None:
            return (
                "market_store",
                str(self._market_store.root),
                self.slot_id,
                self.frequency,
                self.session_view,
                tuple(sorted(self.market_db.items())),
                tuple(sorted(self._factor_identities.items())),
                self._path_stamp(self._market_store.identities_path),
                self._path_stamp(self._market_store.batches_path),
            )
        conn = self._get_conn()
        revision = connection_revision(conn)
        # Independent in-memory databases cannot share a cache namespace.
        memory_id = id(conn) if any(not row[1] for row in revision) else None
        return self.factor_table, revision, memory_id

    def clear_cache(self):
        if self._load_cache is not None:
            self._load_cache.clear()
        self._prefetched.clear()
        self._prefetch_dates.clear()
        self._prefetch_stocks.clear()
        self._prefetch_factors.clear()

    def close(self):
        """Close connection if we own it."""
        if self._conn is not None and self._owns_conn:
            self._conn.close()
            self._conn = None

    def _resolve_market_identities(
        self, factor_names: Sequence[str]
    ) -> Dict[str, str]:
        assert self._market_store is not None
        identities = self._market_store.list_identities(
            frequency=self.frequency,
            slot_id=self.slot_id,
        )
        if self.slot_id is not None and not identities.empty:
            identities = identities.loc[identities["slot_id"].eq(self.slot_id)]
        if self.market_db and not identities.empty:
            identities = identities.loc[
                identities.slot_data_fingerprint.map(lambda value: market_db_identity(value) == self.market_db)
            ]
        selected: Dict[str, str] = {}
        for name in dict.fromkeys(factor_names):
            pinned = self._factor_identities.get(name)
            if pinned is not None:
                matched = identities.loc[identities.factor_identity_hash.eq(pinned)]
                if len(matched) != 1:
                    raise ValueError(
                        f"Pinned factor_identity_hash {pinned!r} for {name!r} was not found "
                        f"for slot_id={self.slot_id!r}, frequency={self.frequency!r}"
                    )
                selected[name] = pinned
                continue
            if _SHA256_RE.fullmatch(name):
                matched = identities.loc[identities.factor_identity_hash.eq(name)]
                if len(matched) == 1:
                    selected[name] = name
                continue
            matched = identities.loc[identities.factor_id.eq(name)]
            if matched.empty:
                continue
            if len(matched) != 1:
                raise ValueError(
                    f"Select one Scheme B identity for {name!r} "
                    f"(slot_id={self.slot_id!r}, frequency={self.frequency!r}); "
                    f"matching hashes: {matched.factor_identity_hash.tolist()}"
                )
            selected[name] = str(matched.iloc[0]["factor_identity_hash"])
        sources = identities.loc[identities.factor_identity_hash.isin(selected.values()), "slot_data_fingerprint"]
        require_single_market_db(sources.tolist())
        return selected

    def _is_session_key(self, key: str) -> bool:
        return (
            self._freq_kind in ("session", "week")
            and bool(_SESSION_DATE_RE.fullmatch(str(key).strip()))
        )

    @staticmethod
    def _parse_utc_key(key: str) -> pd.Timestamp:
        ts = pd.Timestamp(key)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _pivot_market_rows(
        self,
        df: pd.DataFrame,
        stock_list: List[str],
        factor_names: List[str],
        selected: Mapping[str, str],
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(index=stock_list, columns=factor_names, dtype=float)
        hash_to_names: Dict[str, List[str]] = {}
        for req_name, fhash in selected.items():
            hash_to_names.setdefault(fhash, []).append(req_name)
        by_hash = df.pivot_table(
            index="instrument_id",
            columns="factor_identity_hash",
            values="factor_value",
            aggfunc="last",
        )
        result = pd.DataFrame(index=stock_list, columns=factor_names, dtype=float)
        for fhash, names in hash_to_names.items():
            if fhash in by_hash.columns:
                col_vals = pd.to_numeric(by_hash[fhash], errors="coerce").reindex(stock_list)
                for req_name in names:
                    result[req_name] = col_vals
        return result

    @revision_cached
    def prefetch(self, dates: List[str], stock_list: List[str], factor_names: List[str]) -> None:
        """Read frozen raw observations once; preprocessing still uses each exact pool."""
        dates, stocks, factors = sorted(set(dates)), sorted(set(stock_list)), sorted(set(factor_names))
        if not dates or not stocks or not factors:
            return

        if self._market_store is not None:
            selected = self._resolve_market_identities(factors)
            if not selected:
                self._prefetched = {}
                self._prefetch_dates, self._prefetch_stocks, self._prefetch_factors = (
                    set(dates),
                    set(stocks),
                    set(factors),
                )
                return
            all_session = all(self._is_session_key(d) for d in dates)
            if all_session:
                frame = self._market_store.read_factor_values(
                    factor_identity_hash=list(dict.fromkeys(selected.values())),
                    frequency=self.frequency,
                    slot_id=self.slot_id,
                    session_view=self.session_view,
                    instruments=stocks,
                    session_start=min(dates),
                    session_end=max(dates),
                )
                key_col = "session_id"
                key_map = {d: d for d in dates}
            else:
                utc_map = {self._parse_utc_key(d): d for d in dates}
                frame = self._market_store.read_factor_values(
                    factor_identity_hash=list(dict.fromkeys(selected.values())),
                    frequency=self.frequency,
                    slot_id=self.slot_id,
                    session_view=self.session_view,
                    instruments=stocks,
                    start_utc=min(utc_map),
                    end_utc=max(utc_map),
                )
                if not frame.empty:
                    frame = frame.assign(
                        _key=pd.to_datetime(frame["timestamp_utc"], utc=True).map(utc_map)
                    )
                key_col = "_key"
            groups: Dict[str, pd.DataFrame] = {}
            if not frame.empty:
                for key, group in frame.groupby(key_col):
                    if pd.notna(key):
                        groups[str(key)] = group
            self._prefetched = groups
            self._prefetch_dates, self._prefetch_stocks, self._prefetch_factors = (
                set(dates),
                set(stocks),
                set(factors),
            )
            return

        clauses = [
            f"{column} IN ({','.join('?' for _ in values)})"
            for column, values in [("date", dates), ("stock_code", stocks), ("factor_name", factors)]
        ]
        frame = self._get_conn().execute(
            f"SELECT date, stock_code, factor_name, factor_value FROM {self.factor_table} "
            f"WHERE {' AND '.join(clauses)}", dates + stocks + factors,
        ).fetchdf()
        groups = {
            date: group
            for date, group in frame.groupby(pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d"))
        }
        self._prefetched = groups
        self._prefetch_dates, self._prefetch_stocks, self._prefetch_factors = set(dates), set(stocks), set(factors)

    @revision_cached
    def load_factors(
        self, date_str: str, stock_list: List[str], factor_names: List[str]
    ) -> pd.DataFrame:
        """
        Load factor values for a specific session date or UTC timestamp and stock list.

        Returns:
            DataFrame with index=stock_code, columns=factor_names
            Missing values are NaN.
        """
        if not stock_list or not factor_names:
            return pd.DataFrame(index=stock_list, columns=factor_names)

        cache_key: Optional[tuple] = None
        if self._load_cache is not None:
            cache_key = (self._cache_revision, date_str, tuple(stock_list), tuple(factor_names))
            cached = self._load_cache.get(cache_key)
            if cached is not None:
                return cached.copy()

        if self._market_store is not None:
            try:
                selected = self._resolve_market_identities(factor_names)
                if not selected:
                    empty = pd.DataFrame(index=stock_list, columns=factor_names, dtype=float)
                    if cache_key is not None:
                        self._load_cache[cache_key] = empty.copy()
                    return empty
                if (
                    date_str in self._prefetch_dates
                    and set(stock_list) <= self._prefetch_stocks
                    and set(factor_names) <= self._prefetch_factors
                ):
                    df = self._prefetched.get(date_str, pd.DataFrame())
                    if not df.empty:
                        df = df[
                            df["instrument_id"].isin(stock_list)
                            & df["factor_identity_hash"].isin(list(selected.values()))
                        ]
                elif self._is_session_key(date_str):
                    df = self._market_store.read_factor_values(
                        factor_identity_hash=list(dict.fromkeys(selected.values())),
                        frequency=self.frequency,
                        slot_id=self.slot_id,
                        session_view=self.session_view,
                        instruments=list(dict.fromkeys(stock_list)),
                        session_start=date_str,
                        session_end=date_str,
                    )
                else:
                    ts = self._parse_utc_key(date_str)
                    df = self._market_store.read_factor_values(
                        factor_identity_hash=list(dict.fromkeys(selected.values())),
                        frequency=self.frequency,
                        slot_id=self.slot_id,
                        session_view=self.session_view,
                        instruments=list(dict.fromkeys(stock_list)),
                        start_utc=ts,
                        end_utc=ts,
                    )
                result = self._pivot_market_rows(df, stock_list, factor_names, selected)
                if cache_key is not None:
                    self._load_cache[cache_key] = result.copy()
                return result
            except Exception as e:
                self.logger.error(f"[FactorLoader] Failed to load Scheme B factors: {e}")
                raise

        conn = self._get_conn()
        try:
            if (
                date_str in self._prefetch_dates
                and set(stock_list) <= self._prefetch_stocks
                and set(factor_names) <= self._prefetch_factors
            ):
                df = self._prefetched.get(date_str, pd.DataFrame())
                if not df.empty:
                    df = df[df["stock_code"].isin(stock_list) & df["factor_name"].isin(factor_names)]
            else:
                stocks_sql = ",".join("?" for _ in stock_list)
                factors_sql = ",".join("?" for _ in factor_names)
                df = conn.execute(
                    f"SELECT date, stock_code, factor_name, factor_value FROM {self.factor_table} "
                    f"WHERE date = ? AND factor_name IN ({factors_sql}) "
                    f"AND stock_code IN ({stocks_sql})",
                    [date_str, *factor_names, *stock_list],
                ).fetchdf()

            if df.empty:
                self.logger.debug(f"[FactorLoader] No data for {date_str}")
                empty = pd.DataFrame(index=stock_list, columns=factor_names)
                if cache_key is not None:
                    self._load_cache[cache_key] = empty.copy()
                return empty

            result = df.pivot(
                index="stock_code", columns="factor_name", values="factor_value"
            )
            for factor in factor_names:
                if factor not in result.columns:
                    result[factor] = np.nan
            result = result[factor_names]
            result = result.reindex(stock_list)

            if cache_key is not None:
                self._load_cache[cache_key] = result.copy()
            return result
        except Exception as e:
            self.logger.error(f"[FactorLoader] Failed to load factors: {e}")
            raise

    def load_factor_panel(
        self,
        start: Any,
        end: Any,
        stock_list: List[str],
        factor_names: List[str],
        *,
        use_utc_index: Optional[bool] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load wide ``(time x instrument)`` panels for one or more factors."""
        if not factor_names:
            return {}
        if not stock_list:
            return {name: pd.DataFrame(columns=stock_list, dtype=float) for name in factor_names}
        if self._market_store is None:
            end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
            stocks_sql = ",".join("?" for _ in stock_list)
            factors_sql = ",".join("?" for _ in factor_names)
            if start is None:
                df = self._get_conn().execute(
                    f"SELECT date, stock_code, factor_name, factor_value FROM {self.factor_table} "
                    f"WHERE date <= ? AND factor_name IN ({factors_sql}) "
                    f"AND stock_code IN ({stocks_sql}) ORDER BY date",
                    [end_str, *factor_names, *stock_list],
                ).fetchdf()
            else:
                start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
                df = self._get_conn().execute(
                    f"SELECT date, stock_code, factor_name, factor_value FROM {self.factor_table} "
                    f"WHERE date BETWEEN ? AND ? AND factor_name IN ({factors_sql}) "
                    f"AND stock_code IN ({stocks_sql}) ORDER BY date",
                    [start_str, end_str, *factor_names, *stock_list],
                ).fetchdf()
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
            return {
                name: (
                    df.loc[df["factor_name"] == name]
                    .pivot(index="date", columns="stock_code", values="factor_value")
                    .reindex(columns=stock_list)
                    if not df.empty and (df["factor_name"] == name).any()
                    else pd.DataFrame(columns=stock_list, dtype=float)
                )
                for name in factor_names
            }

        selected = self._resolve_market_identities(factor_names)
        if not selected:
            return {name: pd.DataFrame(columns=stock_list, dtype=float) for name in factor_names}
        is_session = (
            self._freq_kind in ("session", "week")
            and isinstance(end, str)
            and bool(_SESSION_DATE_RE.fullmatch(end.strip()))
            and (
                start is None
                or (isinstance(start, str) and bool(_SESSION_DATE_RE.fullmatch(start.strip())))
            )
        )
        if is_session:
            frame = self._market_store.read_factor_values(
                factor_identity_hash=list(dict.fromkeys(selected.values())),
                frequency=self.frequency,
                slot_id=self.slot_id,
                session_view=self.session_view,
                instruments=list(dict.fromkeys(stock_list)) or None,
                session_start=start,
                session_end=end,
            )
            utc_axis = bool(use_utc_index) if use_utc_index is not None else False
        else:
            frame = self._market_store.read_factor_values(
                factor_identity_hash=list(dict.fromkeys(selected.values())),
                frequency=self.frequency,
                slot_id=self.slot_id,
                session_view=self.session_view,
                instruments=list(dict.fromkeys(stock_list)) or None,
                start_utc=None if start is None else self._parse_utc_key(str(start)),
                end_utc=self._parse_utc_key(str(end)),
            )
            utc_axis = True if use_utc_index is None else bool(use_utc_index)

        if frame.empty:
            return {name: pd.DataFrame(columns=stock_list, dtype=float) for name in factor_names}
        index_col = "timestamp_utc" if utc_axis else "session_id"
        if not utc_axis:
            frame = frame.assign(_axis=pd.to_datetime(frame["session_id"]))
            index_col = "_axis"
        panels: Dict[str, pd.DataFrame] = {}
        for name in factor_names:
            fhash = selected.get(name)
            if fhash is None:
                panels[name] = pd.DataFrame(columns=stock_list, dtype=float)
                continue
            sub = frame.loc[frame["factor_identity_hash"].eq(fhash)]
            if sub.empty:
                panels[name] = pd.DataFrame(columns=stock_list, dtype=float)
                continue
            wide = (
                sub.drop_duplicates(subset=[index_col, "instrument_id"], keep="last")
                .pivot(
                    index=index_col,
                    columns="instrument_id",
                    values="factor_value",
                )
                .sort_index()
            )
            panels[name] = wide.reindex(columns=stock_list)
        return panels
