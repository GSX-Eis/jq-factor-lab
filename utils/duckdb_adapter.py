#!/usr/bin/env python3
"""
duckdb_adapter.py - Unified data adapter for local DuckDB/Parquet queries

This adapter provides a high-performance interface for querying price and factor data
from DuckDB databases and Parquet files.

Key Features:
    - Support for both DuckDB tables and Parquet files
    - Arrow-based zero-copy data transfer to pandas
    - Multi-level caching (in-memory + filesystem)

Usage:
    from utils.duckdb_adapter import DuckDBAdapter, get_data_adapter
    
    # Use an explicitly configured DuckDB or Parquet source.
    adapter = get_data_adapter(
        duckdb_path='data/duckdb/factor_database.duckdb',
        parquet_dir='data/duckdb/parquet',
        source_policy='duckdb_then_parquet_on_empty',
    )
    
    # Query price data
    df = adapter.fetch_price_data(
        stocks=['600519.SH', '000001.SZ'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        fields=['close', 'volume']
    )

Cloud callers may omit local sources and use their API path instead.
"""

import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None  # Placeholder for type hints

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


from utils.data_access_config import Dataset, load_data_access_config


CacheKey = Tuple[Any, ...]


from utils.data_revision import connection_revision, parquet_revision, revision_cached


class DataCache:
    """Simple in-memory cache for frequently accessed data"""
    
    def __init__(self, max_size_mb: int = 1000):
        """
        Initialize cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
        """
        self._cache: Dict[CacheKey, pd.DataFrame] = {}
        self._access_times: Dict[CacheKey, float] = {}
        self._max_size_mb = max_size_mb
        
    def _estimate_size_mb(self, df: pd.DataFrame) -> float:
        """Estimate DataFrame size in MB"""
        return df.memory_usage(deep=True).sum() / (1024 * 1024)
        
    def _total_size_mb(self) -> float:
        """Get total cache size in MB"""
        return sum(self._estimate_size_mb(df) for df in self._cache.values())
        
    def _evict_if_needed(self, needed_mb: float):
        """Evict old entries if cache is too large"""
        while self._total_size_mb() + needed_mb > self._max_size_mb and self._cache:
            # Evict least recently accessed
            oldest_key = min(self._access_times, key=self._access_times.get)
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
            
    def get(self, key: CacheKey) -> Optional[pd.DataFrame]:
        """Get item from cache"""
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key].copy()
        return None
        
    def put(self, key: CacheKey, df: pd.DataFrame):
        """Put item in cache"""
        size_mb = self._estimate_size_mb(df)
        if size_mb > self._max_size_mb:
            return
        self._evict_if_needed(size_mb)
        self._cache[key] = df.copy()
        self._access_times[key] = time.time()
        
    def clear(self):
        """Clear all cached data"""
        self._cache.clear()
        self._access_times.clear()


class DuckDBAdapter:
    """
    High-performance data adapter using DuckDB and Parquet.
    
    This adapter provides optimized data fetching for local factor analysis,
    with support for both DuckDB database files and Parquet data lakes.
    """
    
    def __init__(
        self,
        duckdb_path: Optional[str] = None,
        parquet_dir: Optional[str] = None,
        use_cache: bool = True,
        cache_size_mb: int = 1000,
        data_bindings: Optional[dict] = None,
        data_config_path: Optional[str] = None,
        source_policy: Optional[str] = None,
        calendar_duckdb_path: Optional[str] = None,
    ):
        """
        Initialize the adapter.
        
        Args:
            duckdb_path: Path to DuckDB database file. With a binding, must match
                input_db; unrelated paths and retired sources aliases raise.
            parquet_dir: Directory containing Parquet files; cannot be supplied
                with a nonempty explicit or environment binding.
            use_cache: Cache results until a committed import or close().
                Unmanaged Parquet sources do not reuse cached results.
            cache_size_mb: Maximum cache size in MB
            data_bindings: None inherits QUANT_DATA_BINDINGS; {} disables that
                inheritance. A nonempty binding requires input_db. Conflicting
                source arguments raise ValueError instead of being ignored.
            data_config_path: Dataset definitions JSON; loaded once per adapter.
            source_policy: Both sources require 'duckdb_then_parquet_on_empty'.
                Errors propagate; only successful zero-row queries fall back.
        """
        from utils.data_store import binding_contract, resolve_connection_binding

        self.calendar_duckdb_path = calendar_duckdb_path
        self.data_bindings = binding_contract(data_bindings)
        if self.data_bindings and parquet_dir is not None:
            raise ValueError(
                "parquet_dir cannot be combined with data_bindings (including QUANT_DATA_BINDINGS); "
                "remove parquet_dir to use the binding, or pass data_bindings={} to use local paths"
            )
        if self.data_bindings and not self.data_bindings.get("input_db"):
            raise ValueError("Bound data adapter requires input_db")
        resolved_path = self.data_bindings.get("input_db") or duckdb_path
        if self.data_bindings and duckdb_path is not None:
            resolved_path, self.data_bindings = resolve_connection_binding(duckdb_path, self.data_bindings)
            if Path(resolved_path).expanduser().resolve() != Path(self.data_bindings["input_db"]).expanduser().resolve():
                raise ValueError("Data adapter source must select the input store")
        self.duckdb_path = Path(resolved_path) if resolved_path else None
        self.parquet_dir = Path(parquet_dir) if parquet_dir else None
        if source_policy not in (None, 'duckdb_then_parquet_on_empty'):
            raise ValueError(f'Unknown source_policy {source_policy!r}')
        if self.duckdb_path and self.parquet_dir and source_policy is None:
            raise ValueError('Both sources require source_policy="duckdb_then_parquet_on_empty"')
        if source_policy is not None and not (self.duckdb_path and self.parquet_dir):
            raise ValueError('source_policy requires both DuckDB and Parquet sources')
        self.source_policy = source_policy
        self.data_config = load_data_access_config(data_config_path)
        self.use_cache = use_cache
        self._cache = DataCache(max_size_mb=cache_size_mb) if use_cache else None
        self._conn = None
        
        # Validate paths
        if self.duckdb_path and not self.duckdb_path.exists():
            raise FileNotFoundError(self.duckdb_path)
            
        if self.parquet_dir and not self.parquet_dir.exists():
            raise FileNotFoundError(self.parquet_dir)
            
        if not self.duckdb_path and not self.parquet_dir:
            raise ValueError("Either duckdb_path or parquet_dir must be provided")
            
    def trading_calendar(self):
        """Read the same version-bound input calendar as this adapter's data."""
        from utils.trading_calendar import load_trading_calendar
        if self.calendar_duckdb_path is not None:
            return load_trading_calendar(duckdb_path=self.calendar_duckdb_path, bindings=self.data_bindings)
        return load_trading_calendar(self._get_connection())

    def _get_connection(self):
        """Return a fully initialized connection, creating it on first use."""
        if self._conn is not None:
            return self._conn

        from utils.data_store import connect_input, _configure_resources

        if self.duckdb_path:
            conn = connect_input(self.duckdb_path, bindings=self.data_bindings)
        else:
            conn = duckdb.connect()
            try:
                _configure_resources(conn)
            except BaseException:
                conn.close()
                raise

        self._conn = conn
        return conn

    def cache_revision(self):
        revision = connection_revision(self._get_connection())
        if self.parquet_dir:
            parquet = parquet_revision(self.parquet_dir)
            if parquet is None:
                return None
            revision = (revision, parquet)
        return revision

    def clear_cache(self):
        """Discard derived results without closing or changing input bindings."""
        if self._cache is not None:
            self._cache.clear()

    def close(self):
        """End the read session: close the connection and discard cached results."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self.clear_cache()

    def _build_cache_key(
        self,
        table: str,
        stocks: List[str],
        start_date: str,
        end_date: str,
        fields: List[str]
    ) -> CacheKey:
        """Keep query values in the key so hash collisions cannot alias queries."""
        from utils.data_store import digest

        dataset = self.data_config.dataset(table)
        binding_digest = digest(self.data_bindings.get("inputs", {}).get(table, {}))
        return (
            binding_digest, dataset, self.source_policy, tuple(sorted(stocks)),
            start_date, end_date, tuple(fields),
        )
        
    def _query_from_duckdb(
        self,
        table_name: str,
        stocks: List[str],
        start_date: str,
        end_date: str,
        fields: List[str],
        date_column: str = 'date',
        stock_column: str = 'stock_code'
    ) -> pd.DataFrame:
        """Read an inclusive calendar-day range; query errors propagate."""
        from utils.data_store import quote, table_name as quote_table

        if not stocks:
            return pd.DataFrame()
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        columns = [date_column, stock_column] + [
            field for field in fields if field not in (date_column, stock_column)
        ]
        fields_sql = ", ".join(quote(field) for field in columns)
        stock_params = ", ".join("?" for _ in stocks)
        sql = f"""
            SELECT {fields_sql}
            FROM {quote_table(table_name)}
            WHERE {quote(date_column)} >= ?
            AND {quote(date_column)} < ?
            AND {quote(stock_column)} IN ({stock_params})
        """
        # ISO strings also support the importer's VARCHAR date columns.
        params = [start_dt.date().isoformat(),
                  (end_dt + timedelta(days=1)).date().isoformat(), *stocks]
        return self._get_connection().execute(sql, params).fetchdf()

    def _query_from_parquet(
        self,
        table_name: str,
        stocks: List[str],
        start_date: str,
        end_date: str,
        fields: List[str],
        date_column: str = 'date',
        stock_column: str = 'stock_code'
    ) -> pd.DataFrame:
        """Read an inclusive calendar-day range with partition pruning."""
        from utils.data_store import quote

        if not stocks or not self.parquet_dir:
            return pd.DataFrame()

        parquet_path = self.parquet_dir / self.data_config.dataset(table_name).parquet_subdir
        if not parquet_path.is_dir():
            raise FileNotFoundError(f'Parquet dataset directory not found: {parquet_path}')

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        if start_dt > end_dt:
            raise ValueError('start_date must not be after end_date')
        columns = [date_column, stock_column] + [
            field for field in fields
            if field not in (date_column, stock_column)
        ]
        fields_sql = ", ".join(quote(field) for field in columns)
        stock_params = ", ".join("?" for _ in stocks)

        year_months = []
        current = start_dt.replace(day=1)
        while current <= end_dt:
            year_months.append((current.year, current.month))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        partition_filter = " OR ".join('(year = ? AND month = ?)' for _ in year_months)
        sql = f"""
            SELECT {fields_sql}
            FROM read_parquet(?, hive_partitioning=true)
            WHERE ({partition_filter})
            AND {quote(date_column)} >= ?
            AND {quote(date_column)} < ?
            AND {quote(stock_column)} IN ({stock_params})
        """
        params = [str(parquet_path / '**' / '*.parquet'),
                  *(value for pair in year_months for value in pair),
                  start_dt.date().isoformat(),
                  (end_dt + timedelta(days=1)).date().isoformat(), *stocks]
        return self._get_connection().execute(sql, params).fetchdf()

    def _fetch_panel(self, dataset: Dataset, stocks, start_date, end_date, fields):
        """Apply one validation, source, cache and output contract to every panel."""
        fields = dataset.fields(fields)
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        if start_dt > end_dt:
            raise ValueError('start_date must not be after end_date')
        if not isinstance(stocks, (list, tuple)) or any(not isinstance(s, str) or not s for s in stocks):
            raise ValueError('stocks must be a list or tuple of nonempty stock codes')
        start_date, end_date = start_dt.date().isoformat(), end_dt.date().isoformat()
        if not stocks:
            df = pd.DataFrame(columns=['date', 'code', *fields])
            df['date'] = pd.to_datetime(df['date'])
            df.attrs['_sorted_by_code_date'] = True
            return df
        return self._fetch_validated_panel(dataset, stocks, start_date, end_date, fields)

    @revision_cached
    def _fetch_validated_panel(self, dataset, stocks, start_date, end_date, fields):
        """Read/cache a validated nonempty request under one generation guard."""
        cache = self._cache if self.use_cache else None
        cache_key = None
        df = pd.DataFrame()
        if stocks:
            if cache is not None:
                cache_key = self._build_cache_key(dataset.table, stocks, start_date, end_date, fields)
                cached = cache.get(cache_key)
                if cached is not None:
                    return cached
            query_args = (dataset.table, stocks, start_date, end_date, fields,
                          dataset.date_column, dataset.stock_column)
            if self.duckdb_path:
                df = self._query_from_duckdb(*query_args)
            if df.empty and self.parquet_dir:
                df = self._query_from_parquet(*query_args)
        if df.empty:
            df = pd.DataFrame(columns=['date', 'code', *fields])
        else:
            df = df.rename(columns={dataset.date_column: 'date', dataset.stock_column: 'code'})
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df = df.sort_values(['code', 'date'], kind='mergesort', ignore_index=True)
        df.attrs['_sorted_by_code_date'] = True
        if cache_key is not None and not df.empty:
            cache.put(cache_key, df)
        return df

    def fetch_price_data(
        self, stocks: List[str], start_date: str, end_date: str,
        fields: Optional[List[str]] = None, frequency: str = 'daily',
    ) -> pd.DataFrame:
        """Read configured prices over inclusive dates, sorted by code/date.

        None uses configured defaults; [] selects keys only. Missing fields raise.
        Empty results retain date/code, requested fields and a datetime date column.
        """
        return self._fetch_panel(self.data_config.price(frequency), stocks, start_date, end_date, fields)

    @revision_cached
    def fetch_price_factor_ends(self, stocks, start_date, end_date, *,
                               frequency='daily', factor_column='factor', fields=None):
        """Aggregate full-window PIT anchors without materializing the price panel.

        Use the same configured table, keys and zero-row source fallback as prices.
        fields identifies the raw columns used by PIT, so unrequested values do
        not make an empty bar observable. Omit to use all available price columns.
        Empty bars cannot supply an anchor; invalid observed factors fail closed.
        """
        from utils.data_store import quote, table_name

        dataset = self.data_config.price(frequency)
        dataset.fields([factor_column])
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        if start > end:
            raise ValueError('start_date must not be after end_date')
        if not isinstance(stocks, (list, tuple)) or any(not isinstance(s, str) or not s for s in stocks):
            raise ValueError('stocks must be a list or tuple of nonempty stock codes')
        if not stocks:
            return {}
        date, stock, factor = map(quote, (dataset.date_column, dataset.stock_column, factor_column))

        def query(parquet=False):
            params = []
            source = table_name(dataset.table)
            if parquet:
                folder = self.parquet_dir / dataset.parquet_subdir
                if not folder.is_dir():
                    raise FileNotFoundError(f'Parquet dataset directory not found: {folder}')
                source = 'read_parquet(?, hive_partitioning=true)'
                params.append(str(folder / '**' / '*.parquet'))
            sql = f'SELECT * FROM {source} WHERE {date} >= ? AND {date} < ? AND {stock} = ANY(?)'
            params.extend([start.date().isoformat(), (end + timedelta(days=1)).date().isoformat(), list(stocks)])
            relation = self._get_connection().sql(sql, params=params)
            raw_fields = relation.columns if fields is None else fields
            observed_fields = [name for name in ('open', 'high', 'low', 'close', 'volume', 'pre_close', 'avg')
                               if name in raw_fields]
            observed = ' OR '.join(f'{quote(name)} IS NOT NULL' for name in observed_fields) or 'TRUE'
            invalid = f'({factor} IS NULL OR NOT isfinite({factor}) OR {factor} <= 0)'
            if 'money' in raw_fields:
                money_present = f'(money IS NOT NULL AND NOT ({invalid} AND money = 0))'
                if 'paused' in relation.columns:
                    money_present += ' AND NOT (coalesce(paused, 0) = 1 AND money = 0)'
                money_only = not any(name in raw_fields for name in (
                    'open', 'high', 'low', 'close', 'volume', 'pre_close', 'avg', 'high_limit', 'low_limit'))
                observed = money_present if money_only else f'({observed}) OR ({money_present} AND money != 0)'
            return relation.aggregate(
                f'{stock}, arg_max({factor}, {date}) FILTER (WHERE ({observed}) AND NOT {invalid}) AS anchor, '
                f'bool_or(({observed}) AND {invalid}) AS invalid',
                group_expr=stock,
            ).fetchall()

        rows = query() if self.duckdb_path else []
        if not rows and self.parquet_dir:
            rows = query(parquet=True)
        if any(invalid for _, _, invalid in rows):
            raise ValueError('pit_factor_missing_or_invalid')
        return {stock: anchor for stock, anchor, _ in rows if anchor is not None}

    def fetch_factor_data(
        self, stocks: List[str], start_date: str, end_date: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Read input features; supply fields unless defaults are configured."""
        dataset = self.data_config.dataset(self.data_config.factor_table)
        return self._fetch_panel(dataset, stocks, start_date, end_date, fields)

    def fetch_table_data(
        self, table_name: str, stocks: List[str], start_date: str,
        end_date: str, fields: List[str],
    ) -> pd.DataFrame:
        """Read a configured or canonical dated stock panel with the same contract."""
        return self._fetch_panel(self.data_config.dataset(table_name), stocks, start_date, end_date, fields)

    def fetch_index_constituents(self, index_code: str, date: str, *, table_name=None) -> List[str]:
        """Return sorted unique stocks for the exact stored date (no as-of fill).

        Use configured table/columns and the adapter's explicit source policy.
        Successful zero-row queries return []; configuration and query errors raise.
        """
        date = _validate_index_request(index_code, date)
        return self._read_index_constituents(index_code, date, table_name=table_name)

    @revision_cached
    def _read_index_constituents(self, index_code=None, date=None, *, table_name=None, exists_only=False):
        """Share source selection with the pool manager's bounded existence check."""
        dataset, column = self.data_config.index_dataset(table_name)
        conn = self._get_connection()
        args = (conn, dataset, column, index_code, date)
        result = []
        if self.duckdb_path:
            result = _query_index_constituents(*args, exists_only=exists_only)
        if not result and self.parquet_dir:
            result = _query_index_constituents(*args, exists_only=exists_only, parquet_dir=self.parquet_dir)
        return result

    def execute_sql(self, sql: str, params=None) -> pd.DataFrame:
        """Execute SQL with optional positional or named bound parameters.

        Return DuckDB's result columns unchanged, including for zero-row queries.
        Connection and query errors propagate. This low-level entry point does
        not normalize or cache results, scan configured Parquet directories,
        or perform source fallback. SQL reuses the adapter's connection;
        session commands such as SET affect subsequent calls.
        """
        conn = self._get_connection()
        return conn.execute(sql, params).fetchdf()
            
    @revision_cached
    def get_available_stocks(
        self,
        start_date: str,
        end_date: str,
        min_records: int = 10
    ) -> List[str]:
        """List stocks with enough rows in the configured daily dataset.

        Dates include the entire end day. Errors propagate; Parquet fallback
        requires zero primary rows, not merely too few rows per stock. Counts
        are observations, not distinct trading dates. Results are not cached.
        """
        from utils.data_store import quote, table_name as quote_table

        if type(min_records) is not int or min_records < 1:
            raise ValueError('min_records must be a positive integer')
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        if start > end:
            raise ValueError('start_date must not be after end_date')
        dataset = self.data_config.price('daily')
        date, stock = quote(dataset.date_column), quote(dataset.stock_column)
        bounds = [start.date().isoformat(), (end + timedelta(days=1)).date().isoformat()]
        conn = self._get_connection()

        def query(parquet=False):
            relation, params = quote_table(dataset.table), []
            if parquet:
                folder = self.parquet_dir / dataset.parquet_subdir
                if not folder.is_dir():
                    raise FileNotFoundError(f'Parquet dataset directory not found: {folder}')
                relation = 'read_parquet(?, hive_partitioning=true)'
                params.append(str(folder / '**' / '*.parquet'))
            sql = f'''SELECT {stock}, COUNT(*) FROM {relation}
                WHERE {date} >= ? AND {date} < ?
                GROUP BY {stock} ORDER BY {stock}'''
            return conn.execute(sql, [*params, *bounds]).fetchall()

        counts = query() if self.duckdb_path else []
        if not counts and self.parquet_dir:
            counts = query(parquet=True)
        return [code for code, count in counts if count >= min_records]


def _validate_index_code(index_code):
    """Require an exact identifier; reject accidental surrounding whitespace."""
    if not isinstance(index_code, str) or not index_code.strip():
        raise ValueError('index_code must be a nonempty string')
    if index_code != index_code.strip():
        raise ValueError('index_code must not contain leading or trailing whitespace')


def _validate_index_request(index_code, date):
    _validate_index_code(index_code)
    if not isinstance(date, str):
        raise ValueError('date must be YYYY-MM-DD')
    parsed = datetime.strptime(date, '%Y-%m-%d')
    if parsed.strftime('%Y-%m-%d') != date:
        raise ValueError('date must be zero-padded YYYY-MM-DD')
    return date


def _query_index_constituents(conn, dataset, index_column, index_code=None, date=None,
                              *, exists_only=False, parquet_dir=None):
    """Read index membership, also usable with a caller-owned DuckDB connection.

    exists_only omits the date for validation and limits the query to one row.
    Parquet uses no year/month predicate: the importer also writes flat files.
    """
    from utils.data_store import quote, table_name as quote_table

    params = []
    relation = quote_table(dataset.table)
    if parquet_dir is not None:
        folder = Path(parquet_dir) / dataset.parquet_subdir
        if not folder.is_dir():
            raise FileNotFoundError(f'Parquet dataset directory not found: {folder}')
        relation = 'read_parquet(?, hive_partitioning=true)'
        params.append(str(folder / '**' / '*.parquet'))
    conditions = []
    if index_code is not None:
        _validate_index_code(index_code)
        conditions.append(f'{quote(index_column)} = ?')
        params.append(index_code)
    if date is not None:
        _validate_index_request(index_code, date)
        conditions.append(f'{quote(dataset.date_column)} = ?')
        params.append(date)
    where = ' AND '.join(conditions) or 'TRUE'
    stock = quote(dataset.stock_column)
    selection = stock if exists_only else f'DISTINCT {stock}'
    ending = 'LIMIT 1' if exists_only else f'ORDER BY {stock}'
    sql = f'SELECT {selection} FROM {relation} WHERE {where} {ending}'
    return [row[0] for row in conn.execute(sql, params).fetchall()]


def get_data_adapter(
    duckdb_path: Optional[str] = None,
    parquet_dir: Optional[str] = None,
    use_cache: bool = True,
    cache_size_mb: int = 1000,
    data_bindings: Optional[dict] = None,
    data_config_path: Optional[str] = None,
    source_policy: Optional[str] = None,
    calendar_duckdb_path: Optional[str] = None,
) -> Optional[DuckDBAdapter]:
    """
    Return a configured DuckDB/Parquet adapter, or ``None`` when no local source
    was supplied. A caller with a direct DuckDB connection should use that
    connection directly.
    
    Args:
        duckdb_path: Path to DuckDB database; under a binding, must be input_db
            with no retired source aliases. Unmapped paths raise ValueError.
        parquet_dir: Path to Parquet directory. Combining it with a nonempty
            binding raises ValueError; it is never silently ignored.
        use_cache: Enable the adapter's in-memory query cache.
        cache_size_mb: Maximum in-memory cache size in MB.
        data_bindings: None inherits QUANT_DATA_BINDINGS; {} disables environment
            inheritance. A nonempty binding selects its input_db and versions.
        data_config_path: Complete dataset config, loaded once per adapter.
        source_policy: Required when both paths are provided; only
            'duckdb_then_parquet_on_empty' is supported.
        
    Returns:
        DuckDBAdapter instance, or None when no binding or path was configured.
        DuckDB must be installed even for the no-source case.
    """
    if not DUCKDB_AVAILABLE:
        raise ImportError("DuckDB is required for local data reads")
    from utils.data_store import binding_contract

    contract = binding_contract(data_bindings)
    if not contract and duckdb_path is None and parquet_dir is None:
        if source_policy is not None:
            raise ValueError('source_policy requires both DuckDB and Parquet sources')
        return None
    return DuckDBAdapter(
        duckdb_path=duckdb_path,
        parquet_dir=parquet_dir,
        use_cache=use_cache,
        cache_size_mb=cache_size_mb,
        data_bindings=contract,
        data_config_path=data_config_path,
        source_policy=source_policy,
        calendar_duckdb_path=calendar_duckdb_path,
    )


# Convenience function for quick queries
def quick_query(
    sql: str,
    duckdb_path: str = './data/duckdb/factor_database.duckdb',
    *, data_bindings: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Execute a quick SQL query against DuckDB.
    
    Args:
        sql: SQL query
        duckdb_path: Path to DuckDB file
        
    Returns:
        DataFrame with results
    """
    if not DUCKDB_AVAILABLE:
        raise ImportError("duckdb is required. Install with: pip install duckdb")
        
    from utils.data_store import connect_input
    conn = connect_input(duckdb_path, bindings=data_bindings)
    try:
        return conn.execute(sql).fetchdf()
    finally:
        conn.close()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DuckDB adapter")
    parser.add_argument("--db", default="./data/duckdb/factor_database.duckdb")
    parser.add_argument("--parquet", default="./data/duckdb/parquet")
    
    args = parser.parse_args()
    
    adapter = get_data_adapter(
        duckdb_path=args.db,
        parquet_dir=args.parquet,
        source_policy='duckdb_then_parquet_on_empty' if args.db and args.parquet else None,
    )
    
    if adapter:
        # Test query
        stocks = adapter.get_available_stocks('2023-01-01', '2023-12-31')
        print(f"Found {len(stocks)} stocks with data")
        
        if stocks:
            df = adapter.fetch_price_data(
                stocks=stocks[:10],
                start_date='2023-01-01',
                end_date='2023-01-31',
                fields=['close', 'volume']
            )
            print(f"Fetched {len(df)} records")
            print(df.head())
    else:
        print("DuckDB adapter not available")
