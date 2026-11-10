#!/usr/bin/env python3
"""
price_data_to_duckdb.py - Import price data to DuckDB + Parquet

This script imports price data from CSV files to DuckDB database with optional
Parquet export. It mirrors the interface of price_data_to_database.py but uses
DuckDB as the backend for improved analytical query performance.

Usage:
    python -m data_pipeline.loaders.price_data_to_duckdb
    
    # Or with custom paths:
    python -m data_pipeline.loaders.price_data_to_duckdb --db-path data/duckdb/factor_database.duckdb --data-dir /path/to/price_data/
    
    # Export to Parquet as well:
    python -m data_pipeline.loaders.price_data_to_duckdb --export-parquet

Features:
    - 10-50x faster imports compared to SQLite
    - Automatic partitioning for Parquet export
    - Resume support for interrupted imports
    - Parallel CSV reading
"""

import argparse
import csv
import os
import time
import shutil
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

from data_pipeline.loaders._transactions import atomic_import
from utils.data_revision import published_parquet, reset_parquet_dataset

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("Error: duckdb not installed. Install with: pip install duckdb")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available, using pure DuckDB mode")


# Default paths (can be overridden via command line)
DEFAULT_DB_PATH = './data/duckdb/factor_database.duckdb'
DEFAULT_PARQUET_DIR = './data/duckdb/parquet'
DEFAULT_DATA_DIR = './data/import/price_data/'
DEFAULT_DATA_DIR_5M = os.path.join(DEFAULT_DATA_DIR, '5m')
DEFAULT_DATA_DIR_DAILY = os.path.join(DEFAULT_DATA_DIR, 'daily')
BACKUP_DIR = './backups/'
BULK_BATCH_SIZE = 32
REQUIRED_PRICE_COLUMNS = {"date", "open", "close", "high", "low", "volume", "money"}


def _sql_string(value: str) -> str:
    """Return a single-quoted SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def _sql_path_list(paths: List[str]) -> str:
    return "[" + ", ".join(_sql_string(path) for path in paths) + "]"


class DuckDBPriceImporter:
    """Import price data to DuckDB with optional Parquet export"""
    
    def __init__(
        self, 
        db_path: str = DEFAULT_DB_PATH,
        parquet_dir: str = DEFAULT_PARQUET_DIR,
        export_parquet: bool = False
    ):
        """
        Initialize the importer.
        
        Args:
            db_path: Path to DuckDB database file
            parquet_dir: Directory for Parquet exports
            export_parquet: Whether to also export to Parquet
        """
        self.db_path = Path(db_path)
        self.parquet_dir = Path(parquet_dir)
        self.export_parquet = export_parquet
        self.conn = None
        self.import_failed = False
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if export_parquet:
            self.parquet_dir.mkdir(parents=True, exist_ok=True)
            
    def connect(self) -> duckdb.DuckDBPyConnection:
        """Create and configure database connection"""
        self.conn = duckdb.connect(str(self.db_path))
        
        # Optimize for bulk loading
        self.conn.execute("SET threads TO 4")
        self.conn.execute("SET memory_limit = '8GB'")
        self.conn.execute("SET preserve_insertion_order = false")
        
        return self.conn
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def init_tables(self):
        """Initialize database tables"""
        if not self.conn:
            self.connect()
            
        # 5-minute data table (with factor column for PIT adjustment)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS price_data_5m (
                stock_code VARCHAR,
                date TIMESTAMP,
                open DOUBLE,
                close DOUBLE,
                high DOUBLE,
                low DOUBLE,
                volume BIGINT,
                money DOUBLE,
                factor DOUBLE,
                PRIMARY KEY (stock_code, date)
            )
        """)
        
        # Daily data table (with extended fields including factor for PIT adjustment)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS price_data_daily (
                stock_code VARCHAR,
                date DATE,
                open DOUBLE,
                close DOUBLE,
                high DOUBLE,
                low DOUBLE,
                volume DOUBLE,
                money DOUBLE,
                paused INTEGER,
                high_limit DOUBLE,
                low_limit DOUBLE,
                pre_close DOUBLE,
                factor DOUBLE,
                PRIMARY KEY (stock_code, date)
            )
        """)
        
        # Migrate existing tables: add factor column if it doesn't exist
        self._migrate_add_factor_column('price_data_daily')
        self._migrate_add_factor_column('price_data_5m')
        
        # Import progress tracking table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS import_progress (
                file_name VARCHAR PRIMARY KEY,
                imported INTEGER,
                record_count INTEGER,
                import_date TIMESTAMP
            )
        """)
        
        print("Tables initialized successfully")
    
    def _migrate_add_factor_column(self, table_name: str):
        """
        Add factor column to existing table if it doesn't exist.
        
        This migration supports PIT (Point-in-Time) dynamic adjustment by storing
        adjustment factors alongside price data.
        
        Args:
            table_name: Name of the table to migrate
        """
        # DESCRIBE resolves the same catalog/schema and identifier as ALTER TABLE.
        # DuckDB identifiers are case-insensitive, including an existing `Factor`.
        columns = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        if not any(column[0].lower() == "factor" for column in columns):
            self.conn.execute(f"ALTER TABLE {table_name} ADD COLUMN factor DOUBLE")
            print(f"Migration: Added 'factor' column to {table_name} table")

    def create_indices(self):
        """Create indices for faster queries"""
        print("Creating indices...")
        
        # DuckDB automatically creates indices for PRIMARY KEY
        # Add additional indices for common query patterns
        try:
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_5m_date 
                ON price_data_5m(date)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_daily_date 
                ON price_data_daily(date)
            """)
            print("Indices created successfully")
        except Exception as e:
            print(f"Warning: Index creation issue: {e}")
            
    def get_imported_files(self) -> set:
        """Return set of already imported files"""
        result = self.conn.execute("""
            SELECT file_name FROM import_progress WHERE imported = 1
        """).fetchall()
        return {row[0] for row in result}
            
    def mark_file_imported(self, file_name: str, record_count: int):
        """Mark a file as successfully imported"""
        self.conn.execute("""
            INSERT OR REPLACE INTO import_progress
            (file_name, imported, record_count, import_date)
            VALUES (?, 1, ?, CURRENT_TIMESTAMP)
        """, [file_name, record_count])
            
    def import_csv_file(
        self, 
        filepath: str, 
        table_name: str,
        freq_type: str
    ) -> int:
        """
        Import a single CSV file to DuckDB.

        The caller must explicitly open and commit/roll back a transaction.
        This entry point does not make delete/insert or progress updates atomic.
        
        Args:
            filepath: Path to CSV file
            table_name: Target table name
            freq_type: '5m' or 'daily'
            
        Returns:
            Number of rows imported
        """
        code = os.path.splitext(os.path.basename(filepath))[0]
        _, counts = self._bulk_import_batch([filepath], table_name, freq_type)
        return counts.get(code, 0)

    def _csv_header_columns(self, filepath: str) -> set:
        try:
            with open(filepath, newline="", encoding="utf-8-sig") as f:
                header = next(csv.reader(f))
        except Exception:
            return set()
        duplicates = sorted(name for name, count in Counter(header).items() if name and count > 1)
        if duplicates:
            raise ValueError(
                f"{os.path.basename(filepath)} duplicate column(s): {', '.join(duplicates)}"
            )
        return {col.strip() for col in header if col.strip()}

    def _validate_csv_headers(self, filepaths: List[str]) -> Dict[str, set]:
        """Validate each source independently before a union-by-name batch read."""
        headers = {filepath: self._csv_header_columns(filepath) for filepath in filepaths}
        invalid = []
        for filepath, columns in headers.items():
            missing = sorted(REQUIRED_PRICE_COLUMNS - columns)
            if missing:
                invalid.append(f"{os.path.basename(filepath)} missing {', '.join(missing)}")
        if invalid:
            raise ValueError("Invalid price CSV header(s): " + "; ".join(invalid))
        return headers

    def _bulk_import_batch(
        self,
        filepaths: List[str],
        table_name: str,
        freq_type: str,
    ) -> Tuple[int, Dict[str, int]]:
        """Import a batch of CSV files through one DuckDB staging table."""
        if not filepaths:
            return 0, {}

        headers = self._validate_csv_headers(filepaths)
        all_columns = set().union(*headers.values())
        expected_codes = {os.path.splitext(os.path.basename(filepath))[0] for filepath in filepaths}

        factor_expr = (
            "COALESCE(CAST(factor AS DOUBLE), 1.0)"
            if "factor" in all_columns
            else "1.0"
        )
        paths_sql = _sql_path_list(filepaths)

        if freq_type == "5m":
            self.conn.execute(f"""
                CREATE OR REPLACE TEMPORARY TABLE _price_stage_raw AS
                SELECT
                    regexp_extract(filename, '([^/]+)\\.csv$', 1, 'i') AS stock_code,
                    CAST(date AS TIMESTAMP) AS date,
                    CAST(open AS DOUBLE) AS open,
                    CAST(close AS DOUBLE) AS close,
                    CAST(high AS DOUBLE) AS high,
                    CAST(low AS DOUBLE) AS low,
                    CAST(volume AS BIGINT) AS volume,
                    CAST(money AS DOUBLE) AS money,
                    {factor_expr} AS factor
                FROM read_csv({paths_sql},
                    header=true,
                    auto_detect=true,
                    union_by_name=true,
                    ignore_errors=false,
                    filename=true
                )
                WHERE date IS NOT NULL
            """)
        else:
            paused_expr = (
                "CAST(paused AS INTEGER)"
                if "paused" in all_columns
                else "NULL"
            )
            high_limit_expr = (
                "CAST(high_limit AS DOUBLE)"
                if "high_limit" in all_columns
                else "NULL"
            )
            low_limit_expr = (
                "CAST(low_limit AS DOUBLE)"
                if "low_limit" in all_columns
                else "NULL"
            )
            pre_close_expr = (
                "CAST(pre_close AS DOUBLE)"
                if "pre_close" in all_columns
                else "NULL"
            )
            self.conn.execute(f"""
                CREATE OR REPLACE TEMPORARY TABLE _price_stage_raw AS
                SELECT
                    regexp_extract(filename, '([^/]+)\\.csv$', 1, 'i') AS stock_code,
                    CAST(date AS DATE) AS date,
                    CAST(open AS DOUBLE) AS open,
                    CAST(close AS DOUBLE) AS close,
                    CAST(high AS DOUBLE) AS high,
                    CAST(low AS DOUBLE) AS low,
                    CAST(volume AS DOUBLE) AS volume,
                    CAST(money AS DOUBLE) AS money,
                    {paused_expr} AS paused,
                    {high_limit_expr} AS high_limit,
                    {low_limit_expr} AS low_limit,
                    {pre_close_expr} AS pre_close,
                    {factor_expr} AS factor
                FROM read_csv({paths_sql},
                    header=true,
                    auto_detect=true,
                    union_by_name=true,
                    ignore_errors=false,
                    filename=true
                )
                WHERE date IS NOT NULL
            """)

        self.conn.execute("""
            CREATE OR REPLACE TEMPORARY TABLE _price_stage AS
            SELECT * EXCLUDE (rn)
            FROM (
                SELECT *,
                    row_number() OVER (
                        PARTITION BY stock_code, date
                        ORDER BY stock_code
                    ) AS rn
                FROM _price_stage_raw
            )
            WHERE rn = 1
        """)

        counts = dict(self.conn.execute("""
            SELECT stock_code, COUNT(*)::INTEGER
            FROM _price_stage
            GROUP BY stock_code
        """).fetchall())
        missing_records = sorted(expected_codes - counts.keys())
        if missing_records:
            raise ValueError(
                "Price CSV files produced no valid records: " + ", ".join(missing_records)
            )
        total_records = sum(counts.values())

        self.conn.execute(f"""
            DELETE FROM {table_name}
            USING _price_stage
            WHERE {table_name}.stock_code = _price_stage.stock_code
              AND {table_name}.date = _price_stage.date
        """)
        target_columns = "stock_code, date, open, close, high, low, volume, money"
        if freq_type != "5m":
            target_columns += ", paused, high_limit, low_limit, pre_close"
        target_columns += ", factor"
        self.conn.execute(f"""
            INSERT INTO {table_name} ({target_columns})
            SELECT * FROM _price_stage
        """)
        return total_records, counts
            
    @atomic_import
    def import_directory(
        self,
        freq_type: str,
        directory: str,
        table_name: str
    ) -> Tuple[int, int, int]:
        """
        Import all CSV files from a directory.
        
        Args:
            freq_type: '5m' or 'daily'
            directory: Source directory path
            table_name: Target table name
            
        Returns:
            Tuple of (total_files, successful_files, total_records)
        """
        if not os.path.isdir(directory):
            print(f"Directory {directory} does not exist, skipping.")
            return 0, 0, 0
            
        print(f"\n{'='*60}")
        print(f"Processing {freq_type} data from {directory}")
        print(f"{'='*60}")
        
        imported_files = self.get_imported_files()
        
        total_files = 0
        successful_files = 0
        total_records = 0
        start_time = time.time()
        
        csv_files = sorted([
            f for f in os.listdir(directory)
            if f.lower().endswith('.csv') 
            and not f.startswith('.') 
            and not f.startswith('~')
        ])
        
        duplicate_codes = sorted(
            code for code, count in Counter(
                os.path.splitext(name)[0] for name in csv_files
            ).items() if count > 1
        )
        if duplicate_codes:
            raise ValueError(
                "Ambiguous CSV filenames for stock code(s): " + ", ".join(duplicate_codes)
            )

        pending_files = []
        for i, fname in enumerate(csv_files):
            total_files += 1
            file_key = f"{freq_type}/{fname}"
            
            # Skip already imported files
            if file_key in imported_files:
                if (i + 1) % 100 == 0:
                    print(f"Skipped {i + 1} already imported files...")
                continue

            pending_files.append(fname)

        self._validate_csv_headers([
            os.path.join(directory, fname) for fname in pending_files
        ])

        for start in range(0, len(pending_files), BULK_BATCH_SIZE):
            batch_names = pending_files[start:start + BULK_BATCH_SIZE]
            batch_paths = [os.path.join(directory, fname) for fname in batch_names]
            try:
                records, counts = self._bulk_import_batch(
                    batch_paths, table_name, freq_type
                )
                if records > 0:
                    total_records += records
                    for fname in batch_names:
                        stock_code = os.path.splitext(fname)[0]
                        record_count = counts.get(stock_code, 0)
                        if record_count > 0:
                            successful_files += 1
                            self.mark_file_imported(
                                f"{freq_type}/{fname}", record_count
                            )

                processed = min(start + len(batch_names), len(pending_files))
                elapsed = time.time() - start_time
                rate = total_records / max(1, elapsed)
                print(f"Progress: {processed}/{len(pending_files)} pending files "
                      f"({len(csv_files)} total), {total_records:,} records, "
                      f"{rate:.0f} records/sec")
                          
            except KeyboardInterrupt:
                print("\nInterrupted. Database changes will be rolled back.")
                raise
            except Exception as e:
                print(f"Error processing batch starting at {batch_names[0]}: {e}")
                self.import_failed = True
                break
                
        return total_files, successful_files, total_records
        
    @published_parquet
    def export_to_parquet(self, table_name: str, partition_by: List[str] = None):
        """
        Export a table to Parquet format with optional partitioning.
        
        Args:
            table_name: Table to export
            partition_by: Columns to partition by (e.g., ['year', 'month'])
        """
        if not self.export_parquet:
            return
            
        print(f"\nExporting {table_name} to Parquet...")
        
        output_path = self.parquet_dir / table_name
        output_path.mkdir(parents=True, exist_ok=True)
        reset_parquet_dataset(output_path)
        output_sql = str(output_path).replace("'", "''")
        
        try:
            if partition_by:
                partition_cols = ", ".join(partition_by)
                
                # Add partition columns
                self.conn.execute(f"""
                    COPY (
                        SELECT *,
                            EXTRACT(YEAR FROM date)::INTEGER as year,
                            EXTRACT(MONTH FROM date)::INTEGER as month
                        FROM {table_name}
                    )
                    TO '{output_sql}'
                    (FORMAT PARQUET, PARTITION_BY ({partition_cols}), 
                     COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
                """)
            else:
                self.conn.execute(f"""
                    COPY {table_name}
                    TO '{output_sql}/data.parquet'
                    (FORMAT PARQUET, COMPRESSION ZSTD)
                """)
                
            print(f"Exported to {output_path}")
            
        except Exception as e:
            print(f"Error exporting {table_name} to Parquet: {e}")
            raise
            
    def run_import(
        self,
        data_dir_5m: str = DEFAULT_DATA_DIR_5M,
        data_dir_daily: str = DEFAULT_DATA_DIR_DAILY
    ) -> Dict[str, Any]:
        """
        Run the full import process.
        
        Args:
            data_dir_5m: Directory containing 5-minute CSV files
            data_dir_daily: Directory containing daily CSV files
            
        Returns:
            Dictionary with import statistics
        """
        start_time = time.time()
        
        source_dirs = [Path(data_dir_daily), Path(data_dir_5m)]
        if not any(path.is_dir() for path in source_dirs):
            raise FileNotFoundError("No price input directory exists: " + ", ".join(map(str, source_dirs)))
        if not any(
            path.is_dir() and any(
                child.is_file() and child.suffix.lower() == ".csv" and not child.name.startswith((".", "~"))
                for child in path.iterdir()
            )
            for path in source_dirs
        ):
            raise ValueError("No price CSV files found in configured input directories")

        try:
            self.connect()
            self.init_tables()
        
            stats = {
                'daily': {'total': 0, 'success': 0, 'records': 0},
                '5m': {'total': 0, 'success': 0, 'records': 0},
            }
        
            # Import daily data first (smaller)
            total, success, records = self.import_directory(
                'daily', data_dir_daily, 'price_data_daily'
            )
            stats['daily'] = {'total': total, 'success': success, 'records': records}
        
            # Import 5-minute data
            total, success, records = self.import_directory(
                '5m', data_dir_5m, 'price_data_5m'
            )
            stats['5m'] = {'total': total, 'success': success, 'records': records}

            if self.import_failed:
                raise RuntimeError("One or more price import batches failed")
        
            # Create indices
            self.create_indices()
        
            # Export to Parquet if enabled
            if self.export_parquet:
                self.export_to_parquet('price_data_daily', ['year', 'month'])
                self.export_to_parquet('price_data_5m', ['year', 'month'])
            
            elapsed = time.time() - start_time

            # Print summary
            print("\n" + "="*60)
            print("Import Summary")
            print("="*60)
            print(f"Daily data: {stats['daily']['success']}/{stats['daily']['total']} files, "
                  f"{stats['daily']['records']:,} records")
            print(f"5m data:    {stats['5m']['success']}/{stats['5m']['total']} files, "
                  f"{stats['5m']['records']:,} records")
            print(f"Total time: {elapsed:.1f}s")
            print(f"Database:   {self.db_path}")
            if self.export_parquet:
                print(f"Parquet:    {self.parquet_dir}")
            
            return stats
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(
        description="Import price data to DuckDB database"
    )
    parser.add_argument(
        "--db-path", "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to DuckDB database file (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Root directory containing price data (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--export-parquet",
        action="store_true",
        help="Also export data to Parquet format"
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default=DEFAULT_PARQUET_DIR,
        help=f"Directory for Parquet exports (default: {DEFAULT_PARQUET_DIR})"
    )
    
    args = parser.parse_args()
    
    if not DUCKDB_AVAILABLE:
        print("Error: duckdb is required. Install with: pip install duckdb")
        return 1
        
    data_dir_5m = os.path.join(args.data_dir, '5m')
    data_dir_daily = os.path.join(args.data_dir, 'daily')
    
    importer = DuckDBPriceImporter(
        db_path=args.db_path,
        parquet_dir=args.parquet_dir,
        export_parquet=args.export_parquet
    )
    
    try:
        importer.run_import(
            data_dir_5m=data_dir_5m,
            data_dir_daily=data_dir_daily
        )
        return 0
    except KeyboardInterrupt:
        print("\nImport interrupted. Progress has been saved.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
