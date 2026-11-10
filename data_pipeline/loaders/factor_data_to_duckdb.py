#!/usr/bin/env python3
"""
factor_data_to_duckdb.py - Import factor data to DuckDB + Parquet

This script imports factor data from CSV files to DuckDB database with optional
Parquet export. It mirrors the interface of factor_data_to_database.py but uses
DuckDB as the backend for improved analytical query performance.

Usage:
    python -m data_pipeline.loaders.factor_data_to_duckdb
    
    # Or with custom paths:
    python -m data_pipeline.loaders.factor_data_to_duckdb --db-path data/duckdb/factor_database.duckdb --csv-dir /path/to/stock_data/
    
    # Export to Parquet as well:
    python -m data_pipeline.loaders.factor_data_to_duckdb --export-parquet

Features:
    - Handles mixed numeric and text columns automatically
    - Deduplicates data by (stock_code, date)
    - Supports partitioned Parquet export
    - Much faster than SQLite for large datasets
"""

import os
import argparse
import csv
import re
from datetime import datetime
from collections import Counter
import glob
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

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
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas/numpy not available")


# Default paths
DEFAULT_DB_PATH = './data/duckdb/factor_database.duckdb'
DEFAULT_PARQUET_DIR = './data/duckdb/parquet'
DEFAULT_CSV_DIR = './data/import/stock_data'
BULK_BATCH_SIZE = 32
REQUIRED_FACTOR_COLUMNS = {"date"}
DATE_FORMATS = (
    ('\\d{4}-\\d{2}-\\d{2}', '%Y-%m-%d'),
    ('\\d{4}/\\d{2}/\\d{2}', '%Y/%m/%d'),
    ('\\d{8}', '%Y%m%d'),
    ('\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}', '%Y-%m-%d %H:%M:%S'),
    ('\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d{1,6}', '%Y-%m-%d %H:%M:%S.%f'),
    ('\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}', '%Y-%m-%dT%H:%M:%S'),
    ('\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{1,6}', '%Y-%m-%dT%H:%M:%S.%f'),
)


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_path_list(paths: List[str]) -> str:
    return "[" + ", ".join(_sql_string(path) for path in paths) + "]"


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class DuckDBFactorImporter:
    """Import factor data from CSV files to DuckDB"""
    
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
            
    def _detect_schema_from_csvs(self, csv_files: List[str]) -> Dict[str, str]:
        """
        Detect schema by sampling CSV files.
        
        Args:
            csv_files: List of CSV file paths
            
        Returns:
            Dict mapping column names to DuckDB types
        """
        if not PANDAS_AVAILABLE:
            return {}
            
        # Sample up to 10 files
        sample_files = csv_files[:10]
        all_columns = set()
        numeric_columns = set()
        
        for filepath in sample_files:
            try:
                df = pd.read_csv(filepath, nrows=100)
                all_columns.update(df.columns)
                
                for col in df.columns:
                    if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                        numeric_columns.add(col)
            except Exception:
                continue
                
        # Build schema
        schema = {}
        for col in all_columns:
            if col in ['stock_code', 'code', 'date']:
                schema[col] = 'VARCHAR'
            elif col in numeric_columns:
                schema[col] = 'DOUBLE'
            else:
                schema[col] = 'VARCHAR'
                
        return schema
        
    def init_factor_table(self, schema: Dict[str, str] = None):
        """
        Initialize factor_data table.
        
        Args:
            schema: Optional schema dict (column -> type)
        """
        if not self.conn:
            self.connect()
            
        # Check if table exists
        result = self.conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name = 'factor_data'
              AND table_catalog = current_database()
              AND table_schema = current_schema()
        """).fetchone()
        
        if result[0] > 0:
            print("factor_data table already exists")
            return
            
        # Build CREATE TABLE statement
        if schema:
            columns = []
            for col, dtype in schema.items():
                if col not in ['stock_code', 'date']:
                    columns.append(f'{_quote_ident(col)} {dtype}')
                    
            columns_str = ",\n                ".join(columns)
            if columns_str:
                columns_str += ","
            
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS factor_data (
                    stock_code VARCHAR,
                    date VARCHAR,
                    {columns_str}
                    PRIMARY KEY (stock_code, date)
                )
            """)
        else:
            # Create a minimal table that will be expanded
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_data (
                    stock_code VARCHAR,
                    date VARCHAR,
                    PRIMARY KEY (stock_code, date)
                )
            """)
            
        print("factor_data table initialized")

    def _validate_csv_headers(self, csv_files: List[str]) -> None:
        """Require every source to expose its own date field before unioning."""
        invalid = []
        for filepath in csv_files:
            try:
                with open(filepath, newline="", encoding="utf-8-sig") as handle:
                    header = next(csv.reader(handle), [])
                    columns = {column.strip() for column in header}
            except (OSError, UnicodeDecodeError, csv.Error) as exc:
                invalid.append(f"{os.path.basename(filepath)} unreadable ({exc})")
                continue
            # Reject ambiguous identities before pandas and DuckDB rename them differently.
            duplicates = sorted(name for name, count in Counter(header).items() if name and count > 1)
            if duplicates:
                invalid.append(f"{os.path.basename(filepath)} duplicate column(s): {', '.join(duplicates)}")
            missing = sorted(REQUIRED_FACTOR_COLUMNS - columns)
            if missing:
                invalid.append(
                    f"{os.path.basename(filepath)} missing {', '.join(missing)}"
                )
        if invalid:
            raise ValueError("Invalid factor CSV header(s): " + "; ".join(invalid))
        
    @atomic_import
    def import_csv_files(self, csv_dir: str) -> Tuple[int, int, int]:
        """
        Import all CSV files from directory to factor_data table.
        
        This uses DuckDB's native CSV reading and handles schema evolution
        automatically.
        
        Args:
            csv_dir: Directory containing CSV files (one per stock)
            
        Returns:
            Tuple of (total_files, successful_files, total_records)
        """
        if not self.conn:
            self.connect()
            
        if not os.path.isdir(csv_dir):
            raise FileNotFoundError(f"Factor CSV directory not found: {csv_dir}")
        csv_files = sorted(
            path for path in glob.glob(os.path.join(csv_dir, "*.csv"))
            if not os.path.basename(path).startswith((".", "~"))
        )
        if not csv_files:
            raise ValueError(f"No factor CSV files found in {csv_dir}")
        self._validate_csv_headers(csv_files)
        # Validate raw text without changing DuckDB's existing inferred/stored values.
        valid_dates = set()
        for filepath in csv_files:
            with open(filepath, newline="", encoding="utf-8-sig") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                date_column = next(i for i, name in enumerate(header) if name.strip() == "date")
                for row_number, row in enumerate(reader, start=2):
                    value = row[date_column].strip() if len(row) > date_column else ""
                    if not value or value in valid_dates:
                        continue
                    for pattern, date_format in DATE_FORMATS:
                        if re.fullmatch(pattern, value):
                            try:
                                datetime.strptime(value, date_format)
                            except ValueError:
                                continue
                            valid_dates.add(value)
                            break
                    else:
                        raise ValueError(
                            f"Invalid nonempty date in {os.path.basename(filepath)} row {row_number}"
                        )
            
        print(f"\n{'='*60}")
        print(f"Importing {len(csv_files)} CSV files from {csv_dir}")
        print(f"{'='*60}")
        
        # Detect schema from sample files
        print("Detecting schema from sample files...")
        schema = self._detect_schema_from_csvs(csv_files)
        print(f"Detected {len(schema)} columns")
        
        # Initialize table with detected schema
        self.init_factor_table(schema)
        
        total_files = 0
        successful_files = 0
        total_records = 0
        start_time = time.time()
        
        visible_csv_files = csv_files
        total_files = len(csv_files)

        for start in range(0, len(visible_csv_files), BULK_BATCH_SIZE):
            batch = visible_csv_files[start:start + BULK_BATCH_SIZE]
            try:
                paths_sql = _sql_path_list(batch)
                self.conn.execute(f"""
                    CREATE OR REPLACE TEMPORARY TABLE _factor_raw AS
                    SELECT *
                    FROM read_csv({paths_sql},
                        header=true,
                        auto_detect=true,
                        union_by_name=true,
                        filename=true,
                        ignore_errors=false
                    )
                """)

                raw_cols_result = self.conn.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = '_factor_raw'
                    ORDER BY ordinal_position
                """).fetchall()
                raw_types = {name: dtype for name, dtype in raw_cols_result}
                raw_cols = [name for name, _ in raw_cols_result]
                factor_cols = [
                    col for col in raw_cols
                    if col not in {"date", "code", "stock_code", "filename"}
                    and col
                    and not col.startswith("column")
                ]
                select_cols = ",\n                        ".join(
                    [_quote_ident(col) for col in factor_cols]
                )
                trailing_cols = ",\n                        " + select_cols if select_cols else ""
                dropped = self.conn.execute("""
                    SELECT count(*) FROM _factor_raw
                    WHERE date IS NULL OR TRIM(CAST(date AS VARCHAR)) = ''
                """).fetchone()[0]
                if dropped:
                    print(f"Dropped {dropped} rows with missing or empty date")
                self.conn.execute(f"""
                    CREATE OR REPLACE TEMPORARY TABLE _factor_stage_raw AS
                    SELECT
                        regexp_extract(filename, '([^/]+)\\.csv$', 1) AS stock_code,
                        CAST(date AS VARCHAR) AS date
                        {trailing_cols}
                    FROM _factor_raw
                    WHERE date IS NOT NULL
                      AND TRIM(CAST(date AS VARCHAR)) <> ''
                """)
                if factor_cols:
                    payload = "struct_pack(" + ", ".join(
                        f"{_quote_ident(col)} := {_quote_ident(col)}" for col in factor_cols
                    ) + ")"
                    conflicts = self.conn.execute(f"""
                        SELECT count(*) FROM (
                            SELECT stock_code, date FROM _factor_stage_raw
                            GROUP BY stock_code, date
                            HAVING count(DISTINCT {payload}) > 1
                        )
                    """).fetchone()[0]
                    if conflicts:
                        raise ValueError(f"Conflicting factor payloads for {conflicts} duplicate keys")
                self.conn.execute("""
                    CREATE OR REPLACE TEMPORARY TABLE _factor_stage AS
                    SELECT * EXCLUDE (rn)
                    FROM (
                        SELECT *,
                            row_number() OVER (
                                PARTITION BY stock_code, date
                                ORDER BY stock_code
                            ) AS rn
                        FROM _factor_stage_raw
                    )
                    WHERE rn = 1
                """)

                expected_codes = {
                    os.path.splitext(os.path.basename(filepath))[0] for filepath in batch
                }
                imported_codes = {
                    row[0] for row in self.conn.execute(
                        "SELECT DISTINCT stock_code FROM _factor_stage"
                    ).fetchall()
                }
                missing_records = sorted(expected_codes - imported_codes)
                if missing_records:
                    raise ValueError(
                        "Factor CSV files produced no valid records: "
                        + ", ".join(missing_records)
                    )

                # Get existing columns in factor_data
                existing_cols = self.conn.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'factor_data'
                """).fetchall()
                existing_cols = {r[0] for r in existing_cols}
                
                # Add new columns to factor_data if needed
                for col in factor_cols:
                    if col not in existing_cols:
                        raw_type = raw_types.get(col, "VARCHAR").upper()
                        dtype = (
                            "DOUBLE"
                            if any(token in raw_type for token in (
                                "INT", "DOUBLE", "FLOAT", "DECIMAL", "HUGEINT"
                            ))
                            else "VARCHAR"
                        )
                        try:
                            self.conn.execute(
                                f"ALTER TABLE factor_data "
                                f"ADD COLUMN {_quote_ident(col)} {dtype}"
                            )
                        except Exception:
                            pass

                # A unioned batch loses the distinction between absent and NULL.
                # Resolve each file's actual column names using the same CSV reader,
                # then update only those columns. Explicit NULL still clears a value.
                for filepath in batch:
                    source_columns = {
                        row[0].lower() for row in self.conn.execute(
                            "DESCRIBE SELECT * FROM read_csv(?, header=true, "
                            "auto_detect=true, union_by_name=true, filename=true, "
                            "ignore_errors=false)", [filepath]
                        ).fetchall()
                    }
                    file_factors = [col for col in factor_cols if col.lower() in source_columns]
                    insert_cols = ["stock_code", "date", *file_factors]
                    target_cols = [row[0] for row in self.conn.execute(
                        "DESCRIBE factor_data"
                    ).fetchall()]
                    present_columns = {col.lower() for col in insert_cols}
                    missing_cols = [col for col in target_cols if col.lower() not in present_columns]
                    code = os.path.splitext(os.path.basename(filepath))[0]
                    # Snapshot absent values before DELETE. Keep DELETE/INSERT so
                    # existing tables without a primary key remain supported.
                    projection = ", ".join(
                        f"s.{_quote_ident(col)}" for col in insert_cols
                    )
                    join = ""
                    if missing_cols:
                        projection += ", " + ", ".join(
                            f"previous.{_quote_ident(col)}" for col in missing_cols
                        )
                        join = ("LEFT JOIN factor_data previous "
                                "ON previous.stock_code = s.stock_code AND previous.date = s.date")
                    cols_str = ", ".join(_quote_ident(col) for col in insert_cols + missing_cols)
                    self.conn.execute(f"""
                        CREATE OR REPLACE TEMPORARY TABLE _factor_replacement AS
                        SELECT {projection} FROM _factor_stage s {join}
                        WHERE s.stock_code = ?
                    """, [code])
                    self.conn.execute("""
                        DELETE FROM factor_data USING _factor_replacement
                        WHERE factor_data.stock_code = _factor_replacement.stock_code
                          AND factor_data.date = _factor_replacement.date
                    """)
                    self.conn.execute(f"""
                        INSERT INTO factor_data ({cols_str})
                        SELECT {cols_str} FROM _factor_replacement
                    """)

                # Get row count
                count_result = self.conn.execute("""
                    SELECT COUNT(*) FROM _factor_stage
                """).fetchone()
                records = count_result[0] if count_result else 0
                
                total_records += records
                successful_files += len(batch)

                processed = min(start + len(batch), len(visible_csv_files))
                elapsed = time.time() - start_time
                rate = total_records / max(1, elapsed)
                print(f"Progress: {processed}/{len(visible_csv_files)} files, "
                      f"{total_records:,} records, "
                      f"{rate:.0f} records/sec")
                          
            except KeyboardInterrupt:
                print("\nInterrupted. Database changes will be rolled back.")
                raise
            except Exception as e:
                print(f"Error processing batch starting at "
                      f"{os.path.basename(batch[0])}: {e}")
                self.import_failed = True
                raise
                
        elapsed = time.time() - start_time
        print(f"\nImported {successful_files}/{total_files} files, "
              f"{total_records:,} records in {elapsed:.1f}s")
              
        return total_files, successful_files, total_records
        
    def create_indices(self):
        """Create indices for faster queries"""
        print("Creating indices...")
        
        try:
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_factor_date 
                ON factor_data(date)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_factor_stock 
                ON factor_data(stock_code)
            """)
            print("Indices created successfully")
        except Exception as e:
            print(f"Warning: Index creation issue: {e}")
            
    @published_parquet(dataset='factor_data')
    def export_to_parquet(self):
        """Export factor_data to Parquet format"""
        if not self.export_parquet:
            return
            
        print("\nExporting factor_data to Parquet...")
        
        output_path = self.parquet_dir / "factor_data"
        output_path.mkdir(parents=True, exist_ok=True)
        reset_parquet_dataset(output_path)
        output_sql = str(output_path).replace("'", "''")
        
        try:
            self.conn.execute(f"""
                COPY (
                    SELECT *,
                        CAST(EXTRACT(YEAR FROM TRY_CAST(date AS DATE)) AS INTEGER) as year,
                        CAST(EXTRACT(MONTH FROM TRY_CAST(date AS DATE)) AS INTEGER) as month
                    FROM factor_data
                    WHERE date IS NOT NULL
                )
                TO '{output_sql}'
                (FORMAT PARQUET, PARTITION_BY (year, month), 
                 COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
            """)
            print(f"Exported to {output_path}")
        except Exception as e:
            print(f"Error exporting to Parquet: {e}")
            raise
            
    def run_import(self, csv_dir: str = DEFAULT_CSV_DIR) -> Dict[str, Any]:
        """
        Run the full import process.
        
        Args:
            csv_dir: Directory containing CSV files
            
        Returns:
            Dictionary with import statistics
        """
        source_dir = Path(csv_dir)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Factor CSV directory not found: {source_dir}")
        if not any(
            path.is_file() and path.suffix == ".csv"
            and not path.name.startswith((".", "~"))
            for path in source_dir.iterdir()
        ):
            raise ValueError(f"No factor CSV files found in {source_dir}")

        try:
            self.connect()

            total, success, records = self.import_csv_files(csv_dir)

            if self.import_failed:
                raise RuntimeError("One or more factor import batches failed")

            self.create_indices()

            if self.export_parquet:
                self.export_to_parquet()

            stats = {
                'total_files': total,
                'successful_files': success,
                'total_records': records,
                'database': str(self.db_path)
            }

            print("\n" + "="*60)
            print("Import Summary")
            print("="*60)
            print(f"Files:    {success}/{total}")
            print(f"Records:  {records:,}")
            print(f"Database: {self.db_path}")
            if self.export_parquet:
                print(f"Parquet:  {self.parquet_dir}")

            return stats
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(
        description="Import factor data to DuckDB database"
    )
    parser.add_argument(
        "--db-path", "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to DuckDB database file (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=DEFAULT_CSV_DIR,
        help=f"Directory containing CSV files (default: {DEFAULT_CSV_DIR})"
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
        
    importer = DuckDBFactorImporter(
        db_path=args.db_path,
        parquet_dir=args.parquet_dir,
        export_parquet=args.export_parquet
    )
    
    try:
        importer.run_import(csv_dir=args.csv_dir)
        return 0
    except KeyboardInterrupt:
        print("\nImport interrupted.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
