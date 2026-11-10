#!/usr/bin/env python3
"""
import_index_constituents_duckdb.py - Import index constituents data to DuckDB

This script imports index constituents data from CSV files to DuckDB database.
It mirrors the interface of import_index_constituents.py but uses DuckDB as the backend.

Usage:
    python -m data_pipeline.loaders.import_index_constituents_duckdb --csv path/to/index_constituents.csv --db path/to/factor_database.duckdb
    
    # Replace existing data:
    python -m data_pipeline.loaders.import_index_constituents_duckdb --csv data.csv --db db.duckdb --replace

Features:
    - Fast bulk loading using DuckDB's native CSV reader
    - Automatic index creation
    - Data validation and summary statistics
"""

import argparse
from data_pipeline.loaders._transactions import atomic_import
from utils.data_revision import published_parquet, reset_parquet_dataset
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

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


# Default paths
DEFAULT_DB_PATH = './data/duckdb/factor_database.duckdb'
DEFAULT_PARQUET_DIR = './data/duckdb/parquet'


class DuckDBIndexConstituentsImporter:
    """Import index constituents data to DuckDB"""
    
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
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if export_parquet:
            self.parquet_dir.mkdir(parents=True, exist_ok=True)
            
    def connect(self) -> duckdb.DuckDBPyConnection:
        """Create database connection"""
        self.conn = duckdb.connect(str(self.db_path))
        return self.conn
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def init_table(self):
        """Initialize index_constituents table"""
        if not self.conn:
            self.connect()
            
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS index_constituents (
                date VARCHAR NOT NULL,
                stock_code VARCHAR NOT NULL,
                index_code VARCHAR NOT NULL,
                PRIMARY KEY (date, stock_code, index_code)
            )
        """)
        
        print("[INFO] index_constituents table initialized")
        
    def create_indices(self):
        """Create indices for faster queries"""
        print("[INFO] Creating indices...")
        
        try:
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ic_date 
                ON index_constituents(date)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ic_stock 
                ON index_constituents(stock_code)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ic_index 
                ON index_constituents(index_code)
            """)
            print("[INFO] Indices created successfully")
        except Exception as e:
            print(f"[WARNING] Index creation issue: {e}")
            
    @atomic_import
    def import_csv(
        self,
        csv_path: str,
        replace: bool = False
    ) -> Dict[str, Any]:
        """
        Import CSV file to index_constituents table.
        
        Args:
            csv_path: Path to CSV file
            replace: If True, replace existing data; otherwise append
            
        Returns:
            Dictionary with import statistics
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        print(f"[INFO] Reading CSV: {csv_path}")
        start_time = time.time()
        
        if not self.conn:
            self.connect()
            
        self.init_table()
        
        # Validate all keys before replacing any existing rows. Read strings
        # explicitly so leading zeroes in user stock identifiers are preserved.
        self.conn.execute("""
            CREATE OR REPLACE TEMPORARY TABLE _temp_ic AS
            SELECT CAST(TRY_CAST(trim(date) AS DATE) AS VARCHAR) AS date,
                   trim(stock_code) AS stock_code, trim(index_code) AS index_code
            FROM read_csv(?, header=true, all_varchar=true, ignore_errors=false)
        """, [str(csv_path)])
        invalid = self.conn.execute("""
            SELECT COUNT(*) FROM _temp_ic
            WHERE date IS NULL OR stock_code IS NULL OR stock_code = ''
               OR index_code IS NULL OR index_code = ''
        """).fetchone()[0]
        if invalid:
            raise ValueError(
                f"{csv_path}: {invalid} invalid rows; date must be a valid date and "
                "stock_code/index_code must be non-empty after trimming whitespace"
            )
        if not self.conn.execute("SELECT COUNT(*) FROM _temp_ic").fetchone()[0]:
            raise ValueError(f"{csv_path}: no index constituent rows to import")
        if replace:
            self.conn.execute("DELETE FROM index_constituents")

        # Get statistics before insert
        stats_result = self.conn.execute("""
            SELECT 
                COUNT(*) as total_records,
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(DISTINCT date) as unique_dates,
                COUNT(DISTINCT stock_code) as unique_stocks,
                COUNT(DISTINCT index_code) as unique_indices
            FROM _temp_ic
        """).fetchone()
        
        total_records = stats_result[0]
        min_date = stats_result[1]
        max_date = stats_result[2]
        unique_dates = stats_result[3]
        unique_stocks = stats_result[4]
        unique_indices = stats_result[5]
        
        print(f"[INFO] Data summary:")
        print(f"  Records: {total_records:,}")
        print(f"  Date range: {min_date} ~ {max_date}")
        print(f"  Unique dates: {unique_dates}")
        print(f"  Unique stocks: {unique_stocks}")
        print(f"  Unique indices: {unique_indices}")
        
        # Insert into main table
        self.conn.execute("""
            INSERT OR REPLACE INTO index_constituents (date, stock_code, index_code)
            SELECT * FROM _temp_ic
        """)
        
        # Create indices
        self.create_indices()
        
        # Get final row count
        final_count = self.conn.execute("""
            SELECT COUNT(*) FROM index_constituents
        """).fetchone()[0]
        
        # Recent dates sample
        print("\n[INFO] Recent trading days stock counts:")
        recent = self.conn.execute("""
            SELECT date, COUNT(*) as stock_count
            FROM index_constituents
            GROUP BY date
            ORDER BY date DESC
            LIMIT 5
        """).fetchall()
        
        for row in recent:
            print(f"  {row[0]}: {row[1]} stocks")
            
        elapsed = time.time() - start_time
        
        stats = {
            'total_records': total_records,
            'final_count': final_count,
            'min_date': min_date,
            'max_date': max_date,
            'unique_dates': unique_dates,
            'unique_stocks': unique_stocks,
            'unique_indices': unique_indices,
            'elapsed_time': elapsed
        }
        
        print(f"\n[INFO] Import completed in {elapsed:.2f}s")
        print(f"[INFO] Database: {self.db_path}")
        
        return stats
        
    @published_parquet(dataset='index_constituents')
    def export_to_parquet(self):
        """Export index_constituents to Parquet format"""
        if not self.export_parquet:
            return
            
        print("\n[INFO] Exporting to Parquet...")
        
        output_path = self.parquet_dir / "index_constituents"
        output_path.mkdir(parents=True, exist_ok=True)
        reset_parquet_dataset(output_path)
        output_sql = str(output_path).replace("'", "''")
        
        try:
            self.conn.execute(f"""
                COPY index_constituents
                TO '{output_sql}/data.parquet'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
            print(f"[INFO] Exported to {output_path}")
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Import index constituents data to DuckDB"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to DuckDB database file (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing data (default: append)"
    )
    parser.add_argument(
        "--export-parquet",
        action="store_true",
        help="Also export to Parquet format"
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default=DEFAULT_PARQUET_DIR,
        help=f"Directory for Parquet exports (default: {DEFAULT_PARQUET_DIR})"
    )
    
    args = parser.parse_args()
    
    if not DUCKDB_AVAILABLE:
        print("[ERROR] duckdb is required. Install with: pip install duckdb")
        return 1
        
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV file not found: {args.csv}")
        return 1
        
    importer = DuckDBIndexConstituentsImporter(
        db_path=args.db,
        parquet_dir=args.parquet_dir,
        export_parquet=args.export_parquet
    )
    
    try:
        importer.import_csv(
            csv_path=args.csv,
            replace=args.replace
        )
        
        if args.export_parquet:
            importer.export_to_parquet()
            
        return 0
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    finally:
        importer.close()


if __name__ == "__main__":
    exit(main())

