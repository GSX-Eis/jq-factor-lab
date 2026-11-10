"""
Unit tests for price_data_to_duckdb.py schema migration.

Tests:
- Factor column is created in new tables
- Factor column is added to existing tables via migration
- Existing data is preserved after migration
- Factor values are correctly imported from CSV

Requirements: 2.1, 2.4
"""

import tempfile
from pathlib import Path
from typing import Generator

import duckdb
import pytest

from data_pipeline.loaders.price_data_to_duckdb import DuckDBPriceImporter


@pytest.mark.parametrize("frequency", ["daily", "5m"])
@pytest.mark.parametrize("mode", ["single", "bulk"])
def test_import_preserves_unknown_and_zero_money(tmp_path, frequency, mode):
    csv_path = tmp_path / "000001.XSHE.csv"
    time_suffix = " 09:35:00" if frequency == "5m" else ""
    csv_path.write_text(
        "date,open,close,high,low,volume,money,paused,high_limit,low_limit,pre_close,factor\n"
        + "".join(
            f"2019-01-0{day}{time_suffix},10,10,11,9,100,{money},0,11,9,10,1\n"
            for day, money in enumerate(["", "0", "25"], start=2)
        )
    )
    table = "price_data_daily" if frequency == "daily" else "price_data_5m"
    importer = DuckDBPriceImporter(db_path=str(tmp_path / "money.duckdb"))
    try:
        importer.connect()
        importer.init_tables()
        if mode == "single":
            count = importer.import_csv_file(str(csv_path), table, frequency)
        else:
            count, _ = importer._bulk_import_batch([str(csv_path)], table, frequency)
        assert count == 3
        assert importer.conn.execute(
            f"SELECT money FROM {table} ORDER BY date"
        ).fetchall() == [(None,), (0.0,), (25.0,)]
    finally:
        importer.close()


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_factor.duckdb"


@pytest.fixture
def temp_csv_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestFactorColumnCreation:
    """Test that factor column is created in new tables."""

    def test_new_table_has_factor_column(self, temp_db_path: Path) -> None:
        """Verify factor column exists in newly created price_data_daily table."""
        importer = DuckDBPriceImporter(db_path=str(temp_db_path))
        importer.connect()
        importer.init_tables()

        # Check column exists
        result = importer.conn.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns 
            WHERE table_name = 'price_data_daily' 
            AND column_name = 'factor'
        """).fetchone()

        assert result is not None, "factor column should exist in price_data_daily"
        assert result[0] == "factor"
        assert result[1].upper() == "DOUBLE"

        importer.close()

    def test_factor_column_is_nullable(self, temp_db_path: Path) -> None:
        """Verify factor column allows NULL values for backward compatibility."""
        importer = DuckDBPriceImporter(db_path=str(temp_db_path))
        importer.connect()
        importer.init_tables()

        # Insert a row without factor (should use default 1.0 from import logic)
        importer.conn.execute("""
            INSERT INTO price_data_daily 
            (stock_code, date, open, close, high, low, volume, money, paused, 
             high_limit, low_limit, pre_close, factor)
            VALUES ('000001.XSHE', '2024-01-01', 10.0, 10.5, 11.0, 9.5, 
                    1000000, 10500000, 0, 11.55, 9.45, 10.0, NULL)
        """)

        result = importer.conn.execute("""
            SELECT factor FROM price_data_daily WHERE stock_code = '000001.XSHE'
        """).fetchone()

        # NULL is allowed
        assert result[0] is None

        importer.close()


class TestSchemaMigration:
    """Test migration of existing tables to add factor column."""

    def test_migration_adds_factor_to_existing_table(self, temp_db_path: Path) -> None:
        """Verify migration adds factor column to existing table without it."""
        # Step 1: Create old schema without factor column
        conn = duckdb.connect(str(temp_db_path))
        conn.execute("""
            CREATE TABLE price_data_daily (
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
                PRIMARY KEY (stock_code, date)
            )
        """)

        # Insert test data
        conn.execute("""
            INSERT INTO price_data_daily VALUES 
            ('000001.XSHE', '2024-01-01', 10.0, 10.5, 11.0, 9.5, 
             1000000, 10500000, 0, 11.55, 9.45, 10.0)
        """)
        conn.close()

        # Step 2: Run importer which should trigger migration
        importer = DuckDBPriceImporter(db_path=str(temp_db_path))
        importer.connect()
        importer.init_tables()

        # Step 3: Verify factor column was added
        result = importer.conn.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'price_data_daily' 
            AND column_name = 'factor'
        """).fetchone()

        assert result is not None, "factor column should be added by migration"

        importer.close()

    def test_migration_preserves_existing_data(self, temp_db_path: Path) -> None:
        """Verify existing data is preserved after migration adds factor column."""
        # Step 1: Create old schema and insert data
        conn = duckdb.connect(str(temp_db_path))
        conn.execute("""
            CREATE TABLE price_data_daily (
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
                PRIMARY KEY (stock_code, date)
            )
        """)

        # Insert multiple test records
        test_data = [
            ("000001.XSHE", "2024-01-01", 10.0, 10.5, 11.0, 9.5, 1000000, 10500000, 0, 11.55, 9.45, 10.0),
            ("000002.XSHE", "2024-01-01", 20.0, 21.0, 22.0, 19.0, 2000000, 42000000, 0, 23.1, 18.9, 20.0),
            ("000001.XSHE", "2024-01-02", 10.5, 11.0, 11.5, 10.0, 1100000, 12100000, 0, 12.1, 9.9, 10.5),
        ]

        for row in test_data:
            conn.execute(
                "INSERT INTO price_data_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        conn.close()

        # Step 2: Run migration
        importer = DuckDBPriceImporter(db_path=str(temp_db_path))
        importer.connect()
        importer.init_tables()

        # Step 3: Verify all data is preserved
        result = importer.conn.execute("""
            SELECT COUNT(*) FROM price_data_daily
        """).fetchone()
        assert result[0] == 3, "All 3 records should be preserved"

        # Verify specific values
        result = importer.conn.execute("""
            SELECT stock_code, date, close 
            FROM price_data_daily 
            ORDER BY stock_code, date
        """).fetchall()

        assert len(result) == 3
        assert result[0][0] == "000001.XSHE"
        assert result[0][2] == 10.5
        assert result[1][0] == "000001.XSHE"
        assert result[1][2] == 11.0
        assert result[2][0] == "000002.XSHE"
        assert result[2][2] == 21.0

        importer.close()

    def test_migration_is_idempotent(self, temp_db_path: Path) -> None:
        """Verify running migration multiple times doesn't cause errors."""
        importer = DuckDBPriceImporter(db_path=str(temp_db_path))
        importer.connect()

        # Run init_tables multiple times
        importer.init_tables()
        importer.init_tables()
        importer.init_tables()

        # Should still work correctly
        result = importer.conn.execute("""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_name = 'price_data_daily' 
            AND column_name = 'factor'
        """).fetchone()

        assert result[0] == 1, "Should have exactly one factor column"

        importer.close()


class TestFactorImport:
    """Test that factor values are correctly imported from CSV."""

    def test_import_csv_with_factor(
        self, temp_db_path: Path, temp_csv_dir: Path
    ) -> None:
        """Verify factor values are imported from CSV files."""
        # Create test CSV with factor column
        csv_content = """date,open,close,high,low,volume,money,paused,high_limit,low_limit,pre_close,factor
2024-01-01,10.0,10.5,11.0,9.5,1000000,10500000,0,11.55,9.45,10.0,1.5
2024-01-02,10.5,11.0,11.5,10.0,1100000,12100000,0,12.1,9.9,10.5,1.5
2024-01-03,11.0,10.8,11.2,10.5,900000,9720000,0,12.1,9.9,11.0,1.5
"""
        csv_path = temp_csv_dir / "000001.XSHE.csv"
        csv_path.write_text(csv_content)

        # Import
        importer = DuckDBPriceImporter(db_path=str(temp_db_path))
        importer.connect()
        importer.init_tables()

        records = importer.import_csv_file(str(csv_path), "price_data_daily", "daily")

        assert records == 3, "Should import 3 records"

        # Verify factor values
        result = importer.conn.execute("""
            SELECT date, factor FROM price_data_daily 
            WHERE stock_code = '000001.XSHE'
            ORDER BY date
        """).fetchall()

        assert len(result) == 3
        for row in result:
            assert row[1] == 1.5, f"Factor should be 1.5, got {row[1]}"

        importer.close()

    def test_import_csv_without_factor_uses_default(
        self, temp_db_path: Path, temp_csv_dir: Path
    ) -> None:
        """Verify missing factor column defaults to 1.0."""
        # Create test CSV without factor column
        csv_content = """date,open,close,high,low,volume,money,paused,high_limit,low_limit,pre_close
2024-01-01,10.0,10.5,11.0,9.5,1000000,10500000,0,11.55,9.45,10.0
2024-01-02,10.5,11.0,11.5,10.0,1100000,12100000,0,12.1,9.9,10.5
"""
        csv_path = temp_csv_dir / "000002.XSHE.csv"
        csv_path.write_text(csv_content)

        # Import
        importer = DuckDBPriceImporter(db_path=str(temp_db_path))
        importer.connect()
        importer.init_tables()

        records = importer.import_csv_file(str(csv_path), "price_data_daily", "daily")

        assert records == 2, "Should import 2 records"

        # Verify factor defaults to 1.0
        result = importer.conn.execute("""
            SELECT date, factor FROM price_data_daily 
            WHERE stock_code = '000002.XSHE'
            ORDER BY date
        """).fetchall()

        assert len(result) == 2
        for row in result:
            assert row[1] == 1.0, f"Factor should default to 1.0, got {row[1]}"

        importer.close()

    def test_import_csv_with_null_factor_uses_default(
        self, temp_db_path: Path, temp_csv_dir: Path
    ) -> None:
        """Verify NULL factor values default to 1.0."""
        # Create test CSV with some NULL factors
        csv_content = """date,open,close,high,low,volume,money,paused,high_limit,low_limit,pre_close,factor
2024-01-01,10.0,10.5,11.0,9.5,1000000,10500000,0,11.55,9.45,10.0,2.0
2024-01-02,10.5,11.0,11.5,10.0,1100000,12100000,0,12.1,9.9,10.5,
2024-01-03,11.0,10.8,11.2,10.5,900000,9720000,0,12.1,9.9,11.0,2.0
"""
        csv_path = temp_csv_dir / "000003.XSHE.csv"
        csv_path.write_text(csv_content)

        # Import
        importer = DuckDBPriceImporter(db_path=str(temp_db_path))
        importer.connect()
        importer.init_tables()

        records = importer.import_csv_file(str(csv_path), "price_data_daily", "daily")

        assert records == 3, "Should import 3 records"

        # Verify factor values
        result = importer.conn.execute("""
            SELECT date, factor FROM price_data_daily 
            WHERE stock_code = '000003.XSHE'
            ORDER BY date
        """).fetchall()

        assert len(result) == 3
        assert result[0][1] == 2.0, "First row factor should be 2.0"
        assert result[1][1] == 1.0, "Second row factor should default to 1.0"
        assert result[2][1] == 2.0, "Third row factor should be 2.0"

        importer.close()
