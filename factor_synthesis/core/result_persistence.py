"""
Result Persistence Module

Database table utilities for factor synthesis results (DuckDB only).
Extracted from FactorSynthesizer for modularity and reuse.

Author: QuantationAI
Date: 2026-01-12
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def check_table_exists(conn, table_name: str, db_backend: str = 'duckdb') -> bool:
    """检查表是否存在 (DuckDB)
    
    Args:
        conn: DuckDB 数据库连接对象
        table_name: 表名
        db_backend: 数据库后端类型 (保留参数以兼容调用方，仅支持 'duckdb')
        
    Returns:
        表是否存在
    """
    result = conn.execute("SHOW TABLES").fetchall()
    ascii_lower = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
    return table_name.translate(ascii_lower) in [row[0].translate(ascii_lower) for row in result]


def initialize_local_tables(
    conn, 
    db_backend: str = 'duckdb', 
    custom_logger: Optional[logging.Logger] = None
) -> None:
    """初始化本地数据库表结构 (DuckDB)
    
    Args:
        conn: DuckDB 数据库连接对象
        db_backend: 数据库后端类型 (保留参数以兼容调用方，仅支持 'duckdb')
        custom_logger: 可选的自定义 logger
        
    Raises:
        Exception: 初始化失败时抛出
    """
    log = custom_logger or logger
    
    try:
        # DuckDB DDL (无 AUTOINCREMENT，使用 PRIMARY KEY)
        create_factor_values_sql = """
        CREATE TABLE IF NOT EXISTS factor_values (
            date DATE,
            stock_code VARCHAR,
            factor_name VARCHAR,
            factor_value DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, stock_code, factor_name)
        )
        """
        
        create_results_sql = """
        CREATE TABLE IF NOT EXISTS factor_analysis_results (
            factor_name VARCHAR NOT NULL,
            date DATE NOT NULL,
            code VARCHAR NOT NULL,
            period INTEGER NOT NULL,
            quantile INTEGER,
            ic DOUBLE,
            ir DOUBLE,
            return DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (factor_name, date, code)
        )
        """
        
        conn.execute(create_factor_values_sql)
        conn.execute(create_results_sql)
        # DuckDB 自动提交，无需显式 commit
        
        log.info(f"本地数据库表结构初始化完成 (后端: {db_backend})")
        
    except Exception as e:
        log.error(f"初始化本地数据库表失败: {e}")
        raise
