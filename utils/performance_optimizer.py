"""
Performance optimization utilities for factor calculations
"""

import pandas as pd
import numpy as np
import duckdb
from typing import List, Dict, Any, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import time
from contextlib import contextmanager

from utils.data_store import connect_input

class PerformanceOptimizer:
    """性能优化工具类"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    @contextmanager
    def timer(self, operation_name: str):
        """计时上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.logger.info(f"{operation_name} 耗时: {elapsed:.2f}秒")
    
    def optimize_sql_query(self, db_connection, sql: str, chunk_size: int = 10000) -> pd.DataFrame:
        """Execute a DuckDB query and return its frame result.

        DuckDB already streams its vectorized execution internally; callers that
        need an incremental result should use a DuckDB record-batch reader.
        """
        del chunk_size
        if not isinstance(db_connection, duckdb.DuckDBPyConnection):
            raise TypeError("db_connection must be a DuckDBPyConnection")
        try:
            with self.timer("SQL查询"):
                result = db_connection.execute(sql).fetchdf()
                self.logger.info(f"查询完成，总行数: {len(result)}")
                return result
        except Exception as e:
            self.logger.error(f"SQL查询失败: {e}")
            raise
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame内存使用"""
        if df.empty:
            return df
        
        original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # 优化数值类型
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # 优化字符串类型
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype('category')
                except:
                    pass
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        self.logger.info(f"内存优化完成: {original_memory:.1f}MB -> {optimized_memory:.1f}MB "
                        f"(减少 {reduction:.1f}%)")
        
        return df
    
    def parallel_apply(self, df: pd.DataFrame, func, group_col: str = 'code', 
                      n_jobs: int = None) -> pd.DataFrame:
        """并行应用函数到分组数据"""
        if n_jobs is None:
            n_jobs = min(mp.cpu_count(), 8)
        
        # 按指定列分组
        groups = [(name, group) for name, group in df.groupby(group_col)]
        
        if len(groups) <= 1:
            return func(df)
        
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(func, [group for _, group in groups]))
            
            # 合并结果
            valid_results = [r for r in results if r is not None and not r.empty]
            if valid_results:
                return pd.concat(valid_results, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.warning(f"并行处理失败，回退到串行: {e}")
            return func(df)
    
    def batch_process_stocks(self, stock_list: List[str], process_func, 
                           batch_size: int = 50, **kwargs) -> List[Any]:
        """批量处理股票列表"""
        results = []
        
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i:i + batch_size]
            self.logger.info(f"处理批次 {i//batch_size + 1}/{(len(stock_list)-1)//batch_size + 1}")
            
            try:
                batch_result = process_func(batch, **kwargs)
                if batch_result is not None:
                    results.append(batch_result)
            except Exception as e:
                self.logger.error(f"批次处理失败: {e}")
                continue
        
        return results
    
    def create_optimized_index(self, df: pd.DataFrame, index_cols: List[str]) -> pd.DataFrame:
        """创建优化的索引"""
        if not index_cols or df.empty:
            return df
        
        try:
            # 确保索引列存在
            missing_cols = [col for col in index_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"索引列不存在: {missing_cols}")
                return df
            
            # 设置多级索引并排序
            df = df.set_index(index_cols).sort_index()
            self.logger.info(f"创建索引: {index_cols}")
            return df
            
        except Exception as e:
            self.logger.error(f"创建索引失败: {e}")
            return df
    
    def vectorized_rolling_operation(self, data: np.ndarray, window: int, 
                                   operation: str = 'mean') -> np.ndarray:
        """向量化滚动操作"""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        # 创建滚动窗口视图
        shape = (len(data) - window + 1, window)
        strides = (data.strides[0], data.strides[0])
        rolled = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        
        # 应用操作
        if operation == 'mean':
            results = np.mean(rolled, axis=1)
        elif operation == 'std':
            results = np.std(rolled, axis=1)
        elif operation == 'sum':
            results = np.sum(rolled, axis=1)
        elif operation == 'max':
            results = np.max(rolled, axis=1)
        elif operation == 'min':
            results = np.min(rolled, axis=1)
        else:
            raise ValueError(f"不支持的操作: {operation}")
        
        # 在前面填充NaN
        full_results = np.full(len(data), np.nan)
        full_results[window-1:] = results
        
        return full_results
    
    def optimize_database_connection(self, db_path: str, *, data_bindings=None) -> duckdb.DuckDBPyConnection:
        """Open the configured DuckDB input read-only and tune only its session."""
        conn = connect_input(db_path, bindings=data_bindings)
        conn.execute("SET threads=4")
        conn.execute("SET memory_limit='4GB'")
        self.logger.info("DuckDB input connection configured read-only")
        return conn
    
    def profile_function(self, func, *args, **kwargs):
        """性能分析函数"""
        import cProfile
        import pstats
        import io
        
        pr = cProfile.Profile()
        pr.enable()
        
        result = func(*args, **kwargs)
        
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        self.logger.info(f"性能分析结果:\n{s.getvalue()}")
        return result

class DatabaseOptimizer:
    """数据库查询优化器"""
    
    def __init__(self, db_connection, logger: Optional[logging.Logger] = None):
        if not isinstance(db_connection, duckdb.DuckDBPyConnection):
            raise TypeError("db_connection must be a DuckDBPyConnection")
        self.db_connection = db_connection
        self.logger = logger or logging.getLogger(__name__)
    
    def create_indexes(self, table_name: str, index_columns: List[str]):
        """创建数据库索引"""
        for col in index_columns:
            index_name = f"idx_{table_name}_{col}"
            sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({col})"
            try:
                self.db_connection.execute(sql)
                self.logger.info(f"创建索引: {index_name}")
            except Exception as e:
                self.logger.warning(f"创建索引失败 {index_name}: {e}")
    
    def analyze_table_stats(self, table_name: str) -> Dict[str, Any]:
        """分析表统计信息"""
        try:
            # 获取表行数
            count_sql = f"SELECT COUNT(*) as row_count FROM {table_name}"
            row_count = self.db_connection.execute(count_sql).fetchone()[0]
            
            # 获取表结构
            schema_sql = f"DESCRIBE SELECT * FROM {table_name}"
            schema = self.db_connection.execute(schema_sql).fetchdf()
            
            # 获取索引信息
            index_sql = "SELECT index_name FROM duckdb_indexes() WHERE table_name=?"
            indexes = self.db_connection.execute(index_sql, [table_name]).fetchdf()
            
            stats = {
                'row_count': row_count,
                'columns': schema['column_name'].tolist(),
                'indexes': indexes['index_name'].tolist() if not indexes.empty else []
            }
            
            self.logger.info(f"表 {table_name} 统计: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"分析表统计失败: {e}")
            return {}
    
    def optimize_query_plan(self, sql: str) -> str:
        """优化查询计划"""
        # 分析查询计划
        explain_sql = f"EXPLAIN {sql}"
        try:
            plan = self.db_connection.execute(explain_sql).fetchdf()
            self.logger.info(f"查询计划:\n{plan}")
        except Exception as e:
            self.logger.warning(f"分析查询计划失败: {e}")
        
        return sql

# 使用示例
def optimize_factor_calculation(factor_instance):
    """优化因子计算的通用函数"""
    optimizer = PerformanceOptimizer(factor_instance.logger)
    
    # 优化数据库连接
    if hasattr(factor_instance, 'db_connection'):
        db_optimizer = DatabaseOptimizer(factor_instance.db_connection, factor_instance.logger)
        
        # 为常用表创建索引
        db_optimizer.create_indexes('price_data', ['date', 'stock_code'])
        db_optimizer.create_indexes('factor_data', ['date', 'stock_code'])
    
    # 重写获取数据方法以使用优化
    original_fetch = factor_instance._fetch_raw_data
    
    def optimized_fetch(*args, **kwargs):
        with optimizer.timer("数据获取"):
            data = original_fetch(*args, **kwargs)
            return optimizer.optimize_dataframe_memory(data)
    
    factor_instance._fetch_raw_data = optimized_fetch
    
    return factor_instance
