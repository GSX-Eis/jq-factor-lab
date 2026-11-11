from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    'factor_dir': './factors',  # 因子文件所在目录
    'cache_dir': str(_PROJECT_ROOT / 'cache/factor_values'),     # 缓存目录 (统一输出路径)
    'start_date': '2021-01-01', # 分析开始日期
    'end_date': '2022-01-01',   # 分析结束日期
    'code_list': ['002137.XSHE','002692.XSHE','002708.XSHE','002881.XSHE','002144.XSHE'],    # 分析的股票代码列表
    'n_quantiles': 5,           # 分层数量
    'max_workers': 4,           # 并行计算的工作线程数
    
    # ==================== 数据源数据库配置 ====================
    # 数据库后端仅支持 DuckDB。
    'db_backend': 'duckdb',

    # DuckDB 配置
    'duckdb_path': './data/duckdb/input.duckdb',
    'parquet_dir': None,  # 双来源需显式提供目录和 source_policy
    
    # DuckDB 性能优化选项
    'use_duckdb_cache': True,   # 启用内存缓存
    'duckdb_cache_size_mb': 2000,  # 缓存大小 (MB)
    
    # ==================== 数据表配置 ====================
    # 默认使用日频价格表，加速 Alphalens 日度分析；5分钟因子内部仍通过 price_data 视图访问 5m 数据
    'price_table': 'price_data_daily', # 价格数据表
    'output_table': 'factor_analysis_results', # 输出表
    
    # ==================== 结果数据库配置 ====================
    # 结果数据库仅支持 DuckDB。
    'results_db_backend': 'duckdb',

    # 结果数据库 DuckDB 配置
    'results_duckdb_path': str(_PROJECT_ROOT / 'data/duckdb/analysis_results.duckdb'),
    'results_parquet_dir': str(_PROJECT_ROOT / 'outputs/analysis_results_parquet'),
    'results_export_parquet': True,  # 是否导出因子值到Parquet
    
    # 结果表名配置
    'results_output_table': 'factor_analysis_results',
    'results_monthly_output_table': 'monthly_pool_results',
}
