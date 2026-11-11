import argparse
import hashlib
import inspect
import json
import logging
import concurrent.futures
import os
import sys
from pathlib import Path
from contextlib import suppress
from typing import Any, Dict, List, Optional
import pandas as pd
import time
from backtest_and_analysis.config import CONFIG
from backtest_and_analysis.factor_loader import FactorLoader
from backtest_and_analysis.price_fetcher import PriceFetcher
from backtest_and_analysis.alphalens_runner import AlphalensRunner
from backtest_and_analysis.result_writer import ResultWriter
from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
from utils.config_loader import load_and_parse_config, load_pit_config
from utils.data_store import connect_read_only
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# Import monthly pool result writer

try:
    from backtest_and_analysis.monthly_pool_result_writer import MonthlyPoolResultWriter
    MONTHLY_WRITER_AVAILABLE = True
except ImportError:
    MONTHLY_WRITER_AVAILABLE = False
    print("Warning: Monthly pool result writer not available")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_factor_value_cache(factor_class, config):
    """Load a realization cache only when source and research window match."""
    # Old realization receipts do not identify custom datasets or fallback sources.
    if config.get('data_config_path') or config.get('source_policy'):
        return None
    from utils.data_store import binding_contract
    # Legacy realization receipts bind code/window, not the input generation.
    # Keep the evidence, but calculate against the requested database instead.
    if config.get('duckdb_path') or binding_contract().get('input_db'):
        return None
    cache_dir = getattr(factor_class, "FACTOR_VALUE_CACHE_DIR", None)
    source_path = inspect.getsourcefile(factor_class)
    if getattr(factor_class, "DATA_FREQUENCY", None) != "5m" or not cache_dir:
        return None
    if not source_path:
        raise RuntimeError("source-bound factor cache has no source path")
    source_hash = _sha256_file(source_path)
    metadata_path = os.path.join(cache_dir, f"{source_hash}.json")
    if not os.path.isfile(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        expected_window = (
            str(pd.Timestamp(config["start_date"]).date()),
            str(pd.Timestamp(config["end_date"]).date()),
        )
        if metadata.get("schema_version") != "quanta_alpha.factor_value_cache.v1":
            raise RuntimeError("source-bound factor cache schema mismatch")
        if metadata.get("source_sha256") != source_hash:
            raise RuntimeError("source-bound factor cache source mismatch")
        if (metadata.get("start_date"), metadata.get("end_date")) != expected_window:
            raise RuntimeError("source-bound factor cache window mismatch")
        from utils.data_access_config import load_data_access_config
        if metadata.get("data_access") != load_data_access_config().as_dict():
            return None
        if metadata.get("data_file") != f"{source_hash}.parquet":
            raise RuntimeError("source-bound factor cache data path mismatch")
        data_path = os.path.join(cache_dir, metadata["data_file"])
        if not os.path.isfile(data_path) or _sha256_file(data_path) != metadata.get(
            "data_sha256"
        ):
            raise RuntimeError("source-bound factor cache data hash mismatch")
        cached = pd.read_parquet(data_path)
        cached.index = pd.to_datetime(cached.index, errors="coerce")
        return cached.loc[cached.index.notna()].sort_index()
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("source-bound factor cache is invalid") from exc

# Import dynamic stock pool utilities
try:
    from utils.dynamic_stock_pool import DynamicStockPoolManager, StockFilter
    DYNAMIC_POOL_AVAILABLE = True
except ImportError:
    DYNAMIC_POOL_AVAILABLE = False
    print("Warning: Dynamic stock pool utilities not available. Only static pool analysis will work.")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, '__dict__'):
        return _jsonable(vars(value))
    return value


POOL_SNAPSHOT_CONTRACT = 'literal-filters-v3'


def _pool_snapshot_fingerprint(config: Dict[str, Any], pool_configs: List[Dict]) -> str:
    from utils.data_access_config import load_data_access_config
    from utils.data_store import binding_contract, resolve_connection_binding
    # Analysis readers inherit the environment binding, including exact versions.
    bindings = binding_contract()
    db_path = config.get('duckdb_path') or bindings.get('input_db')
    if db_path and bindings:
        db_path, bindings = resolve_connection_binding(db_path, bindings)
    stat = os.stat(db_path) if db_path and os.path.exists(db_path) else None
    from utils.data_revision import database_revision
    revision = database_revision(db_path) if stat else None
    from utils.stock_pool_schedule import load_pool_policy
    payload = {
        'pool_contract': POOL_SNAPSHOT_CONTRACT,
        'pool_policy': load_pool_policy(),
        'start_date': config.get('start_date'),
        'end_date': config.get('end_date'),
        'fundamental_table': config.get('fundamental_table'),
        'price_table': config.get('price_table'),
        'pool_configs': _jsonable(pool_configs),
        'data_access': load_data_access_config(config.get('data_config_path')).as_dict(),
        'input_bindings': {'input_db': bindings.get('input_db'), 'inputs': bindings.get('inputs', {})},
        'source_db': {
            'revision': revision,
            'path': db_path,
            'size': stat.st_size if stat else None,
            'mtime_ns': stat.st_mtime_ns if stat else None,
        },
    }
    if config.get('source_policy'):
        payload['source_policy'] = config['source_policy']
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _load_pool_snapshot(config: Dict[str, Any], pool_configs: List[Dict]) -> Optional[Dict]:
    snapshot_path = config.get('pool_snapshot_path')
    if not snapshot_path:
        return None
    if config.get('pool_snapshot_read_only'):
        if not os.path.exists(snapshot_path):
            raise RuntimeError(f"read-only pool snapshot is missing: {snapshot_path}")
        expected_hash = str(config.get('pool_snapshot_sha256') or '')
        if not expected_hash or _sha256_file(snapshot_path) != expected_hash:
            raise RuntimeError(f"read-only pool snapshot hash mismatch: {snapshot_path}")
        try:
            with open(snapshot_path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            pools = payload.get('pools') if isinstance(payload, dict) else None
            if not isinstance(pools, dict) or not all(isinstance(pool, dict) for pool in pools.values()):
                raise RuntimeError(f"read-only pool snapshot is invalid: {snapshot_path}")
            logger.info("复用只读股票池快照: %s", snapshot_path)
            return pools
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"read-only pool snapshot is invalid: {snapshot_path}") from exc
    if config.get('source_policy'):
        # Mutable Parquet sources have no version in the old pool receipt.
        return None
    if not os.path.exists(snapshot_path):
        return None
    try:
        with open(snapshot_path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        if payload.get('fingerprint') != _pool_snapshot_fingerprint(config, pool_configs):
            logger.info("股票池快照已失效，将重新计算: %s", snapshot_path)
            return None
        pools = payload.get('pools')
        if not isinstance(pools, dict) or not all(isinstance(pool, dict) for pool in pools.values()):
            return None
        logger.info("复用已验证股票池快照: %s", snapshot_path)
        return pools
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("读取股票池快照失败，将重新计算: %s", exc)
        return None


def _save_pool_snapshot(
    config: Dict[str, Any], pool_configs: List[Dict], pools: Dict[str, Dict]
) -> None:
    if config.get('pool_snapshot_read_only') or config.get('source_policy'):
        return
    snapshot_path = config.get('pool_snapshot_path')
    if not snapshot_path or not pools or any(pool is None for pool in pools.values()):
        return
    snapshot_dir = os.path.dirname(snapshot_path)
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)
    temporary_path = f"{snapshot_path}.tmp.{os.getpid()}"
    payload = {
        'schema_version': 1,
        'fingerprint': _pool_snapshot_fingerprint(config, pool_configs),
        'pools': pools,
    }
    with open(temporary_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
    os.replace(temporary_path, snapshot_path)
    logger.info("股票池快照已原子写入: %s", snapshot_path)


def _factor_data_start_date(factor_class: type, config: Dict[str, Any]) -> str:
    """Resolve declared trading-session warmup without silently truncating it."""
    from utils.factor_base import _get_pit_window_from_config
    from utils.trading_calendar import load_trading_calendar, CalendarCoverageError
    analysis_start = pd.Timestamp(config['start_date']).normalize()
    history_floor_value = config.get('factor_history_start_date')
    if not history_floor_value:
        return analysis_start.strftime('%Y-%m-%d')
    calendar = load_trading_calendar(duckdb_path=config.get('duckdb_path'), bindings=config.get('data_bindings'))
    trading_days = _get_pit_window_from_config(factor_class.__name__, config.get('data_frequency', '5m'))
    required_start = calendar.previous_sessions(analysis_start, trading_days)[0]
    if pd.Timestamp(history_floor_value).normalize() > required_start:
        raise CalendarCoverageError('factor_history_start_date truncates declared warmup sessions')
    return required_start.strftime('%Y-%m-%d')


def wait_for_db_connection(
    duckdb_path: Optional[str],
    retry_interval: int = 30,
) -> None:
    """Block until the configured DuckDB database is connectable."""
    if not DUCKDB_AVAILABLE:
        raise RuntimeError("DuckDB is required for factor analysis")
    if not duckdb_path:
        raise ValueError("duckdb_path is required; SQLite is no longer supported")
    if not os.path.isfile(duckdb_path):
        raise FileNotFoundError(f"DuckDB database not found: {duckdb_path}")
    attempt = 0
    while True:
        try:
            conn = connect_read_only(duckdb_path)
            conn.close()
            logger.info("Database connection verified.")
            return
        except Exception as e:
            if "lock" not in str(e).lower():
                raise
            attempt += 1
            logger.warning(
                f"Database unavailable (attempt {attempt}): {e}  "
                f"-- retrying in {retry_interval}s ..."
            )
            time.sleep(retry_interval)


def wait_for_result_db_write_slot(
    duckdb_path: Optional[str],
    retry_interval: float = 1,
) -> None:
    """Block until the result DuckDB can be opened for exclusive writes."""
    if not DUCKDB_AVAILABLE or not duckdb_path:
        return

    from utils.duckdb_results_writer import acquire_exclusive_write_conn

    logger.info("Waiting for result database write slot: %s", duckdb_path)
    with acquire_exclusive_write_conn(duckdb_path, retry_interval_sec=retry_interval) as session:
        logger.info(
            "Result database write slot verified: db_path=%s wait_seconds=%.3f lock_retries=%d",
            duckdb_path,
            session.wait_seconds,
            session.retry_count,
        )


def log_pool_statistics(factor, pool_name: str, description: str):
    """记录股票池统计信息"""
    try:
        # 检测调仓频率（默认为非周频）
        rebalance_schedule = getattr(factor, 'rebalance_schedule', None)

        if hasattr(factor, 'monthly_stock_pools') and factor.monthly_stock_pools:
            total_periods = len(factor.monthly_stock_pools)
            stock_counts = [len(stocks) for stocks in factor.monthly_stock_pools.values()]
            avg_stocks = sum(stock_counts) / len(stock_counts) if stock_counts else 0
            
            logger.info(f"=== 股票池统计 - {pool_name} ===")
            logger.info(f"描述: {description}")
            logger.info(f"覆盖期间数: {total_periods}")
            logger.info(f"平均股票数: {avg_stocks:.1f}")
            logger.info(f"股票数范围: {min(stock_counts) if stock_counts else 0} - {max(stock_counts) if stock_counts else 0}")
            
            # 如果是周频动态股票池，额外打印“当前”股票池的股票数（取最近一个调仓周期）
            if rebalance_schedule == 'weekly':
                try:
                    # key 可能是 'YYYY-MM-DD'，按时间排序后取最后一个作为当前周期
                    sorted_periods = sorted(factor.monthly_stock_pools.keys())
                    current_period = sorted_periods[-1]
                    current_pool = factor.monthly_stock_pools[current_period]
                    logger.info(f"当前周度股票池({current_period})股票数量: {len(current_pool)}")
                except Exception as e:
                    logger.warning("记录当前周度股票池股票数失败: {e}")
            
            # 如果期间数过多（例如周频），只记录概要或部分
            if total_periods > 24:
                logger.debug(f"详细期间股票数已省略（共 {total_periods} 个期间）")
            else:
                # 记录每期股票数
                for period, stocks in factor.monthly_stock_pools.items():
                    logger.debug(f"{period}: {len(stocks)} 只股票")
        else:
            logger.info(f"=== 固定股票池统计 - {pool_name} ===")
            logger.info(f"描述: {description}")
            if hasattr(factor, 'stock_pool') and factor.stock_pool:
                logger.info(f"股票数量: {len(factor.stock_pool)}")
    except Exception as e:
        logger.warning(f"记录股票池统计失败: {e}")


def _close_factor_duckdb_handles(factor: Any, factor_name: str) -> None:
    """Close long-lived DuckDB readers before acquiring the shared write session."""
    try:
        if hasattr(factor, '_duckdb_adapter') and factor._duckdb_adapter is not None:
            factor._duckdb_adapter.close()
            factor._duckdb_adapter = None
            logger.debug("Closed factor DuckDB adapter for %s", factor_name)
    except Exception as exc:
        logger.debug("Failed to close factor DuckDB adapter for %s: %s", factor_name, exc)

    try:
        if hasattr(factor, 'dynamic_pool_manager') and factor.dynamic_pool_manager is not None:
            if hasattr(factor.dynamic_pool_manager, '_duckdb_conn') and factor.dynamic_pool_manager._duckdb_conn is not None:
                factor.dynamic_pool_manager._duckdb_conn.close()
                factor.dynamic_pool_manager._duckdb_conn = None
                logger.debug("Closed dynamic pool DuckDB connection for %s", factor_name)
    except Exception as exc:
        logger.debug("Failed to close dynamic pool DuckDB connection for %s: %s", factor_name, exc)


def _persist_analysis_outputs(
    *,
    factor: Any,
    factor_name: str,
    factor_df_unfiltered: pd.DataFrame,
    combined_results: Dict[str, Any],
    writer: ResultWriter,
    monthly_writer: Optional[Any],
    retry_interval_sec: float = 1,
) -> bool:
    """Commit factor values/analysis/monthly metrics together, then export Parquet.

    A post-commit export failure is reported separately: database rows are durable.
    """
    _close_factor_duckdb_handles(factor, factor_name)

    monthly_using_duckdb = (
        monthly_writer is not None
        and getattr(monthly_writer, "using_duckdb", False)
    )
    if monthly_writer is not None and writer.using_duckdb != monthly_using_duckdb:
        raise RuntimeError(
            "Result writer and monthly writer backend mismatch; both writers must use the same backend."
        )

    if not writer.using_duckdb:
        raise RuntimeError("analysis result persistence requires a DuckDB writer")
    with writer.duckdb_write_session(retry_interval_sec=retry_interval_sec) as session:
        conn = session.conn
        factor_rows = 0
        analysis_rows = 0
        monthly_rows = 0
        stage = "begin"
        logger.info(
            "DuckDB write session acquired: factor=%s wait_seconds=%.3f lock_retries=%d",
            factor_name,
            session.wait_seconds,
            session.retry_count,
        )
        conn.execute("BEGIN TRANSACTION")
        try:
            stage = "factor_values"
            factor_rows = writer.save_factor_values(
                factor_df_unfiltered,
                factor_name,
                duckdb_conn=conn,
                allow_fallback=False,
                skip_parquet=True,
            )
            stage = "factor_analysis_results"
            analysis_rows = writer.write(
                factor_name,
                combined_results,
                duckdb_conn=conn,
                allow_fallback=False,
            )
            if monthly_writer is not None:
                stage = "monthly_pool_results"
                monthly_rows = monthly_writer.write(
                    factor_name,
                    combined_results,
                    duckdb_conn=conn,
                    allow_fallback=False,
                )
            stage = "commit"
            conn.execute("COMMIT")
        except Exception as exc:
            with suppress(Exception):
                conn.execute("ROLLBACK")
            logger.error(
                "Result persistence failed: factor=%s stage=%s wait_seconds=%.3f lock_retries=%d error=%s",
                factor_name,
                stage,
                session.wait_seconds,
                session.retry_count,
                exc,
            )
            raise

        try:
            parquet_rows = writer.export_factor_values_parquet(
                factor_df_unfiltered, factor_name, write_session=session,
            )
        except Exception as exc:
            from utils.duckdb_results_writer import CommittedParquetExportError
            raise CommittedParquetExportError(factor_name) from exc
    logger.info(
        "Persisted analysis outputs: factor=%s backend=duckdb wait_seconds=%.3f lock_retries=%d factor_rows=%d analysis_rows=%d monthly_rows=%d parquet_rows=%d",
        factor_name,
        session.wait_seconds,
        session.retry_count,
        factor_rows,
        analysis_rows,
        monthly_rows,
        parquet_rows,
    )
    return True


def _persist_materialized_factor_values(
    *,
    factor: Any,
    factor_name: str,
    factor_df: pd.DataFrame,
    writer: ResultWriter,
    start_date: str,
    end_date: str,
    retry_interval_sec: float = 1,
) -> bool:
    """Replace one factor's configured DuckDB date range without analysis side effects."""
    if not writer.using_duckdb:
        raise RuntimeError("materialize-only requires an active DuckDB results backend")

    _close_factor_duckdb_handles(factor, factor_name)
    with writer.duckdb_write_session(retry_interval_sec=retry_interval_sec) as session:
        conn = session.conn
        conn.execute("BEGIN TRANSACTION")
        try:
            row_count = writer.save_factor_values(
                factor_df,
                factor_name,
                duckdb_conn=conn,
                allow_fallback=False,
                skip_parquet=True,
                date_range=(start_date, end_date),
            )
            conn.execute("COMMIT")
        except Exception:
            with suppress(Exception):
                conn.execute("ROLLBACK")
            raise

    logger.info(
        "Materialized factor values: factor=%s start_date=%s end_date=%s rows=%d",
        factor_name,
        start_date,
        end_date,
        row_count,
    )
    return True


def _compute_mining_only_ic(
    factor_df: pd.DataFrame,
    pricing: pd.Series,
    periods: tuple[int, ...],
) -> pd.DataFrame:
    """Compute only the daily RankIC required by the mining admission gate."""
    if not isinstance(pricing.index, pd.MultiIndex) or pricing.index.names[0] != "asset":
        raise ValueError("mining-only pricing must use an (asset, date) MultiIndex")
    prices_df = pd.DataFrame(
        {asset: pricing.loc[asset].copy() for asset in pricing.index.get_level_values(0).unique()}
    )
    runner = AlphalensRunner(n_quantiles=5, periods=periods)
    fixed_factor, _ = runner._fix_business_day_alignment(factor_df, prices_df)
    factor_series = runner._break_factor_ties(fixed_factor.stack().dropna())
    factor_series.index.names = ["date", "asset"]
    forward_returns = ForwardReturnsCalculator().calculate_forward_returns(
        pricing,
        periods=periods,
        execution_lag=runner.execution_lag,
    )
    clean = runner._prepare_factor_data_with_precomputed_returns(factor_series, forward_returns)
    return pd.DataFrame(
        {
            column: clean.groupby(level=0, observed=True).apply(
                lambda group, return_column=column: group["factor"].corr(
                    group[return_column], method="spearman"
                )
            )
            for column in (f"{period}D" for period in periods)
        }
    )

def analyze_factor_static(factor, price_fetcher, runner, writer, monthly_writer, config, forward_returns=None):
    """
    对单个因子进行静态股票池分析（保持向后兼容）
    
    Args:
        factor: 因子实例
        price_fetcher: 价格数据获取器实例
        runner: Alphalens 运行器实例
        writer: 结果写入器实例
        config: 配置字典
        forward_returns: 预计算的前瞻收益率DataFrame，可选
    """
    factor_name = factor.__class__.__name__
    logger.info(f"Analyzing factor: {factor_name}")
    
    try:
        # 获取因子值
        factor_df = factor.get_factor_values()
        logger.info(f"Factor values shape: {factor_df.shape}")
        
        if factor_df.empty:
            logger.warning(f"Factor {factor_name} has no data, skipping")
            return False
        
        factor_df_unfiltered = factor_df
        
        # 获取因子覆盖的股票代码
        code_list = factor_df.columns.tolist()
        
        # 获取价格数据
        start_date = factor_df.index.min().strftime('%Y-%m-%d')
        end_date = factor_df.index.max().strftime('%Y-%m-%d')

        # 获取价格数据
        pricing = price_fetcher.fetch(
            code_list,
            start_date,
            end_date,
            future_periods=max(runner.periods) + 1,
        )
        logger.info(f"Price data fetched for {len(code_list)} stocks from {start_date} to {end_date}")
        
        # 打印详细信息进行比对
        logger.debug(f"Factor stock codes: {code_list}")
        
        # 检查哪些股票成功获取了价格
        if isinstance(pricing.index, pd.MultiIndex):
            price_stocks = pricing.index.get_level_values(0).unique()
        else:
            price_stocks = pricing.columns
        logger.debug(f"Price data available for: {list(price_stocks)}")
        
        # 找出缺失的股票
        missing_stocks = set(code_list) - set(price_stocks)
        if missing_stocks:
            logger.warning(f"Missing price data for: {missing_stocks}")

        # 执行 Alphalens 分析（包含IC/IR分析）
        alphalens_results = runner.run(factor_df, pricing, forward_returns)
        if forward_returns is not None:
            logger.info(f"Alphalens analysis complete for {factor_name} using pre-calculated forward returns")
        else:
            logger.info(f"Alphalens analysis complete for {factor_name} (includes IC/IR analysis)")
        
        # 组合结果 - Fix the data structure to prevent KeyError 'ic'
        combined_results = {
            'factor_name': factor_name,
            'pool_type': 'static',
            'factor_statistics': {
                'factor_shape': factor_df.shape,
                'start_date': start_date,
                'end_date': end_date,
                'stock_count': len(code_list)
            }
        }
        
        # Flatten alphalens_results to fix KeyError 'ic' - Handle both dict and DataFrame cases
        if alphalens_results is not None:
            if isinstance(alphalens_results, dict):
                # Expected case - alphalens_results is a dictionary
                if alphalens_results:  # Check if dictionary is not empty
                    combined_results.update(alphalens_results)
            elif isinstance(alphalens_results, pd.DataFrame):
                # Handle case where alphalens_results is a DataFrame
                if not alphalens_results.empty:
                    combined_results['alphalens_data'] = alphalens_results
                    logger.warning(f"alphalens_results is DataFrame instead of dict for factor {factor_name}")
            else:
                logger.warning(f"Unexpected alphalens_results type: {type(alphalens_results)} for factor {factor_name}")
        
        _persist_analysis_outputs(
            factor=factor,
            factor_name=factor_name,
            factor_df_unfiltered=factor_df_unfiltered,
            combined_results=combined_results,
            writer=writer,
            monthly_writer=monthly_writer,
        )
        return True
    except Exception as e:
        logger.error(f"Error analyzing factor {factor_name}: {e}", exc_info=True)
        return False

def analyze_factor_dynamic(
    factor_class,
    pool_config,
    config,
    engine,
    writer,
    monthly_writer,
    monthly_stock_pools=None,
):
    """
    使用动态股票池分析单个因子
    
    Args:
        factor_class: 因子类
        pool_config: 股票池配置字典，包含name, description, filters等
        config: 全局配置
        engine: 数据库引擎
        monthly_stock_pools: 预计算的动态股票池（可选）
    """
    pool_name = pool_config.get('name', 'dynamic_pool')
    pool_description = pool_config.get('description', 'Dynamic stock pool')
    
    # 从配置中提取股票池设置
    filters = []
    if 'filters' in pool_config:
        for filter_config in pool_config['filters']:
            # Check if filter_config is already a StockFilter object
            if isinstance(filter_config, StockFilter):
                filters.append(filter_config)
            else:
                # If it's still a dictionary, convert it to StockFilter
                filters.append(StockFilter(
                    field=filter_config['field'],
                    operator=filter_config['operator'],
                    value=filter_config['value'],
                    description=filter_config.get('description', '')
                ))
    
    max_stocks = pool_config.get('max_stocks', None)
    sort_by = pool_config.get('sort_by', None)
    ascending = pool_config.get('ascending', False)
    
    factor_name = f"{factor_class.__name__}_{pool_name}"
    performance = {'factor_name': factor_name}
    analysis_started = time.perf_counter()
    
    logger.info(f"开始分析因子: {factor_name}")
    
    # ==========================================================================
    # PIT 动态复权配置
    # 因子计算时使用动态复权，确保跨日价格比较的正确性
    # IC/IR 分析时使用静态复权数据（PriceFetcher 默认行为）
    # ==========================================================================
    pit_config = load_pit_config(config)
    pit_enabled = pit_config.get('enabled', True)  # 默认启用动态复权
    pit_factor_column = pit_config.get('factor_column', 'factor')
    
    try:
        # 创建因子实例（使用动态股票池）
        factor = factor_class(
            db_connection=engine,
            factor_name=factor_name,
            start_date=_factor_data_start_date(factor_class, config),
            end_date=config['end_date'],
            use_dynamic_pool=True,
            stock_filters=filters,
            max_stocks_per_month=max_stocks,
            sort_by=sort_by,
            ascending=ascending,
            cache_dir=config.get('cache_dir'),
            rebalance_period_months=pool_config.get('rebalance_period_months', 1),
            rebalance_schedule=pool_config.get('rebalance_schedule', 'quarterly'),
            monthly_stock_pools=monthly_stock_pools,  # 传递预计算的股票池
            fundamental_table=config.get('fundamental_table', 'factor_data'),
            price_table=config.get('price_table', 'price_data'),
            # DuckDB 高性能后端配置
            db_backend=config.get('db_backend', 'auto'),
            duckdb_path=config.get('duckdb_path'),
            parquet_dir=config.get('parquet_dir'),
            data_config_path=config.get('data_config_path'),
            source_policy=config.get('source_policy'),
            use_duckdb_cache=config.get('use_duckdb_cache', True),
            duckdb_cache_size_mb=config.get('duckdb_cache_size_mb', 2000),
            # PIT 动态复权配置 - 因子计算时启用
            pit_enabled=pit_enabled,
            pit_factor_column=pit_factor_column,
        )
        
        # 🚀 优化：只获取一次数据，避免重复查询 23M+ 行
        # 先获取未过滤的因子值（包含所有股票），然后在内存中过滤
        try:
            factor_started = time.perf_counter()
            chunk_size = int(config.get('five_minute_stock_chunk_size', 0) or 0)
            factor_df_unfiltered = _load_factor_value_cache(factor_class, config)
            cache_hit = factor_df_unfiltered is not None
            if not cache_hit:
                factor_df_unfiltered = factor.get_factor_values_unfiltered_chunked(chunk_size)
            analysis_start = pd.Timestamp(config['start_date'])
            factor_df_unfiltered = factor_df_unfiltered.loc[
                pd.to_datetime(factor_df_unfiltered.index) >= analysis_start
            ]
            performance['factor_seconds'] = time.perf_counter() - factor_started
            performance.update(getattr(factor, 'last_performance_metrics', {}))
            performance['factor_value_cache_hit'] = cache_hit
            logger.info(f"因子 {factor_name} 值形状 (未过滤): {factor_df_unfiltered.shape}")
            
            # 在内存中应用动态股票池过滤 - 避免第二次数据库查询
            if hasattr(factor, 'monthly_stock_pools') and factor.monthly_stock_pools:
                # 日期级别动态股票池过滤（复用共享工具函数）
                from utils.dynamic_stock_pool import apply_date_level_pool_filter_to_wide_df
                factor_df = apply_date_level_pool_filter_to_wide_df(
                    factor_df_unfiltered, 
                    factor.monthly_stock_pools
                )
                logger.info(f"因子 {factor_name} 值形状 (已过滤): {factor_df.shape} (日期级别动态股票池过滤)")
            else:
                # 没有动态股票池，直接使用未过滤的值
                factor_df = factor_df_unfiltered.copy()
                logger.info(f"因子 {factor_name} 值形状 (已过滤): {factor_df.shape}")

        except Exception as e:
            if config.get('materialize_only'):
                raise  # Never publish a silently pool-masked fallback panel.
            logger.warning(f"获取未过滤因子值失败，回退到传统方式: {e}")
            performance['factor_value_cache_hit'] = False
            factor_df = factor.get_factor_values()
            logger.info(f"因子 {factor_name} 值形状 (已过滤): {factor_df.shape}")
            factor_df_unfiltered = factor_df

        materialize_range = None
        if config.get('materialize_only'):
            # Store the computed pool-union panel, not the analysis membership
            # mask. A T-day entrant needs its already-known T-1 value. Training
            # and evaluation apply their own dated pool filters downstream.
            factor_df = factor_df_unfiltered
            start_date = pd.Timestamp(config['start_date']).normalize()
            end_date = pd.Timestamp(config['end_date']).normalize()
            factor_dates = pd.to_datetime(factor_df.index).normalize()
            factor_df = factor_df.loc[
                (factor_dates >= start_date) & (factor_dates <= end_date)
            ]
            materialize_range = (
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
            )
        
        # 增强的数据验证
        if factor_df.empty:
            logger.warning(f"因子 {factor_name} 没有数据，跳过分析")
            return False
        
        # 检查因子数据质量
        non_null_count = factor_df.count().sum()
        total_count = factor_df.size
        data_coverage = non_null_count / total_count if total_count > 0 else 0
        
        logger.info(f"因子 {factor_name} 数据覆盖率: {data_coverage:.2%} ({non_null_count}/{total_count})")
        
        # 如果数据覆盖率太低，跳过分析
        if data_coverage < 0.01:  # 少于1%的数据覆盖率
            logger.warning(f"因子 {factor_name} 数据覆盖率过低 ({data_coverage:.2%})，跳过分析")
            return False
        
        # 检查股票数量
        valid_stocks = factor_df.columns[factor_df.count() > 0].tolist()
        if len(valid_stocks) < 5:  # 至少需要5只股票
            logger.warning(f"因子 {factor_name} 有效股票数量过少 ({len(valid_stocks)})，跳过分析")
            return False

        if materialize_range is not None:
            return _persist_materialized_factor_values(
                factor=factor,
                factor_name=factor_name,
                factor_df=factor_df,
                writer=writer,
                start_date=materialize_range[0],
                end_date=materialize_range[1],
            )
        
        # 获取价格数据 (支持 DuckDB 高性能后端)
        # 注意：PIT 动态复权已在 FactorBase 中处理（因子计算时）
        # PriceFetcher 获取静态前复权价格用于 IC/IR 分析（不需要动态复权）
        price_fetcher = PriceFetcher(
            duckdb_path=config['duckdb_path'],
            price_table=config['price_table'],
            parquet_dir=config.get('parquet_dir'),
            data_config_path=config.get('data_config_path'),
        )
        
        # 获取所有涉及的股票代码
        all_stocks = list(set(factor_df.columns))
        start_date = factor_df.index.min().strftime('%Y-%m-%d')
        end_date = factor_df.index.max().strftime('%Y-%m-%d')
        
        # 获取价格数据
        try:
            price_started = time.perf_counter()
            periods = tuple(config.get('ic_periods') or (1, 5, 10))
            pricing = price_fetcher.fetch(
                all_stocks,
                start_date,
                end_date,
                future_periods=0 if config.get('mining_only') else max(periods) + 1,
            )
            logger.info(f"获取了 {len(all_stocks)} 只股票从 {start_date} 到 {end_date} 的价格数据")
            performance['price_fetch_seconds'] = time.perf_counter() - price_started
        except Exception as e:
            logger.error(f"获取价格数据失败: {e}")
            return False
        
        # 验证价格数据
        if isinstance(pricing.index, pd.MultiIndex):
            price_stocks = pricing.index.get_level_values(0).unique()
        else:
            price_stocks = pricing.columns if hasattr(pricing, 'columns') else []
        
        common_stocks = set(all_stocks) & set(price_stocks)
        if len(common_stocks) < 3:  # 至少需要3只股票有价格数据
            logger.warning(f"因子 {factor_name} 与价格数据的股票交集过少 ({len(common_stocks)})，跳过分析")
            return False

        if config.get('mining_only'):
            ic = _compute_mining_only_ic(factor_df, pricing, periods)
            combined_results = {
                'factor_name': factor_name,
                'pool_type': pool_name,
                'factor_statistics': {
                    'factor_shape': factor_df.shape,
                    'start_date': start_date,
                    'end_date': end_date,
                    'stock_count': len(all_stocks),
                    'data_coverage': data_coverage,
                },
                'ic': ic,
                'quantile_returns': pd.DataFrame(),
                'performance_metrics': {**performance, 'mining_only': True},
            }
            persisted = _persist_analysis_outputs(
                factor=factor,
                factor_name=factor_name,
                factor_df_unfiltered=factor_df_unfiltered,
                combined_results=combined_results,
                writer=writer,
                monthly_writer=None,
            )
            if persisted:
                logger.info("PERF_METRICS %s", json.dumps(combined_results['performance_metrics'], ensure_ascii=False, sort_keys=True))
            return persisted
        
        # 执行 Alphalens 分析（包含IC/IR分析）- 使用过滤后的因子值
        runner = AlphalensRunner(
            n_quantiles=config['n_quantiles'],
            periods=tuple(config.get('ic_periods') or (1, 5, 10)),
        )
        
        try:
            alphalens_started = time.perf_counter()
            forward_returns = None
            if config.get('use_precomputed_returns', True):
                returns_calculator = ForwardReturnsCalculator(
                    cache_dir=config.get('forward_returns_cache_dir') or config.get('cache_dir')
                )
                forward_returns = returns_calculator.calculate_or_load_forward_returns(
                    pricing,
                    all_stocks,
                    start_date,
                    end_date,
                    periods,
                    execution_lag=runner.execution_lag,
                    source_signature=_pool_snapshot_fingerprint(config, [pool_config]),
                )
            alphalens_results = runner.run(factor_df, pricing, forward_returns)
            performance['alphalens_seconds'] = time.perf_counter() - alphalens_started
            logger.info(f"完成 {factor_name} 的 Alphalens 分析（包含IC/IR分析）")
        except Exception as e:
            logger.error(f"Alphalens 分析失败: {e}", exc_info=True)
            return False
        
        # 组合结果 - Fix the data structure to prevent KeyError 'ic'
        combined_results = {
            'factor_name': factor_name,
            'pool_type': pool_name,
            'pool_description': pool_description,
            'factor_statistics': {
                'factor_shape': factor_df.shape,
                'start_date': start_date,
                'end_date': end_date,
                'stock_count': len(all_stocks),
                'data_coverage': data_coverage,
                'valid_stocks': len(valid_stocks),
                'common_stocks': len(common_stocks)
            }
        }
        
        # Flatten alphalens_results to fix KeyError 'ic' - Handle both dict and DataFrame cases
        if alphalens_results is not None:
            if isinstance(alphalens_results, dict):
                # Expected case - alphalens_results is a dictionary
                if alphalens_results:  # Check if dictionary is not empty
                    combined_results.update(alphalens_results)
            elif isinstance(alphalens_results, pd.DataFrame):
                # Handle case where alphalens_results is a DataFrame
                if not alphalens_results.empty:
                    combined_results['alphalens_data'] = alphalens_results
                    logger.warning(f"alphalens_results is DataFrame instead of dict for factor {factor_name}")
            else:
                logger.warning(f"Unexpected alphalens_results type: {type(alphalens_results)} for factor {factor_name}")
        
        performance['analysis_before_persist_seconds'] = time.perf_counter() - analysis_started
        combined_results['performance_metrics'] = performance.copy()
        persist_started = time.perf_counter()
        _persist_analysis_outputs(
            factor=factor,
            factor_name=factor_name,
            factor_df_unfiltered=factor_df,
            combined_results=combined_results,
            writer=writer,
            monthly_writer=monthly_writer,
        )
        performance['persist_seconds'] = time.perf_counter() - persist_started
        performance['total_seconds'] = time.perf_counter() - analysis_started
        logger.info("PERF_METRICS %s", json.dumps(performance, ensure_ascii=False, sort_keys=True))
        
        # 记录股票池统计信息
        log_pool_statistics(factor, pool_name, pool_description)
        
        return True
        
    except Exception as e:
        logger.error(f"分析因子 {factor_name} 时出错: {e}", exc_info=True)
        return False

def parse_dynamic_pool_config(config: Dict) -> List[Dict]:
    """解析动态股票池配置"""
    pool_configs = []
    
    dynamic_pools = config.get('dynamic_pools', {})
    for pool_name, pool_settings in dynamic_pools.items():
        if pool_settings.get('enabled', True):
            pool_config = {
                'name': pool_name,
                'description': pool_settings.get('description', pool_name),
                'filters': pool_settings.get('filters', []),
                'max_stocks': pool_settings.get('max_stocks'),
                'sort_by': pool_settings.get('sort_by'),
                'ascending': pool_settings.get('ascending', False),
                'rebalance_period_months': pool_settings.get('rebalance_period_months', 1),
                'rebalance_schedule': pool_settings.get('rebalance_schedule', 'quarterly')
            }
            pool_configs.append(pool_config)
    
    return pool_configs


def _validate_materialize_pool_contract(
    raw_config: Dict[str, Any],
    parsed_pools: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    dynamic_pools = raw_config.get('dynamic_pools')
    if not isinstance(dynamic_pools, dict) or any(
        not isinstance(settings, dict) for settings in dynamic_pools.values()
    ):
        raise RuntimeError("materialize-only requires exactly one enabled dynamic pool")

    enabled = [
        (name, settings)
        for name, settings in dynamic_pools.items()
        if isinstance(settings, dict) and settings.get('enabled', True)
    ]
    if len(enabled) != 1:
        raise RuntimeError("materialize-only requires exactly one enabled dynamic pool")

    pool_name, settings = enabled[0]
    filters = settings.get('filters', [])
    if not isinstance(filters, list) or any(
        not isinstance(filter_config, dict)
        or not {'field', 'operator', 'value'}.issubset(filter_config)
        for filter_config in filters
    ):
        raise RuntimeError(
            "materialize-only pool filters must each define field/operator/value"
        )

    if parsed_pools is not None:
        parsed_filters = parsed_pools.get(pool_name, {}).get('filters', [])
        if set(parsed_pools) != {pool_name} or len(parsed_filters) != len(filters):
            raise RuntimeError("materialize-only parsed filter count mismatch")


def main(config: Dict, *, execution=None):
    """
    主程序入口，支持静态和动态股票池分析
    
    Args:
        config: 配置字典
    """
    logger.info("Starting factor analysis program...")
    if execution is None:
        execution = {}
    execution.update(expected=0, completed=0, failed=0, errors=[])
    if config.get('factor_class') is not None and not config.get('factor_file'):
        raise ValueError("factor_class requires an explicit factor_file")
    materialize_only = bool(config.get('materialize_only'))
    mining_only = bool(config.get('mining_only'))
    if mining_only:
        if str(pd.Timestamp(config['end_date']).date()) > '2023-12-31':
            raise RuntimeError("mining-only cannot access post-2023 data")
        config['results_export_parquet'] = False
        if not config.get('factor_file'):
            raise RuntimeError("mining-only requires an existing factor_file")
        enabled_pools = [
            settings
            for settings in (config.get('dynamic_pools') or {}).values()
            if isinstance(settings, dict) and settings.get('enabled', True)
        ]
        if len(enabled_pools) != 1:
            raise RuntimeError("mining-only requires exactly one enabled dynamic pool")
    if materialize_only:
        factor_file = config.get('factor_file')
        if not factor_file or not os.path.isfile(factor_file):
            raise RuntimeError("materialize-only requires an existing factor_file")
        dynamic_pools = config.get('dynamic_pools')
        if not isinstance(dynamic_pools, dict) or any(
            not isinstance(settings, dict) for settings in dynamic_pools.values()
        ):
            raise RuntimeError("materialize-only requires exactly one enabled dynamic pool")
        enabled_pools = [
            settings
            for settings in dynamic_pools.values()
            if settings.get('enabled', True)
        ]
        if len(enabled_pools) != 1:
            raise RuntimeError("materialize-only requires exactly one enabled dynamic pool")
    
    backend = str(config.get('db_backend', 'auto') or 'auto').lower()
    if backend == 'sqlite':
        raise ValueError("SQLite backend is no longer supported; configure duckdb_path")
    if backend not in {'auto', 'duckdb'}:
        raise ValueError("db_backend must be 'duckdb' or 'auto'")
    if not config.get('duckdb_path'):
        raise ValueError("duckdb_path is required for factor analysis")
    config['db_backend'] = 'duckdb'
    results_backend = str(config.get('results_db_backend', 'auto') or 'auto').lower()
    if results_backend == 'sqlite':
        raise ValueError("SQLite results backend is no longer supported; configure results_duckdb_path")
    if results_backend not in {'auto', 'duckdb'}:
        raise ValueError("results_db_backend must be 'duckdb' or 'auto'")
    if not config.get('results_duckdb_path'):
        raise ValueError("results_duckdb_path is required for analysis results")
    config['results_db_backend'] = 'duckdb'

    # 等待数据库可用（其他进程占用时自动重试）
    wait_for_db_connection(
        duckdb_path=config.get('duckdb_path'),
    )
    if config.get('results_duckdb_path'):
        wait_for_result_db_write_slot(config.get('results_duckdb_path'))
    
    # 创建数据库连接
    engine = connect_read_only(str(config['duckdb_path']))
    

    # ==========================================================================
    # 创建各模块实例 (支持 DuckDB 高性能后端)
    # ==========================================================================
    # 注意：PIT 动态复权已移至 FactorBase 中统一处理，PriceFetcher 仅获取静态前复权价格
    price_fetcher = None
    runner = None
    if not materialize_only:
        price_fetcher = PriceFetcher(
            duckdb_path=config['duckdb_path'],
            price_table=config['price_table'],
            parquet_dir=config.get('parquet_dir'),
            data_config_path=config.get('data_config_path'),
        )
        runner = AlphalensRunner(
            n_quantiles=config['n_quantiles'],
            periods=tuple(config.get('ic_periods') or (1, 5, 10)),
        )
    writer = ResultWriter(
        str(config['results_duckdb_path']),
        config.get('results_output_table', config.get('output_table', 'factor_analysis_results')),
        db_backend='duckdb',
        duckdb_path=config.get('results_duckdb_path'),
        parquet_dir=config.get('results_parquet_dir'),
        export_parquet=config.get('results_export_parquet', True)
    )
    monthly_writer = None
    if materialize_only and not writer.using_duckdb:
        raise RuntimeError("materialize-only requires an active DuckDB results backend")
    if MONTHLY_WRITER_AVAILABLE and not materialize_only:
        monthly_writer = MonthlyPoolResultWriter(
            str(config['results_duckdb_path']),
            config.get('results_monthly_output_table', config.get('monthly_output_table', 'monthly_pool_results')),
            db_backend='duckdb',
            duckdb_path=config.get('results_duckdb_path'),
            parquet_dir=config.get('results_parquet_dir'),
        )
        if writer.using_duckdb != monthly_writer.using_duckdb:
            raise RuntimeError(
                "Result writer and monthly writer initialized with different backends; "
                "shared result persistence requires a single backend."
            )
    
    # 检查是否使用动态股票池
    use_dynamic_pools = config.get('use_dynamic_pools', False) and DYNAMIC_POOL_AVAILABLE
    if config.get('use_dynamic_pools') and not DYNAMIC_POOL_AVAILABLE:
        raise RuntimeError("dynamic stock pool support is unavailable")
    if materialize_only and not use_dynamic_pools:
        raise RuntimeError("materialize-only requires dynamic stock pools")
    
    if use_dynamic_pools:
        logger.info("使用动态股票池模式")
        
        # 解析动态股票池配置
        pool_configs = parse_dynamic_pool_config(config)
        if not pool_configs:
            if materialize_only:
                raise RuntimeError("materialize-only requires at least one enabled dynamic pool")
            raise RuntimeError("no enabled dynamic pools")
        
        logger.info(f"找到 {len(pool_configs)} 个动态股票池配置")
        
        # 加载因子类（不创建实例）
        if config.get('factor_file'):
            factor_classes = FactorLoader.load_factor_class_from_file(config['factor_file'], config.get('factor_class'))
        else:
            factor_classes = FactorLoader.load_factor_classes(config['factor_dir'])
        if materialize_only and len(factor_classes) != 1:
            raise RuntimeError("materialize-only requires exactly one factor class")
        logger.info(f"加载了 {len(factor_classes)} 个因子类")
        if not factor_classes:
            raise RuntimeError("no factor classes loaded")
        
        # 对每个因子类和每个股票池组合进行分析
        success_count = 0
        total_combinations = len(factor_classes) * len(pool_configs)
        execution['expected'] = total_combinations
        
        # 预计算股票池
        logger.info("开始预计算动态股票池...")
        precalculated_pools = _load_pool_snapshot(config, pool_configs) or {}
        
        # 初始化 DynamicStockPoolManager (支持 DuckDB 高性能后端)
        dsp_manager = None
        if any(pool['name'] not in precalculated_pools for pool in pool_configs):
            dsp_manager = DynamicStockPoolManager(
                None,
                fundamental_table=config.get('fundamental_table', 'factor_data'),
                price_table=config.get('price_table', 'price_data'),
                db_backend=config.get('db_backend', 'auto'),
                duckdb_path=config.get('duckdb_path'),
                parquet_dir=config.get('parquet_dir'),
                data_config_path=config.get('data_config_path'),
                source_policy=config.get('source_policy'),
                use_duckdb_cache=config.get('use_duckdb_cache', True),
                duckdb_cache_size_mb=config.get('duckdb_cache_size_mb', 2000)
            )
        
        for pool_config in pool_configs:
            pool_name = pool_config['name']
            logger.info(f"正在计算股票池: {pool_name}")
            if pool_name in precalculated_pools:
                logger.info("股票池 %s 已从快照加载", pool_name)
                continue
            
            try:
                # 提取配置
                filters = []
                if 'filters' in pool_config:
                    for filter_config in pool_config['filters']:
                        if isinstance(filter_config, StockFilter):
                            filters.append(filter_config)
                        else:
                            filters.append(StockFilter(
                                field=filter_config['field'],
                                operator=filter_config['operator'],
                                value=filter_config['value'],
                                description=filter_config.get('description', '')
                            ))
                
                max_stocks = pool_config.get('max_stocks')
                sort_by = pool_config.get('sort_by')
                ascending = pool_config.get('ascending', False)
                rebalance_period_months = pool_config.get('rebalance_period_months', 1)
                rebalance_schedule = pool_config.get('rebalance_schedule', 'quarterly')
                
                # 计算股票池
                monthly_stock_pools = dsp_manager.get_stock_pools_for_period(
                    config['start_date'], 
                    config['end_date'], 
                    filters, 
                    max_stocks, 
                    sort_by, 
                    ascending,
                    rebalance_period_months=rebalance_period_months,
                    rebalance_schedule=rebalance_schedule
                )
                
                precalculated_pools[pool_name] = monthly_stock_pools
                
                # 记录统计信息
                total_periods = len(monthly_stock_pools)
                stock_counts = [len(stocks) for stocks in monthly_stock_pools.values()]
                avg_stocks = sum(stock_counts) / len(stock_counts) if stock_counts else 0
                logger.info(f"股票池 {pool_name} 计算完成: {total_periods} 个期间, 平均 {avg_stocks:.1f} 只股票")
                
                # 记录详细选中日期及股票数量
                sorted_dates = sorted(monthly_stock_pools.keys())
                logger.info(f"详细选中日期及股票数量:")
                for date_str in sorted_dates:
                    count = len(monthly_stock_pools[date_str])
                    logger.info(f"  日期: {date_str}, 股票数: {count}")
                
            except Exception as e:
                logger.error(f"计算股票池 {pool_name} 失败: {e}", exc_info=True)
                precalculated_pools[pool_name] = None

        _close_factor_duckdb_handles(dsp_manager, "precomputed pools")
        _save_pool_snapshot(config, pool_configs, precalculated_pools)

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.get('max_workers', 1)) as executor:
            futures = {}
            
            for factor_class in factor_classes:
                for pool_config in pool_configs:
                    pool_name = pool_config['name']
                    monthly_stock_pools = precalculated_pools.get(pool_name)
                    
                    if monthly_stock_pools is None:
                        logger.warning(f"跳过组合 {factor_class.__name__}_{pool_name} 因为股票池计算失败")
                        execution['errors'].append({'item': f'{factor_class.__name__}_{pool_name}', 'error': 'stock pool calculation failed'})
                        continue
                        
                    future = executor.submit(
                        analyze_factor_dynamic, 
                        factor_class, 
                        pool_config, 
                        config, 
                        engine,
                        writer,
                        monthly_writer,
                        monthly_stock_pools  # 传递预计算的股票池
                    )
                    futures[future] = f"{factor_class.__name__}_{pool_config['name']}"
            
            for future in concurrent.futures.as_completed(futures):
                combination_name = futures[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        execution['errors'].append({'item': combination_name, 'error': 'analysis returned False; see worker log'})
                except Exception as e:
                    logger.error(f"组合 {combination_name} 分析失败: {e}", exc_info=True)
                    execution['errors'].append({'item': combination_name, 'error': str(e)})
        
        logger.info(f"动态股票池分析完成。{success_count} / {total_combinations} 个组合成功分析")
        execution.update(completed=success_count, failed=total_combinations - success_count)
        if success_count != total_combinations:
            raise RuntimeError(
                f"analysis failed for {total_combinations - success_count} combination(s)"
            )
        
    else:
        logger.info("使用静态股票池模式")
        
        # 加载因子实例
        data_options = {key: config[key] for key in (
            'duckdb_path', 'parquet_dir', 'data_config_path', 'source_policy',
            'use_duckdb_cache', 'duckdb_cache_size_mb',
        ) if config.get(key) is not None}
        if config.get('factor_file'):
            factors = FactorLoader.load_factor_from_file(
                config['factor_file'],
                engine,
                config['start_date'],
                config['end_date'],
                config['code_list'],
                config.get('cache_dir'),
                config.get('factor_class'),
                **data_options,
            )
        else:
            factors = FactorLoader.load_all_factors(
                config['factor_dir'],
                engine,
                config['start_date'],
                config['end_date'],
                config['code_list'],
                config.get('cache_dir'),
                **data_options,
            )
        logger.info(f"Loaded {len(factors)} factors")
        if not factors:
            raise RuntimeError("no factors loaded")
        execution['expected'] = len(factors)
        
        # 性能优化：预计算前瞻收益率
        forward_returns = None
        if config.get('use_precomputed_returns', True):  # 默认启用优化
            logger.info("🚀 性能优化：开始预计算前瞻收益率...")
            try:
                # 创建前瞻收益率计算器
                returns_calculator = ForwardReturnsCalculator(
                    cache_dir=config.get('cache_dir', './cache')
                )
                
                # 获取价格数据（扩展时间范围以计算前瞻收益率）
                start_date = config['start_date']
                end_date = config['end_date']
                code_list = config['code_list']
                periods = tuple(runner.periods)
                
                # 获取价格数据
                pricing = price_fetcher.fetch(
                    code_list,
                    start_date,
                    end_date,
                    future_periods=max(periods) + 1,
                )
                logger.info(f"Price data fetched for forward returns calculation")
                
                # 计算或加载前瞻收益率
                forward_returns = returns_calculator.calculate_or_load_forward_returns(
                    pricing, code_list, start_date, end_date, periods
                )
                
                logger.info(f"✅ 前瞻收益率预计算完成！形状: {forward_returns.shape}")
                logger.info(f"包含期间: {list(forward_returns.columns)}")
                
            except Exception as e:
                logger.warning(f"前瞻收益率预计算失败，将使用传统方式: {e}")
                forward_returns = None
        
        # 并行处理所有因子
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.get('max_workers', 4)) as executor:
            futures = {
                executor.submit(analyze_factor_static, factor, price_fetcher, runner, writer, monthly_writer, config, forward_returns): factor.__class__.__name__
                for factor in factors
            }
            
            for future in concurrent.futures.as_completed(futures):
                factor_name = futures[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        execution['errors'].append({'item': factor_name, 'error': 'analysis returned False; see worker log'})
                except Exception as e:
                    logger.error(f"Factor {factor_name} analysis failed: {e}", exc_info=True)
                    execution['errors'].append({'item': factor_name, 'error': str(e)})
        
        logger.info(f"静态股票池分析完成。{success_count} / {len(factors)} 个因子成功分析")
        execution.update(completed=success_count, failed=len(factors) - success_count)
        if success_count != len(factors):
            raise RuntimeError(f"analysis failed for {len(factors) - success_count} factor(s)")
        if forward_returns is not None:
            logger.info("🎯 性能优化生效：使用了预计算的前瞻收益率")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Factor Analysis Program with Dynamic Stock Pool Support')
    parser.add_argument('--config', type=str, help='Path to config file (JSON format for dynamic pools)')
    parser.add_argument('--factor', type=str, default=None,
                        help='Path to a specific factor .py file to analyze (skips full directory scan)')
    parser.add_argument('--use-dynamic', action='store_true', help='Force use dynamic stock pools')
    parser.add_argument(
        '--materialize-only',
        action='store_true',
        help='Persist dynamic-pool factor values for the config date range only',
    )
    parser.add_argument(
        '--mining-only',
        action='store_true',
        help='Persist factor values and daily IC required by the IS mining gate only',
    )
    parser.add_argument('--pools', type=str, nargs='+', help='Specific pool names to analyze (for dynamic mode)')
    
    parser.add_argument("--result-out", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--execution-id", default="")
    args = parser.parse_args()
    from utils.output_paths import execution_directory, process_output_environment
    directory = Path(args.result_out).resolve().parent if args.result_out else execution_directory()
    if not args.result_out:
        args.result_out = str(directory / "result.json")
    if not args.execution_id:
        args.execution_id = directory.name
    if not args.run_id:
        args.run_id = args.execution_id
    with process_output_environment(directory):
        if args.materialize_only and not args.factor:
            parser.error('--materialize-only requires --factor')
        if args.mining_only and not args.factor:
            parser.error('--mining-only requires --factor')
        config = CONFIG.copy()
        from utils.output_paths import execution_directory
        attempt = execution_directory()
        cache = Path(os.environ.get("QUANT_CACHE_DIR", Path(__file__).resolve().parent / "cache"))
        config.update(cache_dir=str(cache / "factor_values"), forward_returns_cache_dir=str(cache / "forward_returns"),
                      results_parquet_dir=str(attempt / "analysis_results_parquet"))  # 复制默认配置
    
        # 如果提供了配置文件，从文件读取配置
        if args.config:
            try:
                import json
                with open(args.config, 'r') as f:
                    raw_file_config = json.load(f)

                if args.materialize_only:
                    _validate_materialize_pool_contract(raw_file_config)

                has_dynamic_pool_config = (
                    DYNAMIC_POOL_AVAILABLE
                    and isinstance(raw_file_config.get('dynamic_pools'), dict)
                    and bool(raw_file_config.get('dynamic_pools'))
                )

                if has_dynamic_pool_config or 'database' in raw_file_config:
                    # Nested database configs also support static stock pools.
                    db_config, analysis_config, pool_configs = load_and_parse_config(args.config)
                
                    # 更新配置 - 正确映射数据库配置
                    # 主数据库配置
                    config['factor_table'] = db_config.get('factor_table', config.get('factor_table'))
                    config['price_table'] = db_config.get('price_table', config.get('price_table'))
                    config['fundamental_table'] = db_config.get('fundamental_table', config.get('fundamental_table'))
                
                    # DuckDB 高性能后端配置
                    config['db_backend'] = db_config.get('db_backend', 'auto')
                    config['duckdb_path'] = db_config.get('duckdb_path')
                    config['parquet_dir'] = db_config.get('parquet_dir')
                    config['use_duckdb_cache'] = db_config.get('use_duckdb_cache', True)
                    config['duckdb_cache_size_mb'] = db_config.get('duckdb_cache_size_mb', 2000)
                
                    # 分析配置
                    config.update(analysis_config)
                    config['dynamic_pools'] = pool_configs
                    config['use_dynamic_pools'] = has_dynamic_pool_config
                    if args.materialize_only:
                        _validate_materialize_pool_contract(raw_file_config, pool_configs)
                
                    logger.info("从JSON配置文件加载数据库与股票池配置")
                else:
                    # 传统配置文件
                    config.update(raw_file_config)
                    if config.get('data_config_path'):
                        config['data_config_path'] = str((Path(args.config).resolve().parent /
                            Path(config['data_config_path']).expanduser()).resolve())
                    
                    logger.info("从配置文件加载传统配置")
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
                sys.exit(1)
    
        # 命令行参数覆盖
        if args.factor:
            config['factor_file'] = args.factor
            logger.info(f"指定分析单个因子文件: {args.factor}")
        if args.use_dynamic:
            config['use_dynamic_pools'] = True
        if args.materialize_only:
            config['materialize_only'] = True
        if args.mining_only:
            config['mining_only'] = True
    
        # 如果指定了特定的股票池
        if args.pools and config.get('dynamic_pools'):
            # 只保留指定的股票池
            filtered_pools = {
                name: settings for name, settings in config['dynamic_pools'].items()
                if name in args.pools
            }
            config['dynamic_pools'] = filtered_pools
            logger.info(f"仅分析指定的股票池: {args.pools}")
    
        # 确保必要的目录存在
        if config.get('cache_dir'):
            os.makedirs(config['cache_dir'], exist_ok=True)
    
        from tools.agent_harness.run_result import execution_result

        with execution_result(args.result_out, command="single-factor-analysis",
                              run_id=args.run_id, execution_id=args.execution_id,
                              config_path=args.config or os.path.join(
                                  os.path.dirname(__file__), "backtest_and_analysis", "config.py"
                              )) as result:
            main(config, execution=result["execution"])
