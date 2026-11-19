import fcntl
import hashlib
import json
import logging
import os
import pickle
import re
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from backtest_and_analysis.evaluation.labels import (
    ForwardReturnsValidationError as ForwardReturnsValidationError,
    label_daily_returns,
    resolve_daily_calendar,
    validate_daily_horizons,
)


def newey_west_mean_test(values, max_lag: int) -> tuple[float | None, float | None]:
    """Bartlett HAC mean test, with asymptotic normal p and no correction.

    The caller must supply ordered, consecutive, equally spaced observations.
    Missing values must not be dropped here, since that would compress time.
    """
    if isinstance(max_lag, (bool, np.bool_)) or not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
        raise ValueError("max_lag must be a non-negative integer")
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("HAC observations must be a finite one-dimensional series")
    if data.size < 2 or np.all(data == data[0]):
        return None, None

    centered = data - data.mean()
    lag_count = min(max_lag, data.size - 1)
    long_run_variance = np.dot(centered, centered) / data.size
    for lag in range(1, lag_count + 1):
        weight = 1 - lag / (lag_count + 1)
        covariance = np.dot(centered[lag:], centered[:-lag]) / data.size
        long_run_variance += 2 * weight * covariance

    if long_run_variance <= 0:
        return None, None
    standard_error = np.sqrt(long_run_variance / data.size)
    t_stat = float(data.mean() / standard_error)
    p_value = float(2 * scipy_stats.norm.sf(abs(t_stat)))
    return t_stat, p_value

class ForwardReturnsCalculator:
    """
    前瞻收益率预计算器 - 优化因子分析性能
    
    预先计算所有股票在所有时间周期的前瞻收益率，避免每个因子分析时重复计算
    """
    
    def __init__(self, cache_dir: Optional[str] = None, *, trading_calendar=None):
        """
        初始化前瞻收益率计算器
        
        Args:
            cache_dir: 缓存目录，如果为None则不使用缓存
        """
        self.cache_dir = cache_dir
        self.trading_calendar = trading_calendar
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _calendar(self, pricing):
        return resolve_daily_calendar(pricing, self.trading_calendar)

    _validate_horizons = staticmethod(validate_daily_horizons)

    def calculate_forward_returns(
        self, 
        pricing: pd.Series, 
        periods: Tuple[int, ...] = (1, 5, 10, 20),
        execution_lag: int = 1
    ) -> pd.DataFrame:
        """
        计算前瞻收益率
        
        Args:
            pricing: 价格数据，MultiIndex (asset, date) 格式；非空值须为有限正实数
            periods: 前瞻期间元组
            execution_lag: 执行滞后期 (默认1天，模拟T+1执行)
            
        Returns:
            前瞻收益率DataFrame，MultiIndex (date, asset) 格式，列为各个期间

        Raises:
            ForwardReturnsValidationError: 非法价格，或所有请求期间均无有效收益
        """
        return label_daily_returns(pricing, self.trading_calendar, execution_lag, periods).returns

    def get_cache_key(
        self,
        code_list: List[str],
        start_date: str,
        end_date: str,
        periods: Tuple[int, ...],
        source_signature: str = "",
    ) -> str:
        """
        生成缓存键
        
        Args:
            code_list: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            periods: 前瞻期间
            
        Returns:
            缓存键字符串
        """
        # 创建唯一标识符
        key_data = {
            # v3 could cache wrapped unsigned lags under the same key as Python integer lags.
            'schema_version': 'forward_returns.v4',
            'code_list': sorted(code_list),  # 排序确保一致性
            'start_date': start_date,
            'end_date': end_date,
            'periods': list(periods),
            'source_signature': source_signature,
        }
        
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def save_to_cache(self, cache_key: str, forward_returns: pd.DataFrame):
        """
        保存前瞻收益率到缓存
        
        Args:
            cache_key: 缓存键
            forward_returns: 前瞻收益率数据
        """
        from utils.local_outputs import cache_lease
        with cache_lease(self.cache_dir):
            if not self.cache_dir:
                return
        
            cache_file = os.path.join(self.cache_dir, f"forward_returns_{cache_key}.pkl")
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode='wb',
                    dir=self.cache_dir,
                    prefix=f".{os.path.basename(cache_file)}.",
                    delete=False,
                ) as f:
                    temporary_path = f.name
                    pickle.dump(forward_returns, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary_path, cache_file)
                self.logger.info(f"前瞻收益率已保存到缓存: {cache_file}")
            except Exception as e:
                self.logger.warning(f"保存缓存失败: {e}")
                if temporary_path and os.path.exists(temporary_path):
                    os.unlink(temporary_path)
    
    def load_from_cache(self, cache_key: str, *, periods: Optional[Tuple[int, ...]] = None) -> Optional[pd.DataFrame]:
        """
        从缓存加载前瞻收益率
        
        Args:
            cache_key: 缓存键
            periods: 如提供，校验收益列与请求的期间及顺序一致
            
        Returns:
            前瞻收益率数据；不存在、读取失败或内容不合约则返回None
        """
        from utils.local_outputs import cache_lease
        with cache_lease(self.cache_dir):
            if not self.cache_dir:
                return None
        
            cache_file = os.path.join(self.cache_dir, f"forward_returns_{cache_key}.pkl")
            if not os.path.exists(cache_file):
                return None
        
            try:
                with open(cache_file, 'rb') as f:
                    forward_returns = pickle.load(f)
                if (
                    not isinstance(forward_returns, pd.DataFrame)
                    or forward_returns.empty
                    or not isinstance(forward_returns.index, pd.MultiIndex)
                    or forward_returns.index.names != ['date', 'asset']
                    or not forward_returns.index.is_unique
                    or not forward_returns.columns.is_unique
                ):
                    raise ValueError("Invalid cached forward-return table or index")
                dates = forward_returns.index.get_level_values('date')
                if (
                    not isinstance(dates, pd.DatetimeIndex)
                    or dates.hasnans or dates.tz is not None
                    or not dates.equals(dates.normalize())
                    or pd.isna(forward_returns.index.get_level_values('asset')).any()
                ):
                    raise ValueError("Invalid cached date/asset keys")
                columns = forward_returns.columns.tolist()
                if any(not isinstance(c, str) or not re.fullmatch(r'[1-9][0-9]*D', c) for c in columns):
                    raise ValueError("Invalid cached return columns")
                if periods is not None and columns != list(dict.fromkeys(f'{p}D' for p in periods)):
                    raise ValueError("Cached return columns differ from requested periods/order")
                if any(
                    not pd.api.types.is_numeric_dtype(dtype)
                    or pd.api.types.is_bool_dtype(dtype)
                    or pd.api.types.is_complex_dtype(dtype)
                    for dtype in forward_returns.dtypes
                ):
                    raise ValueError("Cached returns must be real numeric values")
                values = forward_returns.to_numpy(dtype=float, na_value=np.nan)
                if np.isinf(values).any() or np.isnan(values).all(axis=1).any():
                    raise ValueError("Cached returns contain infinity or rows without valid labels")
                self.logger.info(f"从缓存加载前瞻收益率: {cache_file}")
                return forward_returns
            except Exception as e:
                self.logger.warning(f"加载缓存失败: {e}")
                return None
    
    def calculate_or_load_forward_returns(
        self,
        pricing: pd.Series,
        code_list: List[str],
        start_date: str,
        end_date: str,
        periods: Tuple[int, ...] = (1, 5, 10, 20),
        execution_lag: int = 1,
        source_signature: str = "",
    ) -> pd.DataFrame:
        """
        计算或从缓存加载前瞻收益率
        
        Args:
            pricing: 价格数据
            code_list: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            periods: 前瞻期间
            execution_lag: 执行滞后期 (默认1天)
            
        Returns:
            前瞻收益率DataFrame
        """
        periods = self._validate_horizons(periods, execution_lag)
        if not self.cache_dir:
            return self.calculate_forward_returns(pricing, periods, execution_lag)
        if not isinstance(pricing, pd.Series) or not isinstance(pricing.index, pd.MultiIndex):
            return self.calculate_forward_returns(pricing, periods, execution_lag)
        calendar = self._calendar(pricing)
        signature = hashlib.sha256(json.dumps({
            'source': source_signature,
            'calendar': calendar.fingerprint,
            'index_names': pricing.index.names,
            'index_dtypes': [str(level.dtype) for level in pricing.index.levels],
            'price_dtype': str(pricing.dtype),
        }).encode())
        signature.update(pd.util.hash_pandas_object(pricing, index=True).values.tobytes())
        if pd.api.types.is_object_dtype(pricing.dtype):
            # pandas may hash object floats and their string representations identically.
            value_types = pricing.map(lambda value: f'{type(value).__module__}.{type(value).__qualname__}')
            signature.update(pd.util.hash_pandas_object(value_types, index=False).values.tobytes())
        source_signature = signature.hexdigest()
        from utils.local_outputs import cache_lease
        with cache_lease(self.cache_dir):
            # 生成缓存键 (包含execution_lag以区分不同设置)
            cache_key = self.get_cache_key(
                code_list,
                start_date,
                end_date,
                periods,
                source_signature=source_signature,
            ) + f"_lag{execution_lag}"

            lock_path = os.path.join(self.cache_dir, f"forward_returns_{cache_key}.lock")
            with open(lock_path, 'a+b') as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    forward_returns = self.load_from_cache(cache_key, periods=periods)
                    if forward_returns is not None:
                        self.logger.info(f"使用缓存的前瞻收益率数据 (Lag={execution_lag})")
                        return forward_returns

                    self.logger.info(f"缓存不存在，开始计算前瞻收益率 (Lag={execution_lag})...")
                    forward_returns = self.calculate_forward_returns(pricing, periods, execution_lag)
                    self.save_to_cache(cache_key, forward_returns)
                    return forward_returns
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
