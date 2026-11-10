import fcntl
import hashlib
import json
import logging
import os
import pickle
import re
import tempfile
from numbers import Real
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class ForwardReturnsValidationError(ValueError):
    """Invalid prices or absent labels must not become successful empty analysis."""


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
        from utils.trading_calendar import load_trading_calendar
        calendar = self.trading_calendar or pricing.attrs.get('trading_calendar')
        return calendar if calendar is not None else load_trading_calendar()

    @staticmethod
    def _validate_horizons(
        periods: Tuple[int, ...], execution_lag: int
    ) -> Tuple[int, ...]:
        """Validate discrete trading-row horizons before applying ``shift``."""
        if isinstance(periods, (str, bytes)):
            raise ValueError("periods must be a non-empty iterable of positive integers")
        try:
            normalized_periods = tuple(periods)
        except TypeError as exc:
            raise ValueError("periods must be a non-empty iterable of positive integers") from exc
        if not normalized_periods or any(
            isinstance(period, bool) or not isinstance(period, (int, np.integer)) or period <= 0
            for period in normalized_periods
        ):
            raise ValueError("periods must be a non-empty iterable of positive integers")
        if (
            isinstance(execution_lag, bool)
            or not isinstance(execution_lag, (int, np.integer))
            or execution_lag < 0
        ):
            raise ValueError("execution_lag must be a non-negative integer")
        return tuple(int(period) for period in normalized_periods)
    
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
        periods = self._validate_horizons(periods, execution_lag)
        self.logger.info(f"开始计算前瞻收益率，期间: {periods}, 执行滞后: {execution_lag}天")
        
        # 转换价格数据格式为DataFrame (date, asset)
        if not isinstance(pricing, pd.Series):
            raise ValueError("价格数据必须是Series，并使用MultiIndex")
        if isinstance(pricing.index, pd.MultiIndex) and pricing.index.nlevels == 2:
            # 检查MultiIndex的结构
            if pricing.index.names == ['asset', 'date'] or pricing.index.names == [None, None]:
                asset_level, date_level = 0, 1
            elif pricing.index.names == ['date', 'asset']:
                asset_level, date_level = 1, 0
            else:
                raise ValueError(f"不支持的MultiIndex格式: {pricing.index.names}")

            dates = pd.to_datetime(pricing.index.get_level_values(date_level), errors="coerce")
            if dates.isna().any():
                raise ValueError("价格数据包含无法解析的日期")
            assets = pricing.index.get_level_values(asset_level)
            if pd.isna(assets).any():
                raise ValueError("价格数据包含空的 asset 键")
            canonical_index = pd.MultiIndex.from_arrays(
                [dates, assets], names=["date", "asset"]
            )
            canonical_prices = pd.Series(pricing.to_numpy(), index=canonical_index, name=pricing.name)
            if canonical_prices.index.has_duplicates:
                raise ValueError("价格数据包含重复的 (date, asset) 键")
            # Validate observed values before arithmetic; missing observations
            # remain missing and must never become zero or bridge a trading day.
            if pd.api.types.is_object_dtype(canonical_prices.dtype):
                real_values = canonical_prices.map(
                    lambda value: isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                )
            else:
                real_values = pd.Series(
                    pd.api.types.is_numeric_dtype(canonical_prices.dtype)
                    and not pd.api.types.is_bool_dtype(canonical_prices.dtype)
                    and not pd.api.types.is_complex_dtype(canonical_prices.dtype),
                    index=canonical_index,
                )
            numeric_prices = pd.to_numeric(canonical_prices.where(real_values), errors="coerce")
            numeric_prices = numeric_prices.where(canonical_prices.notna())
            invalid = canonical_prices.notna() & (
                ~real_values | ~np.isfinite(numeric_prices) | numeric_prices.le(0)
            )
            if invalid.any():
                key = invalid[invalid].index[0]
                raise ForwardReturnsValidationError(
                    f"Invalid price at date={key[0]}, asset={key[1]!r}: "
                    f"{canonical_prices.loc[key]!r}; expected a finite positive real number or missing value"
                )
            canonical_prices = numeric_prices
            # Preserve the existing wide-panel shift semantics: every asset
            # shares the same sorted date rows and missing prices remain
            # missing. Sorting only makes equivalent long inputs deterministic.
            prices_df = canonical_prices.unstack("asset").sort_index()
        else:
            raise ValueError("价格数据必须是MultiIndex格式")
        
        calendar = self._calendar(pricing)
        prices_df = calendar.align(prices_df)
        self.logger.info(f"价格数据形状: {prices_df.shape}")
        
        # 计算各期间的前瞻收益率
        forward_returns_dict = {}
        
        for period in periods:
            self.logger.info(f"计算 {period} 天前瞻收益率 (Lag={execution_lag})...")
            
            # 计算收益率:
            # Lag=0: (price_t+period / price_t) - 1
            # Lag=k: (price_t+period+k / price_t+k) - 1
            # 含义: 在 T+k 日收盘买入，持有 period 天，在 T+k+period 日收盘卖出
            if execution_lag == 0:
                forward_prices = prices_df.shift(-period)
                returns = (forward_prices / prices_df) - 1
            else:
                forward_prices = prices_df.shift(-period - execution_lag)
                entry_prices = prices_df.shift(-execution_lag)
                returns = (forward_prices / entry_prices) - 1
            
            # 移除无效值
            returns = returns.replace([np.inf, -np.inf], np.nan)
            
            # 转换为Series格式 (date, asset)
            returns_series = returns.stack().dropna()
            returns_series.index.names = ['date', 'asset']
            
            forward_returns_dict[f'{period}D'] = returns_series
            self.logger.info(f"{period} 天前瞻收益率计算完成，有效数据点: {len(returns_series)}")
        
        # 合并所有期间的收益率
        forward_returns_df = pd.DataFrame(forward_returns_dict)
        if forward_returns_df.empty:
            raise ForwardReturnsValidationError("未能计算出任何前瞻收益率：检查价格覆盖与请求的持有期/执行滞后")
        self.logger.info(f"前瞻收益率计算完成 (Lag={execution_lag})，总形状: {forward_returns_df.shape}")
        return forward_returns_df
    
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
            'schema_version': 'forward_returns.v3',  # Input semantics and validated output schema.
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
