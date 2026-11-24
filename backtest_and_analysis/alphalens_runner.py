from dataclasses import asdict

import pandas as pd
import numpy as np
import re
from typing import Dict
import logging
from numbers import Real
from backtest_and_analysis.evaluation import EvaluationEngine, evaluate_interval_usability
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.forward_returns_calculator import ForwardReturnsValidationError
from utils.market_db.guards import require_single_market_db


def _demean_float32_by_date(source: pd.Series) -> pd.Series:
    """Match the legacy float32 reduction in the original order and day width."""
    codes = source.index.codes[0]
    order = np.argsort(codes, kind="stable")
    _, widths = np.unique(codes[order], return_counts=True)
    starts = np.cumsum(widths) - widths
    values = source.to_numpy()
    excess = np.empty_like(values)
    for width in np.unique(widths):
        positions = order[starts[widths == width, None] + np.arange(width)]
        day = values[positions]
        missing = np.isnan(day)
        # Zero-fill as pandas nanmean does, retaining NumPy's width-dependent summation.
        sums = np.where(missing, np.float32(0), day).sum(axis=1)
        with np.errstate(invalid="ignore"):
            centers = sums / (~missing).sum(axis=1, dtype=np.float32)
            excess[positions] = day - centers[:, None]
    return pd.Series(excess, index=source.index)


class InsufficientCrossSectionAssets(ValueError):
    code = "insufficient_cross_section_assets"

    def __init__(self) -> None:
        super().__init__(
            "Daily cross-sectional analysis requires at least 2 assets. "
            "Provide prices for multiple assets, or supply Bar data to "
            "jq-factor-lab time-series-analysis --analysis-mode time_series for one instrument."
        )


class MaxLossExceededError(ValueError):
    """Too few input observations survived alignment, labels and quantiles."""


class AlphalensRunner:
    """Adapt daily evaluation to the historical Alphalens-style dictionary."""
    
    def __init__(
        self,
        n_quantiles: int = 5,
        periods: tuple = (1, 5, 10),
        execution_lag: int = 1,
        *,
        _force_raw_quantiles: bool = False,
    ):
        """
        初始化 Alphalens 运行器
        
        Args:
            n_quantiles: 分层数量
            periods: 前瞻期间元组，默认为(1, 5, 10)
        """
        self.n_quantiles = n_quantiles
        self.periods = periods
        self.execution_lag = execution_lag
        self._force_raw_quantiles = _force_raw_quantiles
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self, factor_df: pd.DataFrame, pricing: pd.Series, forward_returns: pd.DataFrame = None, max_loss: float = 0.35) -> Dict[str, pd.DataFrame]:
        """
        运行 Alphalens 分析
        
        Args:
            factor_df: 因子值 DataFrame (日期为索引，股票代码为列)
            pricing: 价格 Series (MultiIndex: (asset, datetime))
            forward_returns: 预计算的前瞻收益率 DataFrame，如果提供则跳过收益率计算
            max_loss: 允许的最大数据丢失比例 (0.0-1.0)，默认为0.35。
                分母为原始有限因子对；任一请求周期收益有限且分位有效即保留。
                对齐、常量截面、尾部缺标签及提前返回均计入损失。
            
        Returns:
            包含各种 Alphalens 指标的字典

        Raises:
            MaxLossExceededError: 最终保留比例低于门槛（等于门槛时允许）。
        """
        if (isinstance(max_loss, bool) or not isinstance(max_loss, Real)
                or not np.isfinite(max_loss) or not 0 <= max_loss <= 1):
            raise ValueError('max_loss must be a finite number between 0 and 1')

        # Invalid inputs must not be turned into successful empty analysis.
        from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
        ForwardReturnsCalculator._validate_horizons(self.periods, self.execution_lag)
        if (isinstance(self.n_quantiles, (bool, np.bool_))
                or not isinstance(self.n_quantiles, (int, np.integer)) or self.n_quantiles <= 0):
            raise ValueError('n_quantiles must be a positive integer')
        if not isinstance(factor_df, pd.DataFrame):
            raise TypeError('factor_df must be a DataFrame')
        if not isinstance(pricing, pd.Series):
            raise TypeError('pricing must be a Series')
        if forward_returns is not None and not isinstance(forward_returns, pd.DataFrame):
            raise TypeError('forward_returns must be a DataFrame')
        require_single_market_db(factor_df, pricing, forward_returns)
        if not factor_df.empty:
            if isinstance(factor_df.index, pd.MultiIndex):
                raise ValueError('factor_df must have a one-dimensional date index')
            pd.to_datetime(factor_df.index, errors='raise')
            factor_df.to_numpy(dtype=float, na_value=np.nan)
        if not pricing.empty and (not isinstance(pricing.index, pd.MultiIndex)
                                  or pricing.index.nlevels != 2):
            raise ValueError('pricing must have a two-level asset/date index')
        if forward_returns is not None and not forward_returns.empty:
            if (not isinstance(forward_returns.index, pd.MultiIndex)
                    or forward_returns.index.nlevels != 2):
                raise ValueError('forward_returns must have a two-level date/asset index')
            forward_returns.to_numpy(dtype=float, na_value=np.nan)

        input_count = retained_count = 0

        def check_loss():
            loss = (input_count - retained_count) / input_count if input_count else 0.0
            if loss > max_loss:
                raise MaxLossExceededError(
                    f'max_loss exceeded: loss={loss:.6g}, max_loss={max_loss:.6g}, '
                    f'retained={retained_count}/{input_count}'
                )

        def minimal_results():
            check_loss()
            return self._create_minimal_results()

        # Resolve the calendar before the analysis fallback handlers. A missing
        # calendar must not be disguised as a successful empty analysis.
        if forward_returns is None and not pricing.empty:
            from utils.trading_calendar import load_trading_calendar
            calendar = pricing.attrs.get('trading_calendar')
            if calendar is None:
                calendar = load_trading_calendar()
            pricing = pricing.copy()
            pricing.attrs['trading_calendar'] = calendar
            date_level = 'date' if 'date' in pricing.index.names else -1
            dates = pd.to_datetime(pricing.index.get_level_values(date_level))
            calendar.validate_coverage(dates.min(), dates.max())
        if not factor_df.empty and len(factor_df.columns.unique()) < 2:
            raise InsufficientCrossSectionAssets()
        try:
            input_count = int(np.isfinite(factor_df.to_numpy(dtype=float, na_value=np.nan)).sum())
            # 0. 基本数据验证
            if factor_df.empty:
                self.logger.error("因子数据为空")
                return minimal_results()
            
            # 1. 确保因子数据格式正确
            self.logger.info(f"Original factor_df shape: {factor_df.shape}")
            
            # 检查因子数据质量
            non_null_count = factor_df.count().sum()
            total_count = factor_df.size
            data_coverage = non_null_count / total_count if total_count > 0 else 0
            
            self.logger.info(f"Factor data coverage: {data_coverage:.2%} ({non_null_count}/{total_count})")
            
            # 如果数据覆盖率太低，返回基本结果
            if data_coverage < 0.05:  # 少于5%的数据覆盖率
                self.logger.warning(f"Factor data coverage too low ({data_coverage:.2%}), returning minimal results")
                return minimal_results()
            
            # 2. 确保索引都是DatetimeIndex
            if not isinstance(factor_df.index, pd.DatetimeIndex):
                factor_df.index = pd.to_datetime(factor_df.index)
                self.logger.info("Converted factor index to DatetimeIndex")
            
            # 3. 重要：重新格式化价格数据为DataFrame格式而不是Series
            if isinstance(pricing.index, pd.MultiIndex):
                # Check the structure of the MultiIndex to handle different cases
                index_names = pricing.index.names
                
                # FIXED: Better detection of the MultiIndex structure
                # Check if it's (date, asset) or (asset, date) format
                level_0_sample = pricing.index.get_level_values(0)[0] if len(pricing.index) > 0 else None
                level_1_sample = pricing.index.get_level_values(1)[0] if len(pricing.index) > 0 else None
                
                # Try to determine which level contains dates vs assets
                level_0_is_date = False
                level_1_is_date = False
                
                try:
                    pd.to_datetime(level_0_sample)
                    level_0_is_date = True
                except:
                    pass
                    
                try:
                    pd.to_datetime(level_1_sample)
                    level_1_is_date = True
                except:
                    pass
                
                self.logger.info(f"MultiIndex analysis: level_0_is_date={level_0_is_date}, level_1_is_date={level_1_is_date}")
                self.logger.info(f"Level 0 sample: {level_0_sample}, Level 1 sample: {level_1_sample}")
                
                # Case 1: (date, asset) format
                if level_0_is_date and not level_1_is_date:
                    self.logger.info("Detected (date, asset) MultiIndex format")
                    dates = pricing.index.get_level_values(0).unique()
                    assets = pricing.index.get_level_values(1).unique()
                    
                    # Pivot the data to get DataFrame with dates as index, assets as columns
                    prices_df = pricing.unstack(level=1)
                    
                # Case 2: (asset, date) format  
                elif not level_0_is_date and level_1_is_date:
                    self.logger.info("Detected (asset, date) MultiIndex format")
                    assets = pricing.index.get_level_values(0).unique()
                    
                    # Use dictionary to collect data properly
                    price_dict = {}
                    for asset in assets:
                        try:
                            asset_prices = pricing.loc[asset].copy()
                            
                            # Ensure the index is datetime
                            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                                try:
                                    asset_prices.index = pd.to_datetime(asset_prices.index)
                                    self.logger.debug(f"Converted {asset} price index to DatetimeIndex")
                                except Exception as e:
                                    self.logger.warning(f"Could not convert {asset} price index to datetime: {e}")
                                    continue
                            
                            price_dict[asset] = asset_prices
                        except Exception as e:
                            self.logger.warning(f"Error processing asset {asset}: {e}")
                            continue
                    
                    if price_dict:
                        prices_df = pd.DataFrame(price_dict)
                    else:
                        self.logger.error("No valid price data could be processed")
                        return minimal_results()
                        
                else:
                    # Unknown format - try both approaches
                    self.logger.warning(f"Unknown MultiIndex structure, trying fallback approaches")
                    try:
                        # Try approach 1: assume (asset, date)
                        assets = pricing.index.get_level_values(0).unique()
                        price_dict = {}
                        
                        for asset in assets:
                            try:
                                asset_data = pricing.loc[asset]
                                price_dict[asset] = asset_data
                            except:
                                continue
                        
                        if price_dict:
                            prices_df = pd.DataFrame(price_dict)
                            self.logger.info("Successfully used fallback (asset, date) approach")
                        else:
                            raise ValueError("Fallback approach failed")
                            
                    except Exception as e:
                        self.logger.error(f"All MultiIndex approaches failed: {e}")
                        return minimal_results()
                
                self.logger.info(f"Converted pricing to DataFrame with shape: {prices_df.shape}")
                
            # Check for quantile return data
            elif isinstance(pricing.index, pd.MultiIndex) and ('factor_quantile' in pricing.index.names or 'quantile' in str(pricing.index.names).lower()):
                self.logger.info("Detected quantile return data structure - skipping price conversion")
                # This is likely quantile return data from alphalens, not price data
                self.logger.warning("Received quantile return data instead of price data - this suggests alphalens already processed the data")
                return minimal_results()
            else:
                self.logger.error("Pricing data must have MultiIndex (asset, date)")
                return minimal_results()
            
            # 4. 基本数据验证
            self._validate_data(factor_df, prices_df)
            
            # 5. 使用新的业务日对齐方法
            try:
                factor_df_fixed, prices_df_fixed = self._fix_business_day_alignment(factor_df, prices_df)
                
                # 准备因子数据为Series (date, asset)
                factor_series = factor_df_fixed.stack().dropna()
                factor_series.index.names = ['date', 'asset']
                
                self.logger.info(f"Factor data after cleaning: {len(factor_series)} points")
                
                # 检查是否有足够的数据进行分析
                if len(factor_series) < 50:  # 至少需要50个数据点
                    self.logger.warning(f"Insufficient data points ({len(factor_series)}) for proper analysis")
                    return minimal_results()
                
                # Align daily labels for the evaluation engine
                try:
                    self.logger.info("Preparing daily factor and forward returns...")
                    
                    # Fix timezone issues - ensure both factor and price indices are timezone-naive
                    if hasattr(factor_series.index.get_level_values(0), 'tz') and factor_series.index.get_level_values(0).tz is not None:
                        factor_series.index = factor_series.index.set_levels(
                            factor_series.index.levels[0].tz_localize(None), level=0
                        )
                        self.logger.info("Removed timezone from factor series")
                    
                    if hasattr(prices_df_fixed.index, 'tz') and prices_df_fixed.index.tz is not None:
                        prices_df_fixed.index = prices_df_fixed.index.tz_localize(None)
                        self.logger.info("Removed timezone from price DataFrame")
                    
                    if forward_returns is None:
                        from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator

                        pricing_series = prices_df_fixed.stack().dropna()
                        pricing_series.index.names = ['date', 'asset']
                        pricing_series.attrs = pricing.attrs.copy()
                        forward_returns = ForwardReturnsCalculator(
                            trading_calendar=pricing.attrs.get('trading_calendar')
                        ).calculate_forward_returns(
                            pricing_series,
                            periods=self.periods,
                            execution_lag=self.execution_lag,
                        )
                    
                    # Prepare the legacy eligible sample before evaluating the panel.
                    try:
                        self.logger.info("使用真实交易日预计算前瞻收益率...")
                        clean_factor = self._prepare_factor_data_with_precomputed_returns(
                            factor_series, forward_returns
                        )
                        requested = [col for col in clean_factor.columns
                                     if col in self.periods or col in {f'{p}D' for p in self.periods}]
                        retained_count = int((
                            np.isfinite(clean_factor['factor'].to_numpy(dtype=float, na_value=np.nan))
                            & np.isfinite(clean_factor['factor_quantile'].to_numpy(dtype=float, na_value=np.nan))
                            & np.isfinite(clean_factor[requested].to_numpy(dtype=float, na_value=np.nan)).any(axis=1)
                        ).sum())
                        check_loss()
                        if clean_factor.empty:
                            self.logger.warning("No nonconstant cross-sections with valid forward returns")
                            return minimal_results()
                        return self._process_alphalens_results(
                            clean_factor, trading_calendar=pricing.attrs.get('trading_calendar')
                        )
                        
                    except MaxLossExceededError:
                        raise
                    except Exception as e:
                        self.logger.warning(f"Standard alphalens approach failed: {e}")
                        self.logger.warning("Skipping factor analysis (no fallback)")
                        return minimal_results()
                
                except (ForwardReturnsValidationError, MaxLossExceededError):
                    raise
                except Exception as e:
                    self.logger.warning(f"Standard alphalens approach failed: {e}")
                    self.logger.warning("Skipping factor analysis (no fallback)")
                    return minimal_results()
            
            except (ForwardReturnsValidationError, MaxLossExceededError):
                raise
            except Exception as e:
                self.logger.error(f"Business day alignment failed: {e}")
                self.logger.warning("Skipping factor analysis (no fallback)")
                return minimal_results()
            
        except (ForwardReturnsValidationError, MaxLossExceededError):
            raise
        except Exception as e:
            self.logger.error(f"Error in Alphalens analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.debug(f"Factor data sample:\n{factor_df.head()}")
            self.logger.debug(f"Pricing data sample:\n{pricing.head() if hasattr(pricing, 'head') else 'Not available'}")
            return minimal_results()
    
    def _create_minimal_results(self):
        """创建最小的结果结构，支持多个时间周期"""
        # 为每个周期创建NaN值
        period_labels = [f'{p}D' for p in self.periods]
        
        # 创建IC DataFrame，每个周期一列
        ic_data = {period: [np.nan] for period in period_labels}
        ic_df = pd.DataFrame(ic_data)
        
        # 创建IR DataFrame，每个周期一行
        ir_data = {'ir': [np.nan] * len(period_labels)}
        ir_df = pd.DataFrame(ir_data, index=period_labels)
        ir_df.index.name = 'period'
        
        return {
            'ic': ic_df,
            'ir': ir_df,
            'quantile_returns': pd.DataFrame(),
            'std_err': pd.DataFrame(),
        }
    
    def _break_factor_ties(self, factor_series: pd.Series, tie_threshold: float = 0.25) -> pd.Series:
        """Preserve raw values: jitter would manufacture ranks and change IC.

        Kept for callers of the former helper; quantiles now handle ties per date.
        """
        return factor_series

    def _fix_business_day_alignment(self, factor_df, pricing_df):
        """
        Align factor observations to real price dates without inventing sessions.
        
        Args:
            factor_df: Factor DataFrame with dates as index, stocks as columns
            pricing_df: Pricing DataFrame with dates as index, stocks as columns
            
        Returns:
            tuple: (fixed_factor_df, fixed_pricing_df) with proper business day frequency
        """
        self.logger.info("Fixing business day alignment for alphalens compatibility...")
        
        # Get the date ranges - FIXED: ensure proper datetime handling
        factor_dates = factor_df.index
        price_dates = pricing_df.index
        
        # Ensure all dates are datetime objects
        if not isinstance(factor_dates, pd.DatetimeIndex):
            try:
                factor_dates = pd.to_datetime(factor_dates)
            except Exception as e:
                self.logger.error(f"Cannot convert factor dates to datetime: {e}")
                raise ValueError("Factor dates must be convertible to datetime")
        
        if not isinstance(price_dates, pd.DatetimeIndex):
            try:
                price_dates = pd.to_datetime(price_dates)
            except Exception as e:
                self.logger.error(f"Cannot convert price dates to datetime: {e}")
                raise ValueError("Price dates must be convertible to datetime")
        
        factor_df = factor_df.copy()
        pricing_df = pricing_df.copy()
        factor_df.index = factor_dates
        pricing_df.index = price_dates
        factor_df = factor_df[~factor_df.index.duplicated(keep='last')].sort_index()
        pricing_df = pricing_df[~pricing_df.index.duplicated(keep='last')].sort_index()

        common_dates = factor_df.index.intersection(pricing_df.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates after business day alignment")

        factor_final = factor_df.loc[common_dates].dropna(how='all')
        pricing_final = pricing_df.loc[pricing_df.index >= common_dates.min()]
        
        self.logger.info(f"Final factor data shape: {factor_final.shape}")
        self.logger.info(f"Final pricing data shape: {pricing_final.shape}")
        
        return factor_final, pricing_final

    def _validate_data(self, factor_df, prices_df):
        """
        Validate factor and pricing data
        
        Args:
            factor_df: Factor DataFrame
            prices_df: Pricing DataFrame
        """
        # Prepare factor data for validation
        factor_series = factor_df.stack()
        factor_dates = sorted(factor_series.index.get_level_values(0).unique())
        factor_stocks = sorted(factor_series.index.get_level_values(1).unique())
        price_dates = sorted(prices_df.index.unique())
        price_stocks = sorted(prices_df.columns)
        
        # Calculate overlap ratios
        common_dates = sorted(set(factor_dates) & set(price_dates))
        common_stocks = sorted(set(factor_stocks) & set(price_stocks))
        
        date_overlap_pct = len(common_dates)/len(factor_dates)*100
        stock_overlap_pct = len(common_stocks)/len(factor_stocks)*100
        
        self.logger.info(f"Date coverage: {date_overlap_pct:.1f}% ({len(common_dates)}/{len(factor_dates)})")
        self.logger.info(f"Stock coverage: {stock_overlap_pct:.1f}% ({len(common_stocks)}/{len(factor_stocks)})")
        
        # Check future data sufficiency - FIXED: ensure both dates are datetime objects
        max_forward_period = 10
        
        # Convert all dates to datetime objects for comparison
        try:
            factor_dates_dt = [pd.to_datetime(d) if not pd.isna(d) else None for d in factor_dates]
            factor_dates_dt = [d for d in factor_dates_dt if d is not None]
            last_factor_date = max(factor_dates_dt) if factor_dates_dt else None
        except Exception as e:
            self.logger.warning(f"Error converting factor dates: {e}")
            last_factor_date = None
        
        # Convert price dates to datetime if they aren't already
        price_dates_dt = []
        for d in price_dates:
            try:
                if pd.isna(d):
                    continue
                if isinstance(d, str):
                    price_dates_dt.append(pd.to_datetime(d))
                elif isinstance(d, (pd.Timestamp, pd.datetime)):
                    price_dates_dt.append(pd.to_datetime(d))
                else:
                    # Try to convert other types
                    price_dates_dt.append(pd.to_datetime(d))
            except Exception as e:
                self.logger.debug(f"Could not convert price date {d}: {e}")
                continue
        
        # Calculate future dates only if we have valid dates
        if last_factor_date is not None and price_dates_dt:
            future_dates = [d for d in price_dates_dt if d > last_factor_date]
        else:
            future_dates = []
        future_sufficient = len(future_dates) >= max_forward_period
        
        self.logger.info(f"Future data: {len(future_dates)} days available (sufficient: {future_sufficient})")
        
        # Check data completeness across the overlapping price panel.
        prices = prices_df.loc[common_dates, common_stocks]
        total_points = prices.size
        if total_points:
            missing_data_count = (prices.isna() | prices.eq(0)).to_numpy().sum()
            missing_pct = missing_data_count / total_points * 100
            self.logger.info(f"Price data missing rate: {missing_pct:.1f}% ({missing_data_count}/{total_points})")

        # Log data info
        self.logger.info(f"Factor index type: {type(factor_series.index)}")
        self.logger.info(f"Price index type: {type(prices_df.index)}")
        self.logger.info(f"Factor date range: {factor_series.index.get_level_values(0).min()} to {factor_series.index.get_level_values(0).max()}")
        self.logger.info(f"Price date range: {prices_df.index.min()} to {prices_df.index.max()}")

    def _process_alphalens_results(self, factor_data, *, trading_calendar=None):
        """
        Process alphalens results and compute metrics
        
        Args:
            factor_data: Factor data from alphalens or manual computation
            trading_calendar: Verified snapshot used by downstream monthly HAC tests
            
        Returns:
            dict: Analysis results
        """
        try:
            # Check if this is real alphalens factor_data
            # Real alphalens data has columns like ['1D', '5D', 'factor', 'factor_quantile']
            is_real_alphalens = (
                isinstance(factor_data, pd.DataFrame) and 
                'factor' in factor_data.columns and 
                'factor_quantile' in factor_data.columns and
                any(str(col).endswith('D') or isinstance(col, int) for col in factor_data.columns if col not in ['factor', 'factor_quantile'])
            )
            
            if is_real_alphalens:
                # This is real alphalens results
                self.logger.info("Processing full Alphalens results")
                
                values = factor_data.drop(columns="factor_quantile")
                panel = EvaluationPanel.from_long(pd.DataFrame(
                    values.to_numpy(dtype=float, na_value=np.nan),
                    index=values.index, columns=values.columns,
                ))
                buckets = factor_data["factor_quantile"].unstack().reindex(
                    index=panel.datetimes, columns=panel.assets,
                ).fillna(0).to_numpy(dtype=np.intp)
                names = {h: f"{h}D" if isinstance(h, int) else str(h) for h in panel.returns}
                periods = [h for h in panel.returns
                           if re.fullmatch(r"(?:\d+(?:[Dhms]|ms|us|ns]))+", names[h], re.IGNORECASE)]
                # Legacy pandas produced object standard-deviation arrays for
                # nullable Float32, selecting the raw quantile table.
                raw_quantiles = self._force_raw_quantiles or any(
                    isinstance(factor_data[h].dtype, pd.Float32Dtype) for h in periods
                )
                statistics = EvaluationEngine("cross_sectional").evaluate(
                    panel, n_quantiles=max(2, self.n_quantiles, buckets.max(initial=0)),
                    bucket_labels=buckets, _legacy_ic_rounding=True,
                    return_mode="raw" if raw_quantiles else "cross_section_excess",
                    bucket_standard_errors=not raw_quantiles,
                ).splits["full"].statistics
                ic_data = pd.DataFrame(
                    {names[h]: result.ic.values for h, result in statistics.horizons.items()},
                    index=panel.datetimes,
                )
                ic_data.attrs["hac_inference"] = {
                    names[h]: asdict(evaluate_interval_usability(
                        result.hac, point_estimate=result.ic_summary.mean,
                    ))
                    for h, result in statistics.horizons.items()
                }
                # Use the historical pandas variance and single-IC convention for the dictionary.
                ir_by_period = {}
                for name, values in ic_data.items():
                    observed = values.dropna()
                    deviation = observed.std()
                    ir_by_period[name] = (observed.mean() if len(observed) == 1 else
                                          observed.mean() / deviation if deviation > 0 else np.nan)
                ir_df = pd.Series(ir_by_period, name="ir").rename_axis("period").to_frame()

                if raw_quantiles:
                    # Keep occupied buckets even when a particular horizon lacks labels.
                    occupied = sorted(factor_data["factor_quantile"].unique())
                    periods = [h for h in panel.returns if str(h).endswith("D") or isinstance(h, int)]
                    mean_return_by_q = pd.concat({
                        names[h]: pd.DataFrame(
                            statistics.horizons[h].bucket_returns[:, np.asarray(occupied) - 1],
                            index=panel.datetimes, columns=occupied,
                        ) for h in periods
                    }, axis=1)
                    mean_return_by_q.columns.names = ["period", "quantile"]
                    std_err = pd.DataFrame()
                else:
                    # Match the (bucket, date) layout, including buckets where
                    # every label for a given horizon is missing.
                    present = np.zeros((len(panel.datetimes), statistics.turnover.shape[1] + 1), dtype=bool)
                    rows, cols = np.nonzero(buckets)
                    present[rows, buckets[rows, cols]] = True
                    bucket, date = np.nonzero(present.T)
                    index = pd.MultiIndex.from_arrays(
                        [bucket, panel.datetimes.take(date)], names=["factor_quantile", "date"],
                    )
                    mean_return_by_q = pd.DataFrame({
                        names[h]: statistics.horizons[h].bucket_returns[date, bucket - 1]
                        for h in periods
                    }, index=index)
                    std_err = pd.DataFrame({
                        names[h]: statistics.horizons[h].bucket_standard_errors[date, bucket - 1]
                        for h in periods
                    }, index=index)

                    # The legacy adapter rounded demeaning, bucket means, and bucket
                    # standard deviations in the input's single precision.
                    for h in periods:
                        if factor_data[h].dtype != np.dtype("float32"):
                            continue
                        source = factor_data[h]
                        excess = _demean_float32_by_date(source)
                        moments = excess.groupby([factor_data["factor_quantile"],
                                                  source.index.get_level_values(0)]).agg(["mean", "std", "count"])
                        mean_return_by_q[names[h]] = moments["mean"].astype(float)
                        std_err[names[h]] = moments["std"].astype(float) / np.sqrt(moments["count"])


            else:
                # Fallback to minimal results
                self.logger.warning("Unknown factor_data format, using minimal results")
                return self._create_minimal_results()
            
            if trading_calendar is not None:
                ic_data.attrs['trading_calendar'] = trading_calendar
            return {
                'ic': ic_data,
                'ir': ir_df,
                'quantile_returns': mean_return_by_q,
                'std_err': std_err,
            }
            
        except Exception as e:
            self.logger.error(f"Error processing alphalens results: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return empty results as fallback
            return {
                'ic': pd.DataFrame(),
                'ir': pd.DataFrame(),
                'quantile_returns': pd.DataFrame(),
                'std_err': pd.DataFrame(),
            }

    def _prepare_factor_data_with_precomputed_returns(self, factor_series: pd.Series, forward_returns: pd.DataFrame) -> pd.DataFrame:
        """
        使用预计算的前瞻收益率准备因子数据
        
        Args:
            factor_series: 因子数据Series，MultiIndex (date, asset)
            forward_returns: 预计算的前瞻收益率DataFrame，MultiIndex (date, asset)，列为各期间
            
        Returns:
            合并后的因子数据DataFrame，包含因子值、前瞻收益率和分位数信息
        """
        self.logger.info("开始准备因子数据与预计算前瞻收益率...")
        
        # 确保索引格式一致
        if not isinstance(factor_series.index, pd.MultiIndex):
            raise ValueError("因子数据必须是MultiIndex (date, asset)格式")
        
        if not isinstance(forward_returns.index, pd.MultiIndex):
            raise ValueError("前瞻收益率数据必须是MultiIndex (date, asset)格式")
        
        # 对齐因子数据和前瞻收益率数据
        common_index = factor_series.index.intersection(forward_returns.index)
        self.logger.info(f"因子数据点数: {len(factor_series)}, 前瞻收益率数据点数: {len(forward_returns)}")
        self.logger.info(f"对齐后共同数据点数: {len(common_index)}")
        
        if len(common_index) == 0:
            raise ValueError("因子数据与前瞻收益率数据没有重叠的时间-股票点")
        
        # 获取对齐后的数据
        aligned_factor = factor_series.loc[common_index]
        aligned_returns = forward_returns.loc[common_index]
        
        # 创建合并的DataFrame
        result_data = {'factor': aligned_factor}
        
        # 添加各期间的前瞻收益率
        for col in aligned_returns.columns:
            result_data[col] = aligned_returns[col]
        
        clean_factor = pd.DataFrame(result_data)
        
        # Equal values stay together. Collapsed edges produce fewer bins;
        # constant cross-sections have no quantile or valid IC and are skipped.
        # Never pool dates: future observations must not change historical bins.
        clean_factor['factor_quantile'] = clean_factor.groupby(level=0)['factor'].transform(
            lambda values: (pd.qcut(values, q=self.n_quantiles, labels=False, duplicates='drop') + 1
                            if values.nunique() > 1 else np.nan)
        )

        # Keep rows valid for any requested horizon; each IC column is
        # evaluated pairwise, so a missing 60D return must not erase 20D.
        return_columns = list(aligned_returns.columns)
        clean_factor = clean_factor.dropna(subset=['factor', 'factor_quantile'])
        clean_factor['factor_quantile'] = clean_factor['factor_quantile'].astype(int)
        clean_factor = clean_factor.dropna(subset=return_columns, how='all')
        
        self.logger.info(f"最终因子数据形状: {clean_factor.shape}")
        self.logger.info(f"包含的前瞻收益率期间: {[col for col in clean_factor.columns if str(col).endswith('D')]}")
        
        return clean_factor
