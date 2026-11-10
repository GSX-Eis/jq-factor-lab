import pandas as pd
import numpy as np
import alphalens
from typing import Dict
import logging
from numbers import Real
from backtest_and_analysis.forward_returns_calculator import ForwardReturnsValidationError


class MaxLossExceededError(ValueError):
    """Too few input observations survived alignment, labels and quantiles."""


class AlphalensRunner:
    """执行 Alphalens 分析的类"""
    
    def __init__(
        self,
        n_quantiles: int = 5,
        periods: tuple = (1, 5, 10),
        execution_lag: int = 1,
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
            
            if factor_df.shape[1] == 1:
                # 单股票情况，从pricing中获取正确的股票代码
                if isinstance(pricing.index, pd.MultiIndex):
                    assets = pricing.index.get_level_values(0).unique()
                    if len(assets) == 1:
                        factor_df = factor_df.rename(columns={factor_df.columns[0]: assets[0]})
                        self.logger.info(f"Renamed column to asset code: {assets[0]}")
            
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
                
                # 使用alphalens进行分析
                try:
                    self.logger.info("Attempting alphalens get_clean_factor_and_forward_returns...")
                    
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
                        forward_returns = ForwardReturnsCalculator(
                            trading_calendar=pricing.attrs.get('trading_calendar')
                        ).calculate_forward_returns(
                            pricing_series,
                            periods=self.periods,
                            execution_lag=self.execution_lag,
                        )
                    
                    # Try the main alphalens function
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
        
        # Check data completeness for each stock
        missing_data_count = 0
        total_points = 0
        for stock in common_stocks:
            for date in common_dates:
                total_points += 1
                if date in prices_df.index and (pd.isna(prices_df.loc[date, stock]) or prices_df.loc[date, stock] == 0):
                    missing_data_count += 1
        
        if total_points > 0:
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
                
                # Pairwise Spearman IC keeps shorter horizons available when a
                # longer horizon has already run out of future observations.
                period_columns = [
                    col for col in factor_data.columns
                    if col not in ('factor', 'factor_quantile')
                ]
                ic_data = pd.DataFrame({
                    period: factor_data.groupby(level=0, observed=True).apply(
                        lambda group, column=period: group['factor'].corr(
                            group[column], method='spearman'
                        )
                    )
                    for period in period_columns
                })
                
                # Convert periods to standardized format for IC data
                ic_data_formatted = {}
                for period in ic_data.columns:
                    period_key = f'{period}D' if isinstance(period, int) else str(period)
                    ic_data_formatted[period_key] = ic_data[period]
                
                ic_data = pd.DataFrame(ic_data_formatted)
                
                # Calculate IR (Information Ratio)
                ir_data = {}
                for period in ic_data.columns:
                    period_ic = ic_data[period].dropna()
                    if not period_ic.empty and len(period_ic) > 1:
                        mean_ic = period_ic.mean()
                        ic_std = period_ic.std()
                        ir = mean_ic / ic_std if ic_std != 0 else np.nan
                    else:
                        ir = period_ic.mean() if not period_ic.empty else np.nan
                    ir_data[period] = ir
                
                # Convert periods for IR data  
                ir_data_formatted = {}
                for period, ir_val in ir_data.items():
                    period_key = f'{period}D' if isinstance(period, int) else str(period)
                    ir_data_formatted[period_key] = ir_val
                
                ir_df = pd.DataFrame.from_dict(ir_data_formatted, orient='index', columns=['ir'])
                ir_df.index.name = 'period'
                
                # Calculate quantile returns using alphalens - FIXED: Proper error handling and fallback
                try:
                    mean_return_by_q, std_err = alphalens.performance.mean_return_by_quantile(
                        factor_data, 
                        by_date=True
                    )
                    self.logger.info(f"Successfully calculated quantile returns using alphalens: {mean_return_by_q.shape}")
                    
                    # Ensure column names are properly formatted
                    if hasattr(mean_return_by_q.columns, 'levels'):
                        # MultiIndex columns - rename period level
                        new_columns = []
                        for period, quantile in mean_return_by_q.columns:
                            period_key = f'{period}D' if isinstance(period, int) else str(period)
                            new_columns.append((period_key, quantile))
                        mean_return_by_q.columns = pd.MultiIndex.from_tuples(new_columns, names=['period', 'quantile'])
                    else:
                        # Single level columns - add 'D' suffix if needed
                        new_columns = {}
                        for col in mean_return_by_q.columns:
                            period_key = f'{col}D' if isinstance(col, int) else str(col)
                            new_columns[col] = period_key
                        mean_return_by_q = mean_return_by_q.rename(columns=new_columns)
                    
                except Exception as e:
                    self.logger.warning(f"Standard alphalens quantile returns calculation failed: {e}")
                    # Try manual calculation as fallback
                    try:
                        self.logger.info("Attempting manual quantile returns calculation...")
                        quantile_returns_dict = {}
                        
                        # Get period columns (forward returns)
                        period_cols = [col for col in factor_data.columns if str(col).endswith('D') or isinstance(col, int)]
                        if not period_cols:
                            period_cols = [col for col in factor_data.columns if col not in ['factor', 'factor_quantile']]
                        
                        self.logger.info(f"Found period columns: {period_cols}")
                        
                        for period_col in period_cols:
                            if period_col in factor_data.columns and 'factor_quantile' in factor_data.columns:
                                # Calculate quantile returns for this period
                                quantile_returns = factor_data.groupby([
                                    factor_data.index.get_level_values(0),  # date
                                    'factor_quantile'
                                ])[period_col].mean().unstack(fill_value=np.nan)
                                
                                # Standardize period name
                                period_key = f'{period_col}D' if isinstance(period_col, int) else str(period_col)
                                quantile_returns_dict[period_key] = quantile_returns
                            
                        if quantile_returns_dict:
                            mean_return_by_q = pd.concat(quantile_returns_dict, axis=1)
                            mean_return_by_q.columns.names = ['period', 'quantile']
                            self.logger.info(f"Manual quantile returns calculation succeeded: {mean_return_by_q.shape}")
                        else:
                            self.logger.warning("Manual quantile returns calculation found no data")
                            mean_return_by_q = pd.DataFrame()
                            
                    except Exception as e2:
                        self.logger.error(f"Manual quantile returns calculation also failed: {e2}")
                        mean_return_by_q = pd.DataFrame()
                    
                    std_err = pd.DataFrame()
                
            elif isinstance(factor_data, pd.DataFrame) and 'forward_returns' in factor_data.columns:
                # This is our improved basic fallback result
                self.logger.info("Processing improved basic fallback results")
                
                # Calculate IC for each period
                ic_results = {}
                periods_in_data = factor_data['period'].unique()
                
                for period in periods_in_data:
                    period_data = factor_data[factor_data['period'] == period]
                    if len(period_data) >= 2:
                        try:
                            # Use numpy correlation to avoid Series ambiguity
                            factor_vals = period_data['factor'].values
                            return_vals = period_data['forward_returns'].values
                            
                            # Remove NaN values
                            valid_mask = ~(np.isnan(factor_vals) | np.isnan(return_vals))
                            if np.sum(valid_mask) >= 2:
                                factor_clean = factor_vals[valid_mask]
                                return_clean = return_vals[valid_mask]
                                
                                # Check for sufficient variance
                                if np.std(factor_clean) > 1e-10 and np.std(return_clean) > 1e-10:
                                    correlation = np.corrcoef(factor_clean, return_clean)[0, 1]
                                else:
                                    correlation = np.nan
                            else:
                                correlation = np.nan
                        except Exception as e:
                            self.logger.debug(f"Error calculating IC for period {period}: {e}")
                            correlation = np.nan
                    else:
                        correlation = np.nan
                    
                    ic_results[period] = correlation
                
                # Create IC DataFrame - grouped by date
                ic_data_dict = {}
                for period in periods_in_data:
                    period_data = factor_data[factor_data['period'] == period]
                    dates = period_data.index.get_level_values(0).unique()
                    
                    daily_ic = []
                    for date in dates:
                        date_data = period_data[period_data.index.get_level_values(0) == date]
                        if len(date_data) >= 2:
                            try:
                                # Ensure we get a scalar correlation, not a Series
                                factor_vals = date_data['factor'].values
                                return_vals = date_data['forward_returns'].values
                                
                                # Remove any NaN values
                                valid_mask = ~(np.isnan(factor_vals) | np.isnan(return_vals))
                                if np.sum(valid_mask) >= 2:
                                    factor_clean = factor_vals[valid_mask]
                                    return_clean = return_vals[valid_mask]
                                    
                                    # Calculate correlation using numpy to avoid Series ambiguity
                                    if np.std(factor_clean) > 1e-10 and np.std(return_clean) > 1e-10:
                                        daily_corr = np.corrcoef(factor_clean, return_clean)[0, 1]
                                    else:
                                        daily_corr = np.nan
                                else:
                                    daily_corr = np.nan
                            except Exception as e:
                                self.logger.debug(f"Error calculating daily IC for date {date}: {e}")
                                daily_corr = np.nan
                        else:
                            daily_corr = np.nan
                            
                        daily_ic.append(daily_corr)
                    
                    ic_data_dict[period] = pd.Series(daily_ic, index=dates)
                
                ic_data = pd.DataFrame(ic_data_dict)
                
                # Calculate IR (Information Ratio)
                ir_data = {}
                for period in periods_in_data:
                    if period in ic_results:
                        period_ic_series = ic_data[period].dropna()
                        if len(period_ic_series) > 1:
                            mean_ic = period_ic_series.mean()
                            ic_std = period_ic_series.std()
                            ir = mean_ic / ic_std if ic_std != 0 else np.nan
                        else:
                            ir = ic_results[period] if not pd.isna(ic_results[period]) else 0
                        ir_data[period] = ir
                
                ir_df = pd.DataFrame.from_dict(ir_data, orient='index', columns=['ir'])
                ir_df.index.name = 'period'
                
                # Calculate quantile returns
                quantile_returns_dict = {}
                for period in periods_in_data:
                    period_data = factor_data[factor_data['period'] == period]
                    
                    if 'factor_quantile' in period_data.columns:
                        # Group by quantile and calculate mean returns
                        date_quantile_returns = period_data.groupby([
                            period_data.index.get_level_values(0),  # date
                            'factor_quantile'
                        ])['forward_returns'].mean().unstack(fill_value=np.nan)
                        
                        # Convert period to standardized format (e.g., '1D', '5D', '10D')
                        period_key = f'{period}D' if isinstance(period, int) else str(period)
                        quantile_returns_dict[period_key] = date_quantile_returns
                
                # Combine quantile returns data
                if quantile_returns_dict:
                    combined_qr = pd.concat(quantile_returns_dict, axis=1)
                    combined_qr.columns.names = ['period', 'quantile']
                    mean_return_by_q = combined_qr
                else:
                    mean_return_by_q = pd.DataFrame()
                
                std_err = pd.DataFrame()
                
            elif isinstance(factor_data, pd.DataFrame) and 'factor' in factor_data.columns:
                # This is old basic fallback result (maintain compatibility)
                self.logger.info("Processing basic fallback results")
                
                # Calculate simple correlation as IC substitute
                correlation = np.nan  # Initialize with default value
                if 'return_1d' in factor_data.columns:
                    correlation = factor_data['factor'].corr(factor_data['return_1d'])
                    ic_data = pd.DataFrame({'1D': [correlation]}, index=[factor_data.index.get_level_values(0)[0]])
                else:
                    ic_data = pd.DataFrame({'1D': [np.nan]})
                
                # Simple IR calculation
                ir_data = {'1D': correlation if not pd.isna(correlation) else 0}
                ir_df = pd.DataFrame.from_dict(ir_data, orient='index', columns=['ir'])
                ir_df.index.name = 'period'
                
                # Empty quantile returns data
                mean_return_by_q = pd.DataFrame()
                std_err = pd.DataFrame()
                
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
        self.logger.info(f"包含的前瞻收益率期间: {[col for col in clean_factor.columns if col.endswith('D')]}")
        
        return clean_factor
