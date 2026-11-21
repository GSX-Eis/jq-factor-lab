"""Output normalization and calculation paths for FactorBase."""
import gc
import time
from typing import Callable

import numpy as np
import pandas as pd


def _normalize_factor_dataframe(values: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
    """Preserve keyed outputs; only legacy single-column outputs may use raw row order."""
    if not isinstance(values, pd.DataFrame):
        raise ValueError("因子输出必须是日期×股票 DataFrame")
    if values.empty:
        return values
    if not values.columns.is_unique:
        raise ValueError("因子输出列标签重复")

    if {'date', 'code', 'factor'}.issubset(values.columns):
        if len(values.columns) != 3:
            raise ValueError("因子长表只能包含 date、code、factor 三列")
        long = values.copy()
        if pd.api.types.is_numeric_dtype(long['date'].dtype):
            raise ValueError("因子长表日期不能是数值行号")
        long['date'] = pd.to_datetime(long['date'], errors='raise')
        if long[['date', 'code']].isna().any().any():
            raise ValueError("因子长表包含空日期或股票代码")
        values = long.pivot(index='date', columns='code', values='factor')
    elif values.index.inferred_type in {'integer', 'floating', 'mixed-integer', 'boolean', 'complex'}:
        # The old N×1 form is positional only when its row index is unchanged.
        if (values.index.name == 'date' or values.shape[1] != 1
                or not values.index.is_unique or not values.index.equals(raw_data.index)):
            raise ValueError("无法识别因子 DataFrame：需要日期索引，或与原始行索引一致的单列值")
        values = pd.DataFrame({
            'date': pd.to_datetime(raw_data['date'], errors='raise'),
            'code': raw_data['code'],
            'factor': values.iloc[:, 0].to_numpy(),
        }).pivot(index='date', columns='code', values='factor')

    if isinstance(values.index, pd.MultiIndex):
        raise ValueError("因子宽表需要单层日期索引")
    if not all(isinstance(code, str) and code for code in values.columns):
        raise ValueError("因子宽表的股票代码必须是非空字符串")
    dates = pd.to_datetime(values.index, errors='raise')
    if dates.hasnans or not dates.is_unique:
        raise ValueError("因子宽表日期必须非空且唯一")
    if not isinstance(values.index, pd.DatetimeIndex) or values.index.name != 'date':
        values = values.copy(deep=False)
        values.index = dates.rename('date')
    return values


def get_factor_values(self, factor_base_cls: type, force_recalculate: bool = False) -> pd.DataFrame:
    if getattr(getattr(self, '_frequency_route', None), 'bar_grid', False):
        return self._get_market_factor_values(force_recalculate)
    # Read first: mutable files and borrowed connections have no immutable
    # version token. Cache calculations only after identifying actual inputs.
    raw_data = self._get_raw_data_for_factor()
    cache_key = self._get_cache_key(raw_data) if self.cache_dir else None
    if not force_recalculate and cache_key:
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            self.logger.info(f"从缓存加载因子 {self.factor_name}")
            return cached_result

    # Expose raw inputs to diagnostics while tolerating read-only overrides.
    try:
        self.last_raw_data = raw_data
    except Exception:
        pass
    if raw_data.empty:
        self.logger.warning("原始数据为空，返回空DataFrame")
        return pd.DataFrame()

    if self.pit_enabled:
        raw_data = self._apply_pit_adjustment(raw_data)

    factor_values = self.calculate_factor(raw_data)

    factor_values = factor_base_cls._normalize_factor_dataframe(factor_values, raw_data)
    if factor_values.empty:
        self.logger.warning("因子计算返回空DataFrame")
        return pd.DataFrame()

    if self.use_dynamic_pool and self.stock_pools_by_period and self.dynamic_pool_mode in ('factor', 'both'):
        factor_values = self._apply_dynamic_pool_to_factor_df(factor_values)

    if self.cache_dir:
        try:
            self._save_to_cache(factor_values, cache_key)
        except Exception as e:
            self.logger.warning(f"保存缓存失败: {e}")

    return factor_values


def get_factor_values_unfiltered(self, factor_base_cls: type, force_recalculate: bool = False) -> pd.DataFrame:
    """Compute inside the current stock universe. Only the Bar route uses the disk cache."""
    if getattr(getattr(self, '_frequency_route', None), 'bar_grid', False):
        return self._get_market_factor_values(force_recalculate)
    started = time.perf_counter()
    self.last_performance_metrics = {}
    original_all_stocks = self.all_stocks
    original_use_dynamic_pool = self.use_dynamic_pool
    original_dynamic_pool_mode = self.dynamic_pool_mode

    try:
        self.use_dynamic_pool = False
        self.dynamic_pool_mode = 'factor'  # 不在原始数据阶段过滤

        self.logger.info(f"get_factor_values_unfiltered: 使用当前股票范围 ({len(self.all_stocks)} 只股票)")

        fetch_started = time.perf_counter()
        try:
            raw_data = self._get_raw_data_for_factor()
        finally:
            self.last_performance_metrics['raw_fetch_seconds'] = time.perf_counter() - fetch_started
        self.last_performance_metrics.update({
            'raw_rows': len(raw_data),
            'raw_fields': list(raw_data.columns),
            'raw_memory_mb': float(raw_data.memory_usage(deep=True).sum() / 1024**2),
        })

        if raw_data.empty:
            self.logger.warning("原始数据为空，返回空DataFrame")
            return pd.DataFrame()

        if self.pit_enabled:
            pit_started = time.perf_counter()
            try:
                raw_data = self._apply_pit_adjustment(raw_data)
            finally:
                self.last_performance_metrics['pit_seconds'] = time.perf_counter() - pit_started

        calculate_started = time.perf_counter()
        try:
            factor_values = self.calculate_factor(raw_data)
        finally:
            self.last_performance_metrics['calculate_seconds'] = time.perf_counter() - calculate_started

        factor_values = factor_base_cls._normalize_factor_dataframe(factor_values, raw_data)
        if factor_values.empty:
            self.logger.warning("因子计算返回空DataFrame")
            return pd.DataFrame()

        self.logger.info(f"get_factor_values_unfiltered 返回因子形状: {factor_values.shape} (未应用动态股票池过滤)")
        return factor_values

    finally:
        self.all_stocks = original_all_stocks
        self.use_dynamic_pool = original_use_dynamic_pool
        self.dynamic_pool_mode = original_dynamic_pool_mode
        self.last_performance_metrics['total_seconds'] = time.perf_counter() - started


def get_factor_values_unfiltered_chunked(self, factor_base_cls: type, chunk_size: int) -> pd.DataFrame:
    """Require a concrete stock-independence declaration and a live-data check."""
    stocks = sorted(set(self.all_stocks or []))
    if self.data_frequency != '5m' or chunk_size <= 0 or len(stocks) <= chunk_size:
        return self.get_factor_values_unfiltered()
    if (self.__class__.__dict__.get('STOCK_CHUNK_INDEPENDENT') is not True
            or self.__class__.get_factor_values_unfiltered is not factor_base_cls.get_factor_values_unfiltered):
        self.last_performance_metrics = {}
        result = self.get_factor_values_unfiltered()
        self.last_performance_metrics['stock_chunk_mode'] = 'fallback_full'
        return result

    original_stocks = self.all_stocks
    started = time.perf_counter()
    probe_count = min(len(stocks), chunk_size * 2)
    probe_stocks = stocks[:probe_count]
    try:
        try:
            self.all_stocks = probe_stocks
            probe_full = self.get_factor_values_unfiltered(force_recalculate=True)
            if probe_full.empty:
                raise ValueError("股票探针无有效截面，无法证明股票分块等价")
            probe_chunks = []
            for offset in range(0, probe_count, chunk_size):
                self.all_stocks = probe_stocks[offset:offset + chunk_size]
                probe_chunks.append(self.get_factor_values_unfiltered(force_recalculate=True))
            probe_chunked = pd.concat(probe_chunks, axis=1).sort_index().sort_index(axis=1)
            pd.testing.assert_frame_equal(
                probe_full.sort_index().sort_index(axis=1),
                probe_chunked,
                check_exact=True,
                check_dtype=False,
            )
        except (AssertionError, ValueError) as exc:
            self.logger.warning("股票分块等价检查失败，回退完整计算: %s", exc)
            self.all_stocks = original_stocks
            result = self.get_factor_values_unfiltered(force_recalculate=True)
            self.last_performance_metrics['stock_chunk_mode'] = 'fallback_full'
            return result

        results = [probe_full]
        for offset in range(probe_count, len(stocks), chunk_size):
            self.all_stocks = stocks[offset:offset + chunk_size]
            results.append(self.get_factor_values_unfiltered(force_recalculate=True))
        result = pd.concat(results, axis=1).sort_index().sort_index(axis=1)
        self.last_performance_metrics.update({
            'stock_chunk_mode': 'verified',
            'stock_chunk_size': chunk_size,
            'stock_chunk_seconds': time.perf_counter() - started,
        })
        return result
    finally:
        self.all_stocks = original_stocks


def _get_factor_values_from_prepared_stock_chunks(
    self,
    chunk_size: int,
    prepare_chunk: Callable[[pd.DataFrame], pd.DataFrame],
    calculate_prepared: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    mode: str = "two_stage_intraday",
    min_total_rows: int = 0,
) -> pd.DataFrame:
    """Reduce stock chunks, then calculate once if total input rows suffice.

    min_total_rows applies to all raw rows, never to an individual chunk.
    The full-calculation entry must enforce the same admission threshold.
    """
    stocks = sorted(set(self.all_stocks or []))
    if chunk_size <= 0 or len(stocks) <= chunk_size:
        return self.get_factor_values_unfiltered()

    original_stocks = self.all_stocks
    original_use_dynamic_pool = self.use_dynamic_pool
    original_dynamic_pool_mode = self.dynamic_pool_mode
    prepared_parts = []
    raw_rows = 0
    prepared_rows = 0
    peak_raw_memory_mb = 0.0
    fetch_seconds = 0.0
    prepare_seconds = 0.0
    pit_seconds = 0.0
    calculate_seconds = 0.0
    started = time.perf_counter()
    self.last_performance_metrics = {}

    try:
        self.use_dynamic_pool = False
        self.dynamic_pool_mode = "factor"
        for offset in range(0, len(stocks), chunk_size):
            self.all_stocks = stocks[offset:offset + chunk_size]
            fetch_started = time.perf_counter()
            try:
                raw_data = self._get_raw_data_for_factor()
            finally:
                fetch_seconds += time.perf_counter() - fetch_started
            if raw_data.empty:
                continue

            raw_rows += len(raw_data)
            peak_raw_memory_mb = max(
                peak_raw_memory_mb,
                float(raw_data.memory_usage(deep=True).sum() / 1024**2),
            )
            if self.pit_enabled:
                pit_started = time.perf_counter()
                try:
                    raw_data = self._apply_pit_adjustment(raw_data)
                finally:
                    pit_seconds += time.perf_counter() - pit_started

            prepare_started = time.perf_counter()
            try:
                prepared = prepare_chunk(raw_data)
            finally:
                prepare_seconds += time.perf_counter() - prepare_started
            if not prepared.empty:
                prepared_rows += len(prepared)
                prepared_parts.append(prepared)
            del raw_data, prepared
            gc.collect()

        if raw_rows < min_total_rows or not prepared_parts:
            return pd.DataFrame()
        prepared_all = pd.concat(prepared_parts, ignore_index=True)
        self.all_stocks = original_stocks
        calculate_started = time.perf_counter()
        try:
            return calculate_prepared(prepared_all)
        finally:
            calculate_seconds = time.perf_counter() - calculate_started
    finally:
        self.all_stocks = original_stocks
        self.use_dynamic_pool = original_use_dynamic_pool
        self.dynamic_pool_mode = original_dynamic_pool_mode
        self.last_performance_metrics = {
            "stock_chunk_mode": mode,
            "stock_chunk_size": chunk_size,
            "raw_rows": raw_rows,
            "prepared_rows": prepared_rows,
            "peak_chunk_raw_memory_mb": peak_raw_memory_mb,
            "raw_fetch_seconds": fetch_seconds,
            "pit_seconds": pit_seconds,
            "prepare_seconds": prepare_seconds,
            "calculate_seconds": calculate_seconds,
            "total_seconds": time.perf_counter() - started,
        }


def _get_factor_values_by_rebalance_pool(
    self,
    factor_base_cls: type,
    window_bars: int,
    warmup_calendar_days: int = 0,
    use_last_available: bool = False,
    combine_duplicate_dates: bool = False,
    rebalance_only: bool = True,
) -> pd.DataFrame:
    """Replay each PIT pool in its effective interval.

    Cloud alignment uses the T-1 row. Daily validation requires
    ``rebalance_only=False`` for every factor date in ``[key, next_key)``.
    """
    if not self.use_dynamic_pool or not self.stock_pools_by_period:
        return self.get_factor_values(force_recalculate=True)

    original_start_date = self.start_date
    original_all_stocks = self.all_stocks
    original_use_dynamic_pool = self.use_dynamic_pool
    original_dynamic_pool_mode = self.dynamic_pool_mode
    result_start = pd.to_datetime(getattr(self, "_factor_start_date", original_start_date)).normalize()
    result_end = pd.to_datetime(getattr(self, "_factor_end_date", self.end_date)).normalize()
    pool_key_end = pd.to_datetime(self.end_date).normalize()

    try:
        pool_periods = []
        for rebalance_key in sorted(self.stock_pools_by_period):
            rebalance_date = pd.to_datetime(rebalance_key, errors="coerce")
            if pd.isna(rebalance_date):
                continue
            rebalance_date = rebalance_date.normalize()
            if rebalance_date > pool_key_end:
                continue
            if rebalance_only and rebalance_date <= result_start:
                continue
            pool_stocks = [code for code in (self.stock_pools_by_period.get(rebalance_key) or []) if code]
            # Empty pools still terminate the preceding membership interval.
            pool_periods.append((rebalance_date, pool_stocks))

        if not rebalance_only:
            carry_in = [period for period in pool_periods if period[0] <= result_start]
            pool_periods = carry_in[-1:] + [period for period in pool_periods if period[0] > result_start]

        all_pool_stocks = sorted({code for _, stocks in pool_periods for code in stocks})
        if not all_pool_stocks:
            return pd.DataFrame()

        self.all_stocks = all_pool_stocks
        self.use_dynamic_pool = False
        self.dynamic_pool_mode = "factor"
        if warmup_calendar_days > 0:
            self.start_date = (
                pd.to_datetime(original_start_date)
                - pd.Timedelta(days=int(warmup_calendar_days))
            ).strftime("%Y-%m-%d")

        raw_data = self._get_raw_data_for_factor()
        if raw_data.empty:
            return pd.DataFrame()

        raw_data = raw_data.copy()
        raw_data["date"] = pd.to_datetime(raw_data["date"], errors="coerce")
        if raw_data["date"].isna().any():
            raise ValueError("rebalance raw data contains missing or invalid date values")
        if raw_data["code"].isna().any():
            raise ValueError("rebalance raw data contains missing code values")
        raw_data["code"] = raw_data["code"].astype(str)
        if raw_data["code"].str.strip().eq("").any():
            raise ValueError("rebalance raw data contains blank code values")
        raw_data = raw_data.sort_values(["date", "code"], kind="mergesort")
        available_bars = pd.DatetimeIndex(sorted(raw_data["date"].unique()))
        output_rows = []
        cumulative_pool_stocks = set()

        for period_index, (rebalance_date, pool_stocks) in enumerate(pool_periods):
            if not pool_stocks:
                continue
            cumulative_pool_stocks.update(pool_stocks)
            self.all_stocks = sorted(cumulative_pool_stocks)

            period_start = rebalance_date if rebalance_only else max(rebalance_date, result_start)
            prior_bars = available_bars[available_bars < period_start]
            if len(prior_bars) < window_bars:
                continue

            factor_date = prior_bars[-1].normalize()
            if rebalance_only and (factor_date < result_start or factor_date > result_end):
                continue
            start_bar = prior_bars[-window_bars]
            period_end = (
                pool_periods[period_index + 1][0]
                if period_index + 1 < len(pool_periods)
                else result_end + pd.Timedelta(days=1)
            )
            window_data = raw_data[
                (raw_data["date"] >= start_bar)
                & (
                    (raw_data["date"] < rebalance_date)
                    if rebalance_only
                    else (raw_data["date"] < period_end)
                )
                & raw_data["code"].isin(pool_stocks)
            ].copy()
            if window_data.empty:
                continue
            if self.pit_enabled:
                window_data = self._apply_pit_adjustment(window_data)

            factor_window = self.calculate_factor(window_data)
            factor_window = factor_base_cls._normalize_factor_dataframe(factor_window, window_data)
            if factor_window.empty:
                continue
            output_dates = factor_window.index.normalize()
            if not output_dates.is_unique:
                raise ValueError("因子输出按日期归一化后必须唯一：同一天不能有多行")
            factor_window = factor_window.set_axis(output_dates, axis=0)
            if not rebalance_only:
                period_output = factor_window.loc[
                    (factor_window.index >= period_start)
                    & (factor_window.index < period_end)
                    & (factor_window.index <= result_end),
                    factor_window.columns.intersection(pool_stocks),
                ]
                if not period_output.empty:
                    output_rows.append(period_output.reindex(columns=pool_stocks))
                continue

            output_date = factor_date
            if output_date not in factor_window.index:
                if not use_last_available:
                    continue
                available_outputs = factor_window.index[factor_window.index <= factor_date]
                if len(available_outputs) == 0:
                    continue
                output_date = available_outputs.max()

            row = factor_window.loc[output_date].reindex(pool_stocks)
            row.name = output_date
            output_rows.append(row)

        if not output_rows:
            return pd.DataFrame()

        result = pd.DataFrame(output_rows) if rebalance_only else pd.concat(output_rows, axis=0)
        if combine_duplicate_dates:
            result = result.groupby(level=0, sort=True).first()
        else:
            result = result[~result.index.duplicated(keep="last")].sort_index()
        result = result.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        return result[(result.index >= result_start) & (result.index <= result_end)]
    finally:
        self.start_date = original_start_date
        self.all_stocks = original_all_stocks
        self.use_dynamic_pool = original_use_dynamic_pool
        self.dynamic_pool_mode = original_dynamic_pool_mode


def vectorized_rolling_window(data: np.ndarray, window_size: int, func: Callable):
    """Apply a scalar callback to read-only windows of a one-dimensional array.

    The window must be a positive integer. Incomplete leading windows
    produce NaN; callback exceptions propagate to the caller.
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data must be one-dimensional")
    if (
        isinstance(window_size, (bool, np.bool_))
        or not isinstance(window_size, (int, np.integer))
        or window_size <= 0
    ):
        raise ValueError("window_size must be a positive integer")
    if len(data) < window_size:
        return np.full(len(data), np.nan)

    # Keep the NumPy API supported by the existing JoinQuant bundle.
    shape = (len(data) - window_size + 1, window_size)
    strides = (data.strides[0], data.strides[0])
    rolled = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides, writeable=False)
    results = np.array([func(window) for window in rolled])
    full_results = np.full(len(data), np.nan)
    full_results[window_size - 1:] = results
    return full_results


def optimize_dataframe_operations(self, df: pd.DataFrame) -> pd.DataFrame:
    # Preserve input precision; pandas downcast rules differ across runtimes.
    # attrs can survive reordering; always establish the actual row order.
    if 'date' in df.columns and 'code' in df.columns:
        df = df.sort_values(['code', 'date'], kind='mergesort').reset_index(drop=True)
        df.attrs['_sorted_by_code_date'] = True

    return df
