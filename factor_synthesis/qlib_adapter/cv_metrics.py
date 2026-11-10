# -*- coding: utf-8 -*-
"""
Cross-Validation Metric Evaluator

Provides custom metrics for hyperparameter search in cross-validation,
supporting IC, Top-K Return, Sharpe, and other quantitative metrics.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import logging
from scipy.stats import skew, kurtosis, norm


class CVMetricEvaluator:
    """
    Cross-validation metric evaluator for parameter search
    
    Supports metrics:
    - top_k: Mean return of top-K stocks per day
    - ic: Cross-sectional Spearman/Pearson IC
    - sharpe: Long-Short Sharpe ratio
    - topk_return: Alias for top_k
    - topk_sharpe: Sharpe ratio of top-K returns
    - r2: R-squared (sklearn default)
    
    Supports optional bottom-truncation for long-only evaluation:
    - eval_truncate_bottom_pct: float in [0, 1), removes the bottom N% of
      predictions per date before metric computation. Useful in markets
      without short selling (e.g. A-share) where bottom quantile groups
      are irrelevant to portfolio returns.
    """
    
    SUPPORTED_METRICS = {
        "top_k", "topk_return", "ic", "sharpe", "topk_sharpe", "r2",
        "deflated_sharpe", "net_topk_dsr", "gt_score"
    }
    
    def __init__(
        self,
        metric: str,
        config: dict,
        logger: Optional[logging.Logger] = None,
        n_trials: int = 1,
    ):
        """
        Initialize evaluator
        
        Args:
            metric: Metric name ("top_k", "ic", "sharpe", "topk_sharpe", "r2")
            config: Configuration dict containing:
                - top_k_size: int, number of top stocks to select (default: 10)
                - cv_weight_mode: str, "fixed_k" or "score_weighted"
                - cv_top_pct: float, top percentage for score weighting
                - target_period: int, holding period for annualization
                - eval_truncate_bottom_pct: float, fraction of bottom predictions
                  to discard per date before evaluation (default: 0.0 = no truncation)
            logger: Optional logger
            n_trials: int, number of parameter trials for multiple-testing correction
        """
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unknown metric: {metric}. "
                f"Supported: {self.SUPPORTED_METRICS}"
            )
        
        self.metric = metric
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.n_trials = max(1, int(n_trials))
        
        # Extract config values
        self.top_k = config.get("top_k_size", config.get("lgb_param_search_top_k", 10))
        self.weight_mode = config.get("cv_weight_mode", "fixed_k")
        self.top_pct = config.get("cv_top_pct", 0.2)
        self.target_period = config.get("target_period", 1)
        
        # Long-only truncation: discard bottom N% of predictions per date
        # before computing any metric. Aligns evaluation with A-share long-only
        # portfolio construction where bottom stocks are not actionable.
        self.truncate_bottom_pct = float(
            config.get("eval_truncate_bottom_pct", 0.0)
        )
        if not 0.0 <= self.truncate_bottom_pct < 1.0:
            raise ValueError(
                f"eval_truncate_bottom_pct must be in [0, 1), "
                f"got {self.truncate_bottom_pct}"
            )
    
    def _truncate_bottom(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove bottom N% of predictions per date for long-only evaluation.
        
        In markets without short selling (e.g. A-share), the bottom quantile
        groups are irrelevant to portfolio returns. Discarding them before
        metric computation focuses parameter selection on the actionable
        part of the cross-section.
        
        Theoretical support:
        - Clarke, de Silva & Thorley (2002): long-only constraints reduce
          the transfer coefficient; evaluation should reflect constraints.
        - Haitong Securities Research: A-share factors often have 70-78% of
          long-short return from the short side, which is not executable.
        
        Args:
            df: DataFrame with columns ["y_pred", "y_true", "date"]
            
        Returns:
            pd.DataFrame: Filtered DataFrame with bottom predictions removed
        """
        if self.truncate_bottom_pct <= 0:
            return df
        
        keep_threshold = df.groupby("date")["y_pred"].transform(
            lambda x: x.quantile(self.truncate_bottom_pct)
        )
        truncated = df[df["y_pred"] >= keep_threshold]
        
        if truncated.empty:
            return df  # fallback: keep all if truncation removes everything
        
        return truncated
    
    def evaluate(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        dates: Union[np.ndarray, pd.Index],
        instruments: Optional[Union[np.ndarray, pd.Index]] = None,
    ) -> float:
        """
        Calculate metric score
        
        Args:
            y_pred: Model predictions
            y_true: Actual returns/values
            dates: Date array for cross-sectional grouping
            
        Returns:
            float: Metric score (higher is better for all metrics)
        """
        vectors = (y_pred, y_true, dates)
        if instruments is not None:
            vectors += (instruments,)
        if any(np.ndim(values) != 1 for values in vectors):
            raise ValueError("predictions, labels, dates and instruments must be one-dimensional")
        if any(len(values) != len(y_pred) for values in vectors[1:]):
            raise ValueError("predictions, labels, dates and instruments must have equal length")
        if len(y_pred) == 0:
            return float("-inf")
        
        # Create DataFrame for vectorized operations
        df = pd.DataFrame({
            "y_pred": y_pred,
            "y_true": y_true,
            "date": dates,
            "instrument": instruments if instruments is not None else np.arange(len(y_pred)),
        })
        
        # Drop NaN values
        df = df.dropna()
        
        if df.empty:
            return float("-inf")
        
        # Apply long-only truncation: discard bottom N% of predictions per date
        df = self._truncate_bottom(df)
        
        if self.metric in ("top_k", "topk_return"):
            return self._compute_top_k_return(df)
        elif self.metric == "ic":
            return self._compute_ic(df)
        elif self.metric == "sharpe":
            return self._compute_sharpe(df)
        elif self.metric == "topk_sharpe":
            return self._compute_topk_sharpe(df)
        elif self.metric == "deflated_sharpe":
            return self._compute_deflated_sharpe(df)
        elif self.metric == "net_topk_dsr":
            return self._compute_net_topk_dsr(df)
        elif self.metric == "gt_score":
            return self._compute_gt_score(df)
        elif self.metric == "r2":
            return self._compute_r2(df)
        else:
            raise ValueError(f"Metric not implemented: {self.metric}")
    
    def _compute_top_k_return(self, df: pd.DataFrame) -> float:
        """
        Compute mean return of top-K stocks per day
        
        Args:
            df: DataFrame with columns ["y_pred", "y_true", "date"]
            
        Returns:
            float: Mean daily top-K return
        """
        if self.weight_mode == "score_weighted":
            # Score weighting: weight proportional to excess score over threshold
            threshold = df.groupby("date")["y_pred"].transform(
                lambda x: x.quantile(1 - self.top_pct)
            )
            excess = (df["y_pred"] - threshold).clip(lower=0)
            total_excess = excess.groupby(df["date"]).transform("sum")
            
            # Avoid division by zero
            weights = np.where(total_excess > 1e-8, excess / total_excess, 0.0)
            
            # Weighted return per day
            daily_means = (df["y_true"] * weights).groupby(df["date"]).sum()
        else:
            # Fixed top-K (equal weight)
            rank_mask = df.groupby("date")["y_pred"].rank(
                method="first", ascending=False
            ) <= self.top_k
            
            top_k_df = df[rank_mask]
            daily_means = top_k_df.groupby("date")["y_true"].mean()
        
        if daily_means.empty:
            return float("-inf")
        
        return float(daily_means.mean())
    
    def _compute_ic(self, df: pd.DataFrame) -> float:
        """
        Compute cross-sectional Spearman IC
        
        Args:
            df: DataFrame with columns ["y_pred", "y_true", "date"]
            
        Returns:
            float: Mean daily IC
        """
        def calc_spearman_ic(group):
            if len(group) < 3:
                return np.nan
            rank_true = group["y_true"].rank()
            rank_pred = group["y_pred"].rank()
            if rank_true.std() < 1e-8 or rank_pred.std() < 1e-8:
                return np.nan
            return rank_true.corr(rank_pred)
        
        daily_ics = df.groupby("date").apply(calc_spearman_ic)
        daily_ics = daily_ics.dropna()
        
        if daily_ics.empty:
            return float("-inf")
        
        return float(daily_ics.mean())
    
    def _compute_long_short_daily_diff(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Compute daily long-short returns based on top-K and bottom-K ranking.
        """
        pred_ranks = df.groupby("date")["y_pred"].rank(method="first", ascending=False)
        counts = df.groupby("date")["y_pred"].transform("count")
        
        long_mask = pred_ranks <= self.top_k
        short_mask = pred_ranks > (counts - self.top_k)
        
        long_rets = df[long_mask].groupby("date")["y_true"].mean()
        short_rets = df[short_mask].groupby("date")["y_true"].mean()
        
        combined = pd.concat([long_rets, short_rets], axis=1, keys=["long", "short"])
        combined = combined.dropna()
        if combined.empty:
            return None
        
        return combined["long"] - combined["short"]
    
    def _compute_sharpe(self, df: pd.DataFrame) -> float:
        """
        Compute long-short Sharpe ratio
        
        Long: Top-K stocks
        Short: Bottom-K stocks
        
        Args:
            df: DataFrame with columns ["y_pred", "y_true", "date"]
            
        Returns:
            float: Annualized Sharpe ratio
        """
        daily_diff = self._compute_long_short_daily_diff(df)
        if daily_diff is None or len(daily_diff) < 2:
            return float("-inf")
        
        ret_mean = daily_diff.mean()
        ret_std = daily_diff.std()
        
        if ret_std < 1e-8:
            return 0.0
        
        # Annualize with adjustment for target period
        ann_factor = np.sqrt(252.0 / max(1, self.target_period))
        sharpe = (ret_mean / ret_std) * ann_factor
        
        return float(sharpe)
    
    def _compute_topk_sharpe(self, df: pd.DataFrame) -> float:
        """
        Compute Sharpe ratio of top-K portfolio (long only)
        
        Args:
            df: DataFrame with columns ["y_pred", "y_true", "date"]
            
        Returns:
            float: Annualized Sharpe ratio of top-K returns
        """
        # Get top-K stocks per day
        rank_mask = df.groupby("date")["y_pred"].rank(
            method="first", ascending=False
        ) <= self.top_k
        
        top_k_df = df[rank_mask]
        daily_returns = top_k_df.groupby("date")["y_true"].mean()
        
        if len(daily_returns) < 2:
            return float("-inf")
        
        ret_mean = daily_returns.mean()
        ret_std = daily_returns.std()
        
        if ret_std < 1e-8:
            return 0.0
        
        # Annualize
        ann_factor = np.sqrt(252.0 / max(1, self.target_period))
        sharpe = (ret_mean / ret_std) * ann_factor
        
        return float(sharpe)

    def effective_period_count(self, dates: Union[np.ndarray, pd.Index]) -> int:
        """Count non-overlapping portfolio observations used by this metric."""
        unique_dates = pd.Index(dates).drop_duplicates().sort_values()
        if self.metric != "net_topk_dsr":
            return int(len(unique_dates))
        stride = max(1, int(self.config.get("cv_rebalance_stride", self.target_period)))
        return int(len(unique_dates[::stride]))

    def _compute_net_topk_dsr(self, df: pd.DataFrame) -> float:
        """Deflated Sharpe of non-overlapping Top-K returns after turnover costs."""
        stride = max(1, int(self.config.get("cv_rebalance_stride", self.target_period)))
        cost_rate = max(0.0, float(self.config.get("cv_cost_rate", 0.0)))
        dates = pd.Index(df["date"].drop_duplicates()).sort_values()[::stride]
        previous: dict[object, float] = {}
        net_returns: list[float] = []
        for day in dates:
            group = df[df["date"] == day].nlargest(self.top_k, "y_pred")
            if group.empty:
                continue
            weight = 1.0 / len(group)
            current = {instrument: weight for instrument in group["instrument"]}
            turnover = sum(
                abs(current.get(index, 0.0) - previous.get(index, 0.0))
                for index in set(current) | set(previous)
            )
            net_returns.append(float(group["y_true"].mean()) - cost_rate * turnover)
            previous = current
        return self._deflated_sharpe_from_returns(pd.Series(net_returns, dtype=float))
    
    def _compute_deflated_sharpe(self, df: pd.DataFrame) -> float:
        """
        Compute the existing approximate DSR ranking score.

        Finite scores lie in [0, 1]; they are not calibrated confidence
        probabilities. See _deflated_sharpe_from_returns for the approximation
        and its small-sample fallbacks.
        """
        daily_diff = self._compute_long_short_daily_diff(df)
        if daily_diff is None or len(daily_diff) < 2:
            return float("-inf")
        
        return self._deflated_sharpe_from_returns(daily_diff)

    def _deflated_sharpe_from_returns(self, returns: pd.Series) -> float:
        """Return the existing approximate DSR ranking score, not a calibrated probability.

        For N=n_trials and T=n_obs, the reference threshold is zero for N<=1,
        otherwise norm.ppf(1 - 1/N) / sqrt(T). Thus N=2 also has zero threshold
        and no additional penalty over one trial. With s=mean/std (unannualized),
        the variance approximation is (1 + 0.5*s**2 - skew*s +
        (kurtosis - 3)*s**2/4) / T: the denominator is n_obs, not n_obs-1.
        Nonfinite variance or variance below 1e-8 is set to 1e-8, so short
        samples may saturate at 1; that is not 100% statistical confidence.

        Existing fallbacks remain: fewer than two returns or a nonfinite final
        score yields -inf; standard deviation below 1e-8 yields zero. This
        approximation and configuration are retained for compatibility. A
        paper-based redesign requires a separate versioned proposal; historical
        results are not recomputed or rewritten.
        """
        eps = 1e-8
        if len(returns) < 2:
            return float("-inf")
        ret_mean = returns.mean()
        ret_std = returns.std()
        if ret_std < eps:
            return 0.0
        
        # DSR's sampling-variance formula uses the per-observation Sharpe.
        # Annualising it here can make the variance negative on short,
        # skewed validation folds.
        sharpe_observed = ret_mean / ret_std
        
        n_obs = int(len(returns))
        if n_obs < 2:
            return float("-inf")
        
        skewness = float(skew(returns, bias=False)) if n_obs > 2 else 0.0
        kurt = float(kurtosis(returns, fisher=False, bias=False)) if n_obs > 3 else 3.0
        
        if self.n_trials <= 1:
            e_max_sharpe = 0.0
        else:
            e_max_sharpe = norm.ppf(1 - 1 / self.n_trials) * np.sqrt(1.0 / n_obs)
        
        var_sharpe = (
            1
            + 0.5 * sharpe_observed ** 2
            - skewness * sharpe_observed
            + (kurt - 3) / 4 * sharpe_observed ** 2
        ) / n_obs
        
        if not np.isfinite(var_sharpe) or var_sharpe < eps:
            var_sharpe = eps
        
        dsr = norm.cdf((sharpe_observed - e_max_sharpe) / np.sqrt(var_sharpe))
        return float(dsr) if np.isfinite(dsr) else float("-inf")
    
    def _compute_gt_score(self, df: pd.DataFrame) -> float:
        """
        Compute GT-Score composite metric.
        """
        daily_diff = self._compute_long_short_daily_diff(df)
        if daily_diff is None or len(daily_diff) < 2:
            return float("-inf")
        
        eps = 1e-8
        ret_mean = daily_diff.mean()
        ret_std = daily_diff.std()
        n_obs = int(len(daily_diff))
        
        ann_factor = np.sqrt(252.0 / max(1, self.target_period))
        sharpe = (ret_mean / (ret_std + eps)) * ann_factor
        
        t_stat = ret_mean / (ret_std / np.sqrt(n_obs) + eps)
        t_component = min(t_stat / 2.0, 1.0)
        
        consistency = float((daily_diff > 0).mean())
        downside = daily_diff[daily_diff < 0]
        if len(downside) > 1:
            downside_std = downside.std()
            if downside_std < eps:
                sortino = sharpe
            else:
                sortino = (ret_mean / downside_std) * ann_factor
        else:
            sortino = sharpe
        
        gt_score = 0.3 * sharpe + 0.3 * t_component + 0.2 * consistency + 0.2 * sortino
        return float(gt_score)
    
    def _compute_r2(self, df: pd.DataFrame) -> float:
        """
        Compute R-squared
        
        Args:
            df: DataFrame with columns ["y_pred", "y_true", "date"]
            
        Returns:
            float: R-squared score
        """
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot < 1e-8:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return float(r2)
