# -*- coding: utf-8 -*-
"""
Qlib兼容的线性模型包装器

封装LASSO/ElasticNet，提供与Qlib接口兼容的调用方式。
支持自定义CV指标 (IC, Top-K, Sharpe) 进行参数搜索。
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from contextlib import nullcontext
import pandas as pd
import numpy as np
import logging
import random

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from sklearn.linear_model import (
    Lasso, LassoCV, ElasticNet, ElasticNetCV, Ridge, RidgeCV
)
# StandardScaler removed - data is already rank_zscore normalized
# TimeSeriesSplit replaced by PanelPurgedKFold for panel-aware purged CV

from .quantile_elasticnet import QuantileElasticNet, QuantileElasticNetCV

from .data_converter import FactorPilotDataset
from .cv_metrics import CVMetricEvaluator
from .panel_purged_kfold import (
    PanelPurgedKFold, LabelIsolationError, resolve_label_exit_dates,
)


# ==================== 全局随机种子固定 ====================
def set_random_seed(seed: int = 42):
    """
    固定所有随机种子以确保实验可复现
    
    Args:
        seed: int, 随机种子值（默认42）
    """
    random.seed(seed)
    np.random.seed(seed)
    # 如果使用 torch（未来可能），也固定 torch 的种子
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

# 在模块加载时自动固定种子
set_random_seed(42)
# ==========================================================


def _fit_linear_estimator(model, X, y, *, sample_weight=None):
    """Bound coordinate-descent BLAS overhead without changing solver settings."""
    # Lasso inherits ElasticNet; LassoCV and ElasticNetCV share the same solver.
    if isinstance(model, (ElasticNet, ElasticNetCV, LassoCV)):
        with threadpool_limits(limits=1, user_api="blas"):
            return model.fit(X, y, sample_weight=sample_weight)
    return model.fit(X, y, sample_weight=sample_weight)


def _resolve_alpha_grid(cfg: dict, logger: logging.Logger) -> np.ndarray:
    """
    Resolve alpha grid from config.
    If lasso_alpha_range has exactly 2 elements [min, max], generates a log-spaced
    grid using n_alphas. Otherwise uses the list directly (explicit alpha list).
    """
    raw = cfg.get('lasso_alpha_range', None)
    n_alphas = int(cfg.get('n_alphas', 50))

    if raw is None:
        return np.logspace(-5, 0, n_alphas)

    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 0:
        return np.array([float(arr)])

    # [min, max] -> log-spaced grid
    if len(arr) == 2:
        lo, hi = float(arr[0]), float(arr[1])
        if lo <= 0 or hi <= 0 or lo >= hi:
            logger.warning(
                "[AlphaGrid] Invalid range [%.2e, %.2e], fallback to logspace(-5, 0, %d)",
                lo, hi, n_alphas,
            )
            return np.logspace(-5, 0, n_alphas)
        grid = np.logspace(np.log10(lo), np.log10(hi), n_alphas)
        logger.info(
            "[AlphaGrid] Generated %d alphas in [%.2e, %.2e] (log-spaced)",
            n_alphas, grid[0], grid[-1],
        )
        return grid

    # Explicit list of alphas
    logger.info("[AlphaGrid] Using explicit %d alpha values from config", len(arr))
    return arr


def _fitted_regularization_param(model: Any, name: str, default: float) -> float:
    """Read a fitted regularization parameter across sklearn estimator families.

    CV estimators expose ``alpha_``/``l1_ratio_``; the final refit estimators
    expose the configured values as ``alpha``/``l1_ratio``.  Reading only the
    underscored attributes silently recorded zero/one after a rerank refit.
    """
    for attr in (f"{name}_", name):
        value = getattr(model, attr, None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return float(default)


class QlibLinearWrapper:
    """
    Qlib兼容的线性模型包装器
    
    支持LASSO, ElasticNet, Ridge等线性模型。
    支持自定义CV指标进行参数搜索 (IC, Top-K, Sharpe)。
    """
    
    # Metrics requiring a custom CV loop (cannot use sklearn's internal MSE path).
    # Note: 'r2' and 'gt_score' are intentionally EXCLUDED.
    # They route to _fit_with_sklearn_cv which already uses PanelPurgedKFold
    # pre-computed splits AND sklearn's warm-start coordinate descent — more
    # numerically stable and efficient than independent per-alpha training.
    CUSTOM_CV_METRICS = {
        'ic', 'top_k', 'topk_return', 'sharpe', 'topk_sharpe',
        'deflated_sharpe', 'net_topk_dsr',
    }
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        """
        初始化模型

        Args:
            config: dict, 来自 synthesis_config (如 "lasso_comprehensive")
                - model_type: str ("lasso" 或 "elasticnet" 或 "ridge")
                - lasso_alpha_range: List[float]
                - l1_ratio_range: List[float] (仅elasticnet)
                - lasso_cv_folds: int
                - lasso_max_iter: int
                - param_search_metric: str ("r2", "ic", "top_k", "sharpe", "topk_sharpe")
                - top_k_size: int, number of top stocks for Top-K metrics
                - positive: bool
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        # self.scaler removed - data is already rank_zscore normalized
        self.feature_names: List[str] = []
        
        # 提取模型类型
        self.model_type = config.get('model_type', 'lasso').lower()
        self.use_cv = config.get('lasso_use_cv', True)
        self.param_search_metric = config.get('param_search_metric', 'r2')
        self.quantile_tau = config.get('quantile_tau', 0.8)
        
    def fit(
        self,
        dataset: FactorPilotDataset,
        sample_weight: Optional[np.ndarray] = None
    ) -> dict:
        """
        训练模型
        
        Args:
            dataset: FactorPilotDataset
            sample_weight: Optional[np.ndarray], 样本权重 (已弃用，内部重新计算)
            
        Returns:
            model_info: dict
                - method: str = "lasso" 或 "elasticnet"
                - alpha: float, 最优正则化系数
                - l1_ratio: float
                - coefficients: Dict[str, float]
                - selected_features: List[str], 非零系数的因子
                - r2_score: float
        """
        self.logger.info(f"[QlibLinearWrapper] 开始训练 {self.model_type} 模型...")
        
        # Stable-support mode owns its inner selection and therefore refits on
        # the complete rolling outer-train (the existing train+valid span).
        stability_cfg = self.config.get("support_stability", {})
        stable_mode = bool(stability_cfg.get("enabled", False))
        if stable_mode:
            start = pd.Timestamp(dataset.segments["train"][0])
            end = pd.Timestamp(dataset.segments["valid"][1])
            dates = dataset.feature_df.index.get_level_values("datetime")
            mask = (dates >= start) & (dates <= end)
            if not dataset.feature_df.index.is_unique or not dataset.label_df.index.is_unique:
                raise ValueError("Feature and label indices must be unique for label requests")
            selected_index = dataset.feature_df.index[mask]
            if not selected_index.isin(dataset.label_df.index).all():
                raise ValueError("Selected feature keys are missing from labels")
            df_train = pd.concat(
                {
                    "feature": dataset.feature_df.loc[mask],
                    "label": dataset.label_df.loc[selected_index],
                },
                axis=1,
            )
        else:
            df_train = dataset.prepare("train", col_set=["feature", "label"])
        
        X_train = df_train["feature"]
        y_train = df_train["label"]
        
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        
        self.feature_names = list(X_train.columns)
        
        self.logger.info(
            f"[QlibLinearWrapper] 训练数据: {len(X_train)} 样本, "
            f"{len(self.feature_names)} 特征"
        )
        
        # 不再使用 StandardScaler - 数据已通过 rank_zscore 标准化
        # 这与云端逻辑一致：rank_zscore 后直接用于训练
        X_values = X_train.values
        y_values = y_train.values
        
        # 处理NaN
        valid_mask = ~(np.isnan(X_values).any(axis=1) | np.isnan(y_values))
        X_values = X_values[valid_mask]
        y_values = y_values[valid_mask]
        
        # ---- Sample weight computation ----
        # Deferred import to avoid circular: qlib_adapter <-> pipeline
        from factor_synthesis.pipeline.sample_weighting import (
            compute_return_based_weights,
            combine_weights,
        )

        # Step 1: Time-decay weights (existing behaviour)
        valid_train_index = X_train.index[valid_mask]
        time_weights: np.ndarray | None = None
        weight_decay = self.config.get("sample_weight_decay", 1.0)
        if weight_decay < 1.0:
            try:
                train_dates = pd.to_datetime(valid_train_index.get_level_values("datetime"))
                max_date = train_dates.max()
                days_diff = np.array([(max_date - d).days for d in train_dates], dtype=float)
                time_weights = np.power(weight_decay, days_diff)
                time_weights = time_weights / time_weights.mean()
                self.logger.info(f"[QlibLinearWrapper] 应用时间衰减样本权重 (decay={weight_decay})")
            except LabelIsolationError:
                raise
            except Exception as e:
                self.logger.warning(f"[QlibLinearWrapper] 计算时间衰减权重失败: {e}")
                time_weights = None

        # Step 2: Return-based weights (from raw returns, not preprocessed labels)
        # Preprocessed labels (cs_rank + cs_demean) are demeaned ranks, which
        # distort magnitude-based weighting methods. Use raw returns instead.
        return_weights: np.ndarray | None = None
        raw_ret = getattr(dataset, "raw_returns", None)
        metric_y: np.ndarray | None = None
        baseline_y: np.ndarray | None = None
        try:
            if raw_ret is not None:
                y_for_weights = raw_ret.reindex(valid_train_index).values
                if np.isnan(y_for_weights).any():
                    self.logger.warning(
                        "[QlibLinearWrapper] raw_returns contains NaN, "
                        "falling back to preprocessed labels"
                    )
                    y_for_weights = y_values
            else:
                y_for_weights = y_values

            return_weights = compute_return_based_weights(
                y_for_weights, valid_train_index, self.config, log=self.logger,
            )
        except Exception as e:
            self.logger.warning(f"[QlibLinearWrapper] 计算收益率权重失败: {e}")
            raise

        # Step 3: Combine multiplicatively (configurable)
        sw_cfg = self.config.get("sample_weight_config", {})
        do_combine = sw_cfg.get("combine_with_time_decay", True)
        sample_weight = combine_weights(time_weights, return_weights, combine=do_combine)

        if self.param_search_metric in {"deflated_sharpe", "net_topk_dsr"} and raw_ret is not None:
            candidate = np.asarray(
                raw_ret.reindex(valid_train_index).values, dtype=float
            )
            if np.isfinite(candidate).all():
                metric_y = candidate
            else:
                self.logger.warning(
                    "[QlibLinearWrapper] DSR raw returns unavailable; "
                    "falling back to processed labels"
                )
        baseline_scores = getattr(dataset, "baseline_scores", None)
        if baseline_scores is not None:
            candidate = np.asarray(
                baseline_scores.reindex(valid_train_index).values, dtype=float
            )
            if np.isfinite(candidate).all():
                baseline_y = candidate
        
        self.logger.info(f"[QlibLinearWrapper] 有效样本数: {len(X_values)}")
        
        # 根据配置选择训练方法
        X_index = X_train.index[valid_mask]
        if stable_mode:
            model_info = self._fit_with_stable_support(
                X_values,
                y_values,
                X_index,
                sample_weight,
                metric_y=metric_y,
                baseline_y=baseline_y,
            )
        elif self.param_search_metric in self.CUSTOM_CV_METRICS and self.use_cv:
            # 使用自定义指标进行交叉验证 (PanelPurgedKFold)
            model_info = self._fit_with_custom_cv(
                X_values,
                y_values,
                X_index,
                sample_weight,
                metric_y=metric_y,
                baseline_y=baseline_y,
            )
        else:
            # 使用sklearn内置交叉验证 (PanelPurgedKFold pre-computed splits)
            model_info = self._fit_with_sklearn_cv(
                X_values, y_values, X_index, sample_weight
            )
        model_info["train_r2"] = model_info.get("r2_score")
        model_info["validation_r2"] = (
            None if stable_mode else self._segment_r2(dataset, "valid")
        )
        return model_info

    def _ordered_bagging_members(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_index: pd.MultiIndex,
        sample_weight: Optional[np.ndarray],
        *,
        pair_count: int,
        block_days: int,
        target: int,
        l1_ratio: float,
        alphas: np.ndarray,
        seed: int,
        require_valid: bool,
    ) -> dict:
        dates = pd.DatetimeIndex(X_index.get_level_values(0))
        unique_dates = dates.unique().sort_values()
        blocks = [
            unique_dates[start : start + block_days]
            for start in range(0, len(unique_dates), block_days)
        ]
        if len(blocks) < 2:
            raise ValueError("support_stability requires at least two date blocks")

        all_names = list(self.feature_names)
        support_hits = np.zeros(X.shape[1], dtype=int)
        coefficient_sum = np.zeros(X.shape[1], dtype=float)
        coefficients: List[np.ndarray] = []
        intercepts: List[float] = []
        members: list[dict] = []
        last_model: ElasticNet | None = None
        rng = np.random.default_rng(seed)
        for _ in range(pair_count):
            order = rng.permutation(len(blocks))
            split = len(order) // 2
            for block_ids in (order[:split], order[split:]):
                selected_dates = np.concatenate(
                    [blocks[int(block_id)].to_numpy() for block_id in block_ids]
                )
                rows = dates.isin(selected_dates)
                if not rows.any():
                    raise ValueError("support_stability produced an empty half-sample")
                model = ElasticNet(
                    alpha=float(alphas[0]),
                    l1_ratio=l1_ratio,
                    positive=bool(self.config.get("positive", False)),
                    max_iter=int(self.config.get("lasso_max_iter", 10000)),
                    tol=float(self.config.get("lasso_tol", 1e-4)),
                    random_state=seed,
                    warm_start=True,
                )
                best: tuple[int, float, np.ndarray, float, int] | None = None
                weights = sample_weight[rows] if sample_weight is not None else None
                for alpha in alphas:
                    model.alpha = float(alpha)
                    _fit_linear_estimator(model, X[rows], y[rows], sample_weight=weights)
                    fitted = np.asarray(model.coef_, dtype=float).copy()
                    count = int((np.abs(fitted) > 1e-10).sum())
                    candidate = (
                        abs(count - target),
                        -float(alpha),
                        fitted,
                        float(model.intercept_),
                        int(model.n_iter_),
                    )
                    if best is None or candidate[:2] < best[:2]:
                        best = candidate
                assert best is not None
                selected = np.abs(best[2]) > 1e-10
                if require_valid:
                    if not selected.any() or not (
                        np.isfinite(best[2]).all() and np.isfinite(best[3])
                    ):
                        raise ValueError("support_stability produced an invalid member")
                    if best[4] >= int(self.config.get("lasso_max_iter", 10000)):
                        raise ValueError("support_stability member did not converge")
                support_hits += selected
                coefficient_sum += np.abs(best[2])
                coefficients.append(best[2])
                intercepts.append(best[3])
                members.append(
                    {
                        "alpha": -best[1],
                        "coefficients": {
                            all_names[idx]: float(best[2][idx])
                            for idx in np.flatnonzero(selected)
                        },
                        "active_support": [
                            all_names[idx] for idx in np.flatnonzero(selected)
                        ],
                        "intercept": best[3],
                        "n_iter": best[4],
                    }
                )
                last_model = model
        return {
            "members": members,
            "coefficients": np.asarray(coefficients),
            "intercepts": np.asarray(intercepts),
            "support_hits": support_hits,
            "coefficient_sum": coefficient_sum,
            "last_model": last_model,
            "block_count": len(blocks),
            "unique_dates": unique_dates,
        }

    def _select_nested_pair_count(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_index: pd.MultiIndex,
        sample_weight: Optional[np.ndarray],
        metric_y: np.ndarray,
        baseline_y: np.ndarray,
        *,
        pair_cells: List[int],
        block_days: int,
        target: int,
        l1_ratio: float,
        alphas: np.ndarray,
        seed: int,
        gamma: float,
    ) -> tuple[int, dict]:
        if pair_cells != sorted(set(pair_cells)) or not 3 <= len(pair_cells) <= 6:
            raise ValueError("nested_complementary_pairs must be 3 to 6 increasing cells")
        if pair_cells[0] < 1:
            raise ValueError("nested_complementary_pairs must be positive")
        cv = PanelPurgedKFold(
            n_splits=int(self.config.get("lasso_cv_folds", 5)),
            embargo_td=int(self.config.get("target_period", 5)),
        )
        dates = X_index.get_level_values(0).values
        instruments = X_index.get_level_values(1).values
        label_exits = resolve_label_exit_dates(
            X_index, self.config.get("_label_exit_dates"),
            trading_calendar=self.config.get("_label_trading_calendar"),
            label_period=self.config.get("_label_effective_period"),
            calendar_source=self.config.get("_label_calendar_source"),
        )
        splits = list(cv.split(X, groups=dates, label_exit_dates=label_exits))
        chunks: Dict[int, list] = {cell: [] for cell in pair_cells}
        fold_audit = []
        for train_idx, valid_idx in splits:
            sequence = self._ordered_bagging_members(
                X[train_idx],
                y[train_idx],
                X_index[train_idx],
                sample_weight[train_idx] if sample_weight is not None else None,
                pair_count=pair_cells[-1],
                block_days=block_days,
                target=target,
                l1_ratio=l1_ratio,
                alphas=alphas,
                seed=seed,
                require_valid=True,
            )
            fold_audit.append(
                {
                    "train_end": str(pd.Timestamp(dates[train_idx].max()).date()),
                    "validation_start": str(
                        pd.Timestamp(dates[valid_idx].min()).date()
                    ),
                }
            )
            for cell in pair_cells:
                member_count = 2 * cell
                coefficients = sequence["coefficients"][:member_count].mean(axis=0)
                intercept = float(sequence["intercepts"][:member_count].mean())
                prediction = X[valid_idx] @ coefficients + intercept
                chunks[cell].append(
                    (
                        prediction,
                        metric_y[valid_idx],
                        dates[valid_idx],
                        baseline_y[valid_idx],
                        instruments[valid_idx],
                    )
                )

        evaluator = CVMetricEvaluator(
            "net_topk_dsr", self.config, self.logger, n_trials=len(pair_cells)
        )
        scores = {}
        for cell in pair_cells:
            prediction = np.concatenate([part[0] for part in chunks[cell]])
            realized = np.concatenate([part[1] for part in chunks[cell]])
            oof_dates = np.concatenate([part[2] for part in chunks[cell]])
            baseline = np.concatenate([part[3] for part in chunks[cell]])
            oof_instruments = np.concatenate([part[4] for part in chunks[cell]])
            frame = pd.DataFrame({"prediction": prediction, "date": oof_dates})
            residual = frame.groupby("date")["prediction"].transform(
                lambda values: (values - values.mean()) / (values.std() + 1e-8)
            ).fillna(0.0).to_numpy()
            score = evaluator.evaluate(
                baseline + gamma * residual,
                realized,
                oof_dates,
                oof_instruments,
            )
            if np.isfinite(score):
                scores[cell] = float(score)
        if not scores:
            raise ValueError("nested subagging produced no finite inner OOF score")
        selected = max(pair_cells, key=lambda cell: scores.get(cell, -np.inf))
        return selected, {
            "candidate_pairs": pair_cells,
            "candidate_member_counts": [2 * cell for cell in pair_cells],
            "max_pair_count": pair_cells[-1],
            "ordered_members_fit_per_fold": 2 * pair_cells[-1],
            "selected_pairs": selected,
            "scores": {str(key): value for key, value in scores.items()},
            "folds": fold_audit,
            "outer_model_count": 1,
        }

    def _fit_with_stable_support(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_index: pd.MultiIndex,
        sample_weight: Optional[np.ndarray],
        metric_y: Optional[np.ndarray] = None,
        baseline_y: Optional[np.ndarray] = None,
    ) -> dict:
        """Select a compact ElasticNet support from complementary date blocks."""
        cfg = self.config
        stable = cfg["support_stability"]
        if self.model_type != "elasticnet":
            raise ValueError("support_stability requires model_type=elasticnet")
        l1_values = [float(value) for value in cfg.get("l1_ratio_range", [0.3])]
        if len(l1_values) != 1:
            raise ValueError("support_stability requires one frozen l1_ratio")

        alphas = sorted(_resolve_alpha_grid(cfg, self.logger), reverse=True)
        aggregation = stable.get("aggregation", "stable_support")
        if aggregation not in {"stable_support", "equal_weight_bagging"}:
            raise ValueError(f"unsupported support_stability aggregation: {aggregation}")
        target = int(stable.get("support_target", 10))
        minimum = int(stable.get("minimum_support", 3))
        maximum = int(stable.get("maximum_support", 11))
        frequency_threshold = float(stable.get("selection_frequency", 0.8))
        pair_count = int(stable.get("complementary_pairs", 12))
        block_days = int(stable.get("block_trading_days", 20))
        seed = int(stable.get("random_seed", 42))
        if aggregation == "stable_support" and (
            not 0 < frequency_threshold <= 1 or not 1 <= minimum <= maximum
        ):
            raise ValueError("invalid support_stability bounds")

        all_names = list(self.feature_names)
        nested_audit = None
        pair_cells = stable.get("nested_complementary_pairs") or []
        gamma = float(stable.get("residual_gamma", 0.1))
        if pair_cells:
            if aggregation != "equal_weight_bagging":
                raise ValueError(
                    "nested_complementary_pairs requires equal_weight_bagging"
                )
            if metric_y is None or baseline_y is None:
                raise ValueError(
                    "nested subagging requires finite raw returns and baseline scores"
                )
            pair_cells = [int(value) for value in pair_cells]
            if pair_count != max(pair_cells):
                raise ValueError(
                    "complementary_pairs must equal max nested_complementary_pairs"
                )
            pair_count, nested_audit = self._select_nested_pair_count(
                X,
                y,
                X_index,
                sample_weight,
                np.asarray(metric_y, dtype=float),
                np.asarray(baseline_y, dtype=float),
                pair_cells=pair_cells,
                block_days=block_days,
                target=target,
                l1_ratio=l1_values[0],
                alphas=np.asarray(alphas),
                seed=seed,
                gamma=gamma,
            )

        sequence = self._ordered_bagging_members(
            X,
            y,
            X_index,
            sample_weight,
            pair_count=pair_count,
            block_days=block_days,
            target=target,
            l1_ratio=l1_values[0],
            alphas=np.asarray(alphas),
            seed=seed,
            require_valid=aggregation == "equal_weight_bagging",
        )
        members = sequence["members"]
        support_hits = sequence["support_hits"]
        coefficient_sum = sequence["coefficient_sum"]
        last_model = sequence["last_model"]
        unique_dates = sequence["unique_dates"]
        block_count = sequence["block_count"]
        sample_total = len(members)

        if aggregation == "equal_weight_bagging":
            assert last_model is not None
            mean_coefficients = sequence["coefficients"].mean(axis=0)
            last_model.coef_ = mean_coefficients
            last_model.intercept_ = float(sequence["intercepts"].mean())
            last_model.alpha = float(np.mean([member["alpha"] for member in members]))
            self.model = last_model
            self.feature_names = all_names
            coefficients = {
                name: float(value)
                for name, value in zip(all_names, mean_coefficients)
            }
            selected_features = [
                name for name, value in coefficients.items() if abs(value) > 1e-10
            ]
            model_info = {
                "method": self.model_type,
                "alpha": float(self.model.alpha),
                "l1_ratio": l1_values[0],
                "coefficients": coefficients,
                "selected_features": selected_features,
                "n_selected": len(selected_features),
                "n_features": len(all_names),
                "r2_score": float(self.model.score(X, y)),
                "feature_names": self.feature_names,
                "cv_score": None,
                "cv_metric": "complementary_block_equal_weight_bagging",
                "cv_candidate_count": len(alphas),
                "cv_folds": 0,
                "cv_factor_count": target,
                "factor_count_grid": [target],
                "selected_residual_gamma": gamma,
                "selected_blend_ratio": 1.0 / (1.0 + gamma),
                "support_stability": {
                    "aggregation": aggregation,
                    "block_trading_days": block_days,
                    "block_count": block_count,
                    "complementary_pairs": pair_count,
                    "bootstrap_samples": sample_total,
                    "support_target": target,
                    "members": members,
                    "mean_coefficients": coefficients,
                    "selection_counts": {
                        name: int(support_hits[idx])
                        for idx, name in enumerate(all_names)
                    },
                    "fallback_used": False,
                    "train_start": str(unique_dates.min().date()),
                    "train_end": str(unique_dates.max().date()),
                },
            }
            if nested_audit is not None:
                model_info["nested_subagging_cv"] = nested_audit
                model_info["cv_score"] = nested_audit["scores"][str(pair_count)]
                model_info["cv_metric"] = "nested_complementary_block_net_topk_dsr"
                model_info["cv_candidate_count"] = len(pair_cells)
                model_info["cv_folds"] = len(nested_audit["folds"])
            return model_info

        frequencies = support_hits / sample_total
        ranked = sorted(
            range(len(all_names)),
            key=lambda idx: (-frequencies[idx], -coefficient_sum[idx], all_names[idx]),
        )
        retained = [idx for idx in ranked if frequencies[idx] >= frequency_threshold][
            :maximum
        ]
        if len(retained) < minimum:
            raise ValueError(
                f"support_stability retained {len(retained)} factors; minimum is {minimum}"
            )

        final_target = min(target, len(retained))
        final_model: ElasticNet | None = None
        final_choice: tuple[int, float] | None = None
        for alpha in alphas:
            candidate_model = ElasticNet(
                alpha=float(alpha),
                l1_ratio=l1_values[0],
                positive=bool(cfg.get("positive", False)),
                max_iter=int(cfg.get("lasso_max_iter", 10000)),
                tol=float(cfg.get("lasso_tol", 1e-4)),
                random_state=seed,
            )
            _fit_linear_estimator(
                candidate_model, X[:, retained], y, sample_weight=sample_weight
            )
            count = int((np.abs(candidate_model.coef_) > 1e-10).sum())
            choice = (abs(count - final_target), -float(alpha))
            if final_choice is None or choice < final_choice:
                final_choice = choice
                final_model = candidate_model
        assert final_model is not None

        self.feature_names = [all_names[idx] for idx in retained]
        self.model = final_model
        coefficients = {
            name: float(value)
            for name, value in zip(self.feature_names, self.model.coef_)
        }
        selected_features = [
            name for name, value in coefficients.items() if abs(value) > 1e-10
        ]
        if len(selected_features) < minimum:
            raise ValueError(
                f"support_stability final refit retained {len(selected_features)} factors; "
                f"minimum is {minimum}"
            )
        return {
            "method": self.model_type,
            "alpha": float(self.model.alpha),
            "l1_ratio": l1_values[0],
            "coefficients": coefficients,
            "selected_features": selected_features,
            "n_selected": len(selected_features),
            "n_features": len(all_names),
            "r2_score": float(self.model.score(X[:, retained], y)),
            "feature_names": self.feature_names,
            "cv_score": None,
            "cv_metric": "complementary_block_stability",
            "cv_candidate_count": len(alphas),
            "cv_folds": 0,
            "cv_factor_count": target,
            "factor_count_grid": [target],
            "selected_residual_gamma": gamma,
            "selected_blend_ratio": 1.0 / (1.0 + gamma),
            "support_stability": {
                "block_trading_days": block_days,
                "block_count": block_count,
                "complementary_pairs": pair_count,
                "bootstrap_samples": sample_total,
                "selection_frequency": frequency_threshold,
                "support_target": target,
                "stable_support": self.feature_names,
                "final_nonzero_support": selected_features,
                "support_frequencies": {
                    name: float(frequencies[idx])
                    for idx, name in enumerate(all_names)
                    if frequencies[idx] > 0
                },
                "fallback_used": False,
                "train_start": str(unique_dates.min().date()),
                "train_end": str(unique_dates.max().date()),
            },
        }

    def _segment_r2(self, dataset: FactorPilotDataset, segment: str) -> Optional[float]:
        """Score an already-defined dataset segment without creating a new split."""
        try:
            frame = dataset.prepare(segment, col_set=["feature", "label"])
            features = frame["feature"].reindex(columns=self.feature_names)
            labels = frame["label"]
            if isinstance(labels, pd.DataFrame):
                labels = labels.iloc[:, 0]
            X = features.to_numpy(dtype=float)
            y = labels.to_numpy(dtype=float)
            valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            if int(valid.sum()) < 2:
                return None
            score = float(self.model.score(X[valid], y[valid]))
            return score if np.isfinite(score) else None
        except (KeyError, TypeError, ValueError):
            return None
    
    def _fit_with_sklearn_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_index: pd.MultiIndex,
        sample_weight: Optional[np.ndarray] = None
    ) -> dict:
        """使用sklearn内置交叉验证训练 (with PanelPurgedKFold splits)"""
        cfg = self.config
        
        alphas = _resolve_alpha_grid(cfg, self.logger)
        max_iter = cfg.get('lasso_max_iter', 10000)
        cv_folds = cfg.get('lasso_cv_folds', 5)
        positive = cfg.get('positive', False)
        tol = cfg.get('lasso_tol', 1e-4)
        random_state = cfg.get('random_state', 42)  # 从配置读取，默认42
        
        # 构建 PanelPurgedKFold 并预计算 splits
        # LassoCV/ElasticNetCV accept cv as list-of-(train, test) index tuples
        # Actual label exits govern isolation equally for every scoring metric.
        cv_purged = PanelPurgedKFold(n_splits=cv_folds)
        date_groups = X_index.get_level_values(0).values
        label_exits = resolve_label_exit_dates(
            X_index, self.config.get("_label_exit_dates"),
            trading_calendar=self.config.get("_label_trading_calendar"),
            label_period=self.config.get("_label_effective_period"),
            calendar_source=self.config.get("_label_calendar_source"),
        )
        cv = list(cv_purged.split(X, groups=date_groups, label_exit_dates=label_exits))
        
        self.logger.info(
            "[PurgedCV-sklearn] strict actual-label-exit CV: %d folds",
            len(cv),
        )
        
        if self.model_type == 'lasso':
            self.model = LassoCV(
                alphas=alphas,
                cv=cv,
                max_iter=max_iter,
                tol=tol,
                positive=positive,
                n_jobs=-1,
                random_state=random_state,  # 固定随机种子
            )
        elif self.model_type == 'elasticnet':
            l1_ratios = cfg.get('l1_ratio_range', [0.1, 0.5, 0.7, 0.9, 0.95, 0.99])
            self.model = ElasticNetCV(
                l1_ratio=l1_ratios,
                alphas=alphas,
                cv=cv,
                max_iter=max_iter,
                tol=tol,
                positive=positive,
                n_jobs=-1,
                random_state=random_state,  # 固定随机种子
            )
        elif self.model_type == 'quantile_elasticnet':
            l1_ratios = cfg.get('l1_ratio_range', [0.1, 0.5, 0.7, 0.9, 0.95, 0.99])
            tau = self.quantile_tau
            max_irls = cfg.get('max_irls_iter', 30)
            self.logger.info(
                "[QlibLinearWrapper] Quantile ElasticNet: tau=%.2f, max_irls=%d",
                tau, max_irls,
            )
            self.model = QuantileElasticNetCV(
                l1_ratio=l1_ratios,
                alphas=alphas,
                quantile=tau,
                cv=cv,
                max_iter=max_iter,
                max_irls_iter=max_irls,
                tol=tol,
                positive=positive,
                random_state=random_state,
            )
        elif self.model_type == 'ridge':
            if not cv:
                raise ValueError("No valid purged CV folds for Ridge")
            self.model = RidgeCV(
                alphas=alphas,
                cv=cv,
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 训练 (Stage 1: sklearn warm-start MSE-based alpha selection)
        _fit_linear_estimator(self.model, X, y, sample_weight=sample_weight)
        
        # Stage 2: Post-hoc reranking with truncated evaluation
        # Re-evaluate top-M MSE candidates using a metric that discards the
        # bottom N% of predictions per date, aligning parameter selection
        # with long-only A-share portfolio construction.
        # Only re-trains the final model if reranking picks a different alpha.
        rerank_result = self._rerank_top_alphas(
            X, y, X_index, sample_weight, cv,
        )
        if rerank_result is not None:
            new_alpha, new_l1, rerank_score = rerank_result
            old_alpha = _fitted_regularization_param(self.model, 'alpha', 0.0)
            old_l1 = _fitted_regularization_param(self.model, 'l1_ratio', 1.0)
            if abs(new_alpha - old_alpha) > 1e-12 or abs(new_l1 - old_l1) > 1e-12:
                # Re-train with the reranked-best alpha on full training data
                if self.model_type == 'lasso':
                    self.model = Lasso(
                        alpha=new_alpha, max_iter=max_iter, positive=positive,
                        tol=tol,
                        random_state=random_state,
                    )
                elif self.model_type == 'elasticnet':
                    self.model = ElasticNet(
                        alpha=new_alpha, l1_ratio=new_l1,
                        tol=tol,
                        max_iter=max_iter, positive=positive,
                        random_state=random_state,
                    )
                _fit_linear_estimator(self.model, X, y, sample_weight=sample_weight)
        
        # 提取模型信息
        alpha = _fitted_regularization_param(self.model, 'alpha', 0.0)
        l1_ratio = _fitted_regularization_param(self.model, 'l1_ratio', 1.0)
        
        coefficients = {
            name: float(coef)
            for name, coef in zip(self.feature_names, self.model.coef_)
        }
        
        selected_features = [
            name for name, coef in coefficients.items()
            if abs(coef) > 1e-10
        ]
        
        r2_score = float(self.model.score(X, y))
        
        sklearn_cv_score = getattr(self.model, 'best_score_', None)
        if sklearn_cv_score is None:
            mse_path = getattr(self.model, 'mse_path_', None)
            if mse_path is not None:
                sklearn_cv_score = -float(np.min(np.mean(mse_path, axis=-1)))

        model_info = {
            'method': self.model_type,
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'coefficients': coefficients,
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'n_features': len(self.feature_names),
            'r2_score': r2_score,
            'feature_names': self.feature_names,
            # Persist the selected score and the full search geometry.  For
            # the production deflated-sharpe path this is the post-hoc
            # truncated-CV score; otherwise it is explicitly marked as the
            # sklearn MSE selection score below.
            'cv_score': float(rerank_result[2]) if rerank_result is not None else (
                float(sklearn_cv_score) if sklearn_cv_score is not None else None
            ),
            'cv_metric': self.param_search_metric if rerank_result is not None else 'mse',
            'cv_candidate_count': len(alphas) * (
                len(l1_ratios) if self.model_type == 'elasticnet' else 1
            ),
            'cv_evaluated_candidate_count': (
                self._rerank_evaluated_candidate_count if rerank_result is not None
                else len(alphas) * (len(l1_ratios) if self.model_type == 'elasticnet' else 1)
            ),
            'cv_folds': len(cv),
            'alpha_grid_min': float(np.min(alphas)) if len(alphas) else None,
            'alpha_grid_max': float(np.max(alphas)) if len(alphas) else None,
            'l1_ratio_grid': [
                float(value) for value in (
                    l1_ratios if self.model_type == 'elasticnet' else [1.0]
                )
            ],
        }
        
        # Add reranking metadata if applicable
        if rerank_result is not None:
            model_info['rerank_score'] = float(rerank_result[2])
            model_info['rerank_metric'] = self.param_search_metric
            model_info['rerank_truncate_pct'] = float(
                cfg.get('eval_truncate_bottom_pct', 0.0)
            )
        
        # Add quantile-specific metadata
        if self.model_type == 'quantile_elasticnet':
            model_info['quantile_tau'] = self.quantile_tau
            model_info['n_irls_iterations'] = self.model.n_iter_irls_
            model_info['cv_metric'] = 'pinball_score'
            model_info['cv_candidate_count'] = len(alphas) * len(l1_ratios)
            # Candidates with at least one contributing validation score;
            # cv_folds describes the actual split list, not successful folds.
            model_info['cv_evaluated_candidate_count'] = self.model.n_evaluated_candidates_
            model_info['cv_folds'] = self.model.n_splits_
            model_info['l1_ratio_grid'] = [float(value) for value in l1_ratios]
            score_label = "pinball_score"
        else:
            score_label = "R²"
        
        self.logger.info(
            f"[QlibLinearWrapper] sklearn CV训练完成: "
            f"alpha={alpha:.6f}, l1_ratio={l1_ratio:.2f}, "
            f"selected={len(selected_features)}/{len(self.feature_names)}, "
            f"{score_label}={r2_score:.4f}"
        )
        
        return model_info
    
    def _score_param_grid_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_index: pd.MultiIndex,
        sample_weight: Optional[np.ndarray],
        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
        param_pairs: List[Tuple[float, float]],
        evaluator: CVMetricEvaluator,
        factor_counts: Optional[List[int]] = None,
        metric_y: Optional[np.ndarray] = None,
        baseline_y: Optional[np.ndarray] = None,
        fold_feature_indices: Optional[List[np.ndarray]] = None,
    ) -> Dict[tuple, Dict[str, float]]:
        """
        Evaluate a grid of (alpha, l1_ratio) pairs via cross-validation.

        Shared inner loop for both :meth:`_rerank_top_alphas` and
        :meth:`_fit_with_custom_cv`. Two orthogonal optimisations are applied:

        1. **Warm-start chains** (``cv_warm_start``, default True): for
           coordinate-descent models (Lasso / ElasticNet) the alphas within each
           ``l1_ratio`` group are evaluated in *descending* order on a single
           reused estimator with ``warm_start=True``. Because ElasticNet is a
           convex problem, coordinate descent converges to the same optimum
           regardless of the initial point (up to ``tol``); warm-starting merely
           amortises the iteration count. ``sample_weight`` semantics are fully
           preserved because we still call ``estimator.fit(X, y, sample_weight)``.
        2. **Parallelism** (``cv_n_jobs``): each (l1_ratio, fold) pair is an
           independent unit dispatched via joblib. Units hold their own estimator
           and read-only views of X / y, so there is no shared mutable state.

        Set ``cv_warm_start: false`` in the config to fall back to independent
        cold-start fits (bit-identical to the legacy implementation, parallelism
        aside). Quantile ElasticNet and Ridge always use cold starts.

        Args:
            X: Training feature matrix.
            y: Training labels.
            X_index: MultiIndex (datetime, instrument) for date grouping.
            sample_weight: Optional per-sample weights (aligned to X).
            cv_splits: Pre-computed (train_idx, val_idx) tuples.
            param_pairs: Candidate (alpha, l1_ratio) pairs to evaluate.
            evaluator: Metric evaluator (higher score is better).

        Returns:
            Mapping (alpha, l1_ratio) -> {'score': mean_cv_score, 'n_folds': int}.
            Pairs whose every fold failed/NaN are omitted.
        """
        cfg = self.config
        max_iter = cfg.get('lasso_max_iter', 10000)
        positive = cfg.get('positive', False)
        random_state = cfg.get('random_state', 42)
        tol = cfg.get('lasso_tol', 1e-4)
        warm_start_enabled = bool(cfg.get('cv_warm_start', True)) and \
            self.model_type in ('lasso', 'elasticnet')
        n_jobs = int(cfg.get('cv_n_jobs', 4))
        backend = cfg.get('cv_backend', 'loky')
        gamma_grid = sorted({
            float(value) for value in (cfg.get("residual_gamma_grid") or [])
            if np.isfinite(float(value)) and float(value) >= 0.0
        })
        pooled_overlay = (
            self.param_search_metric in {"deflated_sharpe", "net_topk_dsr"}
            and bool(cfg.get("pooled_oof_dsr", False))
            and baseline_y is not None
            and bool(gamma_grid)
        )

        date_groups = X_index.get_level_values(0).values
        instrument_groups = X_index.get_level_values(1).values

        # Group alphas by l1_ratio; within each group sort DESCENDING so the
        # warm-start chain flows from sparse (large alpha) to dense solutions.
        grid: Dict[float, List[float]] = defaultdict(list)
        for alpha, l1 in param_pairs:
            grid[l1].append(alpha)
        grouped: List[Tuple[float, List[float]]] = [
            (l1, sorted(set(alphas), reverse=True)) for l1, alphas in grid.items()
        ]

        max_irls = cfg.get('max_irls_iter', 30)
        model_type = self.model_type
        quantile_tau = self.quantile_tau

        def _make_estimator(alpha: float, l1: float):
            if model_type == 'lasso':
                return Lasso(
                    alpha=alpha, max_iter=max_iter, tol=tol, positive=positive,
                    random_state=random_state, warm_start=warm_start_enabled,
                )
            if model_type == 'elasticnet':
                return ElasticNet(
                    alpha=alpha, l1_ratio=l1, max_iter=max_iter, tol=tol,
                    positive=positive, random_state=random_state,
                    warm_start=warm_start_enabled,
                )
            if model_type == 'quantile_elasticnet':
                return QuantileElasticNet(
                    alpha=alpha, l1_ratio=l1, quantile=quantile_tau,
                    max_iter=max_iter, max_irls_iter=max_irls,
                    positive=positive, random_state=random_state,
                )
            return Ridge(alpha=alpha, random_state=random_state)

        def _run_group_fold(
            l1: float,
            alphas_desc: List[float],
            train_idx: np.ndarray,
            val_idx: np.ndarray,
            feature_idx: Optional[np.ndarray],
        ) -> List[tuple]:
            X_tr, X_val = X[train_idx], X[val_idx]
            if feature_idx is not None:
                X_tr = X_tr[:, feature_idx]
                X_val = X_val[:, feature_idx]
            y_tr = y[train_idx]
            y_val = metric_y[val_idx] if metric_y is not None else y[val_idx]
            sw = sample_weight[train_idx] if sample_weight is not None else None
            val_dates = date_groups[val_idx]
            val_instruments = instrument_groups[val_idx]

            # One reused estimator per (l1, fold) enables warm-start chaining.
            estimator = _make_estimator(alphas_desc[0], l1) if warm_start_enabled else None
            out: List[tuple] = []
            for alpha in alphas_desc:
                if warm_start_enabled:
                    model = estimator
                    model.set_params(alpha=alpha)
                else:
                    model = _make_estimator(alpha, l1)
                try:
                    _fit_linear_estimator(model, X_tr, y_tr, sample_weight=sw)
                except LabelIsolationError:
                    raise
                except Exception as e:  # noqa: BLE001 - logged, then skip candidate
                    self.logger.debug(
                        "[CV] Fit failed alpha=%.6f, l1=%.2f: %s", alpha, l1, e,
                    )
                    continue
                counts = factor_counts or [0]
                for factor_count in counts:
                    if factor_count and factor_count < X_tr.shape[1]:
                        keep = np.argsort(np.abs(model.coef_))[-factor_count:]
                        y_pred = X_val[:, keep] @ model.coef_[keep] + model.intercept_
                    else:
                        y_pred = model.predict(X_val)
                    if pooled_overlay:
                        out.append(
                            (
                                alpha,
                                l1,
                                int(factor_count),
                                np.asarray(y_pred, dtype=float),
                                np.asarray(y_val, dtype=float),
                                np.asarray(val_dates),
                                np.asarray(baseline_y[val_idx], dtype=float),
                                np.asarray(val_instruments),
                            )
                        )
                    else:
                        score = (
                            evaluator.evaluate(
                                y_pred, y_val, val_dates,
                                X_index.get_level_values('instrument').values[val_idx],
                            )
                            if getattr(evaluator, 'metric', '') == 'net_topk_dsr'
                            else evaluator.evaluate(y_pred, y_val, val_dates)
                        )
                        if np.isfinite(score):
                            out.append((alpha, l1, int(factor_count), float(score)))
            return out

        tasks = [
            (
                l1,
                alphas_desc,
                train_idx,
                val_idx,
                None if fold_feature_indices is None else fold_feature_indices[fold],
            )
            for (l1, alphas_desc) in grouped
            for fold, (train_idx, val_idx) in enumerate(cv_splits)
        ]

        if n_jobs != 1 and len(tasks) > 1:
            # BLAS limits are process-wide. Keep an outer limit alive until
            # all threaded fits finish, so overlapping inner contexts restore
            # to one thread, then the caller's original limit is restored once.
            threaded_coordinate_descent = (
                backend == "threading" and model_type in ("lasso", "elasticnet")
            )
            limits = (
                threadpool_limits(limits=1, user_api="blas")
                if threaded_coordinate_descent else nullcontext()
            )
            with limits:
                task_results = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(_run_group_fold)(*t) for t in tasks
                )
        else:
            task_results = [_run_group_fold(*t) for t in tasks]

        if pooled_overlay:
            oof_acc: Dict[tuple, List[tuple[np.ndarray, ...]]] = defaultdict(list)
            for task_out in task_results:
                for alpha, l1, factor_count, pred, realized, dates, baseline, instruments in task_out:
                    key = (alpha, l1, factor_count) if factor_counts else (alpha, l1)
                    oof_acc[key].append((pred, realized, dates, baseline, instruments))

            pooled_scores: Dict[tuple, Dict[str, float]] = {}
            for key, chunks in oof_acc.items():
                predictions = np.concatenate([chunk[0] for chunk in chunks])
                realized = np.concatenate([chunk[1] for chunk in chunks])
                dates = np.concatenate([chunk[2] for chunk in chunks])
                baseline = np.concatenate([chunk[3] for chunk in chunks])
                instruments = np.concatenate([chunk[4] for chunk in chunks])
                pred_frame = pd.DataFrame({"prediction": predictions, "date": dates})
                residual_score = pred_frame.groupby("date")["prediction"].transform(
                    lambda values: (values - values.mean()) / (values.std() + 1e-8)
                ).fillna(0.0).to_numpy()
                oof_periods = (
                    evaluator.effective_period_count(dates)
                    if hasattr(evaluator, "effective_period_count")
                    else int(pd.Index(dates).nunique())
                )
                for gamma in gamma_grid:
                    args = (baseline + gamma * residual_score, realized, dates)
                    score = (
                        evaluator.evaluate(*args, instruments)
                        if getattr(evaluator, "metric", "") == "net_topk_dsr"
                        else evaluator.evaluate(*args)
                    )
                    if np.isfinite(score):
                        pooled_scores[(*key, gamma)] = {
                            "score": float(score),
                            "n_folds": len(chunks),
                            "oof_periods": oof_periods,
                        }
            return pooled_scores

        score_acc: Dict[tuple, List[float]] = defaultdict(list)
        for task_out in task_results:
            for alpha, l1, factor_count, score in task_out:
                key = (alpha, l1, factor_count) if factor_counts else (alpha, l1)
                score_acc[key].append(score)

        mean_scores = {}
        for key, scores in score_acc.items():
            if scores:
                score = float(np.mean(scores))
                if np.isfinite(score):
                    mean_scores[key] = {'score': score, 'n_folds': len(scores)}
        return mean_scores

    def _rerank_top_alphas(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_index: pd.MultiIndex,
        sample_weight: Optional[np.ndarray],
        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[Tuple[float, float, float]]:
        """
        Post-hoc reranking of top-M alpha candidates using truncated evaluation.
        
        After sklearn's efficient warm-start MSE-based CV selects an alpha,
        this method re-evaluates the top-M MSE candidates using a metric
        that discards the bottom N% of predictions per date. This aligns
        parameter selection with long-only portfolio construction in markets
        without short selling (e.g. A-share).
        
        Theoretical basis:
        - Clarke, de Silva & Thorley (2002): long-only constraints reduce
          the transfer coefficient; evaluation should reflect constraints.
        - Haitong Securities Research: A-share factors often derive 70-78%
          of long-short return from the short side, which is not executable.
        
        Only applies to LassoCV / ElasticNetCV (which expose ``mse_path_``).
        Skipped for QuantileElasticNetCV and RidgeCV.
        
        Args:
            X: Training feature matrix
            y: Training labels
            X_index: MultiIndex (datetime, instrument) for date grouping
            sample_weight: Optional sample weights
            cv_splits: Pre-computed CV splits (train_idx, val_idx) tuples
            
        Returns:
            (best_alpha, best_l1_ratio, best_score) or None if disabled
        """
        cfg = self.config
        truncate_pct = cfg.get('eval_truncate_bottom_pct', 0.0)
        self._rerank_evaluated_candidate_count = 0
        if truncate_pct <= 0:
            return None
        
        # Only supported for models exposing mse_path_ (LassoCV, ElasticNetCV)
        mse_path = getattr(self.model, 'mse_path_', None)
        if mse_path is None:
            self.logger.info(
                "[Rerank] mse_path_ not available for %s, skipping reranking",
                type(self.model).__name__,
            )
            return None
        
        top_m = int(cfg.get('rerank_top_m', 10))

        # ------------------------------------------------------------------
        # Extract candidate (alpha, l1_ratio) pairs from sklearn's mse_path_
        # ------------------------------------------------------------------
        alphas_arr = self.model.alphas_  # shape: (n_alphas,)
        
        candidates: List[Tuple[float, float, float]] = []  # (alpha, l1_ratio, mean_mse)
        
        if self.model_type == 'elasticnet':
            # sklearn squeezes singleton l1/alpha axes; recover them from
            # the fitted grid rather than inferring model type from ndim.
            l1_ratios = np.atleast_1d(self.model.l1_ratio)
            alphas_arr = np.asarray(alphas_arr)
            mean_mse = mse_path.reshape(
                len(l1_ratios), alphas_arr.shape[-1], -1
            ).mean(axis=-1)
            for li, l1 in enumerate(l1_ratios):
                alpha_grid = alphas_arr if alphas_arr.ndim == 1 else alphas_arr[li]
                for ai, alpha in enumerate(alpha_grid):
                    candidates.append((float(alpha), float(l1), float(mean_mse[li, ai])))
        elif mse_path.ndim == 2:
            # LassoCV: mse_path_ shape = (n_alphas, n_folds)
            mean_mse = mse_path.mean(axis=1)
            for i in range(len(alphas_arr)):
                candidates.append((float(alphas_arr[i]), 1.0, float(mean_mse[i])))
        else:
            self.logger.warning(
                "[Rerank] Unexpected mse_path_ ndim=%d, skipping", mse_path.ndim,
            )
            return None
        
        # Sort by MSE (ascending = best first) and take top-M
        candidates.sort(key=lambda x: x[2])
        candidates = candidates[:top_m]
        
        sklearn_alpha = _fitted_regularization_param(self.model, 'alpha', 0.0)
        sklearn_l1 = _fitted_regularization_param(self.model, 'l1_ratio', 1.0)
        
        self.logger.info(
            "[Rerank] Re-evaluating top-%d MSE candidates with truncated %s "
            "(truncate_bottom=%.0f%%). sklearn baseline: alpha=%.6f, l1=%.2f",
            len(candidates), self.param_search_metric,
            truncate_pct * 100, sklearn_alpha, sklearn_l1,
        )
        
        # ------------------------------------------------------------------
        # Re-evaluate each candidate with truncated metric.
        # Uses the shared warm-start + joblib grid scorer. Tie-breaking order is
        # preserved: candidates are already sorted by MSE ascending, and we scan
        # them in that order keeping the first strict maximum.
        # ------------------------------------------------------------------
        evaluator = CVMetricEvaluator(
            self.param_search_metric, cfg, self.logger,
            n_trials=len(candidates),
        )

        param_pairs = [(alpha, l1) for alpha, l1, _ in candidates]
        scores = self._score_param_grid_cv(
            X, y, X_index, sample_weight, cv_splits, param_pairs, evaluator,
        )
        # Count candidates with a finite aggregate score, not requested top-M.
        self._rerank_evaluated_candidate_count = len(scores)

        best_score = -np.inf
        best_alpha = sklearn_alpha
        best_l1 = sklearn_l1

        for alpha, l1_ratio, mse_val in candidates:
            entry = scores.get((alpha, l1_ratio))
            if entry is None:
                continue
            avg = entry['score']
            self.logger.debug(
                "[Rerank] alpha=%.6f, l1=%.2f, MSE=%.6f, %s=%.4f (%d folds)",
                alpha, l1_ratio, mse_val,
                self.param_search_metric, avg, entry['n_folds'],
            )
            if avg > best_score:
                best_score = avg
                best_alpha = alpha
                best_l1 = l1_ratio
        
        # Log reranking outcome
        if abs(best_alpha - sklearn_alpha) > 1e-12 or abs(best_l1 - sklearn_l1) > 1e-12:
            self.logger.info(
                "[Rerank] Alpha overridden: %.6f -> %.6f, l1: %.2f -> %.2f, "
                "truncated %s=%.4f",
                sklearn_alpha, best_alpha, sklearn_l1, best_l1,
                self.param_search_metric, best_score,
            )
        else:
            self.logger.info(
                "[Rerank] sklearn alpha confirmed: %.6f (l1=%.2f), "
                "truncated %s=%.4f",
                best_alpha, best_l1, self.param_search_metric, best_score,
            )
        
        return (best_alpha, best_l1, best_score)
    
    def _fit_with_custom_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_index: pd.MultiIndex,
        sample_weight: Optional[np.ndarray] = None,
        *,
        metric_y: Optional[np.ndarray] = None,
        baseline_y: Optional[np.ndarray] = None,
    ) -> dict:
        """
        使用自定义指标(IC/Top-K/Sharpe)进行交叉验证参数搜索
        
        Uses CVMetricEvaluator for consistent metric calculation across the framework.
        """
        cfg = self.config
        
        alphas = _resolve_alpha_grid(cfg, self.logger)
        l1_ratios = cfg.get('l1_ratio_range', [0.5, 0.9, 0.99])
        max_iter = cfg.get('lasso_max_iter', 10000)
        cv_folds = cfg.get('lasso_cv_folds', 5)
        positive = cfg.get('positive', False)
        random_state = cfg.get('random_state', 42)  # 从配置读取，默认42
        
        # Every validation date is retained; only immature training labels purge.
        cv = PanelPurgedKFold(n_splits=cv_folds)
        
        # Extract date groups from MultiIndex for panel-aware splitting
        date_groups = X_index.get_level_values(0).values
        
        self.logger.info(
            "[PurgedCV] strict actual-label-exit CV: %d folds",
            cv_folds,
        )
        
        # 参数网格搜索
        if self.model_type == 'elasticnet':
            param_grid = [(a, l1) for a in alphas for l1 in l1_ratios]
        else:
            param_grid = [(a, 1.0) for a in alphas]
        factor_counts = sorted({
            max(1, min(int(value), X.shape[1]))
            for value in (cfg.get('factor_count_grid') or [])
        })

        gamma_grid = sorted({
            float(value) for value in (cfg.get("residual_gamma_grid") or [])
            if np.isfinite(float(value)) and float(value) >= 0.0
        })
        pooled_overlay = (
            self.param_search_metric in {"deflated_sharpe", "net_topk_dsr"}
            and bool(cfg.get("pooled_oof_dsr", False))
            and baseline_y is not None
            and bool(gamma_grid)
        )
        hierarchical_cfg = cfg.get("hierarchical_clustering", {})
        nested_thresholds = hierarchical_cfg.get("nested_distance_thresholds") or []
        positive_gamma_count = sum(value > 0.0 for value in gamma_grid)
        pooled_trial_count = (
            len(param_grid) * positive_gamma_count
            + (1 if 0.0 in gamma_grid else 0)
        )
        n_trials = (
            pooled_trial_count if pooled_overlay else len(param_grid)
        ) * max(1, len(factor_counts)) * max(1, len(nested_thresholds))
        evaluator = CVMetricEvaluator(self.param_search_metric, cfg, self.logger, n_trials=n_trials)
        
        best_score = -np.inf
        best_alpha = alphas[len(alphas) // 2] if len(alphas) > 0 else 0.01
        best_l1_ratio = 0.9
        best_factor_count = 0
        
        self.logger.info(
            f"[QlibLinearWrapper] 自定义CV搜索: {len(param_grid)} 参数组合, "
            f"metric={self.param_search_metric}"
        )
        
        # Pre-compute CV splits once (shared across all candidates) so the grid
        # scorer can warm-start and parallelise over (l1_ratio, fold) units.
        label_exits = resolve_label_exit_dates(
            X_index, self.config.get("_label_exit_dates"),
            trading_calendar=self.config.get("_label_trading_calendar"),
            label_period=self.config.get("_label_effective_period"),
            calendar_source=self.config.get("_label_calendar_source"),
        )
        cv_splits = list(cv.split(X, groups=date_groups, label_exit_dates=label_exits))
        nested_audit = None
        if nested_thresholds:
            from factor_synthesis.preselection.hierarchical_clustering_strategy import (
                HierarchicalClusteringStrategy,
                select_nested_fold_representatives,
            )

            X_frame = pd.DataFrame(X, index=X_index, columns=self.feature_names)
            y_series = pd.Series(y, index=X_index)
            nested_folds = select_nested_fold_representatives(
                X_frame, y_series, cv_splits, hierarchical_cfg, self.logger
            )
            score_cells = {}
            for threshold, folds in nested_folds.items():
                feature_indices = [
                    np.asarray(
                        [self.feature_names.index(name) for name in fold["selected_features"]],
                        dtype=int,
                    )
                    for fold in folds
                ]
                score_cells[threshold] = self._score_param_grid_cv(
                    X,
                    y,
                    X_index,
                    sample_weight,
                    cv_splits,
                    param_grid,
                    evaluator,
                    factor_counts=factor_counts or None,
                    metric_y=metric_y,
                    baseline_y=baseline_y,
                    fold_feature_indices=feature_indices,
                )
            nested_audit = {
                "candidate_thresholds": [float(value) for value in nested_folds],
                "folds": {str(key): value for key, value in nested_folds.items()},
                "outer_model_count": 1,
            }
        else:
            score_cells = {
                None: self._score_param_grid_cv(
                    X,
                    y,
                    X_index,
                    sample_weight,
                    cv_splits,
                    param_grid,
                    evaluator,
                    factor_counts=factor_counts or None,
                    metric_y=metric_y,
                    baseline_y=baseline_y,
                )
            }
        
        # Preserve legacy tie-breaking: scan param_grid in its original order and
        # keep the first strict maximum.
        results = []
        best_gamma = 0.0
        best_oof_periods = 0
        best_threshold = None
        min_oof_periods = int(cfg.get("cv_min_oof_periods", 0))
        for threshold, scores in score_cells.items():
            for alpha, l1_ratio in param_grid:
                for factor_count in factor_counts or [0]:
                    for gamma in gamma_grid if pooled_overlay else [None]:
                        base_key = (
                            (alpha, l1_ratio, factor_count)
                            if factor_counts
                            else (alpha, l1_ratio)
                        )
                        key = (*base_key, gamma) if pooled_overlay else base_key
                        entry = scores.get(key)
                        if entry is None:
                            continue
                        avg_score = entry['score']
                        oof_periods = int(entry.get("oof_periods", 0))
                        eligible = not pooled_overlay or oof_periods >= min_oof_periods
                        results.append({
                            'distance_threshold': threshold,
                            'alpha': alpha,
                            'l1_ratio': l1_ratio,
                            'factor_count': factor_count or None,
                            'residual_gamma': gamma,
                            'score': avg_score,
                            'n_folds': entry['n_folds'],
                            'oof_periods': oof_periods or None,
                            'eligible': eligible,
                        })
                        if eligible and avg_score > best_score:
                            best_score = avg_score
                            best_alpha = alpha
                            best_l1_ratio = l1_ratio
                            best_factor_count = factor_count
                            best_gamma = float(gamma or 0.0)
                            best_oof_periods = oof_periods
                            best_threshold = threshold

        if not results:
            raise ValueError(
                f"No finite {self.param_search_metric} CV scores across "
                f"{len(param_grid)} candidates"
            )
        fallback_reason = None
        if pooled_overlay:
            threshold = float(cfg.get("cv_dsr_min_threshold", 0.0))
            has_eligible = any(result["eligible"] for result in results)
            if not has_eligible:
                diagnostic = max(results, key=lambda result: result["score"])
                best_score = float(diagnostic["score"])
                best_alpha = float(diagnostic["alpha"])
                best_l1_ratio = float(diagnostic["l1_ratio"])
                best_factor_count = int(diagnostic["factor_count"] or 0)
                best_oof_periods = int(diagnostic["oof_periods"] or 0)
                best_threshold = diagnostic.get("distance_threshold")
                fallback_reason = "insufficient_oof_history"
                best_gamma = 0.0
            elif best_score < threshold:
                fallback_reason = "dsr_below_threshold"
                best_gamma = 0.0
            elif best_gamma == 0.0:
                fallback_reason = "pure_size_won_inner_cv"
            self.logger.info(
                "[PooledOOF] gamma=%.2f, DSR=%.4f, OOF periods=%d, fallback=%s",
                best_gamma,
                best_score,
                best_oof_periods,
                fallback_reason or "none",
            )
        
        self.logger.info(
            f"[QlibLinearWrapper] 最优参数: alpha={best_alpha:.6f}, "
            f"l1_ratio={best_l1_ratio:.2f}, {self.param_search_metric}={best_score:.4f}"
        )
        
        fit_X = X
        if nested_audit is not None:
            final_config = dict(hierarchical_cfg)
            final_config.pop("nested_distance_thresholds", None)
            final_config["distance_threshold"] = float(best_threshold)
            final_features = HierarchicalClusteringStrategy(
                final_config, self.logger
            ).select(pd.DataFrame(X, index=X_index, columns=self.feature_names), pd.Series(y, index=X_index))
            final_indices = np.asarray(
                [self.feature_names.index(name) for name in final_features], dtype=int
            )
            fit_X = X[:, final_indices]
            self.feature_names = list(final_features)
            nested_audit["selected_threshold"] = float(best_threshold)
            nested_audit["outer_selected_features"] = list(final_features)

        # 使用最优参数训练最终模型（添加 random_state 参数）
        if self.model_type == 'lasso':
            self.model = Lasso(
                alpha=best_alpha, max_iter=max_iter, positive=positive,
                tol=cfg.get('lasso_tol', 1e-4),
                random_state=random_state  # 固定随机种子
            )
        elif self.model_type == 'elasticnet':
            self.model = ElasticNet(
                alpha=best_alpha, l1_ratio=best_l1_ratio,
                tol=cfg.get('lasso_tol', 1e-4),
                max_iter=max_iter, positive=positive,
                random_state=random_state  # 固定随机种子
            )
        elif self.model_type == 'quantile_elasticnet':
            self.model = QuantileElasticNet(
                alpha=best_alpha, l1_ratio=best_l1_ratio,
                quantile=self.quantile_tau,
                max_iter=max_iter,
                max_irls_iter=cfg.get('max_irls_iter', 30),
                positive=positive,
                random_state=random_state,
            )
        else:
            self.model = Ridge(alpha=best_alpha, random_state=random_state)
        
        _fit_linear_estimator(self.model, fit_X, y, sample_weight=sample_weight)
        if best_factor_count and best_factor_count < fit_X.shape[1]:
            keep = np.argsort(np.abs(self.model.coef_))[-best_factor_count:]
            pruned = np.zeros_like(self.model.coef_)
            pruned[keep] = self.model.coef_[keep]
            self.model.coef_ = pruned
        
        # 提取模型信息
        coefficients = {
            name: float(coef)
            for name, coef in zip(self.feature_names, self.model.coef_)
        }
        
        selected_features = [
            name for name, coef in coefficients.items()
            if abs(coef) > 1e-10
        ]
        
        r2_score = float(self.model.score(fit_X, y))
        
        model_info = {
            'method': self.model_type,
            'alpha': float(best_alpha),
            'l1_ratio': float(best_l1_ratio),
            'coefficients': coefficients,
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'n_features': len(self.feature_names),
            'r2_score': r2_score,
            'cv_score': float(best_score),
            'cv_metric': self.param_search_metric,
            'feature_names': self.feature_names,
            'cv_results': results[:10] if results else [],  # Top 10 results for debugging
            # Keep the search geometry with the fitted model.  The deployment
            # bundle used to retain only coefficients, which made it
            # impossible to tell whether a rolling window selected a grid
            # boundary or whether the CV score was comparable across runs.
            'cv_candidate_count': n_trials,
            'cv_factor_count': best_factor_count or None,
            'factor_count_grid': factor_counts,
            'cv_folds': int(cv_folds),
            'selected_residual_gamma': best_gamma if pooled_overlay else None,
            'selected_blend_ratio': (
                1.0 / (1.0 + best_gamma) if pooled_overlay else None
            ),
            'pooled_oof_periods': best_oof_periods if pooled_overlay else None,
            'cv_dsr_min_threshold': (
                float(cfg.get("cv_dsr_min_threshold", 0.0))
                if pooled_overlay
                else None
            ),
            'cv_fallback_reason': fallback_reason,
            'alpha_grid_min': float(np.min(alphas)) if len(alphas) else None,
            'alpha_grid_max': float(np.max(alphas)) if len(alphas) else None,
            'l1_ratio_grid': [float(value) for value in l1_ratios],
        }
        if nested_audit is not None:
            model_info["nested_hierarchical_cv"] = nested_audit
        
        # Add quantile-specific metadata
        if self.model_type == 'quantile_elasticnet':
            model_info['quantile_tau'] = self.quantile_tau
            model_info['n_irls_iterations'] = getattr(self.model, 'n_iter_irls_', 0)
            score_label = "pinball_score"
        else:
            score_label = "R²"
        
        self.logger.info(
            f"[QlibLinearWrapper] 自定义CV训练完成: "
            f"selected={len(selected_features)}/{len(self.feature_names)}, "
            f"{score_label}={r2_score:.4f}"
        )
        
        return model_info
    
    def predict(
        self,
        dataset: FactorPilotDataset,
        segment: str = "test"
    ) -> pd.Series:
        """
        生成预测
        
        Args:
            dataset: FactorPilotDataset
            segment: str, "train"/"valid"/"test"
            
        Returns:
            pd.Series
                - Index: MultiIndex (datetime, instrument)
                - Values: float, 预测分数
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 准备预测数据
        df_segment = dataset.prepare(segment, col_set="feature")
        X = df_segment if isinstance(df_segment, pd.DataFrame) else df_segment["feature"]
        
        # 不再使用 StandardScaler - 数据已通过 rank_zscore 标准化
        # 这与云端逻辑一致：rank_zscore 后直接用权重做线性加权
        X_values = X.reindex(columns=self.feature_names).values
        
        # 处理NaN
        X_values = np.nan_to_num(X_values, nan=0.0)
        
        # 预测
        pred_values = self.model.predict(X_values)
        predictions = pd.Series(pred_values, index=X.index, name='prediction')
        
        self.logger.info(
            f"[QlibLinearWrapper] 预测完成, segment={segment}, "
            f"样本数={len(predictions)}"
        )
        
        return predictions
