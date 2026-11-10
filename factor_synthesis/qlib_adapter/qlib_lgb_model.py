# -*- coding: utf-8 -*-
"""
Qlib LightGBM模型包装器

封装Qlib的LGBModel，提供与现有接口兼容的调用方式。
支持自定义目标函数(IC Loss)和参数搜索(Top-K, Sharpe)。
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import pandas as pd
import numpy as np
import logging
import os
from copy import copy

try:
    import qlib
    from qlib.contrib.model.gbdt import LGBModel
    QLIB_LGB_AVAILABLE = True
except ImportError:
    QLIB_LGB_AVAILABLE = False
    LGBModel = None
    qlib = None

# Qlib initialization flag
_QLIB_INITIALIZED = False


def _ensure_qlib_initialized():
    """
    Ensure Qlib is initialized before using Qlib models.
    Uses a minimal configuration that doesn't require external data.
    """
    global _QLIB_INITIALIZED
    if _QLIB_INITIALIZED:
        return True
    
    if not QLIB_LGB_AVAILABLE or qlib is None:
        return False
    
    try:
        # Check if already initialized
        try:
            from qlib.config import C
            if C.registered:
                _QLIB_INITIALIZED = True
                return True
        except Exception:
            pass
        
        from utils.output_paths import qlib_tracking_config
        # Initialize with minimal config (no external data needed)
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn", exp_manager=qlib_tracking_config())
        _QLIB_INITIALIZED = True
        return True
    except Exception as e:
        # If init fails (e.g., no data), try with skip_if_fail
        try:
            # Use a dummy provider that won't fail
            import tempfile
            import os
            dummy_dir = os.path.join(tempfile.gettempdir(), "qlib_dummy_data")
            os.makedirs(dummy_dir, exist_ok=True)
            qlib.init(provider_uri=dummy_dir, region="cn", exp_manager=qlib_tracking_config())
            _QLIB_INITIALIZED = True
            return True
        except Exception:
            return False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from .data_converter import FactorPilotDataset, DataConverter
from .panel_purged_kfold import LabelIsolationError, resolve_label_exit_dates
from .custom_objectives import QlibCustomObjectives
from .cv_metrics import CVMetricEvaluator


RANKING_OBJECTIVES = {"lambdarank", "rank_xendcg"}


class _NoFiniteSearchWinnerError(ValueError):
    """An explicitly requested search cannot fall back to default training."""


class QlibLGBWrapper:
    """
    Qlib LightGBM模型包装器
    
    封装Qlib的LGBModel或原生lightgbm，提供统一接口。
    支持:
    - 自定义目标函数 (Pearson IC, Cross-Sectional IC)
    - 参数搜索 (基于Top-K, IC, Sharpe等指标)
    """
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        """
        初始化模型
        
        Args:
            config: dict, 来自 synthesis_config (如 "lightgbm_comprehensive")
                - lgb_learning_rate: float
                - lgb_max_depth: int
                - lgb_num_leaves: int
                - lgb_reg_alpha: float
                - lgb_reg_lambda: float
                - lgb_n_estimators: int
                - objective: str ("regression", "lambdarank", or "rank_xendcg")
                - lgb_custom_objective: str ("pearson_ic", "cross_sectional_ic", None)
                - lgb_param_search_metric: str ("topk_return", "ic", "sharpe", "topk_sharpe")
                - lgb_enable_param_search: bool
                - 等等...
                
            logger: logging.Logger
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.qlib_model: Optional[LGBModel] = None
        self.feature_names: List[str] = []
        # Try to initialize Qlib if available
        self._use_qlib = QLIB_LGB_AVAILABLE and _ensure_qlib_initialized()
        self._is_ranking = config.get('objective', 'regression') in RANKING_OBJECTIVES
        
        if self._use_qlib:
            self.logger.info("[QlibLGBWrapper] Qlib已初始化，将使用Qlib LGBModel")
        else:
            self.logger.info("[QlibLGBWrapper] Qlib未初始化，将使用原生LightGBM")
        
        # Custom objective handling
        self.custom_objective_name = config.get('lgb_custom_objective', None)
        if isinstance(self.custom_objective_name, bool) and self.custom_objective_name:
            self.custom_objective_name = "pearson_ic"
        
        # Parameter search settings
        self.enable_param_search = config.get('lgb_enable_param_search', False)
        self.param_search_metric = config.get('lgb_param_search_metric', 'mse')
        
    def _map_config_to_qlib_params(self) -> dict:
        """
        将现有配置映射到Qlib LGBModel参数
        
        配置映射表:
            lgb_learning_rate -> learning_rate
            lgb_max_depth -> max_depth
            lgb_num_leaves -> num_leaves
            lgb_reg_alpha -> lambda_l1
            lgb_reg_lambda -> lambda_l2
            lgb_feature_fraction -> colsample_bytree
            lgb_bagging_fraction -> subsample
            lgb_n_estimators -> num_boost_round
            lgb_bagging_freq -> bagging_freq (explicit only)
            lgb_min_child_samples -> min_child_samples (explicit only)
            lgb_random_state -> random_state (explicit only)
        """
        cfg = self.config
        
        # 目标函数映射
        objective = cfg.get('objective', 'regression')
        loss = 'mse' if objective == 'regression' else 'mse'
        
        params = {
            'loss': loss,
            'learning_rate': cfg.get('lgb_learning_rate', 0.01),
            'max_depth': cfg.get('lgb_max_depth', 3),
            'num_leaves': cfg.get('lgb_num_leaves', 7),
            'lambda_l1': cfg.get('lgb_reg_alpha', 0.1),
            'lambda_l2': cfg.get('lgb_reg_lambda', 0.1),
            'colsample_bytree': cfg.get('lgb_feature_fraction', 0.8),
            'subsample': cfg.get('lgb_bagging_fraction', 0.8),
            'num_boost_round': cfg.get('lgb_n_estimators', 200),
            'num_threads': cfg.get('lgb_n_jobs', -1),
            'early_stopping_rounds': cfg.get('lgb_early_stopping_rounds', 50),
        }
        
        # Preserve Qlib defaults when these keys are absent.
        for source, target in (
            ('lgb_bagging_freq', 'bagging_freq'),
            ('lgb_min_child_samples', 'min_child_samples'),
            ('lgb_random_state', 'random_state'),
        ):
            if source in cfg:
                params[target] = cfg[source]

        # 移除None值
        params = {k: v for k, v in params.items() if v is not None}
        
        return params
    
    def _map_config_to_native_params(self) -> dict:
        """
        将现有配置映射到原生LightGBM参数
        """
        cfg = self.config
        
        objective = cfg.get('objective', 'regression')
        
        params = {
            'objective': objective,
            'metric': 'mae' if objective == 'regression_l1' else cfg.get('metric', 'rmse' if objective == 'regression' else 'ndcg'),
            'learning_rate': cfg.get('lgb_learning_rate', 0.01),
            'max_depth': cfg.get('lgb_max_depth', 3),
            'num_leaves': cfg.get('lgb_num_leaves', 7),
            'reg_alpha': cfg.get('lgb_reg_alpha', 0.1),
            'reg_lambda': cfg.get('lgb_reg_lambda', 0.1),
            'feature_fraction': cfg.get('lgb_feature_fraction', 0.8),
            'bagging_fraction': cfg.get('lgb_bagging_fraction', 0.8),
            'bagging_freq': cfg.get('lgb_bagging_freq', 5),
            'min_child_samples': cfg.get('lgb_min_child_samples', 20),
            'n_estimators': cfg.get('lgb_n_estimators', 200),
            'random_state': cfg.get('lgb_random_state', 42),
            'n_jobs': cfg.get('lgb_n_jobs', -1),
            'verbose': -1,
        }
        
        # Ranking specific
        if objective in RANKING_OBJECTIVES:
            params['eval_at'] = cfg.get('ndcg_eval_at', [1, 3, 5])
            params['label_gain'] = [0, 1, 3, 7, 15]
        
        return params
    
    def _generate_param_grid(self, n_samples: int, n_features: int) -> List[Dict]:
        """
        Generate adaptive parameter grid for search
        """
        cfg = self.config
        is_small_sample = n_samples < 3000
        
        if is_small_sample:
            max_depths = [2, 3]
            num_leaves_options = [3, 7]
            learning_rates = [0.005, 0.01]
        else:
            max_depths = [3, 4]
            num_leaves_options = [7, 15]
            learning_rates = [0.01, 0.05]
        
        # Override from config if provided
        max_depths = cfg.get('lgb_max_depth_grid', max_depths)
        num_leaves_options = cfg.get('lgb_num_leaves_grid', num_leaves_options)
        learning_rates = cfg.get('lgb_lr_grid', learning_rates)
        
        param_grid = []
        for lr in learning_rates:
            for depth in max_depths:
                for leaves in num_leaves_options:
                    params = {
                        'learning_rate': lr,
                        'max_depth': depth,
                        'num_leaves': leaves,
                        'reg_alpha': cfg.get('lgb_reg_alpha', 0.1),
                        'reg_lambda': cfg.get('lgb_reg_lambda', 0.1),
                        'n_estimators': cfg.get('lgb_n_estimators', 200),
                        'feature_fraction': cfg.get('lgb_feature_fraction', 0.8),
                        'bagging_fraction': cfg.get('lgb_bagging_fraction', 0.8),
                        'min_child_samples': cfg.get('lgb_min_child_samples', 20),
                    }
                    param_grid.append(params)
        
        max_candidates = cfg.get('lgb_search_n_candidates', 12)
        if len(param_grid) > max_candidates:
            step = len(param_grid) / max_candidates
            param_grid = [param_grid[int(i * step)] for i in range(max_candidates)]
        
        return param_grid
    
    def _run_param_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[Dict, Dict]:
        """
        Run parameter search with custom CV metric
        
        Returns:
            best_params: dict
            search_info: dict with search results
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        self.logger.info(f"[QlibLGBWrapper] 开始参数搜索, metric={self.param_search_metric}")
        
        param_grid = self._generate_param_grid(len(X_train), len(X_train.columns))
        
        # Initialize CV metric evaluator
        evaluator = CVMetricEvaluator(
            self.param_search_metric, self.config, self.logger, n_trials=len(param_grid)
        )
        
        best_score = float('-inf')
        best_params = param_grid[0] if param_grid else {}
        results = []
        
        for idx, params in enumerate(param_grid):
            try:
                # Train model with these params
                full_params = {
                    **params,
                    'objective': 'regression',
                    'metric': 'rmse',
                    'random_state': self.config.get('lgb_random_state', 42),
                    'n_jobs': self.config.get('lgb_n_jobs', -1),
                    'verbose': -1,
                }
                
                model = lgb.LGBMRegressor(**full_params)
                
                if sample_weight is not None:
                    # Handle sample weight for train set only
                    train_weight = sample_weight[:len(X_train)] if len(sample_weight) >= len(X_train) else None
                    model.fit(X_train.values, y_train.values, sample_weight=train_weight)
                else:
                    model.fit(X_train.values, y_train.values)
                
                # Predict on validation
                y_pred = model.predict(X_valid.values)
                
                # Evaluate with custom metric
                dates = X_valid.index.get_level_values('datetime')
                score = (
                    evaluator.evaluate(
                        y_pred, y_valid.values, dates,
                        X_valid.index.get_level_values('instrument'),
                    )
                    if self.param_search_metric == 'net_topk_dsr'
                    else evaluator.evaluate(y_pred, y_valid.values, dates)
                )
                
                # Track train score for overfitting check
                y_pred_train = model.predict(X_train.values)
                train_dates = X_train.index.get_level_values('datetime')
                train_score = (
                    evaluator.evaluate(
                        y_pred_train, y_train.values, train_dates,
                        X_train.index.get_level_values('instrument'),
                    )
                    if self.param_search_metric == 'net_topk_dsr'
                    else evaluator.evaluate(y_pred_train, y_train.values, train_dates)
                )

                if not (np.isfinite(score) and np.isfinite(train_score)):
                    continue
                
                gap = train_score - score
                max_gap = self.config.get('lgb_max_overfitting_gap', 0.1)
                
                # Apply overfitting penalty
                gap_excess = max(0, gap - max_gap)
                final_score = score - 0.5 * gap_excess

                if not np.isfinite(final_score):
                    continue
                
                results.append({
                    'params': params,
                    'val_score': score,
                    'train_score': train_score,
                    'gap': gap,
                    'final_score': final_score,
                })
                
                if final_score > best_score:
                    best_score = final_score
                    best_params = params
                
                self.logger.debug(
                    f"[参数搜索] {idx+1}/{len(param_grid)}: "
                    f"val={score:.4f}, train={train_score:.4f}, gap={gap:.4f}"
                )
                
            except LabelIsolationError:
                raise
            except Exception as e:
                self.logger.warning(f"[参数搜索] 参数组合 {idx+1} 失败: {e}")
                continue
        
        if not results:
            raise _NoFiniteSearchWinnerError(
                "LightGBM parameter search has no candidate with finite train/validation/final scores"
            )

        # Sort results
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        search_info = {
            'n_candidates': len(param_grid),
            'n_evaluated': len(results),
            'best_val_score': results[0]['val_score'] if results else 0,
            'best_train_score': results[0]['train_score'] if results else 0,
            'metric': self.param_search_metric,
            'top3_results': results[:3],
        }
        
        self.logger.info(
            f"[QlibLGBWrapper] 参数搜索完成, "
            f"最佳score={search_info['best_val_score']:.4f}"
        )
        
        return best_params, search_info
        
    def fit(
        self,
        dataset: FactorPilotDataset,
        sample_weight: Optional[np.ndarray] = None
    ) -> dict:
        """
        训练模型
        
        Args:
            dataset: FactorPilotDataset, 由DataConverter.create_dataset_from_factors创建
            sample_weight: 可选的样本权重 (已弃用，内部重新计算以避免维度不匹配)
            
        Returns:
            model_info: dict
                - method: str = "lightgbm"
                - objective: str
                - feature_importance: Dict[str, float]
                - best_iteration: int
                - n_features: int
                - train_score: float
                - feature_names: List[str]
        """
        self.logger.info("[QlibLGBWrapper] 开始训练模型...")
        
        # 准备训练和验证数据
        df_train = dataset.prepare("train", col_set=["feature", "label"])
        df_valid = dataset.prepare("valid", col_set=["feature", "label"])
        
        # Apply the exact eligible keys before weights, search or any backend fit.
        # Qlib prepares the dataset again, so filtering only local arrays is unsafe.
        exits = resolve_label_exit_dates(
            df_train.index, self.config.get("_label_exit_dates"),
            trading_calendar=self.config.get("_label_trading_calendar"),
            label_period=self.config.get("_label_effective_period"),
            calendar_source=self.config.get("_label_calendar_source"),
        )
        valid_dates = pd.DatetimeIndex(df_valid.index.get_level_values(0)).normalize()
        if not len(valid_dates) or valid_dates.hasnans or valid_dates.tz is not None:
            raise LabelIsolationError("LightGBM validation requires valid nonempty dates")
        eligible = exits < valid_dates.min()
        if not eligible.any():
            raise LabelIsolationError("LightGBM has no training labels exiting before validation")
        if not eligible.all():
            rejected_keys = df_train.index[~eligible.to_numpy()]
            dataset = copy(dataset)
            dataset.feature_df = dataset.feature_df.loc[
                ~dataset.feature_df.index.isin(rejected_keys)
            ]
            dataset._segment_cache = {}
            df_train = dataset.prepare("train", col_set=["feature", "label"])

        X_train = df_train["feature"]
        y_train = df_train["label"]
        X_valid = df_valid["feature"]
        y_valid = df_valid["label"]
        
        # 处理标签维度
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        if isinstance(y_valid, pd.DataFrame):
            y_valid = y_valid.iloc[:, 0]
        
        self.feature_names = list(X_train.columns)
        
        self.logger.info(
            f"[QlibLGBWrapper] 训练数据: {len(X_train)} 样本, {len(self.feature_names)} 特征"
        )
        
        # ---- Sample weight computation (aligned with QlibLinearWrapper) ----
        # Deferred import to avoid circular: qlib_adapter <-> pipeline
        from factor_synthesis.pipeline.sample_weighting import (
            compute_return_based_weights,
            combine_weights,
        )

        # Step 1: Time-decay weights
        time_weights: np.ndarray | None = None
        weight_decay = self.config.get("sample_weight_decay", 1.0)
        half_life = float(self.config.get("sample_weight_half_life", 0.0))
        if weight_decay < 1.0 or half_life > 0:
            try:
                train_dates = pd.to_datetime(
                    X_train.index.get_level_values("datetime")
                )
                max_date = train_dates.max()
                days_diff = np.array(
                    [(max_date - d).days for d in train_dates], dtype=float
                )
                if half_life > 0:
                    time_weights = np.exp(-np.log(2) * days_diff / half_life)
                else:
                    time_weights = np.power(weight_decay, days_diff)
                time_weights = time_weights / time_weights.mean()
                self.logger.info(
                    "[QlibLGBWrapper] 应用时间衰减样本权重 (decay=%.4f, half_life=%.1f, n=%d)",
                    weight_decay, half_life, len(time_weights),
                )
            except LabelIsolationError:
                raise
            except Exception as e:
                self.logger.warning(
                    f"[QlibLGBWrapper] 计算时间衰减权重失败: {e}"
                )
                time_weights = None

        # Step 2: Return-based weights (from raw returns, not preprocessed labels)
        # Preprocessed labels (cs_rank + cs_demean) are demeaned ranks, which
        # distort magnitude-based weighting methods. Use raw returns instead.
        return_weights: np.ndarray | None = None
        try:
            raw_ret = getattr(dataset, 'raw_returns', None)
            if raw_ret is not None:
                y_for_weights = raw_ret.reindex(X_train.index).values
                if np.isnan(y_for_weights).any():
                    self.logger.warning(
                        "[QlibLGBWrapper] raw_returns contains NaN, "
                        "falling back to preprocessed labels"
                    )
                    y_for_weights = y_train.values if isinstance(y_train, pd.Series) else y_train
            else:
                y_for_weights = y_train.values if isinstance(y_train, pd.Series) else y_train

            return_weights = compute_return_based_weights(
                y_for_weights, X_train.index, self.config, log=self.logger,
            )
        except Exception as e:
            self.logger.warning(f"[QlibLGBWrapper] 计算收益率权重失败: {e}")
            raise

        # Step 3: Combine multiplicatively (configurable)
        sw_cfg = self.config.get("sample_weight_config", {})
        do_combine = sw_cfg.get("combine_with_time_decay", True)
        sample_weight = combine_weights(time_weights, return_weights, combine=do_combine)
        
        # Parameter search if enabled
        param_search_info = None
        if self.enable_param_search and self.param_search_metric not in ('mse', 'rmse', None):
            try:
                best_params, param_search_info = self._run_param_search(
                    X_train, y_train, X_valid, y_valid, sample_weight
                )
                # Update config with best params
                for k, v in best_params.items():
                    self.config[f'lgb_{k}'] = v
            except _NoFiniteSearchWinnerError:
                raise
            except LabelIsolationError:
                raise
            except Exception as e:
                self.logger.warning(f"[QlibLGBWrapper] 参数搜索失败: {e}")
        
        # Train with custom objective or native
        if self.custom_objective_name and not self._is_ranking:
            model_info = self._fit_with_custom_objective(
                X_train, y_train, X_valid, y_valid, sample_weight
            )
        elif (self._use_qlib and QLIB_LGB_AVAILABLE and not self._is_ranking
              and not self.custom_objective_name
              and self.config.get('objective', 'regression') != 'regression_l1'):
            keyed_weight = (None if sample_weight is None else
                            pd.Series(sample_weight, index=X_train.index))
            model_info = self._fit_with_qlib(dataset, keyed_weight)
        else:
            model_info = self._fit_with_native(
                X_train, y_train, X_valid, y_valid, sample_weight
            )
        
        if param_search_info:
            model_info['param_search_info'] = param_search_info
        
        return model_info
    
    def _fit_with_custom_objective(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> dict:
        """使用自定义目标函数训练 (Pearson IC Loss / Sharpe Surrogate Loss)"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM未安装")
        
        self.logger.info(
            f"[QlibLGBWrapper] 使用自定义目标函数: {self.custom_objective_name}"
        )
        
        # Get custom objective function
        objective_fn = QlibCustomObjectives.get_lgb_objective(self.custom_objective_name)

        # This wrapper does not supply a nonconstant init_score. IC callbacks
        # return zero gradients at its constant initial predictions.
        if self.custom_objective_name in {'pearson_ic', 'global_ic', 'cross_sectional_ic'}:
            raise ValueError(
                "IC training cannot start from constant predictions. "
                "Set lgb_custom_objective=None to use the configured standard objective; "
                "nonconstant initialization is not supported by this wrapper."
            )
        
        # Select appropriate eval metric based on objective type
        # Sharpe objectives use sharpe eval metric, others use IC
        sharpe_objectives = {'sharpe_surrogate', 'topk_sharpe'}
        if self.custom_objective_name in sharpe_objectives:
            eval_metric_fn = QlibCustomObjectives.get_lgb_eval_metric('sharpe')
            self.logger.info("[QlibLGBWrapper] 使用 Sharpe 评估指标")
        else:
            eval_metric_fn = QlibCustomObjectives.get_lgb_eval_metric('ic')
        
        # Prepare params
        params = self._map_config_to_native_params()
        fit_options = {}
        early_rounds = self.config.get('lgb_early_stopping_rounds')
        if early_rounds is not None and early_rounds > 0:
            fit_options['callbacks'] = [lgb.early_stopping(early_rounds, first_metric_only=True)]
        
        # Check if we need groups for cross-sectional objectives
        # Both cross-sectional IC and Sharpe objectives benefit from group information
        cross_sectional_objectives = {'cross_sectional_ic', 'sharpe_surrogate', 'topk_sharpe'}
        use_native_api = self.custom_objective_name in cross_sectional_objectives
        
        if use_native_api:
            # Use native lgb.train() API for access to groups
            # Group sizes are date-sorted; rows must follow the same order.
            # Keep each date's existing order and align labels/weights by position.
            train_order = np.argsort(X_train.index.get_level_values('datetime'), kind='stable')
            valid_order = np.argsort(X_valid.index.get_level_values('datetime'), kind='stable')
            X_train, y_train = X_train.iloc[train_order], y_train.iloc[train_order]
            X_valid, y_valid = X_valid.iloc[valid_order], y_valid.iloc[valid_order]
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight)[train_order]
            groups = X_train.groupby(level='datetime').size().values
            
            dtrain = lgb.Dataset(
                X_train.values, 
                label=y_train.values,
                group=groups,
                weight=sample_weight
            )
            dvalid = lgb.Dataset(
                X_valid.values,
                label=y_valid.values,
                group=X_valid.groupby(level='datetime').size().values,
            )
            
            # LightGBM 4.0+ requires custom objective via params['objective']
            # instead of the deprecated 'fobj' parameter
            lgb_params = {
                'objective': objective_fn,  # Custom objective function
                'learning_rate': params.get('learning_rate', 0.01),
                'max_depth': params.get('max_depth', 3),
                'num_leaves': params.get('num_leaves', 7),
                'reg_alpha': params.get('reg_alpha', 0.1),
                'reg_lambda': params.get('reg_lambda', 0.1),
                'feature_fraction': params.get('feature_fraction', 0.8),
                'bagging_fraction': params.get('bagging_fraction', 0.8),
                'bagging_freq': params.get('bagging_freq', 5),
                'min_child_samples': params.get('min_child_samples', 20),
                'seed': params.get('random_state', 42),
                'verbose': -1,
            }
            
            # Try modern API first (LightGBM 4.0+), fallback to legacy API
            try:
                self.model = lgb.train(
                    lgb_params,
                    dtrain,
                    num_boost_round=params.get('n_estimators', 200),
                    valid_sets=[dvalid],
                    valid_names=['valid'],
                    feval=eval_metric_fn,
                    **fit_options,
                )
            except TypeError as e:
                # Fallback for older LightGBM versions that use fobj parameter
                if 'fobj' in str(e) or 'objective' in str(e):
                    self.logger.warning(
                        "[QlibLGBWrapper] 检测到旧版 LightGBM API，使用 fobj 参数"
                    )
                    del lgb_params['objective']
                    self.model = lgb.train(
                        lgb_params,
                        dtrain,
                        num_boost_round=params.get('n_estimators', 200),
                        valid_sets=[dvalid],
                        valid_names=['valid'],
                        fobj=objective_fn,
                        feval=eval_metric_fn,
                        **fit_options,
                    )
                else:
                    raise
            
            feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importance().astype(float)
            ))
            best_iteration = self.model.best_iteration or params.get('n_estimators', 200)
            train_score = 0.0  # Not easily available for native API
            
        else:
            # Use sklearn API for global IC (no groups needed)
            params['objective'] = objective_fn
            
            self.model = lgb.LGBMRegressor(**params)
            
            fit_kwargs = {
                'X': X_train.values,
                'y': y_train.values,
                'eval_set': [(X_valid.values, y_valid.values)],
                'eval_names': ['valid'],
                'eval_metric': eval_metric_fn,
            }
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
            
            self.model.fit(**fit_kwargs, **fit_options)
            
            feature_importance = {
                name: float(imp)
                for name, imp in zip(self.feature_names, self.model.feature_importances_)
            }
            best_iteration = getattr(self.model, 'best_iteration_', 0) or self.model.n_estimators
            train_score = 0.0  # Custom objective, score not meaningful
        
        model_info = {
            'method': 'lightgbm',
            'objective': self.custom_objective_name,
            'custom_objective': True,
            'feature_importance': feature_importance,
            'best_iteration': int(best_iteration),
            'n_features': len(self.feature_names),
            'train_score': train_score,
            'feature_names': self.feature_names,
            'lgb_params': params,
        }
        
        self.logger.info(
            f"[QlibLGBWrapper] 自定义目标训练完成, best_iteration={best_iteration}"
        )
        
        return model_info
    
    def _fit_with_qlib(
        self,
        dataset: FactorPilotDataset,
        sample_weight: Optional[pd.Series] = None
    ) -> dict:
        """使用Qlib LGBModel训练"""
        params = self._map_config_to_qlib_params()
        
        self.logger.info(f"[QlibLGBWrapper] 使用Qlib LGBModel, params: {params}")
        
        self.qlib_model = LGBModel(**params)
        if sample_weight is None:
            self.qlib_model.fit(dataset)
        else:
            from qlib.data.dataset.weight import Reweighter

            valid_index = dataset.prepare("valid", col_set=["feature", "label"]).index
            if not sample_weight.index.is_unique or not valid_index.is_unique:
                raise ValueError("Qlib sample weights require unique sample keys")
            if (len(valid_index) == len(sample_weight)
                    and valid_index.isin(sample_weight.index).all()):
                raise ValueError("Qlib weighted train and validation keys must be distinguishable")

            class TrainingReweighter(Reweighter):
                def __init__(self):
                    pass

                def reweight(self, data):
                    # Qlib asks the same reweighter for train and valid. Preserve
                    # unweighted validation; align training by full sample keys.
                    if len(data) == len(valid_index) and data.index.isin(valid_index).all():
                        return None
                    if not data.index.is_unique or not data.index.isin(sample_weight.index).all():
                        raise ValueError("Qlib training keys are missing from sample weights")
                    return sample_weight.reindex(data.index).to_numpy()

            self.qlib_model.fit(dataset, reweighter=TrainingReweighter())
        
        # 提取模型信息
        booster = getattr(self.qlib_model, 'model', None)
        
        feature_importance = {}
        best_iteration = 0
        
        if booster is not None:
            importance = booster.feature_importance()
            feature_importance = {
                name: float(imp) 
                for name, imp in zip(self.feature_names, importance)
            }
            best_iteration = getattr(booster, 'best_iteration', 0) or 0
        
        model_info = {
            'method': 'lightgbm',
            'objective': self.config.get('objective', 'regression'),
            'feature_importance': feature_importance,
            'best_iteration': best_iteration,
            'n_features': len(self.feature_names),
            'train_score': 0.0,
            'feature_names': self.feature_names,
            'lgb_params': params,
        }
        
        self.logger.info(
            f"[QlibLGBWrapper] Qlib训练完成, best_iteration={best_iteration}"
        )
        
        return model_info
    
    def _fit_with_native(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> dict:
        """使用原生LightGBM训练"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM未安装")
        
        params = self._map_config_to_native_params()
        fit_options = {}
        early_rounds = self.config.get('lgb_early_stopping_rounds')
        if early_rounds is not None and early_rounds > 0:
            fit_options['callbacks'] = [lgb.early_stopping(early_rounds, first_metric_only=True)]
        
        self.logger.info(f"[QlibLGBWrapper] 使用原生LightGBM, objective={params.get('objective')}")
        
        if self._is_ranking:
            train_order = np.argsort(X_train.index.get_level_values('datetime'), kind='stable')
            valid_order = np.argsort(X_valid.index.get_level_values('datetime'), kind='stable')
            X_train, y_train = X_train.iloc[train_order], y_train.iloc[train_order]
            X_valid, y_valid = X_valid.iloc[valid_order], y_valid.iloc[valid_order]
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight)[train_order]
            # Ranking模式需要group信息
            group_train = X_train.groupby(level='datetime').size().values
            group_valid = X_valid.groupby(level='datetime').size().values
            
            self.model = lgb.LGBMRanker(**params)
            fit_kwargs = {
                'X': X_train.values,
                'y': y_train.values,
                'group': group_train,
                'eval_set': [(X_valid.values, y_valid.values)],
                'eval_group': [group_valid],
                'eval_names': ['valid'],
            }
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
            self.model.fit(**fit_kwargs, **fit_options)
        else:
            self.model = lgb.LGBMRegressor(**params)
            fit_kwargs = {
                'X': X_train.values,
                'y': y_train.values,
                'eval_set': [(X_valid.values, y_valid.values)],
                'eval_names': ['valid'],
            }
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
            
            self.model.fit(**fit_kwargs, **fit_options)
        
        # 提取模型信息
        feature_importance = {
            name: float(imp)
            for name, imp in zip(self.feature_names, self.model.feature_importances_)
        }
        
        best_iteration = getattr(self.model, 'best_iteration_', 0) or self.model.n_estimators
        
        train_score = 0.0
        if not self._is_ranking:
            train_score = float(self.model.score(X_train.values, y_train.values))
        
        model_info = {
            'method': 'lightgbm',
            'objective': self.config.get('objective', 'regression'),
            'feature_importance': feature_importance,
            'best_iteration': best_iteration,
            'n_features': len(self.feature_names),
            'train_score': train_score,
            'feature_names': self.feature_names,
            'lgb_params': params,
        }
        
        self.logger.info(
            f"[QlibLGBWrapper] 原生训练完成, best_iteration={best_iteration}, "
            f"train_score={train_score:.4f}"
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
        # 准备预测数据
        df_segment = dataset.prepare(segment, col_set="feature")
        X = df_segment if isinstance(df_segment, pd.DataFrame) else df_segment["feature"]

        if not self.feature_names or not pd.Index(self.feature_names).is_unique:
            raise ValueError("Fitted feature names are missing or duplicated")
        if not X.columns.is_unique:
            raise ValueError("Prediction feature columns must be unique")
        if set(X.columns) != set(self.feature_names):
            raise ValueError("Prediction feature columns must exactly match fitted feature names")
        X = X.loc[:, self.feature_names]
        
        # 预测
        # [FIX] Use underlying booster directly instead of qlib_model.predict()
        # to avoid requiring qlib.init() which is not needed for our use case
        if self.qlib_model is not None:
            # Get the underlying LightGBM booster from Qlib model
            booster = getattr(self.qlib_model, 'model', None)
            if booster is not None:
                pred_values = booster.predict(X.values)
                predictions = pd.Series(pred_values, index=X.index, name='prediction')
            else:
                raise ValueError("Qlib模型内部booster为空")
        elif self.model is not None:
            # Handle both sklearn API and native API
            if hasattr(self.model, 'predict'):
                pred_values = self.model.predict(X.values)
            else:
                # Native Booster
                pred_values = self.model.predict(X.values)
            predictions = pd.Series(pred_values, index=X.index, name='prediction')
        else:
            raise ValueError("模型未训练")
        
        self.logger.info(
            f"[QlibLGBWrapper] 预测完成, segment={segment}, "
            f"样本数={len(predictions)}"
        )
        
        return predictions
    
    def save_model(self, path: str):
        """保存模型到文件"""
        if path:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        if self.qlib_model is not None:
            booster = getattr(self.qlib_model, 'model', None)
            if booster is not None:
                booster.save_model(path)
        elif self.model is not None:
            if hasattr(self.model, 'booster_'):
                self.model.booster_.save_model(path)
            elif hasattr(self.model, 'save_model'):
                self.model.save_model(path)
        
        self.logger.info(f"[QlibLGBWrapper] 模型已保存到: {path}")
