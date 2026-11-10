"""
Factor Composer Module

Main class for composing multiple quantitative factors into synthetic factors using
LASSO regression, PCA dimensionality reduction, and LightGBM with rolling window mechanism.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Machine learning imports
from sklearn.linear_model import Lasso, Ridge
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# LightGBM import with fallback
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def _labels_available(available_at, cutoff):
    """Shared deployment/purge rule: equality is observable after the close."""
    values = pd.DatetimeIndex(available_at)
    return values.notna() & (values <= cutoff)


@dataclass
class SynthesisConfig:
    """Configuration class for factor synthesis parameters"""
    method: str = "lasso"  # "lasso", "pca", "ridge", "lightgbm"
    training_window: int = 252  # Trading days for training
    prediction_window: int = 20  # Trading days for prediction
    update_frequency: int = 20  # How often to retrain (in trading days)
    
    # LASSO specific parameters
    lasso_cv_folds: int = 5
    lasso_max_iter: int = 2000
    lasso_alpha_range: Tuple[float, float] = (0.001, 1.0)
    lasso_n_alphas: int = 100
    
    # PCA specific parameters  
    pca_n_components: Optional[int] = None  # None for auto selection based on variance
    pca_variance_threshold: float = 0.95  # Explained variance threshold
    
    # Ridge specific parameters
    ridge_cv_folds: int = 5
    ridge_alpha_range: Tuple[float, float] = (0.1, 100.0)
    ridge_n_alphas: int = 50
    
    # LightGBM specific parameters
    lgb_n_estimators: int = 100  # Number of boosting iterations
    lgb_learning_rate: float = 0.1  # Learning rate
    lgb_max_depth: int = 6  # Maximum tree depth
    lgb_num_leaves: int = 31  # Maximum number of leaves in one tree
    lgb_feature_fraction: float = 0.8  # Feature sampling ratio
    lgb_bagging_fraction: float = 0.8  # Data sampling ratio  
    lgb_bagging_freq: int = 5  # Frequency for bagging
    lgb_min_child_samples: int = 20  # Minimum samples in a leaf
    lgb_reg_alpha: float = 0.0  # L1 regularization
    lgb_reg_lambda: float = 0.0  # L2 regularization
    lgb_early_stopping_rounds: int = 20  # Early stopping patience
    lgb_validation_fraction: float = 0.2  # Validation set ratio for LightGBM
    lgb_random_state: int = 42  # Random seed
    lgb_n_jobs: int = -1  # Number of parallel jobs
    lgb_verbose: int = -1  # Verbosity level
    
    # Data preprocessing
    scaler_type: str = "robust"  # "standard", "robust", "none"
    outlier_threshold: float = 3.0  # Z-score threshold for outliers
    min_observations: int = 50  # Clean sample rows, not dates
    min_training_dates: int = 2
    
    # Performance and validation
    validation_ratio: float = 0.2  # Ratio of data used for validation
    performance_metrics: List[str] = None
    
    def __post_init__(self):
        for name in ("training_window", "update_frequency", "min_training_dates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.performance_metrics is None:
            self.performance_metrics = ["mse", "r2", "ic", "ir"]


class FactorComposer:
    """
    Main class for synthesizing multiple quantitative factors into composite factors.
    
    Supports LASSO regression, Ridge regression, PCA dimensionality reduction, and 
    LightGBM gradient boosting with rolling window mechanism for time-series analysis.
    """
    
    def __init__(
        self,
        config: SynthesisConfig,
        factor_reader,  # FactorReader instance
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the FactorComposer.
        
        Args:
            config: Configuration object with synthesis parameters
            factor_reader: FactorReader instance for database operations
            logger: Optional logger instance
        """
        self.config = config
        self.factor_reader = factor_reader
        self.logger = logger or logging.getLogger(__name__)
        
        # Check LightGBM availability if needed
        if config.method == "lightgbm" and not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Please install it with: pip install lightgbm"
            )
        
        # Initialize models and scalers
        self._initialize_models()
        
        # Storage for results and diagnostics
        self.synthesis_results = {}
        self.performance_metrics = {}
        self.feature_importance = {}
        self.training_history = []
        self.synthesis_diagnostics = {}
        
        self.logger.info(f"FactorComposer initialized with method: {config.method}")
    
    def _initialize_models(self):
        """Initialize machine learning models and preprocessing components."""
        # Initialize scaler
        # NOTE: For lasso/ridge, skip scaler - data is already rank_zscore normalized
        # This aligns with cloud behavior where rank_zscore -> direct linear weighting
        if self.config.method in ["lasso", "ridge"]:
            self.scaler = None  # Skip scaler for linear methods
            self.logger.info("[FactorComposer] Skipping scaler for lasso/ridge (data is rank_zscore normalized)")
        elif self.config.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None
        
        # Initialize models based on method
        if self.config.method == "lasso":
            alphas = np.logspace(
                np.log10(self.config.lasso_alpha_range[0]),
                np.log10(self.config.lasso_alpha_range[1]),
                self.config.lasso_n_alphas
            )
            self.model = Lasso(alpha=alphas[0], max_iter=self.config.lasso_max_iter,
                               random_state=42)
        
        elif self.config.method == "ridge":
            alphas = np.logspace(
                np.log10(self.config.ridge_alpha_range[0]),
                np.log10(self.config.ridge_alpha_range[1]),
                self.config.ridge_n_alphas
            )
            self.model = Ridge(alpha=alphas[0])
        
        elif self.config.method == "pca":
            if self.config.pca_n_components is None:
                # Auto-determine components based on variance threshold
                self.model = PCA(random_state=42)
            else:
                self.model = PCA(
                    n_components=self.config.pca_n_components,
                    random_state=42
                )
        
        elif self.config.method == "lightgbm":
            # Initialize LightGBM regressor with configuration parameters
            self.model = lgb.LGBMRegressor(
                n_estimators=self.config.lgb_n_estimators,
                learning_rate=self.config.lgb_learning_rate,
                max_depth=self.config.lgb_max_depth,
                num_leaves=self.config.lgb_num_leaves,
                feature_fraction=self.config.lgb_feature_fraction,
                bagging_fraction=self.config.lgb_bagging_fraction,
                bagging_freq=self.config.lgb_bagging_freq,
                min_child_samples=self.config.lgb_min_child_samples,
                reg_alpha=self.config.lgb_reg_alpha,
                reg_lambda=self.config.lgb_reg_lambda,
                random_state=self.config.lgb_random_state,
                n_jobs=self.config.lgb_n_jobs,
                verbose=self.config.lgb_verbose,
                objective='regression',
                metric='rmse'
            )
        
        else:
            raise ValueError(f"Unsupported synthesis method: {self.config.method}")
    
    def synthesize_factors(
        self,
        input_factor_names: List[str],
        target_returns: pd.DataFrame,
        start_date: str,
        end_date: str,
        stock_pool: Optional[List[str]] = None,
        composite_factor_name: Optional[str] = None,
        *,
        target_available_at: pd.DataFrame,
        trading_calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Main method to synthesize factors using rolling window approach.
        
        Args:
            input_factor_names: List of factor names to combine
            target_returns: Forward-return labels indexed by the signal date.
                Callers calculate the holding-period return before passing it
                here; this composer does not shift labels again.
            start_date: Start date for synthesis (YYYY-MM-DD)
            end_date: End date for synthesis (YYYY-MM-DD)
            stock_pool: Optional list of stock codes to focus on
            composite_factor_name: Name for the composite factor
            target_available_at: Actual label observation times, with target_returns axes.
            trading_calendar: Complete ordered trading dates, including days with missing X.
            
        Returns:
            DataFrame with composite factor values (date as index, stock codes as columns)
        """
        calendar = pd.DatetimeIndex(trading_calendar)
        if calendar.empty or calendar.hasnans or not calendar.is_unique or not calendar.is_monotonic_increasing:
            raise ValueError("trading_calendar must be nonempty, unique and ordered")
        if (not target_available_at.index.equals(target_returns.index)
                or not target_available_at.columns.equals(target_returns.columns)
                or not target_returns.index.is_unique or not target_returns.columns.is_unique):
            raise ValueError("target_available_at must have the same unique axes as target_returns")
        reserved = {"date", "stock", "target_return"}.intersection(input_factor_names)
        if reserved:
            raise ValueError(f"reserved factor names: {sorted(reserved)}")

        if composite_factor_name is None:
            method_suffix = self.config.method.upper()
            composite_factor_name = f"Composite_{method_suffix}_{len(input_factor_names)}F"
        
        self.logger.info(
            f"Synthesizing {len(input_factor_names)} factors using {self.config.method} "
            f"from {start_date} to {end_date}"
        )
        
        # Load factor data
        factor_data = {}
        for factor_name in input_factor_names:
            factor_df = self.factor_reader.get_factor_values(
                factor_name=factor_name,
                start_date=start_date,
                end_date=end_date,
                stock_pool=stock_pool
            )
            
            if not factor_df.index.is_unique or not factor_df.columns.is_unique:
                raise ValueError(f"duplicate factor keys: {factor_name}")
            if factor_df.empty:
                self.logger.warning(f"No data found for factor {factor_name}")
                continue
                
            factor_data[factor_name] = factor_df
        
        if not factor_data:
            self.logger.error("No valid factor data found")
            return pd.DataFrame()
        
        # Align all factor data to common dates and stocks
        aligned_factors = self._align_factor_data(factor_data)
        
        if aligned_factors.empty:
            self.logger.error("No aligned factor data available")
            return pd.DataFrame()
        
        # Align target returns
        aligned_returns = self._align_target_returns(aligned_factors, target_returns)
        
        availability = self._align_target_returns(aligned_factors, target_available_at)
        aligned_returns["available_at"] = pd.to_datetime(availability["target_return"])
        if not aligned_factors["date"].isin(calendar).all():
            raise ValueError("factor dates must belong to trading_calendar")
        calendar = calendar[(calendar >= pd.Timestamp(start_date)) & (calendar <= pd.Timestamp(end_date))]

        # Perform rolling window synthesis
        composite_factor_df = self._rolling_window_synthesis(
            aligned_factors, aligned_returns, composite_factor_name,
            trading_calendar=calendar,
        )
        
        if composite_factor_df.empty:
            self.synthesis_diagnostics.pop(composite_factor_name, None)

        # Store results for analysis
        self.synthesis_results[composite_factor_name] = {
            "factor_data": composite_factor_df,
            "input_factors": input_factor_names,
            "method": self.config.method,
            "config": self.config,
            "training_history": list(self.training_history),
        }
        
        self.logger.info(
            f"Synthesis completed. Result shape: {composite_factor_df.shape}"
        )
        
        return composite_factor_df
    
    def _align_factor_data(self, factor_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align multiple factor DataFrames to common dates and stocks.
        
        Args:
            factor_data: Dictionary mapping factor names to DataFrames
            
        Returns:
            Multi-level DataFrame with factors as columns
        """
        if not factor_data:
            return pd.DataFrame()
        
        # Find common dates and stocks
        common_dates = None
        common_stocks = None
        
        for factor_name, factor_df in factor_data.items():
            if common_dates is None:
                common_dates = set(factor_df.index)
                common_stocks = set(factor_df.columns)
            else:
                common_dates &= set(factor_df.index)
                common_stocks &= set(factor_df.columns)
        
        if not common_dates or not common_stocks:
            self.logger.error("No common dates or stocks found across factors")
            return pd.DataFrame()
        
        common_dates = sorted(list(common_dates))
        common_stocks = sorted(list(common_stocks))
        
        self.logger.info(
            f"Aligned to {len(common_dates)} dates and {len(common_stocks)} stocks"
        )
        
        # Create aligned DataFrame
        aligned_data = []
        
        for date in common_dates:
            for stock in common_stocks:
                row_data = {"date": date, "stock": stock}
                
                # Extract factor values for this date-stock combination
                for factor_name, factor_df in factor_data.items():
                    if date in factor_df.index and stock in factor_df.columns:
                        value = factor_df.loc[date, stock]
                        row_data[factor_name] = value if pd.notna(value) else np.nan
                    else:
                        row_data[factor_name] = np.nan
                
                aligned_data.append(row_data)
        
        aligned_df = pd.DataFrame(aligned_data)
        aligned_df["date"] = pd.to_datetime(aligned_df["date"])
        
        return aligned_df
    
    def _align_target_returns(
        self, factor_df: pd.DataFrame, returns_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align target returns with factor data.
        
        Args:
            factor_df: Aligned factor DataFrame
            returns_df: Target returns DataFrame
            
        Returns:
            Aligned returns DataFrame matching factor data structure
        """
        aligned_returns = []
        
        for _, row in factor_df.iterrows():
            date = row["date"]
            stock = row["stock"]
            
            # CompositeFactorBase already builds forward labels with
            # price.shift(-prediction_window), indexed by the signal date.
            # Shifting again changes the holding period and treats trading
            # dates as natural calendar dates.
            if (date in returns_df.index and
                stock in returns_df.columns and 
                pd.notna(returns_df.loc[date, stock])):
                return_value = returns_df.loc[date, stock]
            else:
                return_value = np.nan
            
            aligned_returns.append({
                "date": date,
                "stock": stock,
                "target_return": return_value
            })
        
        return pd.DataFrame(aligned_returns)
    
    def _rolling_window_synthesis(
        self, factor_df: pd.DataFrame, returns_df: pd.DataFrame,
        composite_factor_name: str, *, trading_calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Fit only observable historical labels; predict from features alone."""
        feature_cols = [col for col in factor_df if col not in ["date", "stock"]]
        # An object key cannot collide with a caller's legal factor name.
        availability_col = object()
        combined = factor_df.merge(returns_df.rename(columns={"available_at": availability_col}),
                                   on=["date", "stock"], how="left", validate="one_to_one")
        clean = combined.loc[np.isfinite(combined[feature_cols].to_numpy()).all(axis=1)].sort_values("date")
        self.training_history = []
        values, diagnostics = [], None
        model, last_fit, anchor = None, None, None
        for i, current_date in enumerate(trading_calendar):
            window = trading_calendar[max(0, i - self.config.training_window):i]
            training = clean.loc[clean["date"].isin(window)]
            if self.config.method != "pca":
                training = training.loc[
                    np.isfinite(training["target_return"].to_numpy())
                    & _labels_available(training[availability_col], current_date)
                ]
            eligible_rows = training
            if self.config.method != "pca" and self.config.outlier_threshold > 0 and not training.empty:
                kept, _ = self._remove_outliers(training.index.to_numpy(), training["target_return"].to_numpy())
                eligible_rows = training.loc[kept]
            eligible = (len(eligible_rows) >= self.config.min_observations
                        and eligible_rows["date"].nunique() >= self.config.min_training_dates)
            if anchor is None and eligible:
                anchor = i
            due = anchor is not None and (i - anchor) % self.config.update_frequency == 0
            retrained = False
            if due:
                candidate = None
                if eligible:
                    candidate = self._train_model_window(
                        training[feature_cols].to_numpy(), training["target_return"].to_numpy(),
                        dates=training["date"], available_at=training[availability_col],
                    )
                if candidate is not None:
                    model, last_fit, retrained = candidate, i, True
                elif model is not None:
                    self.logger.warning("拟合失败，沿用 %s 的模型", trading_calendar[last_fit])
                else:
                    self.logger.warning("拟合失败，无成功模型，跳过预测: %s", current_date)
            current = clean.loc[clean["date"] == current_date]
            self.training_history.append({
                "date": current_date, "attempted": due, "retrained": retrained,
                "training_date": None if last_fit is None else trading_calendar[last_fit],
                "carried_days": None if last_fit is None else i - last_fit,
                "predicted": model is not None and not current.empty,
            })
            if model is None or current.empty:
                continue
            X = current[feature_cols].to_numpy()
            if self.scaler is not None:
                X = self.scaler.transform(X)
            prediction = model.transform(X)[:, 0] if self.config.method == "pca" else model.predict(X)
            diagnostics = {"method": self.config.method, "features": list(feature_cols)}
            if self.config.method in ["lasso", "ridge"]:
                diagnostics["coefficients"] = model.coef_.copy()
            elif self.config.method == "lightgbm":
                diagnostics["importances"] = model.feature_importances_.copy()
            else:
                diagnostics["explained_variance"] = model.explained_variance_ratio_.copy()
            values.extend({"date": date, "stock": stock, "composite_factor": value}
                          for date, stock, value in zip(current["date"], current["stock"], prediction))
        if not values:
            return pd.DataFrame()
        result = pd.DataFrame(values).pivot(index="date", columns="stock", values="composite_factor")
        self.synthesis_diagnostics[composite_factor_name] = diagnostics
        return result

    def _validation_splits(self, dates, available_at):
        """Expanding complete-date folds, with the deployment maturity predicate."""
        dates = pd.DatetimeIndex(dates)
        unique = dates.unique().sort_values()
        if self.config.method == "lightgbm":
            count = max(1, int(np.ceil(len(unique) * self.config.lgb_validation_fraction)))
            if count >= len(unique):
                raise ValueError("not enough dates for temporal validation")
            blocks = [(unique[:-count], unique[-count:])]
        else:
            count = self.config.lasso_cv_folds if self.config.method == "lasso" else self.config.ridge_cv_folds
            blocks = [(unique[train], unique[val]) for train, val in TimeSeriesSplit(n_splits=count).split(unique)]
        folds = []
        for train_dates, val_dates in blocks:
            train = np.flatnonzero(dates.isin(train_dates) & _labels_available(available_at, val_dates[0]))
            val = np.flatnonzero(dates.isin(val_dates))
            if not len(train) or not len(val):
                raise ValueError("no usable purged training/validation rows")
            folds.append((train, val))
        return folds

    def _train_model_window(self, X_train, y_train, *, dates, available_at):
        """Fit candidates; install the model/scaler pair only after full success."""
        try:
            model = clone(self.model)
            scaler = clone(self.scaler) if self.scaler is not None else None
            if self.config.method == "pca":
                X = scaler.fit_transform(X_train) if scaler is not None else X_train
                if self.config.pca_n_components is None:
                    model = PCA(random_state=42).fit(X)
                    n = int(np.argmax(np.cumsum(model.explained_variance_ratio_) >= self.config.pca_variance_threshold)) + 1
                    model = PCA(n_components=n, random_state=42)
                model.fit(X)
            else:
                folds = self._validation_splits(dates, available_at)
                if self.config.outlier_threshold > 0:
                    # Each fold sees only its own training labels when filtering.
                    # Validation labels do not choose which training rows survive.
                    folds = [(self._remove_outliers(train, y_train[train])[0], val)
                             for train, val in folds]
                    if any(not len(train) for train, _ in folds):
                        raise ValueError("no training rows after fold outlier filtering")
                if self.config.method in ["lasso", "ridge"]:
                    method = self.config.method
                    limits = getattr(self.config, method + "_alpha_range")
                    alphas = np.logspace(np.log10(limits[0]), np.log10(limits[1]), getattr(self.config, method + "_n_alphas"))
                    best_loss, best_alpha = np.inf, None
                    for alpha in alphas:
                        loss, rows = 0., 0
                        for train, val in folds:
                            fold_model = clone(model).set_params(alpha=alpha)
                            fold_scaler = clone(scaler) if scaler is not None else None
                            X_fit = fold_scaler.fit_transform(X_train[train]) if fold_scaler is not None else X_train[train]
                            X_val = fold_scaler.transform(X_train[val]) if fold_scaler is not None else X_train[val]
                            fold_model.fit(X_fit, y_train[train])
                            loss += mean_squared_error(y_train[val], fold_model.predict(X_val)) * len(val)
                            rows += len(val)
                        if loss / rows < best_loss:
                            best_loss, best_alpha = loss / rows, alpha
                    if best_alpha is None:
                        raise ValueError("no finite validation loss")
                    model.set_params(alpha=best_alpha)
                else:
                    train, val = folds[0]
                    fold_scaler = clone(scaler) if scaler is not None else None
                    X_fit = fold_scaler.fit_transform(X_train[train]) if fold_scaler is not None else X_train[train]
                    X_val = fold_scaler.transform(X_train[val]) if fold_scaler is not None else X_train[val]
                    validation_model = clone(model).set_params(n_estimators=self.config.lgb_n_estimators)
                    validation_model.fit(X_fit, y_train[train], eval_set=[(X_val, y_train[val])],
                                         eval_metric="rmse", callbacks=[
                                             lgb.early_stopping(self.config.lgb_early_stopping_rounds),
                                             lgb.log_evaluation(period=0)])
                    model.set_params(n_estimators=validation_model.best_iteration_)
                # Refit with the original full-window outlier rule, after selection.
                fit_X, fit_y = X_train, y_train
                if self.config.outlier_threshold > 0:
                    fit_X, fit_y = self._remove_outliers(X_train, y_train)
                X = scaler.fit_transform(fit_X) if scaler is not None else fit_X
                model.fit(X, fit_y)
            self.model, self.scaler = model, scaler
            return model
        except Exception as exc:
            self.logger.warning("Model training failed for window: %s", exc)
            return None

    def _remove_outliers(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers based on Z-score threshold.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Filtered X and y arrays
        """
        # Calculate Z-scores for target variable
        y_mean = np.mean(y)
        y_std = np.std(y)
        
        if y_std == 0:
            return X, y
        
        z_scores = np.abs((y - y_mean) / y_std)
        mask = z_scores <= self.config.outlier_threshold
        
        return X[mask], y[mask]
    
    def get_feature_importance(self, composite_factor_name: str) -> Dict[str, float]:
        """
        Get feature importance for a composite factor (LASSO/Ridge/LightGBM).
        
        Args:
            composite_factor_name: Name of the composite factor
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if composite_factor_name not in self.synthesis_results:
            return {}
        
        snapshot = self.synthesis_diagnostics.get(composite_factor_name)
        if snapshot is None:
            return {}
        if snapshot["method"] in ["lasso", "ridge"]:
            return dict(zip(snapshot["features"], np.abs(snapshot["coefficients"])))
        if snapshot["method"] == "lightgbm":
            return dict(zip(snapshot["features"], snapshot["importances"]))
        return {}

    def get_explained_variance(self, composite_factor_name: str) -> Dict[str, float]:
        """
        Get explained variance for PCA composite factors.
        
        Args:
            composite_factor_name: Name of the composite factor
            
        Returns:
            Dictionary with variance information
        """
        if composite_factor_name not in self.synthesis_results:
            return {}
        
        snapshot = self.synthesis_diagnostics.get(composite_factor_name)
        if snapshot is None or snapshot["method"] != "pca":
            return {}
        variance = snapshot["explained_variance"]
        return {
            "explained_variance_ratio": variance.tolist(),
            "cumulative_variance": np.cumsum(variance).tolist(),
            "n_components": len(variance)
        }

    def evaluate_performance(
        self,
        composite_factor_name: str,
        actual_returns: pd.DataFrame
    ) -> Dict[str, Union[float, str]]:
        """
        Evaluate the performance of a composite factor.
        
        Args:
            composite_factor_name: Name of the composite factor
            actual_returns: Actual future returns for evaluation
            
        Returns:
            Metrics with daily cross-sectional IC/IR (sample standard deviation,
            no annualization) and an ic_version identifier. MSE/R2 use paired
            samples on their original scale. Undefined IC/IR remain NaN.
        """
        if composite_factor_name not in self.synthesis_results:
            return {}
        
        composite_df = self.synthesis_results[composite_factor_name]["factor_data"]
        
        # Align composite factor with actual returns
        aligned_data = []
        
        for date in composite_df.index:
            for stock in composite_df.columns:
                factor_value = composite_df.loc[date, stock]
                
                if (pd.notna(factor_value) and 
                    date in actual_returns.index and
                    stock in actual_returns.columns):
                    
                    actual_return = actual_returns.loc[date, stock]
                    
                    if pd.notna(actual_return):
                        aligned_data.append({
                            "date": date,
                            "factor_value": factor_value,
                            "actual_return": actual_return
                        })
        
        if not aligned_data:
            return {}
        
        eval_df = pd.DataFrame(aligned_data)
        
        # Calculate performance metrics
        metrics = {}
        
        if "ic" in self.config.performance_metrics or "ir" in self.config.performance_metrics:
            daily_ic = []
            for _, day in eval_df.groupby("date", sort=False):
                if (len(day) >= 2 and day["factor_value"].nunique() > 1
                        and day["actual_return"].nunique() > 1):
                    ic = day["factor_value"].corr(day["actual_return"], method="spearman")
                    if pd.notna(ic):
                        daily_ic.append(ic)
            valid_ic = pd.Series(daily_ic, dtype=float)
            mean_ic = valid_ic.mean()
            if "ic" in self.config.performance_metrics:
                metrics["ic"] = mean_ic
            if "ir" in self.config.performance_metrics:
                std_ic = valid_ic.std(ddof=1)
                metrics["ir"] = mean_ic / std_ic if len(valid_ic) >= 2 and std_ic > 0 else np.nan
            metrics["ic_version"] = "daily_cross_section_v1"

        if "mse" in self.config.performance_metrics:
            # Mean Squared Error
            mse = mean_squared_error(eval_df["actual_return"], eval_df["factor_value"])
            metrics["mse"] = mse
        
        if "r2" in self.config.performance_metrics:
            # R-squared
            r2 = r2_score(eval_df["actual_return"], eval_df["factor_value"])
            metrics["r2"] = r2
        
        self.performance_metrics[composite_factor_name] = metrics
        
        return metrics
