"""
Composite Factor Base Module

Wrapper class to integrate synthesized composite factors with the existing FactorBase
architecture, enabling seamless analysis with Alphalens and other tools.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
from pathlib import Path

# Import the base FactorBase class
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.factor_base import FactorBase
from utils.duckdb_results_writer import CommittedParquetExportError

# Import synthesis components
from .factor_reader import FactorReader
from .factor_composer import FactorComposer, SynthesisConfig


class CompositeFactorBase(FactorBase):
    """
    Wrapper class that integrates composite factors with the existing FactorBase architecture.
    
    This class allows synthesized factors to be used seamlessly with existing analysis
    pipelines, including Alphalens integration and performance evaluation.
    """
    
    def __init__(
        self,
        db_connection,
        composite_factor_name: str,
        input_factor_names: List[str],
        synthesis_config: SynthesisConfig,
        start_date: str,
        end_date: str,
        stock_pool: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        use_dynamic_pool: bool = False,
        stock_filters: Optional[List] = None,
        max_stocks_per_month: Optional[int] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        results_database_path: Optional[str] = None,
        auto_synthesize: bool = True,
        data_bindings: Optional[dict] = None,
        publication_metadata: Optional[dict] = None,
    ):
        """
        Initialize the CompositeFactorBase.
        
        Args:
            db_connection: Database connection for main data
            composite_factor_name: Name for the composite factor
            input_factor_names: List of factor names to combine
            synthesis_config: Configuration for factor synthesis
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)
            stock_pool: Optional fixed stock pool
            cache_dir: Optional cache directory
            use_dynamic_pool: Whether to use dynamic stock pools
            stock_filters: Stock filter conditions for dynamic pools
            max_stocks_per_month: Maximum stocks per month for dynamic pools
            sort_by: Sort field for stock selection
            ascending: Sort direction
            results_database_path: Path to results database (for factor values table)
            auto_synthesize: Whether to automatically synthesize on initialization
        """
        # Initialize the parent FactorBase
        super().__init__(
            db_connection=db_connection,
            factor_name=composite_factor_name,
            start_date=start_date,
            end_date=end_date,
            stock_pool=stock_pool,
            cache_dir=cache_dir,
            use_dynamic_pool=use_dynamic_pool,
            stock_filters=stock_filters,
            max_stocks_per_month=max_stocks_per_month,
            sort_by=sort_by,
            ascending=ascending,
            data_bindings=data_bindings,
        )
        
        # Store synthesis-specific parameters
        self.input_factor_names = input_factor_names
        self.synthesis_config = synthesis_config
        self.results_database_path = results_database_path
        self.auto_synthesize = auto_synthesize

        factor_store = results_database_path or "analysis_results.duckdb"
        # FactorReader is deliberately read-only at construction.  This
        # composite is an explicit output-producing API, so initialize its
        # fresh DuckDB store before attaching the reader.
        if not data_bindings and not Path(factor_store).exists():
            from utils.duckdb_results_writer import DuckDBResultsWriter
            DuckDBResultsWriter(factor_store)
        
        # Initialize synthesis components
        self.factor_reader = FactorReader(
            database_path=factor_store,
            logger=self.logger,
            data_bindings=data_bindings,
            publication_metadata=publication_metadata,
        )
        
        self.factor_composer = FactorComposer(
            config=synthesis_config,
            factor_reader=self.factor_reader,
            logger=self.logger
        )
        
        # Storage for synthesized factor data
        self._synthesized_factor_df = None
        self._synthesis_completed = False
        
        # Perform synthesis if requested
        if auto_synthesize:
            self._perform_synthesis()
        
        self.logger.info(
            f"CompositeFactorBase initialized: {composite_factor_name} "
            f"from {len(input_factor_names)} input factors"
        )
    
    def _perform_synthesis(self) -> bool:
        """
        Perform the factor synthesis process.
        
        Returns:
            True if synthesis succeeded, False otherwise
        """
        try:
            self.logger.info("Starting factor synthesis...")
            
            # Get target returns for training (future returns)
            target_returns = self._get_target_returns()
            
            if target_returns.empty and (self.synthesis_config.method != "pca" or self._trading_calendar.empty):
                self.logger.error("No target returns available for synthesis")
                return False
            
            # Perform synthesis using FactorComposer
            synthesized_df = self.factor_composer.synthesize_factors(
                input_factor_names=self.input_factor_names,
                target_returns=target_returns,
                start_date=self.start_date,
                end_date=self.end_date,
                stock_pool=self.all_stocks,
                composite_factor_name=self.factor_name,
                target_available_at=self._target_available_at,
                trading_calendar=self._trading_calendar,
            )
            
            if synthesized_df.empty:
                self.logger.error("Factor synthesis returned empty result")
                return False
            
            # Save to database if successful
            try:
                success = self.factor_reader.save_composite_factor(
                    composite_factor_df=synthesized_df,
                    factor_name=self.factor_name,
                    overwrite=True
                )
            except CommittedParquetExportError:
                # The database commit succeeded even though its export failed.
                self._synthesized_factor_df = synthesized_df
                self._synthesis_completed = True
                raise
            self._synthesized_factor_df = synthesized_df
            self._synthesis_completed = True
            
            if success:
                self.logger.info(f"Synthesis completed successfully: {synthesized_df.shape}")
            else:
                self.logger.warning("Synthesis completed but failed to save to database")
            
            return True
        
        except CommittedParquetExportError:
            raise
        except Exception as e:
            self.logger.error(f"Factor synthesis failed: {e}")
            return False
    
    def _get_target_returns(self) -> pd.DataFrame:
        """
        Get target returns for synthesis training.
        
        Returns:
            DataFrame with future returns
        """
        # Use the built-in method from FactorBase to get price data and calculate returns
        self._target_available_at = pd.DataFrame()
        self._trading_calendar = pd.DatetimeIndex([])
        try:
            # Get price data for the extended period to calculate future returns
            price_df = self.get_price_data(price_type='close')
            
            if price_df.empty:
                return pd.DataFrame()
            
            # Calculate future returns based on prediction window
            prediction_days = self.synthesis_config.prediction_window
            future_returns = price_df.shift(-prediction_days) / price_df - 1
            
            # The same positional observation used by the label supplies its
            # availability time; never estimate it with natural-day offsets.
            self._trading_calendar = pd.DatetimeIndex(price_df.index).sort_values().unique()
            observed_at = pd.DataFrame({stock: price_df.index for stock in price_df.columns},
                                       index=price_df.index).where(price_df.notna())
            self._target_available_at = observed_at.shift(-prediction_days).where(future_returns.notna())

            # Remove the last few rows that don't have future data
            future_returns = future_returns.iloc[:-prediction_days] if prediction_days > 0 else future_returns
            
            self._target_available_at = self._target_available_at.loc[future_returns.index]
            self.logger.info(f"Generated target returns with shape: {future_returns.shape}")
            
            return future_returns
        
        except Exception as e:
            self.logger.error(f"Failed to get target returns: {e}")
            return pd.DataFrame()
    
    def _get_raw_data_for_factor(self) -> pd.DataFrame:
        """
        Implementation of FactorBase abstract method.
        
        For composite factors, this returns the input factor data in the format
        expected by FactorBase (long format with date, code, and factor columns).
        
        Returns:
            Raw data DataFrame in long format
        """
        # If synthesis hasn't been performed yet, do it now
        if not self._synthesis_completed:
            if not self._perform_synthesis():
                return pd.DataFrame()
        
        # Check if we already have synthesized factor data
        if self._synthesized_factor_df is None or self._synthesized_factor_df.empty:
            # Try to load from database
            synthesized_df = self.factor_reader.get_factor_values(
                factor_name=self.factor_name,
                start_date=self.start_date,
                end_date=self.end_date,
                stock_pool=self.all_stocks,
                pivot_format=True
            )
            
            if synthesized_df.empty:
                self.logger.error("No synthesized factor data available")
                return pd.DataFrame()
            
            self._synthesized_factor_df = synthesized_df
        
        # Convert pivoted DataFrame to long format expected by FactorBase
        factor_df = self._synthesized_factor_df
        
        # Create long format DataFrame
        long_data = []
        
        for date in factor_df.index:
            for stock_code in factor_df.columns:
                factor_value = factor_df.loc[date, stock_code]
                
                if pd.notna(factor_value):
                    long_data.append({
                        'date': date,
                        'code': stock_code,
                        'composite_factor': factor_value
                    })
        
        if not long_data:
            return pd.DataFrame()
        
        raw_data_df = pd.DataFrame(long_data)
        raw_data_df['date'] = pd.to_datetime(raw_data_df['date'])
        
        self.logger.info(f"Prepared raw data for factor calculation: {raw_data_df.shape}")
        
        return raw_data_df
    
    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of FactorBase abstract method.
        
        For composite factors, this simply extracts the pre-calculated factor values.
        
        Args:
            data: Raw data DataFrame with composite_factor column
            
        Returns:
            Factor values in the format expected by FactorBase
        """
        if data.empty or 'composite_factor' not in data.columns:
            self.logger.warning("Invalid data for composite factor calculation")
            return pd.DataFrame()
        
        return data.pivot(index='date', columns='code', values='composite_factor')
    
    def get_synthesis_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostics about the factor synthesis process.
        
        Returns:
            Dictionary with synthesis diagnostics and performance metrics
        """
        if not self._synthesis_completed:
            return {"error": "Synthesis not completed"}
        
        diagnostics = {
            "synthesis_config": {
                "method": self.synthesis_config.method,
                "training_window": self.synthesis_config.training_window,
                "prediction_window": self.synthesis_config.prediction_window,
                "update_frequency": self.synthesis_config.update_frequency
            },
            "input_factors": self.input_factor_names,
            "composite_factor_name": self.factor_name,
            "training_history": self.factor_composer.synthesis_results.get(
                self.factor_name, {}
            ).get("training_history", []),
            "data_shape": self._synthesized_factor_df.shape if self._synthesized_factor_df is not None else None
        }
        
        # Add feature importance for LASSO/Ridge methods
        if self.synthesis_config.method in ["lasso", "ridge"]:
            feature_importance = self.factor_composer.get_feature_importance(self.factor_name)
            if feature_importance:
                diagnostics["feature_importance"] = feature_importance
        
        # Add explained variance for PCA method
        if self.synthesis_config.method == "pca":
            variance_info = self.factor_composer.get_explained_variance(self.factor_name)
            if variance_info:
                diagnostics["explained_variance"] = variance_info
        
        # Add performance metrics if available
        if hasattr(self.factor_composer, 'performance_metrics'):
            performance = self.factor_composer.performance_metrics.get(self.factor_name, {})
            if performance:
                diagnostics["performance_metrics"] = performance
        
        return diagnostics
    
    def evaluate_synthesis_performance(self) -> Dict[str, float]:
        """
        Evaluate the performance of the synthesized factor.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self._synthesis_completed:
            return {"error": "Synthesis not completed"}
        
        try:
            # Get actual returns for evaluation
            actual_returns = self._get_target_returns()
            
            if actual_returns.empty:
                return {"error": "No actual returns available for evaluation"}
            
            # Evaluate performance using FactorComposer
            performance = self.factor_composer.evaluate_performance(
                composite_factor_name=self.factor_name,
                actual_returns=actual_returns
            )
            
            return performance
        
        except Exception as e:
            self.logger.error(f"Failed to evaluate synthesis performance: {e}")
            return {"error": str(e)}
    
    def get_input_factor_correlation(self) -> pd.DataFrame:
        """
        Get correlation matrix of input factors.
        
        Returns:
            Correlation matrix of input factors
        """
        try:
            # Get all input factors
            factor_data = self.factor_reader.get_multiple_factors(
                factor_names=self.input_factor_names,
                start_date=self.start_date,
                end_date=self.end_date,
                stock_pool=self.all_stocks
            )
            
            if not factor_data:
                return pd.DataFrame()
            
            # Align and calculate correlations
            aligned_data = {}
            
            # Get common dates and stocks
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
                return pd.DataFrame()
            
            # Create aligned factor vectors
            for factor_name, factor_df in factor_data.items():
                factor_values = []
                for date in sorted(common_dates):
                    for stock in sorted(common_stocks):
                        if date in factor_df.index and stock in factor_df.columns:
                            value = factor_df.loc[date, stock]
                            factor_values.append(value)
                
                aligned_data[factor_name] = factor_values
            
            # Calculate correlation matrix
            correlation_df = pd.DataFrame(aligned_data).corr()
            
            return correlation_df
        
        except Exception as e:
            self.logger.error(f"Failed to calculate input factor correlation: {e}")
            return pd.DataFrame()
    
    def __repr__(self) -> str:
        """String representation of the composite factor."""
        return (
            f"CompositeFactorBase(name='{self.factor_name}', "
            f"method='{self.synthesis_config.method}', "
            f"input_factors={len(self.input_factor_names)}, "
            f"period='{self.start_date}' to '{self.end_date}')"
        )
