"""
Synthesis Visualizer Module

Comprehensive visualization tools for factor synthesis analysis, including
performance metrics, feature importance, correlation analysis, and time series plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner plots
warnings.filterwarnings('ignore', category=FutureWarning)
plt.style.use('default')
sns.set_palette("husl")


class SynthesisVisualizer:
    """
    Comprehensive visualization toolkit for factor synthesis analysis.
    
    Provides static (matplotlib/seaborn) and interactive (plotly) visualizations
    for composite factor analysis, performance evaluation, and diagnostics.
    """
    
    def __init__(
        self,
        factor_reader,  # FactorReader instance
        logger: Optional[logging.Logger] = None,
        style_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the SynthesisVisualizer.
        
        Args:
            factor_reader: FactorReader instance for data access
            logger: Optional logger instance
            style_config: Optional styling configuration
        """
        self.factor_reader = factor_reader
        self.logger = logger or logging.getLogger(__name__)
        
        # Default style configuration
        self.style_config = {
            "figsize": (12, 8),
            "dpi": 300,
            "color_palette": "husl",
            "font_size": 10,
            "title_size": 14,
            "grid": True,
            "alpha": 0.7
        }
        
        # Update with user configuration
        if style_config:
            self.style_config.update(style_config)
        
        # Apply matplotlib styling
        plt.rcParams.update({
            'figure.figsize': self.style_config["figsize"],
            'figure.dpi': self.style_config["dpi"],
            'font.size': self.style_config["font_size"],
            'axes.titlesize': self.style_config["title_size"],
            'axes.grid': self.style_config["grid"]
        })
        
        self.logger.info("SynthesisVisualizer initialized")
    
    def plot_factor_correlation_matrix(
        self,
        factor_names: List[str],
        start_date: str,
        end_date: str,
        stock_pool: Optional[List[str]] = None,
        method: str = "spearman",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot correlation matrix of input factors.
        
        Args:
            factor_names: List of factor names to analyze
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD) 
            stock_pool: Optional list of stocks to focus on
            method: Correlation method ("spearman", "pearson", "kendall")
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info(f"Creating correlation matrix for {len(factor_names)} factors")
        
        # Get factor data
        factor_data = self.factor_reader.get_multiple_factors(
            factor_names=factor_names,
            start_date=start_date,
            end_date=end_date,
            stock_pool=stock_pool
        )
        
        if not factor_data:
            raise ValueError("No factor data available for correlation analysis")
        
        # Align factors and calculate correlations
        aligned_values = {}
        
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
        
        # Extract aligned values for each factor
        for factor_name, factor_df in factor_data.items():
            factor_values = []
            
            for date in sorted(common_dates):
                for stock in sorted(common_stocks):
                    if (date in factor_df.index and 
                        stock in factor_df.columns and
                        pd.notna(factor_df.loc[date, stock])):
                        factor_values.append(factor_df.loc[date, stock])
            
            aligned_values[factor_name] = factor_values
        
        # Create correlation DataFrame
        correlation_df = pd.DataFrame(aligned_values).corr(method=method)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.style_config["figsize"])
        
        # Create heatmap
        sns.heatmap(
            correlation_df,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'label': f'{method.title()} Correlation'},
            ax=ax
        )
        
        ax.set_title(f'Factor Correlation Matrix ({method.title()})', 
                    fontsize=self.style_config["title_size"])
        ax.set_xlabel('Factors')
        ax.set_ylabel('Factors')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.style_config["dpi"], bbox_inches='tight')
            self.logger.info(f"Correlation matrix saved to: {save_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance from LASSO/Ridge synthesis.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if not feature_importance:
            raise ValueError("No feature importance data provided")
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        feature_names = [item[0] for item in sorted_features]
        importance_values = [item[1] for item in sorted_features]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.style_config["figsize"])
        
        # Create horizontal bar plot
        bars = ax.barh(
            range(len(feature_names)),
            importance_values,
            alpha=self.style_config["alpha"]
        )
        
        # Color bars based on positive/negative values
        for bar, value in zip(bars, importance_values):
            bar.set_color('green' if value >= 0 else 'red')
        
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Importance Score')
        ax.set_title(title, fontsize=self.style_config["title_size"])
        
        # Add value labels on bars
        for i, value in enumerate(importance_values):
            ax.text(value + (0.01 if value >= 0 else -0.01), i,
                   f'{value:.3f}', ha='left' if value >= 0 else 'right',
                   va='center', fontsize=8)
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.style_config["dpi"], bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to: {save_path}")
        
        return fig
    
    def plot_explained_variance(
        self,
        explained_variance: Dict[str, Any],
        title: str = "PCA Explained Variance",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot explained variance for PCA synthesis.
        
        Args:
            explained_variance: Dictionary with variance information from PCA
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if not explained_variance or "explained_variance_ratio" not in explained_variance:
            raise ValueError("No explained variance data provided")
        
        variance_ratio = explained_variance["explained_variance_ratio"]
        cumulative_variance = explained_variance.get("cumulative_variance", np.cumsum(variance_ratio))
        n_components = len(variance_ratio)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Individual explained variance
        components = range(1, n_components + 1)
        
        bars = ax1.bar(components, variance_ratio, alpha=self.style_config["alpha"])
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Individual Explained Variance')
        ax1.set_xticks(components)
        
        # Add value labels
        for i, (component, var_ratio) in enumerate(zip(components, variance_ratio)):
            ax1.text(component, var_ratio + 0.005, f'{var_ratio:.3f}',
                    ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Cumulative explained variance
        line = ax2.plot(components, cumulative_variance, 'o-', 
                       linewidth=2, markersize=6, alpha=self.style_config["alpha"])
        ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Explained Variance Ratio')
        ax2.set_title('Cumulative Explained Variance')
        ax2.set_xticks(components)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add final percentage label
        final_variance = cumulative_variance[-1]
        ax2.text(components[-1], final_variance + 0.01, f'{final_variance:.1%}',
                ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(title, fontsize=self.style_config["title_size"])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.style_config["dpi"], bbox_inches='tight')
            self.logger.info(f"Explained variance plot saved to: {save_path}")
        
        return fig
    
    def plot_composite_factor_time_series(
        self,
        composite_factor_df: pd.DataFrame,
        factor_name: str,
        stock_sample: Optional[List[str]] = None,
        n_stocks: int = 5,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot time series of composite factor values for selected stocks.
        
        Args:
            composite_factor_df: DataFrame with composite factor values
            factor_name: Name of the composite factor
            stock_sample: Optional specific stocks to plot
            n_stocks: Number of stocks to sample if stock_sample not provided
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if composite_factor_df.empty:
            raise ValueError("Empty composite factor DataFrame provided")
        
        # Select stocks to plot
        if stock_sample is None:
            available_stocks = composite_factor_df.columns.tolist()
            # Select stocks with good coverage (fewer NaN values)
            coverage = composite_factor_df.count()
            best_stocks = coverage.nlargest(min(n_stocks, len(available_stocks))).index.tolist()
            stock_sample = best_stocks
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.style_config["figsize"])
        
        # Plot each stock's factor time series
        for stock in stock_sample:
            if stock in composite_factor_df.columns:
                stock_data = composite_factor_df[stock].dropna()
                if not stock_data.empty:
                    ax.plot(stock_data.index, stock_data.values, 
                           label=stock, alpha=self.style_config["alpha"], linewidth=1.5)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Factor Value')
        ax.set_title(f'{factor_name} - Time Series by Stock', 
                    fontsize=self.style_config["title_size"])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.style_config["dpi"], bbox_inches='tight')
            self.logger.info(f"Time series plot saved to: {save_path}")
        
        return fig
    
    def plot_performance_metrics_dashboard(
        self,
        performance_metrics: Dict[str, float],
        factor_name: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard of performance metrics.
        
        Args:
            performance_metrics: Dictionary with performance metrics
            factor_name: Name of the composite factor
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if not performance_metrics:
            raise ValueError("No performance metrics provided")
        
        # Create subplot layout
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Metric 1: Information Coefficient (if available)
        if "ic" in performance_metrics:
            ax1 = fig.add_subplot(gs[0, 0])
            ic_value = performance_metrics["ic"]
            
            # Create gauge-like plot for IC
            ax1.bar([0], [abs(ic_value)], color='green' if ic_value >= 0 else 'red',
                   alpha=self.style_config["alpha"])
            ax1.set_ylim(0, 0.1)
            ax1.set_xlim(-0.5, 0.5)
            ax1.set_title(f'Information Coefficient\n{ic_value:.4f}')
            ax1.set_xticks([])
            ax1.axhline(y=0.02, color='orange', linestyle='--', label='Threshold')
            ax1.legend()
        
        # Metric 2: Information Ratio (if available)
        if "ir" in performance_metrics:
            ax2 = fig.add_subplot(gs[0, 1])
            ir_value = performance_metrics["ir"]
            
            ax2.bar([0], [abs(ir_value)], color='blue' if ir_value >= 0.5 else 'orange',
                   alpha=self.style_config["alpha"])
            ax2.set_ylim(0, 2.0)
            ax2.set_xlim(-0.5, 0.5)
            ax2.set_title(f'Information Ratio\n{ir_value:.4f}')
            ax2.set_xticks([])
            ax2.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
            ax2.legend()
        
        # Metric 3: R-squared (if available)
        if "r2" in performance_metrics:
            ax3 = fig.add_subplot(gs[0, 2])
            r2_value = performance_metrics["r2"]
            
            ax3.bar([0], [r2_value], color='purple',
                   alpha=self.style_config["alpha"])
            ax3.set_ylim(0, 1.0)
            ax3.set_xlim(-0.5, 0.5)
            ax3.set_title(f'R-squared\n{r2_value:.4f}')
            ax3.set_xticks([])
        
        # Metric 4: MSE (if available)
        if "mse" in performance_metrics:
            ax4 = fig.add_subplot(gs[1, 0])
            mse_value = performance_metrics["mse"]
            
            ax4.bar([0], [mse_value], color='red',
                   alpha=self.style_config["alpha"])
            ax4.set_title(f'Mean Squared Error\n{mse_value:.6f}')
            ax4.set_xticks([])
        
        # Performance summary table
        ax5 = fig.add_subplot(gs[1, 1:])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create table data
        table_data = []
        for metric, value in performance_metrics.items():
            table_data.append([metric.upper(), f'{value:.6f}'])
        
        table = ax5.table(cellText=table_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax5.set_title('Performance Summary', fontsize=self.style_config["title_size"])
        
        plt.suptitle(f'Performance Dashboard: {factor_name}', 
                    fontsize=self.style_config["title_size"] + 2)
        
        if save_path:
            plt.savefig(save_path, dpi=self.style_config["dpi"], bbox_inches='tight')
            self.logger.info(f"Performance dashboard saved to: {save_path}")
        
        return fig
    
    def create_interactive_factor_comparison(
        self,
        factor_data: Dict[str, pd.DataFrame],
        title: str = "Interactive Factor Comparison"
    ):
        """
        Create interactive plotly visualization comparing multiple factors.
        
        Args:
            factor_data: Dictionary mapping factor names to DataFrames
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if not factor_data:
            raise ValueError("No factor data provided")
        
        # Create subplots
        n_factors = len(factor_data)
        rows = (n_factors + 1) // 2
        
        fig = make_subplots(
            rows=rows,
            cols=2,
            subplot_titles=list(factor_data.keys()),
            vertical_spacing=0.08
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (factor_name, factor_df) in enumerate(factor_data.items()):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            if factor_df.empty:
                continue
            
            # Calculate average factor value over time
            avg_values = factor_df.mean(axis=1)
            
            fig.add_trace(
                go.Scatter(
                    x=avg_values.index,
                    y=avg_values.values,
                    mode='lines',
                    name=factor_name,
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate=f'<b>{factor_name}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Average Value: %{y:.4f}<extra></extra>'
                ),
                row=row,
                col=col
            )
        
        fig.update_layout(
            title=title,
            height=300 * rows,
            showlegend=False,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Factor Value")
        
        return fig
    
    def create_correlation_heatmap_interactive(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Interactive Correlation Heatmap"
    ):
        """
        Create interactive correlation heatmap using plotly.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{y} vs %{x}</b><br>' +
                         'Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Factors",
            yaxis_title="Factors",
            width=600,
            height=600
        )
        
        return fig
    
    def generate_synthesis_report(
        self,
        composite_factor_name: str,
        input_factors: List[str],
        synthesis_config: Dict[str, Any],
        performance_metrics: Dict[str, float],
        diagnostics: Dict[str, Any],
        save_directory: str
    ):
        """
        Generate a comprehensive synthesis report with multiple visualizations.
        
        Args:
            composite_factor_name: Name of the composite factor
            input_factors: List of input factor names
            synthesis_config: Configuration used for synthesis
            performance_metrics: Performance metrics dictionary
            diagnostics: Additional diagnostics information
            save_directory: Directory to save all plots
        """
        self.logger.info(f"Generating comprehensive report for {composite_factor_name}")
        
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. Correlation matrix of input factors
        if len(input_factors) > 1:
            try:
                fig1 = self.plot_factor_correlation_matrix(
                    factor_names=input_factors,
                    start_date=synthesis_config.get("start_date", "2020-01-01"),
                    end_date=synthesis_config.get("end_date", "2024-12-31"),
                    save_path=os.path.join(save_directory, f"{composite_factor_name}_correlation_matrix.png")
                )
                plt.close(fig1)
            except Exception as e:
                self.logger.warning(f"Failed to create correlation matrix: {e}")
        
        # 2. Feature importance (for LASSO/Ridge)
        if "feature_importance" in diagnostics:
            try:
                fig2 = self.plot_feature_importance(
                    feature_importance=diagnostics["feature_importance"],
                    title=f"Feature Importance: {composite_factor_name}",
                    save_path=os.path.join(save_directory, f"{composite_factor_name}_feature_importance.png")
                )
                plt.close(fig2)
            except Exception as e:
                self.logger.warning(f"Failed to create feature importance plot: {e}")
        
        # 3. Explained variance (for PCA)
        if "explained_variance" in diagnostics:
            try:
                fig3 = self.plot_explained_variance(
                    explained_variance=diagnostics["explained_variance"],
                    title=f"Explained Variance: {composite_factor_name}",
                    save_path=os.path.join(save_directory, f"{composite_factor_name}_explained_variance.png")
                )
                plt.close(fig3)
            except Exception as e:
                self.logger.warning(f"Failed to create explained variance plot: {e}")
        
        # 4. Performance dashboard
        if performance_metrics:
            try:
                fig4 = self.plot_performance_metrics_dashboard(
                    performance_metrics=performance_metrics,
                    factor_name=composite_factor_name,
                    save_path=os.path.join(save_directory, f"{composite_factor_name}_performance_dashboard.png")
                )
                plt.close(fig4)
            except Exception as e:
                self.logger.warning(f"Failed to create performance dashboard: {e}")
        
        self.logger.info(f"Report generated in directory: {save_directory}")
    
    def __del__(self):
        """Cleanup matplotlib resources."""
        plt.close('all')