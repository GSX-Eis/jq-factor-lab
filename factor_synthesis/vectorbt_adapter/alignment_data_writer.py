"""
Alignment Data Writer

Collects and serializes backtest data for alignment with cloud JoinQuant strategy.

4-Stage Alignment System:
- Stage 1: Input data (raw price/volume for factor calculation)
- Stage 2: Raw factor values (computed factor values before preprocessing)
- Stage 3: Preprocessed values (after zscore/rank_zscore, before blend)
- Stage 4: Composite scores (final blended scores for ranking)

Output files:
- stock_pool_snapshot.csv: Stock pool per rebalance date
- stage2_raw_factors.parquet: Raw factor values per stock
- stage3_preprocessed.parquet: Preprocessed factor values
- stage4_composite.parquet: Final composite scores
- holdings_snapshot.csv: Target holdings and actions
- summary.json: Metadata
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RebalanceSnapshot:
    """Snapshot of data for a single rebalance date."""
    rebalance_date: str  # T-day (execution day)
    factor_date: str     # T-1 day (factor calculation day)
    stock_pool: List[str]
    raw_factors: Optional[pd.DataFrame] = None      # stocks x factors (raw values)
    processed_factors: Optional[pd.DataFrame] = None  # stocks x factors (preprocessed)
    composite_scores: Optional[pd.Series] = None    # stocks -> score
    target_holdings: List[str] = field(default_factory=list)
    target_values: Dict[str, float] = field(default_factory=dict)
    actions: Dict[str, str] = field(default_factory=dict)  # stock -> BUY/SELL/HOLD
    pool_size: int = 0
    nav: float = 0.0  # Net asset value at this point
    # 4-Stage alignment data
    stage1_input_data: Optional[Dict[str, pd.DataFrame]] = None  # factor -> df with price/volume
    stage2_raw_factors: Optional[pd.DataFrame] = None            # stocks x factors (raw from DB)
    stage3_preprocessed: Optional[pd.DataFrame] = None           # stocks x factors (after zscore)
    stage4_composite: Optional[pd.Series] = None                 # final scores (post-blend)
    lasso_weights: Optional[Dict[str, float]] = None             # Lasso model weights
    blend_info: Optional[Dict[str, Any]] = None                  # Blend configuration


class AlignmentDataWriter:
    """
    Collects backtest snapshots and serializes them for cloud alignment.
    
    4-Stage Alignment System:
        Stage 1: Input data (raw price/volume used for factor calculation)
        Stage 2: Raw factor values (computed values before preprocessing)
        Stage 3: Preprocessed values (after zscore/rank_zscore, before blend)
        Stage 4: Composite scores (final blended scores)
    
    Usage:
        writer = AlignmentDataWriter(output_dir, experiment_name)
        
        for date in rebalance_dates:
            # Add 4-stage alignment data
            writer.add_stage2_raw_factors(rebalance_date, factor_date, raw_factors_df)
            writer.add_stage3_preprocessed(rebalance_date, factor_date, preprocessed_df)
            writer.add_stage4_composite(rebalance_date, factor_date, composite_scores)
            # Also add full snapshot for holdings
            writer.add_rebalance_snapshot(...)
        
        writer.save()
    """
    
    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        output_format: str = "parquet",  # "parquet" or "csv"
        logger: Optional[logging.Logger] = None,
        alignment_stages: Optional[Dict[str, Any]] = None,
        data_bindings: Optional[dict] = None,
        publication_metadata: Optional[dict] = None,
    ):
        """
        Initialize alignment data writer.
        
        Args:
            output_dir: Base output directory (e.g., "outputs/synthesis/alignment")
            experiment_name: Experiment identifier (e.g., "exp48_refined")
            output_format: Format for factor values ("parquet" or "csv")
            logger: Optional logger instance
            alignment_stages: Config for which stages to save, e.g.:
                {
                    "stage1_enabled": True,
                    "stage2_enabled": True,
                    "stage3_enabled": True,
                    "stage4_enabled": True,
                    "sample_stocks_limit": None  # None = all stocks
                }
        """
        self.output_dir = Path(output_dir) / experiment_name
        self.experiment_name = experiment_name
        self.output_format = output_format
        self.logger = logger or logging.getLogger(__name__)
        from utils.data_store import binding_contract, output_context
        self.data_bindings = binding_contract(data_bindings)
        self.publication_metadata = output_context(publication_metadata)
        if self.data_bindings.get("output_db") and self.data_bindings.get("publish_outputs") is not False:
            if not self.publication_metadata:
                raise ValueError("Alignment publication requires frozen output provenance")
            self.publication_metadata = {**self.publication_metadata, "schema_version": "alignment_snapshots.v1",
                                         "experiment_name": experiment_name}
        
        # Stage configuration
        self.alignment_stages = alignment_stages or {
            "stage1_enabled": False,  # Input data can be very large
            "stage2_enabled": True,
            "stage3_enabled": True,
            "stage4_enabled": True,
            "sample_stocks_limit": None,
        }
        
        # Collected snapshots
        self.snapshots: List[RebalanceSnapshot] = []
        
        # 4-stage data collectors (keyed by factor_date for lookup)
        self._stage2_data: Dict[str, pd.DataFrame] = {}  # factor_date -> raw factors df
        self._stage3_data: Dict[str, pd.DataFrame] = {}  # factor_date -> preprocessed df
        self._stage4_data: Dict[str, pd.Series] = {}     # factor_date -> composite scores
        self._stage4_dates: Dict[str, str] = {}
        self._lasso_weights: Dict[str, Dict[str, float]] = {}  # factor_date -> weights
        self._blend_info: Dict[str, Dict[str, Any]] = {}  # factor_date -> blend config
        
        # Metadata
        self.config_path: Optional[str] = None
        self.analysis_period: Dict[str, str] = {}
        self.factors_list: List[str] = []
        
    def set_metadata(
        self,
        config_path: str,
        analysis_period: Dict[str, str],
        factors_list: List[str],
    ):
        """Set experiment metadata."""
        self.config_path = config_path
        self.analysis_period = analysis_period
        self.factors_list = factors_list
    
    # =========================================================================
    # 4-Stage Alignment Data Collection Methods
    # =========================================================================
    
    def add_stage2_raw_factors(
        self,
        rebalance_date: str,
        factor_date: str,
        raw_factors: pd.DataFrame,
        lasso_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Add Stage 2 data: Raw factor values (computed values before preprocessing).
        
        Args:
            rebalance_date: T-day (execution day)
            factor_date: T-1 day (factor calculation day)
            raw_factors: DataFrame with index=stock_code, columns=factor_names
            lasso_weights: Optional Lasso model weights
        """
        if not self.alignment_stages.get("stage2_enabled", True):
            return
        
        if raw_factors is None or raw_factors.empty:
            return
        
        # Apply sample limit if configured
        limit = self.alignment_stages.get("sample_stocks_limit")
        if limit and len(raw_factors) > limit:
            raw_factors = raw_factors.head(limit)
        
        # Store with metadata columns
        df = raw_factors.copy()
        df['rebalance_date'] = rebalance_date
        df['factor_date'] = factor_date
        df = df.reset_index()
        if df.columns[0] != 'stock_code':
            df = df.rename(columns={df.columns[0]: 'stock_code'})
        
        self._stage2_data[factor_date] = df
        
        if lasso_weights:
            self._lasso_weights[factor_date] = lasso_weights.copy()
        
        self.logger.debug(
            f"[AlignmentWriter] Stage2: {factor_date} - {len(raw_factors)} stocks, "
            f"{len(raw_factors.columns)} factors"
        )
    
    def add_stage3_preprocessed(
        self,
        rebalance_date: str,
        factor_date: str,
        preprocessed: pd.DataFrame,
        preprocess_config: Optional[Dict[str, str]] = None,
    ):
        """
        Add Stage 3 data: Preprocessed factor values (after zscore/rank_zscore).
        
        Args:
            rebalance_date: T-day (execution day)
            factor_date: T-1 day (factor calculation day)
            preprocessed: DataFrame with index=stock_code, columns=factor_names
            preprocess_config: Preprocessing method config (e.g., {"ALL": "rank_zscore"})
        """
        if not self.alignment_stages.get("stage3_enabled", True):
            return
        
        if preprocessed is None or preprocessed.empty:
            return
        
        # Apply sample limit if configured
        limit = self.alignment_stages.get("sample_stocks_limit")
        if limit and len(preprocessed) > limit:
            preprocessed = preprocessed.head(limit)
        
        # Store with metadata columns
        df = preprocessed.copy()
        df['rebalance_date'] = rebalance_date
        df['factor_date'] = factor_date
        df = df.reset_index()
        if df.columns[0] != 'stock_code':
            df = df.rename(columns={df.columns[0]: 'stock_code'})
        
        self._stage3_data[factor_date] = df
        
        self.logger.debug(
            f"[AlignmentWriter] Stage3: {factor_date} - {len(preprocessed)} stocks"
        )
    
    def add_stage4_composite(
        self,
        rebalance_date: str,
        factor_date: str,
        composite_scores: pd.Series,
        blend_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Add Stage 4 data: Final composite scores (post-blend).
        
        Args:
            rebalance_date: T-day (execution day)
            factor_date: T-1 day (factor calculation day)
            composite_scores: Series with index=stock_code, values=scores
            blend_info: Blend configuration (ratio, weights, etc.)
        """
        if not self.alignment_stages.get("stage4_enabled", True):
            return
        
        if composite_scores is None or composite_scores.empty:
            return
        
        # Apply sample limit if configured
        limit = self.alignment_stages.get("sample_stocks_limit")
        if limit and len(composite_scores) > limit:
            composite_scores = composite_scores.head(limit)
        
        # Store as series with factor_date key
        self._stage4_data[factor_date] = composite_scores.copy()
        self._stage4_dates[factor_date] = rebalance_date
        
        if blend_info:
            self._blend_info[factor_date] = blend_info.copy()
        
        self.logger.debug(
            f"[AlignmentWriter] Stage4: {factor_date} - {len(composite_scores)} stocks"
        )
    
    def add_rebalance_snapshot(
        self,
        rebalance_date: str,
        factor_date: str,
        stock_pool: List[str],
        raw_factors: Optional[pd.DataFrame] = None,
        processed_factors: Optional[pd.DataFrame] = None,
        composite_scores: Optional[pd.Series] = None,
        target_holdings: Optional[List[str]] = None,
        target_values: Optional[Dict[str, float]] = None,
        actions: Optional[Dict[str, str]] = None,
        pool_size: int = 0,
        nav: float = 0.0,
    ):
        """
        Add a rebalance snapshot.
        
        Args:
            rebalance_date: T-day (execution day) in YYYY-MM-DD format
            factor_date: T-1 day (factor calculation day) in YYYY-MM-DD format
            stock_pool: List of stocks in the pool for this date
            raw_factors: DataFrame of raw factor values (stocks x factors)
            processed_factors: DataFrame of preprocessed factor values
            composite_scores: Series of composite scores (index=stock_code)
            target_holdings: List of target holding stocks
            target_values: Dict of {stock_code: target_value}
            actions: Dict of {stock_code: action} where action is BUY/SELL/HOLD
            pool_size: Size of the stock pool
            nav: Net asset value at this rebalance
        """
        snapshot = RebalanceSnapshot(
            rebalance_date=rebalance_date,
            factor_date=factor_date,
            stock_pool=stock_pool.copy() if stock_pool else [],
            raw_factors=raw_factors.copy() if raw_factors is not None else None,
            processed_factors=processed_factors.copy() if processed_factors is not None else None,
            composite_scores=composite_scores.copy() if composite_scores is not None else None,
            target_holdings=target_holdings.copy() if target_holdings else [],
            target_values=target_values.copy() if target_values else {},
            actions=actions.copy() if actions else {},
            pool_size=pool_size,
            nav=nav,
        )
        # One final snapshot per execution date, including two-pass overlays.
        # ponytail: linear lookup; index dates if histories exceed thousands.
        for index, previous in enumerate(self.snapshots):
            if previous.rebalance_date == rebalance_date:
                self.snapshots[index] = snapshot
                break
        else:
            self.snapshots.append(snapshot)
        
        self.logger.debug(
            f"[AlignmentWriter] Added snapshot: {rebalance_date} "
            f"(pool={len(stock_pool)}, holdings={len(target_holdings or [])})"
        )
    
    def save(self) -> Dict[str, str]:
        """
        Save all collected data to files.
        
        Returns:
            Dict mapping output type to file path
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # 1. Save stock pool snapshot
        pool_path = self._save_stock_pool_snapshot()
        output_files['stock_pool_snapshot'] = str(pool_path)
        
        # 2. Save 4-stage alignment data
        if self._stage2_data:
            stage2_path = self._save_stage2_raw_factors()
            output_files['stage2_raw_factors'] = str(stage2_path)
        
        if self._stage3_data:
            stage3_path = self._save_stage3_preprocessed()
            output_files['stage3_preprocessed'] = str(stage3_path)
        
        if self._stage4_data:
            stage4_path = self._save_stage4_composite()
            output_files['stage4_composite'] = str(stage4_path)
        
        # 3. Save Lasso weights if available
        if self._lasso_weights:
            weights_path = self._save_lasso_weights()
            output_files['lasso_weights'] = str(weights_path)
            weights = pd.DataFrame([(date, name, value) for date, values in self._lasso_weights.items()
                                    for name, value in values.items()], columns=['factor_date', 'factor_name', 'weight'])
            weights_path = self.output_dir / 'lasso_weights.parquet'
            weights.to_parquet(weights_path, index=False)
            output_files['lasso_weights_table'] = str(weights_path)
        
        # 4. Save legacy factor values snapshot (for backward compatibility)
        factor_path = self._save_factor_values_snapshot()
        output_files['factor_values_snapshot'] = str(factor_path)
        
        # 5. Save holdings snapshot
        holdings_path = self._save_holdings_snapshot()
        output_files['holdings_snapshot'] = str(holdings_path)
        
        self.logger.info(
            f"[AlignmentWriter] Saved {len(self.snapshots)} snapshots to {self.output_dir}"
        )
        
        from utils.data_store import publish_outputs
        targets = {"summary": "meta.alignment_summary", "lasso_weights": "meta.alignment_weights",
                   "lasso_weights_table": "alignment.lasso_weights"}
        refs = publish_outputs({targets.get(kind, "alignment." + kind): path for kind, path in output_files.items()},
                               metadata=self.publication_metadata, bindings=self.data_bindings)
        # The final summary freezes the complete set of successfully published stages.
        summary_path = self._save_summary(output_files, refs)
        output_files['summary'] = str(summary_path)
        refs.update(publish_outputs({"meta.alignment_summary": summary_path},
                    metadata=self.publication_metadata, bindings=self.data_bindings))
        from utils.local_outputs import write_json
        write_json(self.output_dir / "publication.json", {"schema_version": "publication.v1",
                   "output_db": self.data_bindings.get("output_db"), "data_refs": refs})
        return output_files
    
    # =========================================================================
    # 4-Stage Save Methods
    # =========================================================================
    
    def _save_stage2_raw_factors(self) -> Path:
        """Save Stage 2: Raw factor values to file."""
        if not self._stage2_data:
            return self.output_dir / 'stage2_raw_factors.parquet'
        
        # Concatenate all dates
        all_dfs = list(self._stage2_data.values())
        df = pd.concat(all_dfs, ignore_index=True)
        
        # Reorder columns: metadata first, then factors
        meta_cols = ['rebalance_date', 'factor_date', 'stock_code']
        factor_cols = [c for c in df.columns if c not in meta_cols]
        df = df[meta_cols + factor_cols]
        
        if self.output_format == 'parquet':
            path = self.output_dir / 'stage2_raw_factors.parquet'
            df.to_parquet(path, index=False, compression='snappy')
        else:
            path = self.output_dir / 'stage2_raw_factors.csv'
            df.to_csv(path, index=False)
        
        self.logger.info(
            f"[AlignmentWriter] Stage2 raw factors: {len(df)} rows, "
            f"{len(factor_cols)} factors -> {path}"
        )
        return path
    
    def _save_stage3_preprocessed(self) -> Path:
        """Save Stage 3: Preprocessed factor values to file."""
        if not self._stage3_data:
            return self.output_dir / 'stage3_preprocessed.parquet'
        
        # Concatenate all dates
        all_dfs = list(self._stage3_data.values())
        df = pd.concat(all_dfs, ignore_index=True)
        
        # Reorder columns: metadata first, then factors
        meta_cols = ['rebalance_date', 'factor_date', 'stock_code']
        factor_cols = [c for c in df.columns if c not in meta_cols]
        df = df[meta_cols + factor_cols]
        
        if self.output_format == 'parquet':
            path = self.output_dir / 'stage3_preprocessed.parquet'
            df.to_parquet(path, index=False, compression='snappy')
        else:
            path = self.output_dir / 'stage3_preprocessed.csv'
            df.to_csv(path, index=False)
        
        self.logger.info(
            f"[AlignmentWriter] Stage3 preprocessed: {len(df)} rows, "
            f"{len(factor_cols)} factors -> {path}"
        )
        return path
    
    def _save_stage4_composite(self) -> Path:
        """Save Stage 4: Final composite scores to file."""
        if not self._stage4_data:
            return self.output_dir / 'stage4_composite.parquet'
        
        rows = []
        for factor_date, scores in self._stage4_data.items():
            rebalance_date = self._stage4_dates[factor_date]
            
            # Sort by score descending for rank calculation
            sorted_scores = scores.sort_values(ascending=False)
            
            for rank, (stock, score) in enumerate(sorted_scores.items(), 1):
                blend_info = self._blend_info.get(factor_date, {})
                rows.append({
                    'rebalance_date': rebalance_date,
                    'factor_date': factor_date,
                    'stock_code': stock,
                    'composite_score': score,
                    'rank': rank,
                    'blend_ratio': blend_info.get('blend_ratio', 0.0),
                })
        
        df = pd.DataFrame(rows)
        
        if self.output_format == 'parquet':
            path = self.output_dir / 'stage4_composite.parquet'
            df.to_parquet(path, index=False, compression='snappy')
        else:
            path = self.output_dir / 'stage4_composite.csv'
            df.to_csv(path, index=False)
        
        self.logger.info(
            f"[AlignmentWriter] Stage4 composite: {len(df)} rows -> {path}"
        )
        return path
    
    def _save_lasso_weights(self) -> Path:
        """Save Lasso weights per date for alignment verification."""
        if not self._lasso_weights:
            return self.output_dir / 'lasso_weights.json'
        
        path = self.output_dir / 'lasso_weights.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._lasso_weights, f, indent=2, ensure_ascii=False)
        
        self.logger.info(
            f"[AlignmentWriter] Lasso weights: {len(self._lasso_weights)} dates -> {path}"
        )
        return path
    
    def _save_stock_pool_snapshot(self) -> Path:
        """Save stock pool data to CSV."""
        rows = []
        
        for snap in self.snapshots:
            for stock in snap.stock_pool:
                rows.append({
                    'rebalance_date': snap.rebalance_date,
                    'factor_date': snap.factor_date,
                    'stock_code': stock,
                    'in_pool': True,
                    'is_target': stock in snap.target_holdings,
                })
        
        df = pd.DataFrame(rows, columns=['rebalance_date', 'factor_date', 'stock_code', 'in_pool', 'is_target'])
        path = self.output_dir / 'stock_pool_snapshot.csv'
        df.to_csv(path, index=False)
        
        self.logger.info(f"[AlignmentWriter] Stock pool: {len(df)} rows -> {path}")
        return path
    
    def _save_factor_values_snapshot(self) -> Path:
        """Save factor values to Parquet or CSV."""
        rows = []
        
        for snap in self.snapshots:
            if snap.composite_scores is None or snap.composite_scores.empty:
                continue
            
            for stock in snap.composite_scores.index:
                row = {
                    'rebalance_date': snap.rebalance_date,
                    'factor_date': snap.factor_date,
                    'stock_code': stock,
                    'composite_score': snap.composite_scores.get(stock, np.nan),
                    'rank': self._get_rank(snap.composite_scores, stock),
                }
                
                # Add raw factor values if available
                if snap.raw_factors is not None and stock in snap.raw_factors.index:
                    for col in snap.raw_factors.columns:
                        row[f'{col}_raw'] = snap.raw_factors.loc[stock, col]
                
                # Add processed factor values if available
                if snap.processed_factors is not None and stock in snap.processed_factors.index:
                    for col in snap.processed_factors.columns:
                        row[f'{col}_processed'] = snap.processed_factors.loc[stock, col]
                
                rows.append(row)
        
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            'rebalance_date', 'factor_date', 'stock_code', 'composite_score', 'rank'])
        
        if self.output_format == 'parquet':
            path = self.output_dir / 'factor_values_snapshot.parquet'
            df.to_parquet(path, index=False, compression='snappy')
        else:
            path = self.output_dir / 'factor_values_snapshot.csv'
            df.to_csv(path, index=False)
        
        self.logger.info(f"[AlignmentWriter] Factor values: {len(df)} rows -> {path}")
        return path
    
    def _save_holdings_snapshot(self) -> Path:
        """Save holdings data to CSV."""
        rows = []
        
        for snap in self.snapshots:
            # Sort by composite score for ranking
            if snap.composite_scores is not None and not snap.composite_scores.empty:
                ranked = snap.composite_scores.sort_values(ascending=False)
            else:
                ranked = pd.Series(dtype=float)
            
            # Add target holdings with their ranks
            for rank, stock in enumerate(snap.target_holdings, 1):
                score = snap.composite_scores.get(stock, np.nan) if snap.composite_scores is not None else np.nan
                target_value = snap.target_values.get(stock, 0.0)
                action = snap.actions.get(stock, 'UNKNOWN')
                
                rows.append({
                    'rebalance_date': snap.rebalance_date,
                    'factor_date': snap.factor_date,
                    'stock_code': stock,
                    'rank': rank,
                    'composite_score': score,
                    'target_value': target_value,
                    'target_weight': 1.0 / len(snap.target_holdings) if snap.target_holdings else 0.0,
                    'action': action,
                    'pool_size': snap.pool_size,
                    'nav': snap.nav,
                })
        
        df = pd.DataFrame(rows, columns=['rebalance_date', 'factor_date', 'stock_code', 'rank',
                         'composite_score', 'target_value', 'target_weight', 'action', 'pool_size', 'nav'])
        path = self.output_dir / 'holdings_snapshot.csv'
        df.to_csv(path, index=False)
        
        self.logger.info(f"[AlignmentWriter] Holdings: {len(df)} rows -> {path}")
        return path
    
    def _save_summary(self, output_files: Dict[str, str], data_refs: dict) -> Path:
        """Save summary JSON with metadata."""
        summary = {
            'experiment_name': self.experiment_name,
            'config_path': self.config_path,
            'generated_at': self.publication_metadata.get('created_at') or datetime.now().isoformat(),
            'analysis_period': self.analysis_period,
            'total_rebalance_dates': len(self.snapshots),
            'factors_count': len(self.factors_list),
            'factors_list': self.factors_list,
            'output_format': self.output_format,
            'output_files': {k: os.path.basename(v) for k, v in output_files.items()},
            'data_refs': data_refs,
            # 4-Stage alignment info
            'alignment_stages': {
                'stage2_dates': len(self._stage2_data),
                'stage3_dates': len(self._stage3_data),
                'stage4_dates': len(self._stage4_data),
                'lasso_weights_dates': len(self._lasso_weights),
            },
            'alignment_config': self.alignment_stages,
        }
        
        path = self.output_dir / 'summary.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"[AlignmentWriter] Summary -> {path}")
        return path
    
    @staticmethod
    def _get_rank(scores: pd.Series, stock: str) -> int:
        """Get 1-indexed rank of stock in scores (higher score = lower rank number)."""
        if stock not in scores.index:
            return -1
        ranked = scores.sort_values(ascending=False)
        try:
            return list(ranked.index).index(stock) + 1
        except ValueError:
            return -1
