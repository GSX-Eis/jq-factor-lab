import json
import os
from collections.abc import Mapping
from typing import Dict, Unpack

import numpy as np
import pandas as pd
from .preprocessing import MarketInputOptions, bind_synthesis_panel, selected_market_inputs
from ..core.feature_preprocessor import TIME_SERIES_METHODS, time_series_standardize


def _save_weighted_sum_bundle(
    output_dir: str,
    synthesis_name: str,
    weights: Dict[str, float],
    feature_names: list,
    preprocess_cfg: Dict[str, str],
    logger,
) -> str:
    """
    Create a bundle file for weighted_sum synthesis, compatible with VectorBT backtest runner.
    
    The bundle mimics Lasso bundle format with a single model entry that's always valid.
    
    Args:
        output_dir: Directory to save the bundle file
        synthesis_name: Name of the synthesis method
        weights: Dict of factor_name -> weight
        feature_names: List of factor names
        preprocess_cfg: Dict of factor_name -> preprocess method
        logger: Logger instance
        
    Returns:
        Path to the saved bundle file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create bundle structure matching Lasso bundle format
    # Use train_end = "1900-01-01" to ensure model is always selected for any date
    bundle = {
        "index": {
            "models": [
                {
                    "train_end": "1900-01-01",  # Always valid
                    "weights": weights,
                    "feature_names": feature_names,
                    "blend_only": False,
                    "blend_with_original_ratio": 0.0,
                    "blend_weights": {},
                    "standardization_method": preprocess_cfg,
                }
            ]
        },
        "metadata": {
            "method": "weighted_sum",
            "synthesis_name": synthesis_name,
        }
    }
    
    bundle_path = os.path.join(output_dir, f"weighted_sum_bundle_{synthesis_name}.json")
    
    with open(bundle_path, 'w', encoding='utf-8') as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[weighted_sum] Bundle saved: {bundle_path}")
    return bundle_path


def run_weighted_sum_synthesis(
    synthesizer,
    synthesis_name: str,
    synthesis_config: dict,
    factor_df: pd.DataFrame | None,
    **market_inputs: Unpack[MarketInputOptions],
) -> None:
    """
    Fixed-weight linear composite factor (no training).

    Extracted from `FactorSynthesizer.weighted_sum_synthesis` so that the
    main orchestrator stays slim and only delegates to this function.
    """
    factor_df, synthesis_config = bind_synthesis_panel(
        synthesizer, factor_df, synthesis_config, market_inputs,
    )
    logger = synthesizer.logger

    try:
        if factor_df.empty:
            raise RuntimeError(f"weighted_sum: no factor data for {synthesis_name}")

        raw_synth_cfg = getattr(synthesizer, "config", None)
        synth_cfg = raw_synth_cfg if isinstance(raw_synth_cfg, Mapping) else {}
        factor_group = synthesis_config.get("factor_group")
        group_cfg: Dict = synth_cfg.get("factor_groups", {}).get(factor_group, {})
        factor_names = group_cfg.get("factors", [])

        if not factor_names:
            raise RuntimeError(f"weighted_sum: empty factor group {factor_group}")

        weights_cfg: Dict = synthesis_config.get("weights", {})
        preprocess_cfg: Dict = synthesis_config.get("preprocess", {})

        missing_cols = [f for f in factor_names if f not in factor_df.columns]
        if missing_cols:
            logger.warning(
                "weighted_sum[%s]: 下列因子在数据中缺失: %s",
                synthesis_name,
                missing_cols,
            )

        for factor in factor_names:
            if factor in factor_df.columns and not np.isfinite(float(weights_cfg.get(factor, 0.0))):
                raise ValueError(f"weighted_sum: non-finite weight for {factor}")

        used_factors = [
            f
            for f in factor_names
            if f in factor_df.columns and abs(float(weights_cfg.get(f, 0.0))) > 0.0
        ]

        if not used_factors:
            raise RuntimeError(f"weighted_sum: no usable weighted factors for {synthesis_name}")

        df = factor_df[used_factors].copy()

        # Preserve the historical A-share weighting input lag.
        inputs = factor_df.attrs.get("market_factor_inputs", {}).get("factors", [])
        source_slot = market_inputs.get("slot") or synthesis_config.get("slot")
        fallback_slot = source_slot.slot_id if source_slot is not None else "jq_ashare_local"
        slot_id = inputs[0]["slot_id"] if inputs else market_inputs.get(
            "slot_id", synthesis_config.get("slot_id", fallback_slot),
        )
        freq = inputs[0]["frequency"] if inputs else market_inputs.get(
            "frequency", synthesis_config.get("frequency", "1d"),
        )
        if slot_id == "jq_ashare_local" and freq in {"1d", "daily"}:
            shifted = df.unstack(level="stock_code").shift(1).stack(level="stock_code", dropna=True)
            df = shifted.reindex(df.index).dropna()

        def _zscore(series: pd.Series) -> pd.Series:
            m = series.mean()
            std = series.std()
            if std and std > 1e-8:
                return (series - m) / std
            return pd.Series(0.0, index=series.index)

        def _rank_zscore(series: pd.Series) -> pd.Series:
            ranks = series.rank(method="average", pct=True)
            m = ranks.mean()
            std = ranks.std()
            if std and std > 1e-8:
                return (ranks - m) / std
            return pd.Series(0.0, index=series.index)

        for f in used_factors:
            default_mode = preprocess_cfg.get("ALL")
            mode = preprocess_cfg.get(f, default_mode if default_mode in TIME_SERIES_METHODS else "none")
            if mode in TIME_SERIES_METHODS:
                df[f] = time_series_standardize(
                    df[f], mode, window=preprocess_cfg.get("_ts_window", 20),
                    min_periods=preprocess_cfg.get("_ts_min_periods"),
                )
            elif mode == "zscore":
                logger.info(
                    "weighted_sum[%s]: 对因子 %s 使用按日截面 z-score 预处理",
                    synthesis_name,
                    f,
                )
                df[f] = df[f].groupby(level="date").transform(_zscore)
            elif mode == "rank_zscore":
                logger.info(
                    "weighted_sum[%s]: 对因子 %s 使用按日截面 rank+z-score 预处理",
                    synthesis_name,
                    f,
                )
                df[f] = df[f].groupby(level="date").transform(_rank_zscore)
            elif mode == "rank":
                logger.info(
                    "weighted_sum[%s]: 对因子 %s 使用按日截面百分位 rank 预处理",
                    synthesis_name,
                    f,
                )
                df[f] = df[f].groupby(level="date").transform(
                    lambda s: s.rank(method="average", pct=True)
                )
            else:
                logger.info(
                    "weighted_sum[%s]: 因子 %s 不做预处理 (mode=%s)",
                    synthesis_name,
                    f,
                    mode,
                )

        composite = None
        for f in used_factors:
            w = float(weights_cfg.get(f, 0.0))
            if composite is None:
                composite = w * df[f]
            else:
                composite = composite.add(w * df[f], fill_value=0.0)

        if composite is None or composite.empty:
            raise RuntimeError(f"weighted_sum: empty composite for {synthesis_name}")

        composite = composite.dropna()
        composite.index = composite.index.set_names(["date", "stock_code"])

        composite_name = (
            f"{synth_cfg['output_settings']['composite_factor_prefix']}_"
            f"{synthesis_name}"
        )
        composite_series = composite.copy()

        composite_series.name = composite_name
        if inputs:
            composite_series.attrs["market_factor_inputs"] = selected_market_inputs(factor_df, used_factors)
        logger.info(
            "weighted_sum[%s]: 合成因子 %s 计算完成, 数据点=%d",
            synthesis_name,
            composite_name,
            len(composite_series),
        )

        # [FIX] Save bundle file for VectorBT backtest (same format as Lasso)
        output_settings = synth_cfg.get("output_settings", {})
        weights_output_dir = output_settings.get("weights_output_dir", "./results")
        
        # Build weights dict with only used factors
        final_weights = {f: float(weights_cfg.get(f, 0.0)) for f in used_factors}
        
        _save_weighted_sum_bundle(
            output_dir=weights_output_dir,
            synthesis_name=synthesis_name,
            weights=final_weights,
            feature_names=used_factors,
            preprocess_cfg=preprocess_cfg,
            logger=logger,
        )

        synthesizer.direct_analyze_composite_factor(
            composite_series,
            composite_name,
            run_alphalens=not bool(
                synth_cfg.get("performance_evaluation", {}).get(
                    "skip_alphalens_analysis", False
                )
            ),
        )
        logger.info("完成 weighted_sum 合成因子分析: %s", composite_name)

    except Exception as e:
        logger.error("weighted_sum[%s]: 合成失败: %s", synthesis_name, e)
        import traceback

        logger.error(traceback.format_exc())
        raise

