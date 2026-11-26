import json
import os
from collections.abc import Mapping
from typing import Any, Dict, Unpack

import numpy as np
import pandas as pd

from utils.strategy_kernel import (
    preprocess_cross_section_series,
    resolve_preprocess_method,
    standardize_composite_scores,
    weighted_score,
)
from .preprocessing import MarketInputOptions, bind_synthesis_panel, selected_market_inputs
from ..core.factor_data_loader import (
    PANEL_DATE_ALIGNMENT_ATTR,
    PANEL_DATE_ALIGNMENT_RAW,
    PANEL_DATE_ALIGNMENT_TRADE_DATE,
)
from ..core.feature_preprocessor import TIME_SERIES_METHODS, time_series_standardize


_PREPROCESS_PARAM_KEYS = (
    "_binning_n_bins",
    "_winsorize_percentile",
    "_ts_window",
    "_ts_min_periods",
)


def _resolve_weighted_sum_preprocess(
    preprocess_cfg: Mapping[str, Any],
    feature_names: list[str],
) -> Dict[str, Any]:
    """Resolve explicit per-factor preprocessing modes and parameters for local/bundle parity."""
    default_mode = resolve_preprocess_method(preprocess_cfg, "ALL", default="none")
    resolved: Dict[str, Any] = {"ALL": default_mode}
    for factor in feature_names:
        resolved[factor] = resolve_preprocess_method(
            preprocess_cfg, factor, default=default_mode,
        )
    for key in _PREPROCESS_PARAM_KEYS:
        if key in preprocess_cfg:
            resolved[key] = preprocess_cfg[key]
    return resolved


def _trim_composite_to_window(
    composite: pd.Series,
    *,
    start_date: object | None,
    end_date: object | None,
) -> pd.Series:
    """Clip composite scores strictly to [start_date, end_date], excluding warmup bars."""
    if composite.empty or (start_date in (None, "") and end_date in (None, "")):
        return composite
    dates = composite.index.get_level_values("date")
    date_values = dates if isinstance(dates, pd.DatetimeIndex) else pd.to_datetime(dates)
    use_utc = isinstance(date_values, pd.DatetimeIndex) and date_values.tz is not None
    mask = pd.Series(True, index=composite.index)
    if start_date not in (None, ""):
        start_ts = pd.Timestamp(start_date)
        if use_utc:
            start_ts = (
                start_ts.tz_localize("UTC")
                if start_ts.tz is None
                else start_ts.tz_convert("UTC")
            )
        mask &= date_values >= start_ts
    if end_date not in (None, ""):
        raw_end = pd.Timestamp(end_date)
        if use_utc:
            if raw_end.tz is None:
                end_ts = raw_end.tz_localize("UTC")
                if end_ts == end_ts.normalize():
                    end_ts += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            else:
                end_ts = raw_end.tz_convert("UTC")
        else:
            end_ts = raw_end
        mask &= date_values <= end_ts
    return composite.loc[mask]


def _save_weighted_sum_bundle(
    output_dir: str,
    synthesis_name: str,
    weights: Dict[str, float],
    feature_names: list,
    preprocess_cfg: Dict[str, Any],
    logger,
    *,
    standardize: bool = True,
    postprocess_cs_standardize: bool = False,
) -> str:
    """
    Create a bundle file for weighted_sum synthesis, compatible with VectorBT backtest runner.
    
    The bundle mimics Lasso bundle format with a single model entry that's always valid.
    
    Args:
        output_dir: Directory to save the bundle file
        synthesis_name: Name of the synthesis method
        weights: Dict of factor_name -> weight
        feature_names: List of factor names
        preprocess_cfg: Explicit resolved dict of factor_name / ALL -> preprocess method
        logger: Logger instance
        standardize: Whether per-factor preprocessing is active
        postprocess_cs_standardize: Whether composite scores undergo cross-sectional z-score
        
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
                    "standardize": bool(standardize),
                    "standardization_method": preprocess_cfg,
                    "postprocess_cs_standardize": bool(postprocess_cs_standardize),
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

        weights_cfg: Dict = synthesis_config.get("weights") or {}
        preprocess_cfg: Dict = synthesis_config.get("preprocess") or {}
        standardize = bool(synthesis_config.get("standardize", True))
        postprocess_cs_standardize = bool(
            synthesis_config.get("postprocess_cs_standardize", False)
        )

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

        inputs = factor_df.attrs.get("market_factor_inputs", {}).get("factors", [])
        is_trade_date_aligned = (
            factor_df.attrs.get(PANEL_DATE_ALIGNMENT_ATTR) == PANEL_DATE_ALIGNMENT_TRADE_DATE
        )
        df = factor_df[used_factors].copy()

        # Shift raw signal panels once so trading day T uses T-1 signal; panels
        # loaded through factor_data_loader are already trade-date aligned.
        if not is_trade_date_aligned:
            if factor_df.attrs.get(PANEL_DATE_ALIGNMENT_ATTR) is None:
                logger.warning(
                    "weighted_sum[%s]: factor_df 缺少 %s 属性，按原始信号面板执行单次 shift(1) 对齐；"
                    "直接传入未对齐面板时建议显式标注 attrs[%r] = %r",
                    synthesis_name,
                    PANEL_DATE_ALIGNMENT_ATTR,
                    PANEL_DATE_ALIGNMENT_ATTR,
                    PANEL_DATE_ALIGNMENT_RAW,
                )
            wide = df.unstack(level="stock_code").sort_index()
            shifted = wide.shift(1).stack(level="stock_code", future_stack=True)
            shifted.index = shifted.index.set_names(["date", "stock_code"])
            df = shifted.dropna(how="all").sort_index()
            if df.empty:
                raise RuntimeError(f"weighted_sum: empty composite for {synthesis_name}")

        resolved_preprocess = _resolve_weighted_sum_preprocess(preprocess_cfg, used_factors)
        if standardize:
            for f in used_factors:
                mode = resolved_preprocess[f]
                if mode in TIME_SERIES_METHODS:
                    df[f] = time_series_standardize(
                        df[f],
                        mode,
                        window=preprocess_cfg.get("_ts_window", 20),
                        min_periods=preprocess_cfg.get("_ts_min_periods"),
                    )
                elif mode in (None, "", "none"):
                    logger.info(
                        "weighted_sum[%s]: 因子 %s 不做预处理 (mode=%s)",
                        synthesis_name,
                        f,
                        mode,
                    )
                else:
                    logger.info(
                        "weighted_sum[%s]: 对因子 %s 使用按日截面 %s 预处理",
                        synthesis_name,
                        f,
                        mode,
                    )
                    df[f] = df[f].groupby(level="date").transform(
                        lambda s, m=mode: preprocess_cross_section_series(
                            s, m, resolved_preprocess,
                        )
                    )
        else:
            logger.info("weighted_sum[%s]: standardize=False，跳过全部逐因子预处理", synthesis_name)

        final_weights = {f: float(weights_cfg.get(f, 0.0)) for f in used_factors}
        valid_rows = df[used_factors].notna().any(axis=1)
        if not valid_rows.any():
            raise RuntimeError(f"weighted_sum: empty composite for {synthesis_name}")

        composite = weighted_score(df.loc[valid_rows, used_factors], final_weights).dropna()
        if composite.empty:
            raise RuntimeError(f"weighted_sum: empty composite for {synthesis_name}")

        composite.index = composite.index.set_names(["date", "stock_code"])
        if postprocess_cs_standardize:
            composite = composite.groupby(level="date").transform(standardize_composite_scores)

        analysis_period = synth_cfg.get("analysis_period")
        ap = analysis_period if isinstance(analysis_period, Mapping) else {}
        out_start = ap.get("start_date") if ap.get("start_date") is not None else synthesis_config.get("start_date")
        out_end = ap.get("end_date") if ap.get("end_date") is not None else synthesis_config.get("end_date")
        composite = _trim_composite_to_window(
            composite, start_date=out_start, end_date=out_end,
        )
        if composite.empty:
            raise RuntimeError(f"weighted_sum: empty composite for {synthesis_name}")

        composite_name = (
            f"{synth_cfg['output_settings']['composite_factor_prefix']}_"
            f"{synthesis_name}"
        )
        composite_series = composite.copy()

        composite_series.name = composite_name
        composite_series.attrs[PANEL_DATE_ALIGNMENT_ATTR] = PANEL_DATE_ALIGNMENT_TRADE_DATE
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

        _save_weighted_sum_bundle(
            output_dir=weights_output_dir,
            synthesis_name=synthesis_name,
            weights=final_weights,
            feature_names=used_factors,
            preprocess_cfg=resolved_preprocess,
            logger=logger,
            standardize=standardize,
            postprocess_cs_standardize=postprocess_cs_standardize,
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

