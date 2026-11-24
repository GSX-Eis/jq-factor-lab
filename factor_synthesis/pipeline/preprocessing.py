# -*- coding: utf-8 -*-
"""
Shared preprocessing utilities for synthesis methods.

Uses Qlib-style processors internally for consistent preprocessing logic.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Optional, Tuple, TypedDict
import logging

import numpy as np
import pandas as pd

from utils.market_db.guards import require_single_market_db
from utils.market_db.identity import slot_market_db_identity

from utils.market_factor import MarketFactorValueStore
from utils.slots import MarketSlot


class MarketInputOptions(TypedDict, total=False):
    slot: MarketSlot
    slot_id: str
    frequency: str
    factor_values_store: MarketFactorValueStore | str | Path
    factor_identities: Mapping[str, str]


def bind_synthesis_panel(synthesizer, panel, config, market_inputs, labels=None):
    """Consume a loaded panel or load the configured instruments and interval from Scheme B."""
    from ..core.factor_data_loader import load_factor_data
    from ..factor_reader import resolve_market_input

    raw_synth_cfg = getattr(synthesizer, "config", None)
    synth_cfg = raw_synth_cfg if isinstance(raw_synth_cfg, Mapping) else {}
    config = {} if config is None else config
    if not isinstance(config, Mapping):
        raise TypeError("synthesis config must be a mapping or None")
    source_keys = MarketInputOptions.__annotations__
    options = {key: synth_cfg[key] for key in source_keys if key in synth_cfg}
    options.update({key: config[key] for key in source_keys if key in config})
    if market_inputs:
        options.update(market_inputs)
    metadata = panel.attrs.get("market_factor_inputs", {}) if panel is not None else {}
    sources = metadata.get("factors", [])
    if sources:
        options.setdefault("slot_id", sources[0]["slot_id"])
        options.setdefault("frequency", sources[0]["frequency"])
    slot_id, frequency = resolve_market_input(
        options.get("slot"), options.get("slot_id"), options.get("frequency", "1d"),
    )
    if panel is None:
        required = ("factor_names", "start_date", "end_date")
        missing = [key for key in required if not config.get(key)]
        if missing:
            raise ValueError(f"Supply a synthesis panel or configure {', '.join(missing)}")
        panel = load_factor_data(
            None, "factor_values", config["factor_names"], config["start_date"], config["end_date"],
            stock_filter=set(config.get("instruments") or []),
            config=synth_cfg, neutralization_config=config.get("neutralization"), **options,
        )
        metadata = panel.attrs.get("market_factor_inputs", {})
    elif any((item["slot_id"], item["frequency"]) != (slot_id, frequency) for item in sources):
        raise ValueError("Scheme B panel sources do not match the requested slot and frequency")
    require_single_market_db(panel, labels, slot_market_db_identity(options.get("slot")))
    if metadata:
        pins = options.get("factor_identities") or {}
        for name, identity in metadata["identities"].items():
            if name in pins and pins[name] != identity:
                raise ValueError(f"Scheme B panel identity does not match factor_identities[{name!r}]")

    label_exits = labels.attrs.get("label_exit_dates") if labels is not None else None
    if label_exits is not None:
        def _norm_exit_time(value):
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                raise ValueError("label_exit_dates contains NaT")
            return ts.normalize() if ts.tz is None else ts.tz_convert("UTC")

        def _normalize_exit_map(raw_map):
            if not isinstance(raw_map, (Mapping, pd.Series)):
                raise ValueError("_label_exit_dates must be a mapping or Series matching producer label_exit_dates")
            normalized = {}
            for k, v in raw_map.items():
                day = _norm_exit_time(k)
                exit_day = _norm_exit_time(v)
                if day in normalized:
                    raise ValueError("Duplicate normalized entry date in label_exit_dates")
                normalized[day] = exit_day
            return normalized

        producer_map = _normalize_exit_map(label_exits)
        configured = config.get("_label_exit_dates")
        if configured is not None:
            configured_map = _normalize_exit_map(configured)
            valid_labels = labels.dropna()
            if valid_labels.empty or panel.empty:
                active_dates = set()
            elif (
                isinstance(valid_labels.index, pd.MultiIndex)
                and isinstance(panel.index, pd.MultiIndex)
                and valid_labels.index.nlevels == panel.index.nlevels
            ):
                effective_index = valid_labels.index.intersection(panel.index)
                active_dates = {_norm_exit_time(day) for day in effective_index.get_level_values(0).unique()}
            else:
                panel_dates = {_norm_exit_time(day) for day in panel.index.get_level_values(0).unique()}
                label_dates = {_norm_exit_time(day) for day in valid_labels.index.get_level_values(0).unique()}
                active_dates = panel_dates & label_dates
            if any(
                day not in producer_map or day not in configured_map or configured_map[day] != producer_map[day]
                for day in active_dates
            ):
                raise ValueError("Configured _label_exit_dates conflicts with label producer label_exit_dates")
        config = dict(config, _label_exit_dates=label_exits)
    if frequency != "1d" and not panel.empty and not freq_uses_utc(panel):
        raise ValueError("Non-daily synthesis requires UTC timestamps")
    if (slot_id != "jq_ashare_local" and labels is not None and not panel.empty
            and panel.index.get_level_values("stock_code").nunique() == 1
            and "target_cs_demean" not in config):
        config = dict(config, target_cs_demean=False)
    return panel, config


def selected_market_inputs(panel: pd.DataFrame, features) -> list[dict]:
    return [
        item for item in panel.attrs.get("market_factor_inputs", {}).get("factors", [])
        if item["factor_id"] in features
    ]


def freq_uses_utc(panel: pd.DataFrame) -> bool:
    return (isinstance(panel.index, pd.MultiIndex)
            and isinstance(panel.index.get_level_values("date"), pd.DatetimeIndex)
            and panel.index.get_level_values("date").tz is not None)


def format_panel_time(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat() if timestamp.tz is not None else timestamp.strftime("%Y-%m-%d")


def training_bounds(panel: pd.DataFrame, start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if not freq_uses_utc(panel):
        return first, last
    first = first.tz_localize("UTC") if first.tz is None else first.tz_convert("UTC")
    if last.tz is None:
        last = last.tz_localize("UTC")
        if last == last.normalize():
            last += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    else:
        last = last.tz_convert("UTC")
    return first, last


def prediction_bounds(panel: pd.DataFrame, train_end, window_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    _, last = training_bounds(panel, train_end, train_end)
    first = last + (pd.Timedelta(nanoseconds=1) if freq_uses_utc(panel) else pd.Timedelta(days=1))
    return first, first + pd.Timedelta(days=window_days)


def preprocess_synthesis_features(synthesizer, frame: pd.DataFrame, config, *, history: pd.DataFrame) -> pd.DataFrame:
    from ..core.factor_ops import apply_feature_preprocessing
    from ..core.feature_preprocessor import TIME_SERIES_METHODS

    if TIME_SERIES_METHODS.intersection(value for value in config.values() if isinstance(value, str)):
        return apply_feature_preprocessing(frame, config, synthesizer.logger, history=history)
    return synthesizer._apply_feature_preprocessing(frame, config)


def residualize_target_by_factor(
    target: pd.Series,
    baseline: pd.Series,
    *,
    min_samples: int = 10,
) -> pd.Series:
    """Remove the same-date linear baseline exposure from a return target."""
    aligned = pd.concat(
        [
            pd.to_numeric(target, errors="coerce").rename("target"),
            pd.to_numeric(baseline, errors="coerce").rename("baseline"),
        ],
        axis=1,
    )

    def residualize(group: pd.DataFrame) -> pd.Series:
        valid = group.dropna()
        output = pd.Series(np.nan, index=group.index, dtype=float)
        if len(valid) < min_samples:
            output.loc[valid.index] = valid["target"] - valid["target"].mean()
            return output
        centered_x = valid["baseline"] - valid["baseline"].mean()
        denominator = float(centered_x.pow(2).sum())
        if denominator <= 1e-12:
            output.loc[valid.index] = valid["target"] - valid["target"].mean()
            return output
        centered_y = valid["target"] - valid["target"].mean()
        beta = float((centered_x * centered_y).sum() / denominator)
        output.loc[valid.index] = centered_y - beta * centered_x
        return output

    level = "date" if "date" in aligned.index.names else 0
    groups = [residualize(group) for _, group in aligned.groupby(level=level, sort=False)]
    residual = pd.concat(groups) if groups else pd.Series(index=target.index, dtype=float)
    residual.name = target.name
    return residual.reindex(target.index)


def apply_target_cs_preprocessing(
    y: pd.Series,
    cs_demean: bool = True,
    cs_rank: bool = False,
    cs_standardize: bool = False,
    eps: float = 1e-8,
    logger: Optional[logging.Logger] = None,
) -> pd.Series:
    """
    Apply cross-sectional preprocessing to target variable.
    
    Uses Qlib TargetCSProcessor logic internally.
    
    Args:
        y: Target Series with MultiIndex (date, stock_code)
        cs_demean: Whether to demean cross-sectionally
        cs_rank: Whether to rank cross-sectionally (then demean)
        cs_standardize: Whether to standardize cross-sectionally
        eps: Small value to avoid division by zero
        logger: Optional logger
        
    Returns:
        Preprocessed target Series
    """
    try:
        from factor_synthesis.qlib_adapter.processors import TargetCSProcessor
        
        if not (cs_demean or cs_rank or cs_standardize):
            return y
        
        # Convert to DataFrame for processor
        y_df = pd.DataFrame({'LABEL0': y})
        
        # Rename index for Qlib compatibility
        original_names = y_df.index.names
        if 'date' in original_names:
            y_df.index = y_df.index.rename(['datetime' if n == 'date' else n for n in original_names])
        
        # Apply processor
        processor = TargetCSProcessor(
            cs_demean=cs_demean,
            cs_rank=cs_rank,
            cs_standardize=cs_standardize,
            eps=eps,
            fields_group="label"
        )
        
        y_processed = processor(y_df)['LABEL0']
        
        # Restore index names
        y_processed.index = y_processed.index.rename(original_names)
        
        if logger:
            logger.info(
                "[目标处理] 使用Qlib Processor: 去均值=%s, 标准化=%s, Rank化=%s",
                cs_demean, cs_standardize, cs_rank
            )
        
        return y_processed
        
    except ImportError:
        # Fallback to manual implementation
        if logger:
            logger.warning("[目标处理] Qlib processor不可用，使用回退实现")
        return _apply_target_cs_preprocessing_fallback(
            y, cs_demean, cs_rank, cs_standardize, eps, logger
        )


def _apply_target_cs_preprocessing_fallback(
    y: pd.Series,
    cs_demean: bool,
    cs_rank: bool,
    cs_standardize: bool,
    eps: float,
    logger: Optional[logging.Logger],
) -> pd.Series:
    """Fallback implementation without Qlib dependency."""
    result = y.copy()
    
    if cs_rank:
        y_rank = y.groupby(level="date").rank(pct=True)
        result = y_rank - y_rank.groupby(level="date").transform("mean")
        if logger:
            logger.info("[目标处理] 已执行Rank变换并按日去均值")
    else:
        if cs_demean:
            result = y - y.groupby(level="date").transform("mean")
        if cs_standardize:
            cs_std = result.groupby(level="date").transform("std")
            result = result / (cs_std.replace(0, np.nan).fillna(eps))
    
    return result


def compute_sample_weights(
    df: pd.DataFrame,
    weight_decay: float = 1.0,
    half_life: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    """
    Compute time-decay sample weights.
    
    Uses Qlib SampleWeightProcessor logic internally.
    
    Args:
        df: DataFrame with MultiIndex (date, stock_code)
        weight_decay: Exponential decay factor (1.0 = no decay)
        half_life: Half-life for decay (0.0 = use weight_decay instead)
        logger: Optional logger
        
    Returns:
        Sample weights array, or None if no decay
    """
    if not np.isfinite(half_life):
        raise ValueError("half_life must be finite")
    if half_life <= 0:
        if weight_decay >= 1.0:
            return None
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("Active weight_decay must be nonnegative and finite")
    
    try:
        from factor_synthesis.qlib_adapter.processors import SampleWeightProcessor
        
        # Rename index for Qlib compatibility
        df_qlib = df.copy()
        original_names = df_qlib.index.names
        if 'date' in original_names:
            df_qlib.index = df_qlib.index.rename(['datetime' if n == 'date' else n for n in original_names])
        
        processor = SampleWeightProcessor(
            weight_decay=weight_decay,
            half_life=half_life
        )
        processor(df_qlib)
        weights = processor.get_sample_weights()
        
        if logger:
            logger.info(
                "[样本权重] 使用Qlib Processor: decay=%.4f, half_life=%.1f",
                weight_decay, half_life
            )
        
        return weights
        
    except ImportError:
        # Fallback to manual implementation
        if logger:
            logger.warning("[样本权重] Qlib processor不可用，使用回退实现")
        return _compute_sample_weights_fallback(df, weight_decay, half_life, logger)


def _compute_sample_weights_fallback(
    df: pd.DataFrame,
    weight_decay: float,
    half_life: float,
    logger: Optional[logging.Logger],
) -> np.ndarray:
    """Fallback implementation without Qlib dependency."""
    dates = df.index.get_level_values("date")
    dates = pd.to_datetime(dates)
    max_date = dates.max()
    days_diff = (max_date - dates).days.values.astype(float)
    
    if half_life > 0:
        weights = np.exp(-np.log(2) * days_diff / half_life)
    else:
        weights = np.power(weight_decay, days_diff)
    
    # Normalize
    weights = weights / weights.mean()
    
    if logger:
        logger.info("[样本权重] 时间衰减权重计算完成")
    
    return weights






