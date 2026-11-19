"""Training-only quantile and normalization fits for time-series evaluation."""

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import numpy as np
import pandas as pd

from .kernels import QuantileResult, _positive_int
from .types import FloatArray

QuantileScheme = Literal["per_cross_section", "fitted_on_train"]
Standardization = Literal["zscore", "median_mad"]


@dataclass(frozen=True)
class InstrumentFit:
    # Finite training edges, including the observed extremes.
    edges: tuple[float, ...]
    center: float | None
    scale: float | None


@dataclass(frozen=True)
class QuantileFit:
    """Immutable parameters in training asset-axis order."""

    assets: tuple[str | int, ...]
    instruments: tuple[InstrumentFit, ...]
    n_quantiles: int
    direction: int
    standardization: Standardization | None

    def identity(self) -> dict:
        """Return JSON-safe parameters and the SHA-256 of their canonical JSON."""
        payload = {
            "schema": "evaluation.quantile_fit.v1",
            "scheme": "fitted_on_train",
            "n_quantiles": self.n_quantiles,
            "direction": self.direction,
            "standardization": self.standardization,
            "instruments": [
                {"asset": asset, "training_edges": list(fit.edges), "center": fit.center,
                 "scale": fit.scale}
                for asset, fit in zip(self.assets, self.instruments, strict=True)
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {**payload, "sha256": sha256(encoded.encode("utf-8")).hexdigest()}

    def assign(self, factor: FloatArray, assets: pd.Index, eligible: np.ndarray) -> QuantileResult:
        """Apply frozen raw-value boundaries with right-closed, extended endpoint buckets."""
        if tuple(assets) != self.assets:
            raise ValueError("Evaluation assets must match the frozen training asset axis")
        labels = np.zeros(factor.shape, dtype=np.intp)
        edges = np.full((self.n_quantiles + 1, len(assets)), np.nan)
        n_bins = np.zeros(len(assets), dtype=np.intp)
        for col, fit in enumerate(self.instruments):
            if len(fit.edges) < 3:
                continue
            cutoffs = fit.edges[1:-1]
            edges[:len(fit.edges), col] = (-np.inf, *cutoffs, np.inf)
            n_bins[col] = len(fit.edges) - 1
            rows = eligible[:, col]
            labels[rows, col] = np.searchsorted(cutoffs, factor[rows, col], side="left") + 1
        return QuantileResult(labels, edges, n_bins)

    def directional_signal(self, factor: FloatArray) -> FloatArray:
        """Compute the sign of the standardized factor without overflow or underflow."""
        if self.standardization is None:
            return factor
        centers = np.array([np.nan if fit.center is None else fit.center for fit in self.instruments])
        return (factor > centers).astype(float) - (factor < centers)


def _training_edges(values: FloatArray, probabilities: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        edges = np.quantile(values, probabilities, method="linear")
    failed = ~np.isfinite(edges)
    if failed.any():
        # Opposite-sign endpoints may have a difference larger than float64.
        ordered = np.sort(values)
        positions = (len(values) - 1) * probabilities[failed]
        left = positions.astype(np.intp)
        weight = positions - left
        edges[failed] = (
            ordered[left] * (1 - weight) + ordered[np.minimum(left + 1, len(values) - 1)] * weight
        )
    return np.unique(edges)


def _median(values: FloatArray) -> float:
    middle = len(values) // 2
    if len(values) % 2:
        return float(np.partition(values, middle)[middle])
    lower, upper = map(float, np.partition(values, (middle - 1, middle))[middle - 1:middle + 1])
    total = lower + upper
    return total / 2 if np.isfinite(total) else lower / 2 + upper / 2


def fit_quantiles(
    factor: FloatArray,
    assets: pd.Index,
    eligible: np.ndarray,
    *,
    n_quantiles: int,
    direction: int,
    standardization: Standardization | None,
) -> QuantileFit:
    """Fit one set of parameters per instrument from eligible training factors."""
    n_quantiles = _positive_int(n_quantiles, "n_quantiles", minimum=2)
    if isinstance(direction, (bool, np.bool_)) or direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if standardization not in (None, "zscore", "median_mad"):
        raise ValueError("standardization must be None, 'zscore', or 'median_mad'")
    if (factor.ndim != 2 or factor.shape[1] != len(assets)
            or eligible.shape != factor.shape or eligible.dtype.kind != "b" or not assets.is_unique):
        raise ValueError("Expected a factor panel, a matching boolean mask and a unique asset axis")
    asset_ids = tuple(asset.item() if isinstance(asset, np.generic) else asset for asset in assets)
    if any(type(asset) not in (str, int) for asset in asset_ids):
        raise ValueError("Fitted quantiles require string or integer asset identifiers")

    probabilities = np.arange(n_quantiles + 1) / n_quantiles
    fits = []
    for col in range(len(asset_ids)):
        values = factor[eligible[:, col] & np.isfinite(factor[:, col]), col]
        if not values.size:
            fits.append(InstrumentFit((), None, None))
            continue
        edges = _training_edges(values, probabilities)
        center, scale = None, None
        if standardization is not None and edges.size == 1:
            center, scale = float(edges[0]), 0.0
        elif standardization == "zscore":
            # Power-of-two scaling keeps squared deviations in range.
            _, exponent = np.frexp(np.max(np.abs(values)))
            scaled = np.ldexp(values, -exponent)
            center, scale = (float(value) for value in np.ldexp([scaled.mean(), scaled.std(ddof=0)], exponent))
        elif standardization == "median_mad":
            center = _median(values)
            # Overflowing deviations belong to the upper tail, beyond the middle ranks.
            with np.errstate(over="ignore"):
                scale = _median(np.abs(values - center))
        fits.append(InstrumentFit(tuple(edges.tolist()), center, 1.0 if scale == 0 else scale))
    return QuantileFit(asset_ids, tuple(fits), n_quantiles, int(direction), standardization)
