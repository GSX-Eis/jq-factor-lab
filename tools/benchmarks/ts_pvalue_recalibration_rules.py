"""Preregistered decision statistics for the NERE-1279 warp-speed experiment."""
from math import sqrt

import numpy as np
from scipy.stats import binom, binomtest, norm

from backtest_and_analysis.evaluation.calibration import adjust_pvalues


def size_summary(critical: np.ndarray, observed: np.ndarray, alpha: float) -> dict:
    """Combine evaluation and estimated-quantile noise with a local delta method."""
    cutoff, low, high = np.quantile(critical, [1 - alpha, 1 - 1.2 * alpha, 1 - 0.8 * alpha],
                                    method="inverted_cdf")
    rejected = int(np.count_nonzero(observed > cutoff))
    estimate = rejected / len(observed)
    # F_obs(high) - F_obs(low) divided by the fixed 0.4*alpha CDF bandwidth.
    ratio = np.count_nonzero((low < observed) & (observed <= high)) / (len(observed) * 0.4 * alpha)
    if not low < cutoff < high:
        raise ValueError("The preregistered density-ratio window collapsed")
    evaluation_variance = estimate * (1 - estimate) / len(observed)
    critical_variance = ratio**2 * alpha * (1 - alpha) / len(critical)
    se = sqrt(evaluation_variance + critical_variance)
    if not se:
        raise ValueError("Degenerate Monte Carlo variance")
    width = norm.ppf(0.995) * se
    upper = min(1.0, estimate + norm.ppf(0.99) * se)
    return dict(alpha=alpha, rejected=rejected, rejection_rate=estimate, critical_value=float(cutoff),
                repetitions=len(observed), critical_repetitions=len(critical),
                evaluation_variance=evaluation_variance, critical_variance=critical_variance,
                density_ratio=ratio, ci99=[max(0.0, estimate - width), min(1.0, estimate + width)],
                upper99=upper, hard_cap_pass=bool(upper <= 1.5 * alpha),
                deviation_p=float(2 * norm.sf(abs(estimate - alpha) / se)))


def power_summary(critical: np.ndarray, observed: np.ndarray, alpha: float, sign: int) -> dict:
    cutoff = float(np.quantile(critical, 1 - alpha, method="inverted_cdf"))
    rejections = int(np.count_nonzero(np.abs(observed) > cutoff))
    signed = int(np.count_nonzero(sign * observed > cutoff))
    power = signed / len(observed)
    lower, upper = binom.ppf([0.0025, 0.9975], len(critical), 1 - alpha).astype(int)
    ordered = np.sort(critical)
    largest_cutoff = ordered[upper] if upper < len(ordered) else np.inf
    smallest_cutoff = ordered[lower - 1] if lower else -np.inf
    at_high = int(np.count_nonzero(sign * observed > largest_cutoff))
    at_low = int(np.count_nonzero(sign * observed > smallest_cutoff))
    ci_low = binomtest(at_high, len(observed)).proportion_ci(0.995).low
    ci_high = binomtest(at_low, len(observed)).proportion_ci(0.995).high
    return dict(alpha=alpha, critical_value=cutoff, rejected=rejections, repetitions=len(observed),
                rejection_rate=rejections / len(observed), signed_rejections=signed, power=power,
                power_ci99=[ci_low, ci_high], passes=power >= 0.8)


def apply_holm(cells: list[dict]) -> None:
    null_rows = [row for cell in cells if not cell["sign"] for row in cell["thresholds"]]
    adjusted = adjust_pvalues([row["deviation_p"] for row in null_rows], method="holm")
    for row, p in zip(null_rows, adjusted, strict=True):
        row.update(holm_p=float(p), holm_pass=bool(p > 0.01), passes=bool(p > 0.01 and row["hard_cap_pass"]))


def pooled_size(cells: list[dict]) -> list[dict]:
    totals: dict[tuple[int, float], list[int]] = {}
    for cell in cells:
        if cell["sign"]:
            continue
        for row in cell["thresholds"]:
            group = totals.setdefault((cell["grid_size"], row["alpha"]), [0, 0])
            group[0] += row["rejected"]
            group[1] += row["repetitions"]
    return [
        dict(grid_size=size, alpha=alpha, rejected=count, repetitions=n, rejection_rate=count / n,
             passes=count / n <= 1.1 * alpha)
        for (size, alpha), (count, n) in sorted(totals.items())
    ]


def check_scope(cells: list[dict], nested_pass: bool) -> dict:
    """Apply the registered eliminations once, retaining the original Holm family."""
    excluded = sorted({
        cell["feature"] for cell in cells if cell["feature"] != "gaussian"
        and any(not row["passes"] for row in cell["thresholds"])
    })
    retained = [cell for cell in cells if cell["feature"] not in excluded]
    full_pooled, retained_pooled = pooled_size(cells), pooled_size(retained)
    failures = {(cell["grid_size"], row["alpha"]) for cell in retained
                for row in cell["thresholds"] if not row["passes"]}
    failures |= {(row["grid_size"], row["alpha"]) for row in retained_pooled if not row["passes"]}
    exclude_small_high_alpha = failures == {(512, 0.05)}
    passes = nested_pass and (not failures or exclude_small_high_alpha)
    return dict(passes=passes, excluded_features=excluded, exclude_512_alpha_005=exclude_small_high_alpha,
                full_pooled=full_pooled, retained_pooled=retained_pooled,
                retained_failures=sorted(failures), nested_pass=nested_pass)
