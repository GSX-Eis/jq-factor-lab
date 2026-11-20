"""Small synthetic-data example using the synchronized synthesis pipelines."""
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any


METHODS = ("weighted_sum", "pca", "lasso", "ridge", "elasticnet", "quantile_elasticnet", "lightgbm")


def synthetic_panel():
    """Independent random features and an artificial response; no market observations."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=100)
    index = pd.MultiIndex.from_product(
        [dates, [f"DEMO{i:03d}" for i in range(12)]], names=["date", "stock_code"]
    )
    features = pd.DataFrame(rng.normal(size=(len(index), 4)), index=index,
                            columns=[f"signal_{i}" for i in range(4)])
    response = (0.02 * features.signal_0 - 0.01 * features.signal_1
                + rng.normal(scale=0.005, size=len(index))).rename("response")
    return features, response


def run_demo(
    method: str, output_dir: Path | None = None, *, overwrite: bool = False,
) -> dict[str, Any]:
    """Run one teaching window with a completion marker for its outputs."""
    import numpy as np
    from utils.local_outputs import new_run_directory, stage_demo_output

    if method not in METHODS:
        raise ValueError(f"Unknown synthesis method: {method}")
    # The artificial same-row response has no verified forward-label exit.
    # PCA uses the same strict pipeline context, despite not fitting y.
    if method != "weighted_sum":
        from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
        raise LabelIsolationError(
            f"{method} cannot run on synthetic_panel: its same-row artificial target has no "
            "verified label exit context. Supervised selection must isolate label exits; "
            "PCA shares this strict window contract. Run jq-factor-lab factor-synthesis-demo --method "
            "weighted_sum for this example, or python3 -m examples.local_factor_workflow to produce "
            "price-backed forward labels. Training requires wiring those producer-verified "
            "exit dates into the pipeline context; local_factor_workflow demonstrates "
            "fixed-weight synthesis and labels, not supervised training. Do not invent "
            "exit dates or disable strict validation. See docs/SYNTHESIS.md."
        )
    from factor_synthesis.core.factor_ops import apply_feature_preprocessing
    from factor_synthesis.pipeline import run_weighted_sum_synthesis

    X, _ = synthetic_panel()
    captured = []
    context = SimpleNamespace(
        logger=logging.getLogger(__name__),
        config={
            "rolling_window": {"predict_window_days": 20, "min_train_samples": 100},
            "factor_groups": {"demo": {"factors": list(X.columns)}},
            "output_settings": {"composite_factor_prefix": "demo"},
            "performance_evaluation": {"skip_alphalens_analysis": True},
        },
        _apply_feature_preprocessing=apply_feature_preprocessing,
        direct_analyze_composite_factor=lambda values, *_args, **_kwargs: captured.append(values),
    )
    if output_dir is None:
        target = new_run_directory(Path("outputs/synthesis"))
    else:
        path = Path(output_dir).expanduser()
        target = path.parent.resolve() / path.name
    with stage_demo_output(target, overwrite=overwrite or output_dir is None) as staging:
        context.config["output_settings"]["weights_output_dir"] = str(staging)
        run_weighted_sum_synthesis(context, "demo", {
            "factor_group": "demo", "weights": {name: 0.25 for name in X.columns},
            "preprocess": {name: "rank_zscore" for name in X.columns},
        }, X)
        predictions = captured[0]
        if predictions is None or predictions.empty or not np.isfinite(predictions.to_numpy()).all():
            raise RuntimeError("Synthesis returned no finite predictions")
        predictions.rename("score").to_csv(staging / "scores.csv")
        summary = {"method": method, "data": "synthetic", "rows": len(predictions),
                   "features": list(X.columns), "training_end": None, "output_dir": str(target),
                   "purpose": "API demonstration; not historical investment performance"}
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
