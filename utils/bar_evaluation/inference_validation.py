"""Validate and project the v2 inference report published in summary.json."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from hashlib import sha256
from typing import Any

import pandas as pd

from backtest_and_analysis.evaluation.calibration import Trial, TrialLedger
from backtest_and_analysis.evaluation.types import BOOTSTRAP_V2, BOOTSTRAP_V3_P, BootstrapInterval, HACInterval

from .grid_validation import LabelGrid, build_label_grid, matching_calendars, resolve_bucket_strategy
from .interval_validation import _interval_fields, _usability, _verified_interval
from .labels_validation import SampleMinimum, StreamKey, check_point, check_split_labels
from .normalization import (
    _array, _canonical_json, _coerce_optional_json_str, _count, _metric, _nullable_text, _object, _require,
)
from .schema import _SPLIT_NAMES, _SPLIT_TABLE_COLUMNS, BarEvaluationIntegrityError


def _utc(value: Any, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value) if isinstance(value, str) else None
        if stamp is not None and stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0:
            return stamp
    except (TypeError, ValueError, OverflowError):
        pass
    raise BarEvaluationIntegrityError(f"summary.json {field}: expected a UTC timestamp")


def _check_ledger(value: Any, split: str | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    field = "trial_ledger"
    ledger = _object(value, field, "method", "alpha", "attempt_count", "selection_complete",
                     "best", "pending_candidates", "missing_p_value_candidates", "candidates", "split")
    method = ledger["method"]
    alpha = _metric(ledger["alpha"], f"{field}.alpha", probability=True)
    _require(method in ("holm", "bh"), field, "Holm or BH correction")
    _require(alpha is not None and 0 < alpha < 1, field, "alpha between 0 and 1")
    _require(split is not None and ledger["split"] == split, field, "the report's selection split")
    candidates = _array(ledger["candidates"], f"{field}.candidates")
    _require(_count(ledger["attempt_count"], f"{field}.attempt_count", minimum=1) == len(candidates),
             field, "attempt_count matching candidates")
    trials: dict[tuple[str, int], Trial | None] = {}
    for candidate in candidates:
        row = _object(candidate, f"{field}.candidates[]", "factor_id", "horizon", "statistic", "p_value",
                      "n_effective", "n_observations", "n_dependence_adjusted", "interval", "interval_usability",
                      "reason", "adjusted_p_value", "inference_adjusted_p_value", "significant")
        factor, horizon = row["factor_id"], _count(row["horizon"], f"{field}.horizon", minimum=1)
        _require(isinstance(factor, str) and bool(factor.strip()), field, "a nonempty factor_id")
        key = (factor, horizon)
        _require(key not in trials, field, "unique candidate keys")
        _nullable_text(row["reason"], f"{field}.reason")
        p_method = row.get("p_value_method")
        _require(
            p_method is None or (
                row["p_value"] is not None and p_method in ("two_sided_normal_hac", "two_sided_student_t_hac")
            ), field, "a known method for a reported p value",
        )
        _require(type(row["significant"]) is bool, f"{field}.significant", "a boolean")
        for col in ("p_value", "adjusted_p_value", "inference_adjusted_p_value"):
            _metric(row[col], f"{field}.{col}", probability=True)
        _metric(row["n_dependence_adjusted"], f"{field}.n_dependence_adjusted")
        if row["reason"] == "not_evaluated":
            _require(all(row[col] is None for col in (
                "statistic", "p_value", "interval", "interval_usability", "n_effective",
                "n_observations", "n_dependence_adjusted",
            )), field, "null statistics on pending candidates")
            trials[key] = None
            continue
        n = _count(row["n_observations"], f"{field}.n_observations")
        _require(_count(row["n_effective"], f"{field}.n_effective") == n, field, "matching observation counts")
        bounds = row["interval"]
        if bounds is not None:
            _require(isinstance(bounds, list) and len(bounds) == 2, field, "a pair of interval bounds")
            bounds = tuple(_metric(bound, f"{field}.interval") for bound in bounds)
            _require(all(bound is not None for bound in bounds), field, "finite interval bounds")
        use = row["interval_usability"]
        try:
            trials[key] = Trial(
                _metric(row["statistic"], f"{field}.statistic"), row["p_value"], n, bounds,
                None if use is None else _usability(use, f"{field}.interval_usability", horizon),
                p_value_method=p_method,
            )
        except (TypeError, ValueError) as exc:
            raise BarEvaluationIntegrityError(f"summary.json {field}: invalid trial {key!r}: {exc}") from exc

    pending = [list(key) for key, trial in trials.items() if trial is None]
    missing = [list(key) for key, trial in trials.items() if trial is not None and trial.p_value is None]
    _require(ledger["pending_candidates"] == pending and ledger["missing_p_value_candidates"] == missing,
             field, "pending and missing keys matching candidate records")
    complete = not (pending or missing)
    _require(ledger["selection_complete"] is complete, field, "selection_complete matching available p values")
    selection = None
    if complete:
        registry = TrialLedger(trials)
        for key, trial in trials.items():
            registry.record(key, trial)
        selection = registry.select_best(method=method, alpha=alpha)
    best = list(selection.best) if selection is not None and selection.best is not None else None
    _require(ledger["best"] == best, field, "best matching gated selection")
    for index, (key, row) in enumerate(zip(trials, candidates, strict=True)):
        adjusted = selection.trials[index] if selection is not None else None
        _require(
            row["adjusted_p_value"] == (adjusted.adjusted_p_value if adjusted is not None else None)
            and row["inference_adjusted_p_value"] == (
                selection.inference_adjusted_p_values.get(key) if selection is not None else None
            )
            and row["significant"] is (adjusted.significant if adjusted is not None else False),
            field, "candidate adjustments and significance matching the registered family",
        )
    return candidates


def _key(row: dict[str, Any], field: str, time_series: bool) -> tuple[str, int, str | None]:
    frequency, instrument = row.get("frequency"), row.get("instrument_id")
    _require(isinstance(frequency, str) and bool(frequency.strip()), field, "a nonempty frequency")
    _require(
        isinstance(instrument, str) and bool(instrument.strip()) if time_series else instrument is None,
        field, "an instrument_id for time series or null for cross sections",
    )
    return frequency, _count(row.get("horizon_bars"), f"{field}.horizon_bars", minimum=1), instrument


def _check_fits(fits: Any, keys: set[tuple[str, int, str | None]], field: str) -> None:
    actual = set()
    for frequency, instruments in _object(fits, field).items():
        for instrument, fit in _object(instruments, field).items():
            payload = _object(fit, field, "sha256", "schema", "scheme", "instruments",
                              "n_quantiles", "direction", "standardization")
            body = {k: v for k, v in payload.items() if k != "sha256"}
            encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
            _require(payload["sha256"] == sha256(encoded.encode("utf-8")).hexdigest(),
                     field, "a verified fit identity digest")
            _require(payload["schema"] == "evaluation.quantile_fit.v1" and payload["scheme"] == "fitted_on_train",
                     field, "a fitted_on_train quantile identity")
            actual.add((frequency, instrument))
    _require(actual == {(f, i) for f, _, i in keys}, field, "fits for the reported instruments")


def _check_instrument_pvalue_summaries(
    summaries: list[Any], bootstraps: dict[tuple[str, int], list[dict[str, Any]]], field: str,
) -> None:
    expected = {
        key: (rows[0]["p_value"], rows[0]["p_value_reason"]) if len(rows) == 1
        else (None, "multi_asset_pvalue_undefined")
        for key, rows in bootstraps.items()
        if any(row.get("method_version") == BOOTSTRAP_V3_P[0] for row in rows)
    }
    p_fields = {"p_value", "p_value_reason"}
    if not expected and not any(isinstance(row, dict) and p_fields & row.keys() for row in summaries):
        return
    seen = set()
    for raw in summaries:
        row_field = f"{field}[]"
        row = _object(raw, row_field)
        frequency, horizon, _ = _key(row, row_field, time_series=False)
        key = frequency, horizon
        _require(key in bootstraps and key not in seen, row_field, "a unique verified frequency/horizon group")
        seen.add(key)
        if key not in expected:
            _require(not p_fields & row.keys(), row_field, "P value fields only for verified v3 bootstrap groups")
            continue
        _object(row, row_field, "p_value", "p_value_reason")
        p = _metric(row["p_value"], f"{row_field}.p_value", probability=True)
        _nullable_text(row["p_value_reason"], f"{row_field}.p_value_reason")
        _require((p, row["p_value_reason"]) == expected[key], row_field,
                 "P value and reason matching the verified instrument details")
    _require(expected.keys() <= seen, field, "one summary for every verified v3 frequency/horizon group")


def _split_row(
    row: dict[str, Any], *, split: str, key: StreamKey, time_series: bool,
    trial: dict[str, Any], fit: dict[str, Any] | None, field: str, labels: pd.DataFrame, grid: LabelGrid,
    cutoffs: tuple[pd.Timestamp | None, pd.Timestamp | None], oos_grid_start: pd.Timestamp | None,
    minimum: SampleMinimum, config: dict[str, Any], code_version: str, code_version_source: str | None,
    n_quantiles: int | None,
) -> dict[str, Any]:
    frequency, horizon, instrument = key
    method = "bootstrap" if time_series else "hac"
    ic_field, count_field = ("spearman_ic", "sample_count") if time_series else ("mean", "n_valid")
    ic = _object(row.get("ic"), f"{field}.ic", ic_field, count_field, "reason")
    statistic = _metric(ic[ic_field], f"{field}.ic.{ic_field}")
    n = _count(ic[count_field], f"{field}.ic.{count_field}")
    _nullable_text(ic["reason"], f"{field}.ic.reason")
    _require(not time_series or (statistic is None) == (ic["reason"] is not None),
             field, "Spearman IC reason matching point availability")
    interval_type = BootstrapInterval if time_series else HACInterval
    raw_interval = _object(row.get(method), f"{field}.{method}", "calibration_status", "calibration_evidence")
    interval = _interval_fields(raw_interval, f"{field}.{method}", interval_type)
    legacy = "method_version" not in raw_interval
    if time_series:
        _require(row.get("intervals_verified") is True, field, "verified Bar holding intervals")
    _nullable_text(interval["reason"], f"{field}.{method}.reason")
    alpha = _metric(interval["alpha"], f"{field}.{method}.alpha", probability=True)
    _require(alpha is not None and 0 < alpha < 1, field, "alpha between 0 and 1")
    grid_size = _count(interval["grid_size"], f"{field}.{method}.grid_size")
    _require(n <= grid_size,
             field, "observation count within the interval grid")
    lower = _metric(interval["lower"], f"{field}.{method}.lower")
    upper = _metric(interval["upper"], f"{field}.{method}.upper")
    bounds = [lower, upper] if lower is not None and upper is not None else None
    _require((lower is None and upper is None) or (bounds is not None and lower <= upper),
             field, "null or ordered interval bounds")
    use = _usability(row.get("interval_usability"), f"{field}.interval_usability", horizon)
    start, end = (_utc(row.get(k), f"{field}.{k}") for k in ("grid_start_utc", "grid_end_utc"))
    if split == "train":
        _require(oos_grid_start is not None and end < oos_grid_start, f"{field}.grid_end_utc",
                 "a clock strictly before the matching OOS grid_start_utc")
    actual, observed_count, section_sizes, section_ics, pair_pattern = check_split_labels(
        labels, grid=grid, time_series=time_series, split=split, start=start, end=end, train_end=cutoffs[0],
        oos_start=cutoffs[1], oos_grid_start=oos_grid_start, grid_size=grid_size,
        minimum=minimum, claimed_count=n, point_reason=ic["reason"], field=field,
        bootstrap_v2=time_series and interval["method_version"] == BOOTSTRAP_V2[0],
        row=row, n_quantiles=n_quantiles, fit=fit,
    )
    check_point(statistic, actual, f"{field}.ic.{ic_field}")
    _require(n == observed_count, field, "IC observation count matching the split label pairs")
    p = _metric(row.get("p_value"), f"{field}.p_value", probability=True)
    p_method = row.get("p_value_method")
    _require(time_series or (p is None) == (bounds is None), field, "HAC p value matching interval availability")
    _verified_interval(
        interval, use, time_series=time_series, horizon=horizon,
        statistic=statistic if statistic is not None else float("nan"),
        point_reason=ic["reason"] if time_series else None, n=n, cross_section_sizes=section_sizes,
        section_ics=section_ics, pair_pattern=pair_pattern, bounds=bounds, p_value=p, p_method=p_method, alpha=alpha,
        config=config, field=f"{field}.interval_usability", legacy=legacy,
        raw_interval=raw_interval, code_version=code_version, code_version_source=code_version_source,
    )
    if trial:
        _require(
            (trial["statistic"], trial["n_observations"], trial["p_value"], trial["interval"],
             trial["interval_usability"]) == (statistic, n, p, bounds, row["interval_usability"])
            and trial.get("p_value_method", p_method if legacy else None) == p_method,
            "trial_ledger", "the current candidate matching its unique split IC, sample, p value, interval and usability",
        )
    purge = _count(row.get("split_purged_count"), f"{field}.split_purged_count")
    return dict(
        split=split, frequency=frequency, instrument_id=instrument, horizon_bars=horizon,
        split_purged_count=purge, ic=statistic, ic_reason=ic["reason"], n_observations=n,
        n_dependence_adjusted=trial.get("n_dependence_adjusted"), interval_method=method,
        interval_lower=lower, interval_upper=upper, calibration_status=use.calibration_status,
        descriptive_stats_usable=use.descriptive_stats_usable,
        significance_interpretation_usable=use.significance_interpretation_usable,
        p_value=p, p_value_method=p_method, adjusted_p_value=trial.get("adjusted_p_value"),
        inference_adjusted_p_value=trial.get("inference_adjusted_p_value"), significant=trial.get("significant"),
        fit_identity_json=_canonical_json(fit) if fit is not None else None,
        metrics_json=_coerce_optional_json_str(row, field),
    )


def prepare_inference(
    summary: dict[str, Any], identity_hash: str, label_streams: dict[StreamKey, pd.DataFrame], minimum: SampleMinimum,
    *, instruments: pd.Index, frequencies: pd.Index, code_version: str, quantile_counts: dict[StreamKey, int],
) -> tuple[dict[str, Any], list[tuple[Any, ...]]]:
    """Check the summary-only v2 sections and retain the whole report alongside projections."""
    _object(summary, "", "splits", "interval_usability", "trial_ledger")
    time_series = summary["analysis_mode"] == "time_series"
    code_version_source = summary["result_identity"].get("code_version_source")
    config = summary.get("config") or {}
    ic_rows = [row for row in summary["ic_by_horizon"] if not time_series or row.get("scope") == "per_instrument"]
    keys = {_key(row, "ic_by_horizon", time_series) for row in ic_rows}
    _require(len(keys) == len(ic_rows), "ic_by_horizon", "unique stream/horizon keys")
    calibration = _array(summary["interval_usability"], "interval_usability")
    calibration_keys = set()
    for raw in calibration:
        row = _object(raw, "interval_usability[]", "instrument_id")
        key = _key(row, "interval_usability[]", time_series)
        _require(key not in calibration_keys, "interval_usability", "unique stream/horizon keys")
        calibration_keys.add(key)
        use = _usability(row, "interval_usability[]", key[1])
        _require(use.calibration_status == "uncalibrated", "interval_usability[]",
                 "uncalibrated descriptive IC without split interval evidence")
    _require(calibration_keys == keys, "interval_usability", "the same stream/horizon keys as IC")

    splits = _object(summary["splits"], "splits")
    has_split = set(splits) == {"train", "oos"}
    _require(not splits or set(splits) == {"full"} or has_split, "splits", "full, train and oos, or empty")
    selection_split = ("train" if has_split else "full") if splits else None
    ledger = summary["trial_ledger"]
    if splits and (not time_series or has_split or any(config.get(option) is not None for option in (
        "trial_selection", "bootstrap", "hac_lag", "direction", "standardization", "calibration_scenario",
    ))):
        _require(ledger is not None, "trial_ledger", "the candidate family for the inference splits")
    candidates = _check_ledger(ledger, selection_split)
    harness_input = "bars_csv" in config or "continuous_futures" in config
    factor = config.get("factor_id", "factor_value" if harness_input else None)
    factors = {c["factor_id"] for c in candidates if c["reason"] != "not_evaluated"}
    if not factors:
        factors = {c["factor_id"] for c in candidates}
    if factor is None and len(factors) == 1:
        factor = next(iter(factors))
    candidates = {row["horizon"]: row for row in candidates if row["factor_id"] == factor}
    stream_count = Counter(h for _, h, _ in keys)
    calendars = matching_calendars(summary) if splits else None
    bucket_strategy = resolve_bucket_strategy(summary, calendars, instruments=instruments, frequencies=frequencies)
    grids: dict[tuple[str, str | None], LabelGrid] = {}

    oos_grid_starts = {}
    if has_split:
        oos = _object(splits["oos"], "splits.oos", "horizons")
        for raw in _array(oos["horizons"], "splits.oos.horizons"):
            row = _object(raw, "splits.oos.horizons[]")
            oos_grid_starts[_key(row, "splits.oos", time_series)] = _utc(
                row.get("grid_start_utc"), "splits.oos.grid_start_utc",
            )

    split_config = {}
    tuples = []
    for split in _SPLIT_NAMES:
        if split not in splits:
            continue
        field = f"splits.{split}"
        report = _object(splits[split], field, "train_end", "oos_start", "quantile_scheme",
                         "fit_identity", "split_purged_count", "horizons")
        split_config[split] = {k: v for k, v in report.items() if k != "horizons"}
        cutoffs = (None, None)
        train_end, oos_start = report["train_end"], report["oos_start"]
        if has_split:
            cutoffs = _utc(train_end, field), _utc(oos_start, field)
            _require(cutoffs[0] < cutoffs[1], field, "train_end preceding oos_start")
        else:
            _require(train_end is None and oos_start is None, field, "null bounds for full sample")
        scheme = "per_cross_section"
        summaries = []
        bootstraps: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        if time_series:
            scheme = "fitted_on_train" if has_split else "full_sample_descriptive_only"
            summaries = _array(report.get("instrument_ic_summaries"), f"{field}.instrument_ic_summaries")
        _require(report["quantile_scheme"] == scheme, field, f"quantile_scheme={scheme}")
        fits = report["fit_identity"]
        if scheme == "fitted_on_train":
            _check_fits(fits, keys, f"{field}.fit_identity")
        else:
            _require(fits is None, field, "null fit_identity outside fitted_on_train")

        counts = _object(report["split_purged_count"], f"{field}.split_purged_count")
        _require(all(k.isdigit() and not k.startswith("0") for k in counts),
                 field, "positive horizon keys for split_purged_count")
        purged = dict.fromkeys(counts, 0)
        seen = set()
        for raw in _array(report["horizons"], f"{field}.horizons"):
            row = _object(raw, f"{field}.horizons[]")
            key = _key(row, field, time_series)
            frequency, horizon, instrument = key
            _require(key in keys and key not in seen and str(horizon) in counts,
                     field, "unique known stream/horizon keys and purge totals")
            if split == "train":
                _require(key in oos_grid_starts, field, "a matching OOS stream")
            trial = {}
            if ledger is not None and factor is not None and split == selection_split and stream_count[horizon] == 1:
                trial = candidates.get(horizon)
                _require(trial is not None and trial["reason"] != "not_evaluated",
                         "trial_ledger", f"an evaluated candidate for the current factor {factor!r}, horizon {horizon}")
            seen.add(key)
            grid_key = frequency, instrument
            if grid_key not in grids:
                grids[grid_key] = build_label_grid(
                    summary, label_streams[key], calendars or {}, bucket_strategy=bucket_strategy,
                )
            record = _split_row(
                row, split=split, key=key, time_series=time_series, field=f"{field}.horizons[]",
                trial=trial,
                fit=fits[frequency][instrument] if fits is not None else None, labels=label_streams[key],
                grid=grids[grid_key],
                cutoffs=cutoffs, oos_grid_start=oos_grid_starts.get(key), minimum=minimum, config=config,
                code_version=code_version, code_version_source=code_version_source,
                n_quantiles=quantile_counts.get(key),
            )
            if time_series:
                bootstraps[frequency, horizon].append(row["bootstrap"])
            purged[str(horizon)] += record["split_purged_count"]
            record.update(identity_hash=identity_hash, row_index=len(tuples))
            tuples.append(tuple(record[col] for col in _SPLIT_TABLE_COLUMNS))
        _require(seen == keys, field, "the same stream/horizon keys as IC")
        if time_series:
            _check_instrument_pvalue_summaries(summaries, bootstraps, f"{field}.instrument_ic_summaries")
        _require(all(_count(n, f"{field}.split_purged_count") == purged[h] for h, n in counts.items()),
                 field, "purge totals matching horizon rows")
        _require(split == "train" or not any(purged.values()), field, "purging only in train")
    if has_split:
        for key in ("train_end", "oos_start", "fit_identity"):
            _require(splits["train"][key] == splits["oos"][key], f"splits.{key}", "matching train and oos metadata")
    return {
        "has_split": has_split,
        "bucket_strategy": bucket_strategy,
        # The frozen fit digest covers the original JSON number representation.
        "split_config_json": _canonical_json(split_config),
        "trial_ledger_json": _coerce_optional_json_str(ledger, "trial_ledger"),
        "calibration_summary_json": _coerce_optional_json_str(calibration, "interval_usability"),
    }, tuples
