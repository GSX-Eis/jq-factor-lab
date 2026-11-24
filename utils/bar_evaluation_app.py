"""Read-only Streamlit viewer for Bar evaluation results (`bar_evaluation_*`).

Launch:
    streamlit run utils/bar_evaluation_app.py -- --results-db <path-to-duckdb>
or via environment variable:
    QUANT_BAR_RESULTS_DB=<path-to-duckdb> streamlit run utils/bar_evaluation_app.py

Importing this module performs no database access and no writes.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from utils.bar_evaluation_store import (
    get_bar_evaluation_filter_options,
    query_bar_evaluation_detail,
    query_bar_evaluation_runs,
)

RESULTS_DB_ENV_VAR = "QUANT_BAR_RESULTS_DB"
st = None


def _load_streamlit() -> None:
    global st
    if st is None:
        try:
            import streamlit as st
        except ModuleNotFoundError as exc:
            if exc.name != "streamlit":
                raise
            raise SystemExit("Run python3 -m pip install 'jq-factor-lab[viz]'") from None


def resolve_results_db_path(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the explicit read-only results DuckDB path from CLI args or env."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-db", default=None)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    known, _ = parser.parse_known_args(raw_argv)

    env = os.environ if environ is None else environ
    raw_path = known.results_db if known.results_db is not None else env.get(RESULTS_DB_ENV_VAR)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(
            f"Explicit results DB is required: pass --results-db <path> or set {RESULTS_DB_ENV_VAR}"
        )
    return Path(raw_path).expanduser().resolve()


def _format_run_label(run: Mapping[str, Any]) -> str:
    short_hash = str(run.get("identity_hash", ""))[:12]
    mode = run.get("analysis_mode", "")
    freq = run.get("frequency", "")
    cal = run.get("calendar_id") or "-"
    slot = run.get("slot_id") or "-"
    ingested = run.get("ingested_at_utc") or ""
    return f"{short_hash} | {mode} | {freq} | {cal} | {slot} | {ingested}"


def _extract_build_id_summary(data_identity: Mapping[str, Any]) -> Any:
    found: dict[str, Any] = {}
    for key in ("db_id", "profile_hash", "market_domain", "slot_data_fingerprint", "build_id", "build_ids", "roll_build_ids", "sources", "data_versions",
                "continuous_futures"):
        if key in data_identity and data_identity[key] is not None:
            found[key] = data_identity[key]
    return found


def render_bar_evaluation_page(db_path: Path) -> None:
    """Render the read-only Streamlit dashboard for one results DuckDB."""
    _load_streamlit()
    st.set_page_config(page_title="Bar Evaluation Results (Read-Only)", layout="wide")
    st.title("Bar Evaluation Results (Read-Only)")
    st.caption(f"Database (read-only): `{db_path}`")

    options = get_bar_evaluation_filter_options(db_path)

    st.sidebar.header("Filters")
    mode_choice = st.sidebar.selectbox(
        "Analysis Mode",
        ["All", *options["analysis_modes"]],
    )
    freq_choice = st.sidebar.selectbox(
        "Frequency",
        ["All", *options["frequencies"]],
    )
    cal_choice = st.sidebar.selectbox(
        "Calendar ID",
        ["All", *options["calendar_ids"]],
    )
    slot_choice = st.sidebar.selectbox("Slot ID", [None, *options["slot_ids"]], format_func=lambda slot: slot or "All")

    runs = query_bar_evaluation_runs(
        db_path,
        analysis_mode=None if mode_choice == "All" else mode_choice,
        frequency=None if freq_choice == "All" else freq_choice,
        calendar_id=None if cal_choice == "All" else cal_choice,
        slot_id=slot_choice,
    )

    st.subheader(f"Evaluation Runs ({len(runs)})")
    if not runs:
        st.info("No evaluation runs match the current filters.")
        return

    run_summary_rows = [
        {
            "identity_hash": r["identity_hash"],
            "analysis_mode": r["analysis_mode"],
            "frequency": r["frequency"],
            "view": r.get("view"),
            "boundary_policy": r.get("boundary_policy"),
            "bucket_strategy": r.get("bucket_strategy"),
            "calendar_id": r.get("calendar_id"),
            "slot_id": r.get("slot_id"),
            "code_version": r.get("code_version"),
            "result_schema_version": r["result_schema_version"],
            "has_split": r.get("has_split"),
            "run_id": r.get("run_id"),
            "ingested_at_utc": r.get("ingested_at_utc"),
        }
        for r in runs
    ]
    st.dataframe(pd.DataFrame(run_summary_rows), use_container_width=True)

    run_by_hash = {r["identity_hash"]: r for r in runs}
    selected_hash = st.selectbox(
        "Select Evaluation Run",
        options=list(run_by_hash.keys()),
        format_func=lambda h: _format_run_label(run_by_hash[h]),
    )
    detail = query_bar_evaluation_detail(db_path, selected_hash)
    run = detail["run"]

    st.subheader("Identity & Provenance")
    col1, col2 = st.columns(2)
    with col1:
        st.json(
            {
                "identity_hash": run["identity_hash"],
                "analysis_mode": run["analysis_mode"],
                "frequency": run["frequency"],
                "return_unit": run["return_unit"],
                "view": run.get("view"),
                "boundary_policy": run.get("boundary_policy"),
                "bucket_strategy": run.get("bucket_strategy"),
                "calendar_id": run.get("calendar_id"),
                "calendar_hash": run["calendar_hash"],
                "slot_id": run.get("slot_id"),
                "slot_config_hash": run.get("slot_config_hash"),
                "config_hash": run["config_hash"],
                "code_version": run["code_version"],
                "code_version_source": run.get("code_version_source"),
                "result_schema_version": run["result_schema_version"],
                "has_split": run.get("has_split"),
                "run_id": run.get("run_id"),
                "ingested_at_utc": run.get("ingested_at_utc"),
            }
        )
    with col2:
        data_identity = run.get("data_identity") or {}
        st.json(
            {
                "data_identity": data_identity,
                "source_and_build": _extract_build_id_summary(data_identity),
                "input_file_hashes": run.get("input_file_hashes"),
                "exclusion_summary": run.get("exclusion_summary"),
                "artifacts": {
                    "summary": {"path": run["summary_path"], "sha256": run["summary_sha256"]},
                    "per_bar_labels": {
                        "path": run["per_bar_labels_path"],
                        "sha256": run["per_bar_labels_sha256"],
                    },
                    "ic_by_horizon": {
                        "path": run["ic_by_horizon_path"],
                        "sha256": run["ic_by_horizon_sha256"],
                    },
                    "quantile_stats": {
                        "path": run["quantile_stats_path"],
                        "sha256": run["quantile_stats_sha256"],
                    },
                },
            }
        )

    if run.get("trial_ledger_json") is not None:
        st.subheader("Trial Family and Significance Gates")
        st.json(run["trial_ledger_json"])

    if detail["split_metrics"]:
        st.subheader("Split Metrics")
        columns = [
            "frequency", "instrument_id", "horizon_bars", "ic", "ic_reason", "n_observations",
            "n_dependence_adjusted", "split_purged_count", "interval_method", "interval_lower",
            "interval_upper", "calibration_status", "descriptive_stats_usable",
            "significance_interpretation_usable", "p_value", "adjusted_p_value",
            "inference_adjusted_p_value", "significant",
        ]
        for split, rows in pd.DataFrame(detail["split_metrics"]).groupby("split", sort=False):
            st.caption(split)
            st.dataframe(rows[columns], use_container_width=True)
    elif run.get("calibration_summary_json"):
        st.subheader("Interval Usability")
        st.json(run["calibration_summary_json"])

    st.subheader("IC by Horizon")
    ic_df = pd.DataFrame(detail["ic_by_horizon"])
    if ic_df.empty:
        st.info("No IC rows.")
    else:
        st.dataframe(ic_df.dropna(axis=1, how="all"), use_container_width=True)

    st.subheader("Quantile Stats")
    q_df = pd.DataFrame(detail["quantile_stats"])
    if q_df.empty:
        st.info("No quantile rows.")
    else:
        st.dataframe(q_df.dropna(axis=1, how="all"), use_container_width=True)

    if detail["cross_sections"]:
        st.subheader("Cross Sections")
        cs_df = pd.DataFrame(detail["cross_sections"])
        st.dataframe(cs_df, use_container_width=True)


def main(argv: Sequence[str] | None = None) -> None:
    _load_streamlit()
    try:
        db_path = resolve_results_db_path(argv=argv)
        render_bar_evaluation_page(db_path)
    except Exception as exc:
        st.error(f"Unable to open Bar evaluation results DB (read-only): {exc}")


if __name__ == "__main__":
    main()
