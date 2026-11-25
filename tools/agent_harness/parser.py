"""Define the agent Harness command-line arguments."""
from __future__ import annotations

import argparse
from pathlib import Path

from tools.agent_harness import runner
from tools.agent_harness._cli_compat import _cli_bindings


def build_parser(*, public_bar_cli: bool = False, parser_class: type[argparse.ArgumentParser] = argparse.ArgumentParser) -> argparse.ArgumentParser:
    cli = _cli_bindings(__name__, runner)
    parser = parser_class(
        prog="jq-factor-lab" if public_bar_cli else None,
        description="Agent-native harness wrappers for quantation_factor_database"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    gc = subparsers.add_parser("outputs-gc", help="Delegate owned tmp/cache cleanup to the public Task Manager GC")
    gc.add_argument("--task-manager", default="task-manager", help="Installed Task Manager CLI executable")
    gc.add_argument("--task-root", default=str(cli.REPO_ROOT), help="Owner root to collect")
    gc.add_argument("--scope", choices=("temporary", "cache", "local"), default="local")
    mode = gc.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    gc.add_argument("--completed-hours", type=float, default=24)
    gc.add_argument("--failed-days", type=float, default=7)
    gc.add_argument("--cache-days", type=float, default=30)


    def add_common(subparser: argparse.ArgumentParser, *, subprocess_options: bool = True) -> None:
        subparser.add_argument(
            "--run-id", default="", help="Stable run identifier for output colocation"
        )
        subparser.add_argument(
            "--output-root", default="", help="Override output root for summaries"
        )
        subparser.add_argument(
            "--json-out",
            default="",
            help="Optional path to write the final JSON payload",
        )
        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help="Plan the command and emit structured JSON only",
        )
        if not subprocess_options:
            return
        subparser.add_argument(
            "--fake-result",
            choices=("success", "failure", "degraded", "alignment_failed"),
            default="",
            help="Bypass heavy execution and emit a deterministic fake result",
        )
        subparser.add_argument(
            "--timeout-sec",
            type=int,
            default=1800,
            help="Subprocess timeout in seconds",
        )

    market_db = subparsers.add_parser("market-db", help="Manage a local market database profile")
    market_actions = market_db.add_subparsers(dest="market_db_action", required=True)
    for name in ("init", "adopt", "show", "migrate-profile"):
        market_action = market_actions.add_parser(name)
        add_common(market_action, subprocess_options=False)
        if name == "show":
            selection = market_action.add_mutually_exclusive_group(required=True)
            selection.add_argument("--db", help="Local DuckDB file")
            selection.add_argument("--market-db", help="db_id from QUANT_DATA_BINDINGS")
        else:
            market_action.add_argument("--db", required=True, help="Local DuckDB file")
            market_action.add_argument("--profile", required=True, help="Market database YAML or JSON declaration")
        if name == "adopt":
            market_action.add_argument("--sidecar", action="store_true", required=True)

    for name in ('build-bars', 'audit-bars', 'import-bars', 'build-roll-schedule'):
        storage = subparsers.add_parser(name, help='Explicit local Bar storage operation')
        add_common(storage)
        storage.add_argument('--db', required=True, help='Explicit owned DuckDB destination')
        storage.add_argument('--bucket-strategy', choices=('segment_anchored', 'session_clock', 'session'))
        if name != 'import-bars':
            storage.add_argument('--instruments', required=name == 'build-bars', help='Comma-separated concrete IDs')
            storage.add_argument('--start', required=name in ('build-bars', 'build-roll-schedule'))
            storage.add_argument('--end', required=name in ('build-bars', 'build-roll-schedule'))
        else:
            storage.set_defaults(instruments=None)
            source = storage.add_mutually_exclusive_group(required=True)
            source.add_argument('--bars', help='Canonical UTC Bar CSV (requires --metadata)')
            source.add_argument('--manifest', help='Declarative slot YAML or JSON')
            storage.add_argument('--metadata', help='Instrument metadata JSON array for --bars')
            storage.add_argument('--derive', default='', help='Optional comma-separated target frequencies')
        if name == 'build-bars':
            storage.add_argument('--source', required=True)
            storage.add_argument('--targets', required=True)
            storage.add_argument('--check-only', action='store_true')
        if name == 'build-roll-schedule':
            storage.add_argument('--product-roots', required=True, help='Comma-separated futures product roots')
            storage.add_argument('--rule', choices=('max_oi', 'max_volume', 'days_before_expiry'), default='max_oi')
            storage.add_argument('--confirm-days', type=int, default=1)
            storage.add_argument('--forward-only', action=argparse.BooleanOptionalAction, default=True)

    write_fv = subparsers.add_parser('write-factor-values', help='Append market factor values to Scheme B store')
    add_common(write_fv)
    write_fv.add_argument('--factor-values-store', required=True, help='Explicit Scheme B store directory')
    write_fv.add_argument('--input', required=True, help='Factor values CSV or Parquet file')
    write_fv.add_argument('--factor-id', default=None)
    write_fv.add_argument('--frequency', default=None)
    write_fv.add_argument('--session-view', choices=('ETH', 'RTH'), default='ETH')
    write_fv.add_argument('--analysis-mode', choices=('time_series', 'cross_sectional'), default=None)
    write_fv.add_argument('--bucket-strategy', choices=('segment_anchored', 'session_clock', 'session'), default=None)
    write_fv.add_argument('--calendar', default=None)
    write_fv.add_argument('--calendar-start', default=None)
    write_fv.add_argument('--calendar-end', default=None)
    write_fv.add_argument('--rth-reference-calendar', default=None)
    write_fv.add_argument('--slot-config', help='SlotConfig YAML or JSON with its calendar declaration')
    write_fv.add_argument('--slot-db', help='Read-only DuckDB source for a native futures slot calendar')
    write_fv.add_argument('--slot-id')
    write_fv.add_argument('--slot-config-hash')
    write_fv.add_argument('--source-build-id', default=None)
    write_fv.add_argument('--slot-data-fingerprint', help='Source version string or JSON object')
    write_fv.add_argument('--data-version', default=None)
    write_fv.add_argument('--params-json', default='{}')
    write_fv.add_argument('--factor-class', default='external_cli')
    write_fv.add_argument('--factor-source-sha256', help='Stable SHA-256 of the factor definition for slot imports')
    policy = write_fv.add_mutually_exclusive_group()
    policy.add_argument('--adjust', choices=('none', 'ratio_back_adjust', 'diff_back_adjust'), default=None)
    policy.add_argument('--adjustment-policy')

    read_fv = subparsers.add_parser('read-factor-values', help='Read-only query over committed Scheme B factor store')
    add_common(read_fv)
    read_fv.add_argument('--factor-values-store', required=True, help='Explicit Scheme B store directory')
    read_fv.add_argument('--session-view', choices=('ETH', 'RTH'), default=None)
    read_fv.add_argument('--instruments', default=None, help='Comma-separated instrument IDs')
    read_fv.add_argument('--start-utc', default=None)
    read_fv.add_argument('--end-utc', default=None)
    read_fv.add_argument('--session-start', default=None)
    read_fv.add_argument('--session-end', default=None)
    read_fv.add_argument('--list-identities', action='store_true')
    read_fv.add_argument('--out', default=None, help='Optional CSV or Parquet export path')
    read_fv.add_argument('--limit', type=int, default=0, help='Optional preview row count in JSON metrics')

    verify_fv = subparsers.add_parser('verify-factor-values', help='Read-only integrity check on Scheme B factor store')
    add_common(verify_fv)
    verify_fv.add_argument('--factor-values-store', required=True, help='Explicit Scheme B store directory')
    for command in (read_fv, verify_fv):
        command.add_argument('--factor-id')
        command.add_argument('--factor-identity-hash')
        command.add_argument('--frequency', help='Stored output frequency')
        command.add_argument('--slot-id')
    for command in (write_fv, read_fv, verify_fv):
        command.add_argument('--json', action='store_true', help='Emit JSON to stdout (default)')

    export = subparsers.add_parser("export-strategy", help="Export an offline deployment candidate")
    verify = subparsers.add_parser("verify-target-alignment", help="Verify an offline deployment candidate")
    for deploy in (export, verify):
        deploy.add_argument("--bundle-dir", required=True)
        deploy.add_argument("--target", required=True, choices=("jq_cloud", "signal_export"))
    export.add_argument("--output-dir", required=True)
    export.add_argument("--factor-file", action="append", help="JoinQuant factor source file, repeatable")
    export.add_argument("--requirements", help="JoinQuant factor requirements JSON")
    export.add_argument("--target-positions", help="Signal export target positions in Parquet or JSONL")
    export.add_argument("--current-positions", help="Signal export current weights as an instrument-to-weight JSON object")
    verify.add_argument("--artifact-dir", required=True)
    verify.add_argument("--reference-positions", help="JoinQuant weights JSON or signal export positions in Parquet or JSONL")
    verify.add_argument("--reference-factors", help="JoinQuant factor values in Parquet, indexed by stock code")
    verify.add_argument("--trusted-manifest-hash", help="Manifest hash recorded separately from a trusted JoinQuant export")
    verify.add_argument("--replay-inputs", help="JoinQuant offline replay JSON with stocks, asof and bars_by_frequency")

    alignment = subparsers.add_parser(
        "alignment-run",
        help="Build, replay and compare factor or frozen-strategy alignment",
    )
    add_common(alignment)
    target = alignment.add_mutually_exclusive_group(required=True)
    target.add_argument("--factor")
    target.add_argument("--strategy-config")
    alignment.add_argument("--factor-class")
    alignment.add_argument("--output-dir", required=True)
    alignment.add_argument("--real-cloud", action="store_true")
    alignment.add_argument("--resume", action="store_true")
    alignment.add_argument("--iteration", type=int, default=1)
    alignment.add_argument("--max-iterations", type=int, default=1)
    alignment.add_argument("--result-file")
    alignment.add_argument("--work-log")

    analyze = subparsers.add_parser(
        "single-factor-analysis", help="Wrap run_analysis_main.py"
    )
    add_common(analyze)
    analyze.add_argument(
        "--config",
        default="",
        help="Optional JSON config path passed to run_analysis_main.py",
    )
    analyze.add_argument("--factor", default="", help="Optional factor module path")
    analyze.add_argument(
        "--use-dynamic", action="store_true", help="Forward --use-dynamic"
    )
    analyze.add_argument(
        "--materialize-only",
        action="store_true",
        help="Forward --materialize-only",
    )
    analyze.add_argument(
        "--mining-only",
        action="store_true",
        help="Forward --mining-only",
    )
    analyze.add_argument(
        "--pool", action="append", default=[], help="Append a dynamic pool name"
    )
    analyze.add_argument(
        "--causal-gate-policy",
        choices=("reject", "fallback"),
        default=None,
        help="Policy when CAUSAL_VECTORIZED fast-path verification fails",
    )

    bar_analysis = subparsers.add_parser("time-series-analysis", help="Evaluate CSV or MarketSlot factors in an explicit mode")
    add_common(bar_analysis)
    bar_analysis.add_argument("--config", default="", help="Bar analysis JSON config")
    bar_analysis.add_argument("--diagnostics-report", action=argparse.BooleanOptionalAction,
                              help="Publish time-series diagnostics in JSON, CSV and Markdown")
    bar_analysis.add_argument("--factor", default="", help="Optional factor module path for Slot evaluation")
    bar_analysis.add_argument("--factor-class", default=None, help="Optional factor class name in --factor")
    bar_analysis.add_argument(
        "--causal-gate-policy",
        choices=("reject", "fallback"),
        default=None,
        help="Policy when CAUSAL_VECTORIZED fast-path verification fails",
    )

    verify_causal = subparsers.add_parser(
        "verify-factor-causality",
        help="Run automatic causal verification gate for fast-path Slot/FactorBase factors",
    )
    add_common(verify_causal)
    verify_causal.add_argument("--factor", default="", help="Factor module .py path")
    verify_causal.add_argument("--factor-class", default=None, help="Optional factor class name")
    verify_causal.add_argument("--slot", "--slot-config", dest="slot", default=None, help="SlotConfig YAML/JSON path or jq_ashare_local")
    verify_causal.add_argument("--slot-id", default=None, help="Configured MarketSlot identity")
    verify_causal.add_argument("--slot-db", default=None, help="Read-only DuckDB path for native slots")
    verify_causal.add_argument("--market-db", default=None, help="Database ID in QUANT_DATA_BINDINGS.market_dbs")
    verify_causal.add_argument("--start-date", default=None, help="Evaluation start date YYYY-MM-DD")
    verify_causal.add_argument("--end-date", default=None, help="Evaluation end date YYYY-MM-DD")
    verify_causal.add_argument("--frequency", default=None, help="Slot frequency (e.g. 60min, 1d)")
    verify_causal.add_argument("--analysis-mode", choices=("time_series", "cross_sectional"), default="time_series")
    verify_causal.add_argument("--view", choices=("ETH", "RTH"), default="ETH")
    verify_causal.add_argument("--instruments", default=None, help="Comma-separated instrument IDs")
    verify_causal.add_argument("--slot-execution-mode", choices=("auto", "vectorized", "incremental", "pit_replay"), default=None)
    verify_causal.add_argument("--n-truncations", type=int, default=3)
    verify_causal.add_argument("--n-pit-samples", type=int, default=5)
    verify_causal.add_argument("--seed", type=int, default=42)
    verify_causal.add_argument("--on-failure", choices=("reject", "fallback"), default="reject")
    for subparser in (bar_analysis, analyze):
        subparser.add_argument("--slot", "--slot-config", dest="slot", help="SlotConfig YAML/JSON path or jq_ashare_local")
        subparser.add_argument("--market-db", help="Database ID in QUANT_DATA_BINDINGS.market_dbs")
        subparser.add_argument("--slot-id", help="Configured MarketSlot identity")
        subparser.add_argument("--bars-csv")
        subparser.add_argument("--factors-csv")
        subparser.add_argument("--analysis-mode", help="time_series or cross_sectional (required for Bars)")
        subparser.add_argument("--view", help="ETH or RTH")
        subparser.add_argument("--calendar", help="Session calendar name, e.g. NQ or XSHG")
        subparser.add_argument("--horizons", help="Comma-separated positive Bar counts")
        subparser.add_argument("--entry-offset-bars", type=int)
        subparser.add_argument("--n-quantiles", type=int)
        subparser.add_argument("--output-dir", help="Evaluation artifact directory")
        subparser.add_argument("--results-db", default=None, help="Explicit DuckDB results database path")

    read_bar_eval = subparsers.add_parser(
        "read-bar-evaluations",
        help="Read-only query over ingested Bar evaluation results",
    )
    add_common(read_bar_eval)
    read_bar_eval.add_argument("--results-db", required=True, help="Explicit DuckDB results database path")
    read_bar_eval.add_argument("--identity-hash", default=None)
    read_bar_eval.add_argument("--analysis-mode", choices=("time_series", "cross_sectional"), default=None)
    read_bar_eval.add_argument("--frequency", default=None)
    read_bar_eval.add_argument("--calendar-id", default=None)
    read_bar_eval.add_argument("--slot-id", default=None)
    read_bar_eval.add_argument("--start-utc", default=None, help="Inclusive UTC start timestamp or YYYY-MM-DD")
    read_bar_eval.add_argument("--end-utc", default=None, help="Inclusive UTC end timestamp or YYYY-MM-DD")
    read_bar_eval.add_argument("--split", choices=("full", "train", "oos"), help="Filter split metrics and matching runs")
    read_bar_eval.add_argument("--calibration-status", choices=("uncalibrated", "calibrated_pass", "calibrated_fail"))
    read_bar_eval.add_argument("--descriptive-stats-usable", action=argparse.BooleanOptionalAction)
    read_bar_eval.add_argument("--significance-interpretation-usable", action=argparse.BooleanOptionalAction)
    read_bar_eval.add_argument("--has-trial-ledger", action=argparse.BooleanOptionalAction)
    read_bar_eval.add_argument("--factor-name", help="Recorded factor ID")
    read_bar_eval.add_argument("--limit", type=int, help="Maximum latest matching runs")
    read_bar_eval.add_argument("--offset", type=int, default=0, help="Number of matching runs to skip")
    if public_bar_cli:
        output_root = str(Path("outputs/agent_harness").resolve())
        for command in (bar_analysis, read_bar_eval):
            command.set_defaults(output_root=output_root)
        bar_analysis.add_argument("--overwrite", action="store_true",
                                  help="Allow updating the evaluation files in an existing output directory")

    symbolic = subparsers.add_parser(
        "symbolic-factor-mining", help="Wrap run_symbolic_factor_mining.py"
    )
    add_common(symbolic)
    symbolic.add_argument(
        "--config",
        default=str(cli.REPO_ROOT / "config" / "symbolic_mining" / "default.json"),
    )

    quanta_alpha = subparsers.add_parser(
        "quanta-alpha-realize",
        help="Compile and fast-screen one three-expression QuantaAlpha proposal",
    )
    add_common(quanta_alpha)
    quanta_alpha.add_argument("--input", required=True, help="Frozen proposal JSON")
    quanta_alpha.add_argument(
        "--output", default="", help="Frozen realization JSON output path"
    )
    symbolic.add_argument(
        "--novelty-context", default="", help="Run-local symbolic novelty context JSON"
    )
    symbolic.add_argument(
        "--marginal-context",
        default="",
        help="Run-local symbolic marginal-aware search context JSON",
    )
    symbolic.add_argument(
        "--skip-validation", action="store_true", help="Forward --skip-validation"
    )
    symbolic.add_argument(
        "--materialize-candidate-rank",
        type=int,
        default=0,
        help="Materialize and validate a rerank candidate rank from an existing symbolic run",
    )
    symbolic.add_argument(
        "--factor-path",
        default="",
        help="Override final factor path for candidate materialization",
    )

    synthesis = subparsers.add_parser(
        "factor-synthesis", help="Wrap run_factor_synthesis.py"
    )
    add_common(synthesis)
    synthesis.add_argument("--market-db", help="Database ID in QUANT_DATA_BINDINGS.market_dbs")
    synthesis.add_argument(
        "--config", default=str(cli.REPO_ROOT / "config" / "factor_synthesis_config.json")
    )
    synthesis.add_argument(
        "--return-series-output-dir",
        default="",
        help="Override output directory for per-candidate return CSV artifacts",
    )
    synthesis.add_argument(
        "--slot-config",
        default="",
        help="Optional SlotConfig YAML/JSON path or jq_ashare_local",
    )
    synthesis.add_argument(
        "--slot-id",
        default="",
        help="Optional MarketSlot identity override",
    )
    synthesis.add_argument(
        "--factor-values-store",
        default="",
        help="Optional Scheme B MarketFactorValueStore root directory",
    )
    synthesis.add_argument(
        "--frequency",
        default="",
        help="Optional synthesis and backtest frequency (e.g. 1d, 5min, 1h)",
    )

    parent_screen = subparsers.add_parser(
        "parent-factor-screen",
        help="Rank eligible factors by residual RankIC against one exact parent",
    )
    add_common(parent_screen)
    parent_screen.add_argument("--job", required=True, help="Frozen parent screen job JSON")
    parent_screen.add_argument("--output", required=True, help="Evidence JSON output path")

    monitor = subparsers.add_parser(
        "monitor-refresh", help="Build a structured monitor readiness summary"
    )
    add_common(monitor)
    monitor.add_argument("--database-path", default=str(cli.DEFAULT_MONITOR_DB))

    return parser
