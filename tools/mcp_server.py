"""Local stdio tools for offline factor research."""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from typing import Any

INSTALL = "python3 -m pip install 'jq-factor-lab[mcp]'"


def _run(command: str, *flags: str, **options: str | bool | int | None) -> dict[str, Any]:
    # A separate process also isolates logging from native libraries and trusted factor modules.
    argv = [sys.executable, "-B", "-m", "tools.demo_cli", command, *flags]
    for name, value in options.items():
        option = "--" + name.replace("_", "-")
        if value is True:
            argv.append(option)
        elif value is not None and value is not False:
            argv.append(f"{option}={value}")
    try:
        process = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, check=False)
        result = json.loads(process.stdout)
        if isinstance(result, dict) and result.get("exit_code") == process.returncode:
            return result
    except (OSError, ValueError):
        pass
    return {"ok": False, "exit_code": 1, "error": {
        "code": "cli_execution_failed",
        "message": f"{command} did not return a valid CLI response.",
        "fix": f"Run jq-factor-lab {command} directly and inspect stderr.",
        "retryable": False,
    }}


def run_daily_demo(output_dir: str | None = None, overwrite: bool = False,
                   dry_run: bool = False, compact: bool = False) -> dict[str, Any]:
    """Evaluate fictional daily prices. Use dry_run to check the output plan."""
    return _run("daily-demo", output_dir=output_dir, overwrite=overwrite, dry_run=dry_run, compact=compact)


def run_bar_demo(output_dir: str | None = None, overwrite: bool = False,
                 dry_run: bool = False, compact: bool = False) -> dict[str, Any]:
    """Evaluate fictional minute Bars and write a local results database."""
    return _run("bar-demo", output_dir=output_dir, overwrite=overwrite, dry_run=dry_run, compact=compact)


def run_single_factor_analysis(config_path: str, output_dir: str | None = None,
                               overwrite: bool = False, dry_run: bool = False,
                               compact: bool = False) -> dict[str, Any]:
    """Evaluate local daily prices. Execution imports Python code from the trusted local config."""
    return _run("single-factor-analysis", config=config_path, output_dir=output_dir,
                overwrite=overwrite, dry_run=dry_run, compact=compact)


def run_time_series_analysis(config_path: str, output_root: str | None = None, results_db: str | None = None,
                             diagnostics_report: bool = True, overwrite: bool = False,
                             dry_run: bool = False) -> dict[str, Any]:
    """Evaluate local Bar data. An explicit results_db requests ingestion into that database."""
    diagnostics = "--diagnostics-report" if diagnostics_report else "--no-diagnostics-report"
    return _run("time-series-analysis", diagnostics, config=config_path, output_root=output_root,
                results_db=results_db, overwrite=overwrite, dry_run=dry_run)


def read_bar_evaluations(results_db: str, factor_name: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Read up to limit latest runs from a local evaluation database, optionally by recorded factor ID."""
    return _run("read-bar-evaluations", results_db=results_db, factor_name=factor_name, limit=limit)


def run_factor_synthesis_demo(method: str = "weighted_sum", output_dir: str | None = None,
                              overwrite: bool = False, dry_run: bool = False,
                              compact: bool = False) -> dict[str, Any]:
    """Combine fictional features using the installed synthesis pipeline."""
    return _run("factor-synthesis-demo", method=method, output_dir=output_dir, overwrite=overwrite,
                dry_run=dry_run, compact=compact)


def _register_tools(server: Any, annotate: Callable[..., Any]) -> None:
    for tool in (run_daily_demo, run_bar_demo, run_single_factor_analysis, run_time_series_analysis,
                 read_bar_evaluations, run_factor_synthesis_demo):
        read_only = tool is read_bar_evaluations
        server.add_tool(tool, annotations=annotate(readOnlyHint=read_only, destructiveHint=not read_only,
                                                   openWorldHint=False))


def main() -> int:
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ImportError:
        print(f"The jq-factor-lab MCP server requires the [mcp] extra. Run {INSTALL}", file=sys.stderr)
        return 1
    server = FastMCP("jq-factor-lab")
    _register_tools(server, ToolAnnotations)
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
