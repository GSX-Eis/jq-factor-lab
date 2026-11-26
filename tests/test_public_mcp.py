import asyncio
import builtins
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from examples.generate_data import make_prices
from examples.synthetic_bars import write_demo_bars
from test_public_entry_harness_e2e import _prepare_external_workspace
from tools import mcp_server


ROOT = Path(__file__).resolve().parents[1]
DEMO_OPTIONS = {"output_dir": None, "overwrite": False, "dry_run": False, "compact": False}
REQUIRED = inspect.Parameter.empty
SCHEMAS = {
    "run_daily_demo": DEMO_OPTIONS,
    "run_bar_demo": DEMO_OPTIONS,
    "run_single_factor_analysis": {"config_path": REQUIRED, **DEMO_OPTIONS},
    "run_time_series_analysis": {
        "config_path": REQUIRED, "output_root": None, "results_db": None,
        "diagnostics_report": True, "overwrite": False, "dry_run": False,
    },
    "read_bar_evaluations": {"results_db": REQUIRED, "factor_name": None, "limit": 20},
    "run_factor_synthesis_demo": {"method": "weighted_sum", **DEMO_OPTIONS},
}


def test_tool_registration_and_signatures():
    server = Mock()
    mcp_server._register_tools(server, dict)
    registrations = {call.args[0].__name__: call for call in server.add_tool.call_args_list}
    assert set(registrations) == set(SCHEMAS)
    assert server.add_tool.call_count == len(SCHEMAS)
    for name, expected in SCHEMAS.items():
        call = registrations[name]
        params = inspect.signature(call.args[0]).parameters
        assert list(params) == list(expected)
        assert {key: param.default for key, param in params.items()} == expected
        assert all(param.annotation is not inspect.Parameter.empty for param in params.values())
        assert call.kwargs["annotations"]["readOnlyHint"] is (name == "read_bar_evaluations")


def test_missing_sdk_produces_install_hint(monkeypatch, capsys):
    original = builtins.__import__

    def import_without_mcp(name, *args, **kwargs):
        if name.split(".")[0] == "mcp":
            raise ModuleNotFoundError("mcp unavailable", name="mcp")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_mcp)
    assert mcp_server.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "python3 -m pip install 'jq-factor-lab[mcp]'" in output.err
    assert "Traceback" not in output.err


def test_cli_tool_dry_run_and_invalid_limit_are_structured(tmp_path, monkeypatch, capfd):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    result = mcp_server.run_daily_demo(output_dir="out", dry_run=True, compact=True)
    assert result["dry_run"] and result["ok"] and result["exit_code"] == 0
    assert result["plan"]["output_dir"] == str(tmp_path / "out")
    assert not list(tmp_path.iterdir())
    spec = write_demo_bars(Path("bars.csv"))
    spec["diagnostics_report"] = True
    Path("bars.json").write_text(json.dumps(spec))
    bar = mcp_server.run_time_series_analysis("bars.json", "harness", "result.duckdb",
                                             diagnostics_report=False, dry_run=True)
    assert bar["ok"] and bar["plan"]["diagnostics_report"] is False
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bars.csv", "bars.json"]

    failure = mcp_server.read_bar_evaluations("missing.duckdb", limit=0)
    assert failure["ok"] is False and failure["exit_code"] == 2
    assert failure["error"]["fix"]
    assert capfd.readouterr().out == ""


def test_tool_call_contains_python_and_native_stdout(tmp_path, monkeypatch, capfd):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    make_prices(sessions=80).to_csv("prices.csv", index=False)
    Path("noisy.py").write_text(
        "import ctypes\nimport os\nfrom examples.momentum import DemoMomentum\n"
        "print('python stdout log')\nos.write(1, b'native stdout log\\n')\n"
        "ctypes.CDLL(None).puts(b'buffered native log')\n"
        "class NoisyMomentum(DemoMomentum):\n    pass\n"
    )
    Path("config.json").write_text(json.dumps({
        "input_csv": "prices.csv", "factor_file": "noisy.py", "factor_class": "NoisyMomentum",
        "calendar_file": str(ROOT / "config/demo_calendar.json"), "periods": [1, 5],
    }))
    result = mcp_server.run_single_factor_analysis("config.json", output_dir="out")
    assert result["ok"] and result["factor"] == "NoisyMomentum"
    output = capfd.readouterr()
    assert output.out == ""
    assert "buffered native log" in output.err
    assert "python stdout log" in output.err and "native stdout log" in output.err


def test_mcp_external_lookahead_rejection_remedies_factor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not tmp_path.resolve().is_relative_to(ROOT)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    make_prices(sessions=80).to_csv("prices.csv", index=False)
    Path("lookahead.py").write_text(
        "from examples.momentum import DemoMomentum\n"
        "class LookaheadMomentum(DemoMomentum):\n"
        "    CAUSAL_VECTORIZED = True\n"
        "    def calculate_factor(self, data):\n"
        '        prices = data.pivot(index="date", columns="code", values="close").sort_index()\n'
        "        return prices.shift(-2) / prices.shift(-1) - 1\n",
        encoding="utf-8",
    )
    Path("daily.json").write_text(json.dumps({
        "input_csv": "prices.csv", "factor_file": "lookahead.py", "factor_class": "LookaheadMomentum",
        "calendar_file": str(ROOT / "config/demo_calendar.json"), "periods": [1, 5],
    }), encoding="utf-8")

    result = mcp_server.run_single_factor_analysis("daily.json", output_dir="rejected")
    assert result["ok"] is False
    assert result["exit_code"] == 1
    assert result["error"]["code"] == "causal_fast_path_gate_failed"
    fix = result["error"]["fix"].lower()
    assert "lookahead" in fix
    assert "disable" not in fix
    assert "causal_vectorized" not in fix
    assert not Path("rejected").exists()


def test_mcp_external_time_series_repeats_and_reads_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller = _prepare_external_workspace(tmp_path)
    monkeypatch.chdir(caller)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    config.update(slot="slot.json", factor_file="my_probe_factor.py")
    Path("config.json").write_text(json.dumps(config), encoding="utf-8")
    database = "results/evaluation.duckdb"

    first = mcp_server.run_time_series_analysis("config.json", output_root="first", results_db=database)
    second = mcp_server.run_time_series_analysis("config.json", output_root="second", results_db=database)
    assert first["ok"] is True, first
    assert second["ok"] is True, second
    assert first["metrics"]["causal_gate"]["status"] == "passed"
    inserted = first["metrics"]["results_db"]
    repeated = second["metrics"]["results_db"]
    assert inserted["status"] == "inserted"
    assert repeated["status"] == "noop"
    assert repeated["identity_hash"] == inserted["identity_hash"]

    read = mcp_server.read_bar_evaluations(database, factor_name="ProbeRollingFactor")
    assert read["ok"] is True, read
    assert read["metrics"]["run_count"] == 1
    assert read["metrics"]["runs"][0]["identity_hash"] == inserted["identity_hash"]


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="wire test requires [mcp]")
def test_fastmcp_stdio_protocol_and_structured_output(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def check():
        server = StdioServerParameters(command=sys.executable, args=["-B", "-m", "tools.mcp_server"],
                                       cwd=str(tmp_path), env={**os.environ, "PYTHONPATH": str(ROOT)})
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                assert set(tools) == set(SCHEMAS)
                for name, expected in SCHEMAS.items():
                    schema = tools[name].inputSchema
                    assert set(schema["properties"]) == set(expected)
                    assert set(schema.get("required", [])) == {k for k, v in expected.items() if v is REQUIRED}
                result = await session.call_tool("run_bar_demo", {"dry_run": True})
                assert not result.isError and result.structuredContent["plan"]["data"] == "synthetic"
                assert not list(tmp_path.iterdir())

    asyncio.run(check())
