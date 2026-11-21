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
