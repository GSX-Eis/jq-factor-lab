"""Check the requested date range in the prepared dumper without running it."""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/alignment/prepare_cloud_dumper.py"


def prepare(
    directory: Path, source: str, target_date: str = "2023-01-03"
) -> tuple[subprocess.CompletedProcess[str], Path]:
    config = directory / "config"
    config.mkdir()
    (config / "factor_requirements.json").write_text('{"factors": {"DemoMomentum": {}}}')
    (config / "small_mkt_config.json").write_text("{}")
    source_path = directory / "source.py"
    source_path.write_text(source)
    output = directory / "prepared.py"
    command = [
        sys.executable,
        str(SCRIPT),
        "--factor", "DemoMomentum",
        "--source", str(source_path),
        "--output", str(output),
    ]
    if target_date:
        command.extend(["--target-date", target_date])
    process = subprocess.run(
        command, cwd=directory, capture_output=True, text=True, timeout=30
    )
    return process, output


@pytest.mark.parametrize("quote", ["'", '"'], ids=["single", "double"])
@pytest.mark.parametrize("in_main", [False, True], ids=["module", "main"])
def test_prepare_dumper_sets_both_dates(tmp_path: Path, quote: str, in_main: bool) -> None:
    dates = (
        f"start_date = {quote}2022-01-01{quote}  # start comment\n"
        f"end_date = {quote}2023-12-31{quote}\n"
    )
    source = "REQ = {}\nTARGET_FACTORS = []\n"
    source += "def main():\n" + textwrap.indent(dates, "    ") if in_main else dates
    result, output = prepare(tmp_path, source)

    assert result.returncode == 0, result.stderr
    tree = ast.parse(output.read_text())
    assert {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"start_date", "end_date"}
    } == {"start_date": "2023-01-03", "end_date": "2023-01-03"}
    assert "# start comment" in output.read_text()


def test_prepare_dumper_keeps_default_window_and_surrounding_code(tmp_path: Path) -> None:
    source = '标记 = 1; start_date = "2022-02-02"; end_date = "2023-02-03"  # dates\n'
    result, output = prepare(tmp_path, source, target_date="")

    assert result.returncode == 0, result.stderr
    assert output.read_text() == (
        "标记 = 1; start_date = '2022-01-01'; end_date = '2023-12-31'  # dates\n"
    )


@pytest.mark.parametrize(
    "source",
    [
        'start_date = "2022-01-01"\n# end_date = "2023-12-31"\n',
        'start_date = "2022-01-01"\nend_date = "2023-12-31"\nend_date = "2023-01-04"\n',
        'start_date = "2022-01-01"\nend_date = choose_end()\n',
        'start_date = "2022-01-01"\ndef other():\n    end_date = "2023-12-31"\n',
    ],
    ids=["missing", "duplicate", "dynamic", "different-scopes"],
)
def test_prepare_dumper_rejects_unconfirmed_dates(tmp_path: Path, source: str) -> None:
    result, output = prepare(tmp_path, source)

    assert result.returncode != 0
    assert "alignment_dumper_date_assignments_required" in result.stderr
    assert not output.exists()
