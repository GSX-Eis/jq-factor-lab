"""Publish Bar evaluation tables, with summary.json as the completion marker."""
from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ContractValidationError, ResultIdentity
from .evaluation_v2 import EVALUATION_V2_FILENAME
from .evaluator import EvaluationResult, RESULT_SCHEMA_VERSION, _mode, _prepare


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Mapping):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat().replace('+00:00', 'Z')
    if isinstance(value, Enum):
        return value.value
    return value


def _csv_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy()
    for name in table:
        if isinstance(table[name].dtype, pd.DatetimeTZDtype):
            table[name] = table[name].map(_json_value)
        elif name in ('exclusion_counts', 'nonzero_exclusion_counts'):
            table[name] = table[name].map(lambda v: json.dumps(_json_value(v), allow_nan=False)
                                         if isinstance(v, Mapping) else None)
        elif name == 'exclusion_reason':
            table[name] = table[name].map(lambda v: v.value if isinstance(v, Enum) else v)
    return table


DIAGNOSTICS_CSV = 'diagnostics_report.csv'
DIAGNOSTICS_MARKDOWN = 'diagnostics_report.md'


def write_evaluation_artifacts(evaluation_result: EvaluationResult, output_dir: str | Path, *,
                               result_identity: ResultIdentity,
                               config: Mapping[str, Any] | None = None,
                               code_version_source: str | None = None,
                               diagnostics_report: Mapping[str, Any] | None = None,
                               split_diagnostics_report: Mapping[str, Any] | None = None) -> dict[str, Path]:
    """Stage the evaluation files on the destination filesystem, publish summary last.

    Each rename is atomic. Readers must require summary.json; a failed retry
    removes that completion marker even if older CSV files remain for diagnosis.
    Label-only evaluation cannot reconstruct source/calendar provenance: supply
    metadata on the result, or use evaluate_bars to retain it automatically.
    IC inference metadata is available on the result; v1 persists it only
    through the opt-in diagnostics report when supplied.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / 'summary.json'
    summary_path.unlink(missing_ok=True)
    if not isinstance(evaluation_result, EvaluationResult):
        raise ContractValidationError('evaluation_result must be an EvaluationResult')
    if not isinstance(result_identity, ResultIdentity):
        raise ContractValidationError('result_identity must be a ResultIdentity')
    result = evaluation_result
    _mode(result.analysis_mode)
    _prepare(result.per_bar_labels, None)
    if result_identity.result_schema_version != RESULT_SCHEMA_VERSION:
        raise ContractValidationError(f'result_schema_version must be {RESULT_SCHEMA_VERSION}')
    calendar_hash = result.calendar_identity.get('calendar_hash')
    if calendar_hash is not None and calendar_hash != result_identity.calendar_hash:
        raise ContractValidationError('result_identity calendar_hash does not match evaluation')
    if config is not None and not isinstance(config, Mapping):
        raise ContractValidationError('config must be a mapping')
    if code_version_source is not None and code_version_source not in ('git', 'env', 'package', 'unknown'):
        raise ContractValidationError('Invalid code_version_source')
    payload = dict(
        result_schema_version=RESULT_SCHEMA_VERSION, analysis_mode=result.analysis_mode,
        config=dict(config) if config is not None else {}, data_identity=result.data_identity,
        calendar_identity=dict(calendar_id=None, calendar_hash=result_identity.calendar_hash,
                               view=None, boundary_policy=None) | result.calendar_identity,
        result_identity={name: getattr(result_identity, name)
                         for name in ResultIdentity.__dataclass_fields__}
                        | {'identity_hash': result_identity.identity_hash()},
        ic_by_horizon=result.ic_by_horizon.to_dict('records'),
        quantile_stats=result.quantile_stats.to_dict('records'),
        exclusion_summary=result.exclusion_summary,
        cross_sections=result.cross_sections.to_dict('records'))
    if code_version_source is not None:
        payload['result_identity']['code_version_source'] = code_version_source
    if diagnostics_report is not None:
        payload['diagnostics_report'] = diagnostics_report
    if split_diagnostics_report is not None and result.evaluation_v2 is None:
        raise ContractValidationError('split_diagnostics_report requires evaluation.v2')
    v2_json = None
    try:
        if result.evaluation_v2 is not None:
            v2 = dict(**asdict(result.evaluation_v2), analysis_mode=result.analysis_mode,
                      data_identity=payload['data_identity'], calendar_identity=payload['calendar_identity'],
                      result_identity=payload['result_identity'])
            if split_diagnostics_report is not None:
                v2['diagnostics_report'] = split_diagnostics_report
            v2_json = json.dumps(_json_value(v2), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        payload = _json_value(payload)
        summary = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    except (ValueError, TypeError) as exc:
        raise ContractValidationError(f'Evaluation metadata must be JSON serializable: {exc}') from exc
    tables = {'per_bar_labels': result.per_bar_labels, 'ic_by_horizon': result.ic_by_horizon,
              'quantile_stats': result.quantile_stats}
    paths = {name: destination / f'{name}.csv' for name in tables}
    if v2_json is not None:
        paths['evaluation_v2'] = destination / EVALUATION_V2_FILENAME
    if diagnostics_report is not None or split_diagnostics_report is not None:
        paths.update(diagnostics_report_csv=destination / DIAGNOSTICS_CSV,
                     diagnostics_report_markdown=destination / DIAGNOSTICS_MARKDOWN)
    paths['summary'] = summary_path
    # Publication staging is retained only inside this synchronous transaction.
    with TemporaryDirectory(prefix='.bar-evaluation-', dir=destination) as staging:
        staged = Path(staging)
        for name, table in tables.items():
            _csv_table(table).to_csv(staged / paths[name].name, index=False)
        if v2_json is not None:
            (staged / EVALUATION_V2_FILENAME).write_text(v2_json, encoding='utf-8')
        if 'diagnostics_report_csv' in paths:
            from .diagnostics_report import DIAGNOSTICS_CSV_COLUMNS, diagnostics_rows, render_diagnostics_markdown
            report = (_json_value(split_diagnostics_report) if split_diagnostics_report is not None
                      else payload['diagnostics_report'])
            columns = DIAGNOSTICS_CSV_COLUMNS
            if split_diagnostics_report is not None:
                columns = ('split', *columns, 'split_purged_count')
            rows = diagnostics_rows(report)
            with (staged / DIAGNOSTICS_CSV).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            (staged / DIAGNOSTICS_MARKDOWN).write_text(render_diagnostics_markdown(report, rows), encoding='utf-8')
        (staged / 'summary.json').write_text(summary, encoding='utf-8')
        try:
            for path in paths.values():
                os.replace(staged / path.name, path)
        except BaseException:
            summary_path.unlink(missing_ok=True)
            raise
    return paths
