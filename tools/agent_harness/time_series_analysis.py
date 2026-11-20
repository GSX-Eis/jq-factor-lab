"""CSV-only Bar evaluation worker; the CLI owns execution and completion markers."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


_ADVANCED_OPTIONS = ('split', 'bootstrap', 'hac_lag', 'direction', 'standardization',
                     'calibration_scenario', 'trial_selection')


def evaluation_v2_requested(config: dict[str, Any]) -> bool:
    return config.get('evaluation_v2') is True or any(config.get(key) is not None for key in _ADVANCED_OPTIONS)


class AnalysisPreconditionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('Bar analysis config must be a JSON object')
    block = payload.get('time_series_analysis', {})
    if not isinstance(block, dict):
        raise ValueError('time_series_analysis must be a JSON object')
    payload.pop('time_series_analysis', None)
    if 'calendar' in block:
        payload.pop('calendar_by_instrument', None)
    if 'calendar_by_instrument' in block:
        payload.pop('calendar', None)
    return {**payload, **block}


def evaluate(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    from backtest_and_analysis.time_series import (
        BarReturnLabeler, ResultIdentity, evaluate_bars, read_bar_csv, read_factor_csv,
        sha256_file, write_evaluation_artifacts,
    )
    from backtest_and_analysis.time_series.evaluator import RESULT_SCHEMA_VERSION
    from utils.session_calendar import SessionCalendar

    mode = config.get('analysis_mode')
    if mode is None:
        raise AnalysisPreconditionError('missing_analysis_mode', 'Explicit analysis_mode is required')
    if mode not in ('time_series', 'cross_sectional'):
        raise AnalysisPreconditionError('invalid_analysis_mode',
            'analysis_mode must be time_series or cross_sectional')
    diagnostics_requested = config.get('diagnostics_report', False)
    if not isinstance(diagnostics_requested, bool) or (diagnostics_requested and mode != 'time_series'):
        raise ValueError('diagnostics_report must be a boolean and requires time_series when enabled')
    paths = {}
    continuous = config.get('continuous_futures')
    if continuous is not None:
        from utils.continuous_bars import ADJUSTMENTS
        if not isinstance(continuous, dict) or continuous.get('adjust') not in ADJUSTMENTS:
            raise ValueError(f'continuous_futures requires explicit adjust: {ADJUSTMENTS}')
        if not isinstance(continuous.get('db'), str) or not continuous['db'].strip():
            raise ValueError('continuous_futures requires an explicit db path')
        if config.get('bars_csv') is not None:
            raise ValueError('Supply continuous_futures or bars_csv, not both')
        paths['continuous_db'] = Path(continuous['db']).expanduser().resolve()
    for name in ('bars_csv', 'factors_csv'):
        if name == 'bars_csv' and continuous is not None:
            continue
        value = config.get(name)
        if value is None and name == 'factors_csv':
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a nonempty CSV path')
        paths[name] = Path(value).expanduser().resolve()
    hashes = {str(path): sha256_file(path) for path in paths.values()}
    bars = read_bar_csv(paths['bars_csv']) if continuous is None else None
    factors = read_factor_csv(paths['factors_csv']) if 'factors_csv' in paths else None
    calendar_options = {name: config[name] for name in ('start_date', 'end_date') if name in config}
    for name, value in calendar_options.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a nonempty session date')

    def calendar(identifier: str) -> SessionCalendar:
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError('calendar must be a nonempty calendar name')
        try:
            return SessionCalendar(identifier, **calendar_options)
        except RuntimeError as exc:
            raise AnalysisPreconditionError('time_series_precondition_failed',
                f'Invalid calendar {identifier}: {exc}') from exc

    if continuous is not None:
        calendars = None  # Bound to concrete-contract metadata below.
    elif 'calendar_by_instrument' in config:
        mapping = config['calendar_by_instrument']
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('calendar_by_instrument must be a nonempty instrument-to-calendar object')
        calendars = {identifier: calendar(name) for identifier, name in mapping.items()}
    else:
        calendars = calendar(config.get('calendar'))
    from contextlib import ExitStack
    with ExitStack() as stack:
        source = None
        if continuous is not None:
            from utils.continuous_bars import ContinuousBarSource
            from backtest_and_analysis.time_series.factor_bridge import continuous_factor_input
            source = stack.enter_context(ContinuousBarSource(duckdb_path=paths['continuous_db']))
            bars = continuous_factor_input(source, continuous)
            calendars = source.get_calendars(bars)
        labeler = BarReturnLabeler(calendars,
            view=continuous.get('view', 'ETH') if continuous is not None else config.get('view', 'ETH'),
            horizons=config.get('horizons', [1, 5, 15, 60]),
            entry_offset_bars=config.get('entry_offset_bars', 1),
            bucket_strategy=(bars.attrs['continuous']['bucket_strategy'] if source else config.get('bucket_strategy')),
            boundary_policy=config.get('boundary_policy', 'same_session'), continuous_source=source)
        result = evaluate_bars(bars, factors, labeler, analysis_mode=mode,
            n_quantiles=config.get('n_quantiles', 5), min_ic_samples=config.get('min_ic_samples', 3),
            min_cross_section_size=config.get('min_cross_section_size', 3),
            factor_id=config.get('factor_id', 'factor_value'), evaluation_v2=config.get('evaluation_v2', False),
            **{key: config[key] for key in _ADVANCED_OPTIONS if key in config})
    diagnostics_report = split_diagnostics_report = None
    if diagnostics_requested:
        from backtest_and_analysis.time_series.diagnostics_report import evaluate_diagnostics
        options = dict(n_quantiles=config.get('n_quantiles', 5), min_samples=config.get('min_ic_samples', 3))
        diagnostics_report = evaluate_diagnostics(result, labeler, **options)
        if diagnostics_report['schema'] == 'time_series_diagnostics.v2':
            split_diagnostics_report = diagnostics_report
            diagnostics_report = evaluate_diagnostics(result, labeler, legacy_full=True, **options)
    # A changing source must not be bound to the digest from before it was read.
    if hashes != {str(path): sha256_file(path) for path in paths.values()}:
        raise ValueError('Input CSV changed during evaluation')
    from utils.code_version import resolve_code_version
    code_version, code_version_source = resolve_code_version(Path(__file__).resolve().parents[2])
    identity = ResultIdentity(
        config_hash=hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                            allow_nan=False).encode()).hexdigest(),
        input_file_hashes=hashes, calendar_hash=result.calendar_identity['calendar_hash'],
        code_version=code_version, result_schema_version=RESULT_SCHEMA_VERSION)
    outputs = write_evaluation_artifacts(result, destination, result_identity=identity, config=config,
                                         code_version_source=code_version_source, diagnostics_report=diagnostics_report,
                                         split_diagnostics_report=split_diagnostics_report)
    summary = json.loads(outputs['summary'].read_text(encoding='utf-8'))
    metrics = {key: summary[key] for key in ('analysis_mode', 'exclusion_summary', 'ic_by_horizon',
                                            'quantile_stats', 'cross_sections', 'result_identity')}
    if diagnostics_requested:
        diagnostics_source = (json.loads(outputs['evaluation_v2'].read_text(encoding='utf-8'))
                              if split_diagnostics_report is not None else summary)
        metrics['diagnostics_report'] = diagnostics_source['diagnostics_report']
    return dict(ok=True, exit_code=0, metrics=metrics,
        artifacts=[dict(artifact_type='evaluation_summary' if name == 'summary' else name,
                        path=str(path), sha256=sha256_file(path)) for name, path in outputs.items()])


def main() -> int:
    destination = Path(sys.argv[2])
    try:
        config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
        payload = evaluate(config, destination)
    except Exception as exc:
        destination.joinpath('summary.json').unlink(missing_ok=True)
        precondition = isinstance(exc, (ValueError, TypeError, KeyError, FileNotFoundError))
        payload = dict(ok=False, exit_code=2 if precondition else 3,
            error=dict(code=getattr(exc, 'code', 'time_series_precondition_failed' if precondition
                                    else 'time_series_execution_failed'), message=str(exc),
                       details={'exception_type': type(exc).__name__}))
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return payload['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
