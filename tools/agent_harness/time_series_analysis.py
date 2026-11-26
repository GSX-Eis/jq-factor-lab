"""CSV and MarketSlot evaluation worker; the CLI owns execution and completion markers."""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import pandas as pd

    from backtest_and_analysis.price_fetcher import PriceFetcher
    from backtest_and_analysis.time_series import EvaluationResult


_INFERENCE_OPTIONS = ('split', 'bootstrap', 'hac_lag', 'direction', 'standardization',
                      'calibration_scenario', 'trial_selection')


class AnalysisPreconditionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def revoke_evaluation_completion(destinations: Iterable[Path]) -> dict[str, str]:
    errors = []
    for destination in destinations:
        try:
            (destination / "summary.json").unlink(missing_ok=True)
        except OSError as exc:
            errors.append(str(exc))
    return {"completion_cleanup_error": "; ".join(errors)} if errors else {}


class BarEvaluationStoreFailure(Exception):
    """Classify a failed Harness write and revoke its completion markers."""

    def __init__(self, cause: Exception, destinations: Iterable[Path]) -> None:
        from tools.agent_harness.runner import EXIT_EXECUTION, EXIT_PRECONDITION, EXIT_QUALITY_GATE
        from utils.bar_evaluation_store import (
            BarEvaluationConflictError, BarEvaluationIntegrityError, BarEvaluationPreconditionError,
        )

        super().__init__(str(cause))
        if isinstance(cause, BarEvaluationConflictError):
            self.code, self.exit_code = "bar_evaluation_store_conflict", EXIT_QUALITY_GATE
        elif isinstance(cause, (BarEvaluationPreconditionError, BarEvaluationIntegrityError, ValueError, FileNotFoundError)):
            self.code, self.exit_code = "bar_evaluation_store_precondition_failed", EXIT_PRECONDITION
        else:
            self.code, self.exit_code = "bar_evaluation_store_write_failed", EXIT_EXECUTION
        self.details = {"exception_type": type(cause).__name__, "error": str(cause)}
        self.cleanup = revoke_evaluation_completion(destinations)


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


def resolve_slot_request(config: dict[str, Any]):
    from utils.market_db.research import resolve_research_slot

    conflicts = ('bars_csv', 'factors_csv', 'continuous_futures', 'calendar', 'calendar_by_instrument')
    if any(config.get(name) is not None for name in conflicts):
        raise ValueError('Slot analysis supplies execution bars, factors and calendar through the Slot and store')
    if config.get('slot') is not None and config.get('slot_config') is not None:
        raise ValueError('Supply either slot or slot_config')
    return resolve_research_slot(
        config.get('slot_config', config.get('slot')), config.get('slot_id'),
        Path(config.get('slot_base_dir', '.')).expanduser().resolve(),
    )


def _default_slot_instruments(slot) -> list[str]:
    static = getattr(slot.universe, 'static', None)
    if isinstance(static, dict) and static:
        first_pool = next(iter(static.values()))
        if first_pool:
            return list(first_pool)
    specs = getattr(slot.universe, 'specs', None)
    if isinstance(specs, dict) and specs:
        return sorted(specs.keys())
    root = getattr(slot.bar_feed, 'root', None)
    if isinstance(root, str) and root:
        return [root]
    raise ValueError('instruments must be provided when the Slot has no static instruments')


def _resolve_slot_factor_id(config: Mapping[str, Any], factor_cls: type[Any]) -> str:
    """Resolve explicit or class-derived factor_id for Slot factor_file evaluation."""
    if 'factor_id' not in config or config['factor_id'] is None:
        return factor_cls.__name__
    raw = config['factor_id']
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError('factor_id must be a nonempty string when provided')
    return raw


def _canonicalize_config_order(value: Any) -> Any:
    """Recursively sort mapping keys so summary.json matches sorted config_hash order."""
    if isinstance(value, Mapping):
        return {key: _canonicalize_config_order(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_config_order(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize_config_order(item) for item in value)
    return value


def _evaluate_slot(config: dict[str, Any], options: dict[str, Any]):
    import pandas as pd

    from backtest_and_analysis.factor_loader import FactorLoader
    from backtest_and_analysis.time_series.evaluator import evaluate_slot
    from backtest_and_analysis.time_series.factor_bridge import bar_factor_frame, read_slot_factor_values
    from slots import build_slot_from_config
    from tools.agent_harness.causal_gate import (
        CausalFastPathGateError, causal_gate_semantic_summary, verify_factor_causality,
    )
    from utils.data_revision import frame_digest
    from utils.duckdb_results_writer import connect_read_only_with_retry
    from utils.factor_core.bar_bridge import _session_rollup
    from utils.market_db.identity import slot_market_db_identity
    from utils.market_db.research import open_research_slot
    from utils.time_axis import Frequency

    frozen, base_dir = resolve_slot_request(config)
    factor_file = config.get('factor_file')

    required_fields = ('start_date', 'end_date') if factor_file else ('start_date', 'end_date', 'factor_store_root', 'factor_id')
    for name in required_fields:
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f'{name} must be a nonempty string for Slot evaluation')
    frequency = Frequency.parse(config.get('frequency', '1d'))
    if not frequency.is_valid:
        raise ValueError('Slot evaluation requires a valid frequency')
    instruments = config.get('instruments')
    if instruments is not None and (not isinstance(instruments, list) or not instruments or
            any(not isinstance(item, str) or not item.strip() for item in instruments) or
            len(set(instruments)) != len(instruments)):
        raise ValueError('instruments must contain distinct nonempty instrument IDs')

    slot_options = dict(options)
    causal_gate_policy: str | None = None
    causal_gate_report: dict[str, Any] | None = None
    with ExitStack() as stack:
        connection = None
        database = config.get('slot_db', config.get('duckdb_path'))
        if config.get('market_db') is not None:
            slot, _ = stack.enter_context(open_research_slot(
                config['market_db'], frozen, base_dir, declared_db=database,
            ))
        else:
            if frozen.bar_feed.provider in ('jq_ashare_local', 'cn_futures', 'global_futures'):
                if not isinstance(database, str) or not database.strip():
                    raise ValueError('Native Slot analysis requires a read-only slot_db path')
                connection = connect_read_only_with_retry(Path(database).expanduser().resolve())
                stack.callback(connection.close)
                connection.execute('BEGIN TRANSACTION')
            slot = build_slot_from_config(frozen, connection=connection, base_dir=base_dir)
        selected_db = slot_market_db_identity(slot)
        if "market_db_identity" in config and config["market_db_identity"] != selected_db:
            raise ValueError("Market database identity changed since preflight")
        if config.get('view', getattr(slot.bar_feed, 'view', 'ETH')) != getattr(slot.bar_feed, 'view', 'ETH'):
            raise ValueError('view disagrees with the Slot')
        profile = slot.microstructure.execution_profile
        overrides = config.get('execution_profile', {})
        if not isinstance(overrides, dict):
            raise ValueError('execution_profile overrides must be an object')
        if 'boundary_policy' in config:
            if 'boundary_policy' in overrides and overrides['boundary_policy'] != config['boundary_policy']:
                raise ValueError('Conflicting boundary_policy overrides')
            overrides = {**overrides, 'boundary_policy': config['boundary_policy']}
        profile = replace(profile, **overrides)

        if factor_file:
            factor_path = Path(factor_file).expanduser().resolve()
            classes = FactorLoader.load_factor_class_from_file(str(factor_path), config.get('factor_class'))
            if len(classes) != 1:
                raise ValueError(f'Expected one factor class in {factor_path}; pass factor_class to disambiguate')
            factor_cls = classes[0]
            factor_instruments = instruments or _default_slot_instruments(slot)
            factor_id = _resolve_slot_factor_id(config, factor_cls)
            slot_options['factor_id'] = factor_id
            causal_gate_policy = config.get('causal_gate_policy', 'reject')
            factor = factor_cls(
                None,
                factor_id,
                config['start_date'],
                config['end_date'],
                stock_pool=factor_instruments,
                slot=slot,
                data_frequency=frequency.canonical,
                analysis_mode=options['analysis_mode'],
                session_view=getattr(slot.bar_feed, 'view', 'ETH'),
                slot_execution_mode=config.get('slot_execution_mode'),
            )
            factor._require_causal_verification = True
            gate = verify_factor_causality(
                factor,
                on_failure=causal_gate_policy,
                apply_to_factor=True,
            )
            causal_gate_report = gate.to_dict()
            if not gate.ok:
                raise CausalFastPathGateError(
                    f'Causal fast-path gate rejected {gate.factor_name}: {gate.failure_reason}',
                    gate_result=causal_gate_report,
                )
            if factor._frequency_route.bar_grid and not _session_rollup(factor):
                values = bar_factor_frame(factor.get_bar_factor_frame())
            else:
                panel = factor.get_factor_values()
                if getattr(panel.index, 'tz', None) is None and not panel.empty:
                    grid = slot.calendar.bar_grid(
                        frequency, config['start_date'], config['end_date'], view=getattr(slot.bar_feed, 'view', 'ETH')
                    )
                    keys = grid.groupby(pd.to_datetime(grid.session_id))['timestamp_utc'].max()
                    mapped = pd.to_datetime(panel.index).map(keys)
                    if mapped.isna().any():
                        raise ValueError('Factor panel dates could not be mapped to Slot session timestamps')
                    panel = panel.set_axis(pd.DatetimeIndex(mapped))
                values = bar_factor_frame(panel)
            factor_identity_hash = hashlib.sha256(factor_path.read_bytes()).hexdigest()
            slot = replace(slot, slot_data_fingerprint={
                **selected_db,
                'factor_identity_hash': factor_identity_hash,
                'factor_values_sha256': frame_digest(values),
            })
            eval_instruments = instruments or sorted(values.instrument_id.unique())
        else:
            values, identity = read_slot_factor_values(
                config['factor_store_root'], slot, factor_id=config['factor_id'], frequency=frequency.canonical,
                start_date=config['start_date'], end_date=config['end_date'],
                factor_identity_hash=config.get('factor_identity_hash'), instruments=instruments,
            )
            factor_identity_hash = identity.factor_identity_hash
            slot = replace(slot, slot_data_fingerprint={
                **selected_db,
                'factor_identity_hash': factor_identity_hash,
                'factor_input': identity.slot_data_fingerprint,
                'factor_source_build_ids': sorted(values.source_build_id.dropna().unique().tolist()),
            })
            eval_instruments = instruments or sorted(values.instrument_id.unique())
        result = evaluate_slot(
            slot, values[['timestamp_utc', 'instrument_id', 'factor_value']], frequency=frequency.canonical,
            start_date=config['start_date'], end_date=config['end_date'],
            instruments=eval_instruments,
            horizons=config.get('horizons', [1, 5, 15, 60]), entry_offset_bars=config.get('entry_offset_bars'),
            bucket_strategy=config.get('bucket_strategy'), profile=profile,
            apply_tradability_mask=config.get('apply_tradability_mask', False), **slot_options,
        )
    resolved = {key: value for key, value in config.items() if key != 'slot'}
    resolved.update(slot_config=frozen.as_dict(), slot_id=slot.slot_id, frequency=frequency.canonical,
                    factor_identity_hash=factor_identity_hash, slot_base_dir=str(base_dir))
    if factor_file:
        resolved['factor_id'] = slot_options['factor_id']
    if causal_gate_report is not None:
        resolved['causal_gate_policy'] = causal_gate_policy
        # Keep runtime diagnostics (elapsed_sec, cutoffs, samples) out of config_hash / summary.json.
        resolved['causal_gate'] = causal_gate_semantic_summary(causal_gate_report)
    if overrides:
        resolved['execution_profile'] = overrides
    resolved = _canonicalize_config_order(resolved)
    fingerprints = result.data_identity['slot_data_fingerprint']
    hashes = {name: fingerprints[name] for name in ('execution_bars_sha256', 'factor_values_sha256')}
    return result, resolved, hashes, causal_gate_report


def evaluate(config: dict[str, Any], destination: Path | None) -> dict[str, Any]:
    if 'evaluation_v2' in config:
        raise ValueError(
            'The Bar config key "evaluation_v2" has been removed. Remove it and set '
            '"trial_selection": {"method": "holm"} to request inference and the trial ledger.'
        )
    mode = config.get('analysis_mode')
    if mode is None:
        raise AnalysisPreconditionError('missing_analysis_mode', 'Explicit analysis_mode is required')
    if mode not in ('time_series', 'cross_sectional'):
        raise AnalysisPreconditionError('invalid_analysis_mode',
            'analysis_mode must be time_series or cross_sectional')
    diagnostics_requested = config.get('diagnostics_report', False)
    if not isinstance(diagnostics_requested, bool) or (diagnostics_requested and mode != 'time_series'):
        raise ValueError('diagnostics_report must be a boolean and requires time_series when enabled')
    options = dict(
        analysis_mode=mode, n_quantiles=config.get('n_quantiles', 5), min_ic_samples=config.get('min_ic_samples', 3),
        min_cross_section_size=config.get('min_cross_section_size', 3),
        factor_id=config.get('factor_id', 'factor_value'), diagnostics=diagnostics_requested,
        **{key: config[key] for key in _INFERENCE_OPTIONS if key in config},
    )
    causal_gate_report = None
    if any(config.get(key) is not None for key in ('slot', 'slot_id', 'slot_config', 'factor_store_root', 'market_db')):
        result, config, hashes, causal_gate_report = _evaluate_slot(config, options)
    else:
        result, hashes = _evaluate_csv(config, options, destination=destination)
        if destination is None:
            return result
    diagnostics_report = None
    if diagnostics_requested:
        from backtest_and_analysis.time_series.diagnostics_report import evaluate_diagnostics
        diagnostics_report = evaluate_diagnostics(result)
    published = publish_evaluation(result, config, hashes, destination, diagnostics_report=diagnostics_report)
    if causal_gate_report is not None:
        published['metrics']['causal_gate'] = causal_gate_report
    return published


def publish_evaluation(result: EvaluationResult, config: dict[str, Any], hashes: dict[str, str], destination: Path, *,
                       diagnostics_report: dict[str, Any] | None = None) -> dict[str, Any]:
    from backtest_and_analysis.time_series import (
        RESULT_SCHEMA_VERSION, ResultIdentity, sha256_file, write_evaluation_artifacts,
    )
    from utils.code_version import resolve_code_version
    from utils.market_db.identity import market_db_identity

    if identity := market_db_identity(result.data_identity):
        config = {**config, "market_db_identity": identity}
    code_version, code_version_source = resolve_code_version(Path(__file__).resolve().parents[2])
    identity = ResultIdentity(
        config_hash=hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                            allow_nan=False).encode()).hexdigest(),
        input_file_hashes=hashes, calendar_hash=result.calendar_identity['calendar_hash'],
        code_version=code_version, result_schema_version=RESULT_SCHEMA_VERSION)
    outputs = write_evaluation_artifacts(result, destination, result_identity=identity, config=config,
                                         code_version_source=code_version_source, diagnostics_report=diagnostics_report)
    metrics = json.loads(outputs['summary'].read_text(encoding='utf-8'))
    metrics.pop('config')
    return dict(ok=True, exit_code=0, metrics=metrics,
        artifacts=[dict(artifact_type='evaluation_summary' if name == 'summary' else name,
                        path=str(path), sha256=sha256_file(path)) for name, path in outputs.items()])



def publish_legacy_daily_evaluation(
    destination: Path, *, pricing: pd.Series, factors: pd.DataFrame, price_fetcher: PriceFetcher,
    factor_id: str, pool_name: str,
    horizons: tuple[int, ...], entry_offset_bars: int, n_quantiles: int,
) -> None:
    """Use the same daily price snapshot and pool values as the historical runner."""
    import pandas as pd

    from backtest_and_analysis.time_series.evaluator import evaluate_slot
    from backtest_and_analysis.time_series.factor_bridge import bar_factor_frame
    from slots.jq_ashare_local import daily_slot_config
    from utils.market_factor.bridge import factor_rows_on_grid

    verified = pricing.attrs['trading_calendar']
    dates = pd.DatetimeIndex(factors.index).tz_localize(None).normalize()
    on_grid = factor_rows_on_grid(
        pd.Series(dates.isin(verified.sessions()), index=factors.index), factors.notna().any(axis=1),
    )
    factors, dates = factors.loc[on_grid], dates[on_grid.to_numpy()]
    if factors.empty:
        raise ValueError('No factor rows on the daily evaluation grid')
    start = dates.min()
    end = max(dates.max(), pricing.index.get_level_values('date').max())
    config = daily_slot_config(start=start.date().isoformat(), end=end.date().isoformat())
    slot = price_fetcher.daily_evaluation_slot(pricing, config)
    if not slot.calendar.sessions().equals(verified.sessions(start, end)):
        raise ValueError('Legacy trading dates disagree with the A-share Slot calendar')
    grid = slot.calendar.bar_grid('1d')
    keys = grid.set_index(pd.to_datetime(grid.session_id)).timestamp_utc
    signals = factors.set_axis(pd.DatetimeIndex(dates.map(keys)))
    result = evaluate_slot(
        slot, bar_factor_frame(signals), frequency='1d', start_date=config.calendar.options['start'],
        end_date=config.calendar.options['end'], instruments=signals.columns.tolist(),
        analysis_mode='cross_sectional', horizons=horizons, entry_offset_bars=entry_offset_bars,
        n_quantiles=n_quantiles, factor_id=factor_id,
    )
    fingerprint = result.data_identity['slot_data_fingerprint']
    hashes = {name: fingerprint[name] for name in ('execution_bars_sha256', 'factor_values_sha256')}
    publish_evaluation(result, dict(
        slot_config=config.as_dict(), analysis_mode='cross_sectional', frequency='1d', factor_id=factor_id,
        pool_name=pool_name, price_source=price_fetcher.price_table, entry_offset_bars=entry_offset_bars,
        horizons=list(horizons), n_quantiles=n_quantiles,
    ), hashes, destination)


def ingest_legacy_daily_evaluations(
    directory: Path, results_db: Path, *, expected_count: int, run_id: str,
) -> dict[str, Any]:
    """Ingest every publication from the completed single-factor execution."""
    from backtest_and_analysis.time_series import sha256_file
    from utils.bar_evaluation_store import ingest_evaluation

    evaluations: list[Path] = []
    runs, artifacts = [], []
    try:
        evaluations = sorted(directory.iterdir())
        if len(evaluations) != expected_count:
            raise ValueError('Daily evaluation publication count does not match completed analyses')
        for path in evaluations:
            publication = [
                dict(artifact_type=kind, path=str(path / name), sha256=sha256_file(path / name))
                for kind, name in (
                    ('evaluation_summary', 'summary.json'), ('per_bar_labels', 'per_bar_labels.csv'),
                    ('ic_by_horizon', 'ic_by_horizon.csv'), ('quantile_stats', 'quantile_stats.csv'),
                )
            ]
            receipt = ingest_evaluation(results_db, evaluation_dir=path, artifacts=publication, run_id=run_id)
            runs.append(dict(identity_hash=receipt.identity_hash, status=receipt.status))
            artifacts.extend(publication)
    except Exception as exc:
        raise BarEvaluationStoreFailure(exc, evaluations[len(runs):]) from exc
    database = dict(path=str(results_db), runs=runs)
    if len(runs) == 1:
        database.update(runs[0])
    return dict(results_db=database, artifacts=artifacts)

def _evaluate_csv(config: dict[str, Any], options: dict[str, Any], *, destination: Path | None = None):
    from backtest_and_analysis.time_series import BarReturnLabeler, evaluate_bars, read_bar_csv, read_factor_csv, sha256_file
    from utils.session_calendar import SessionCalendar

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
    hashes = {str(path): sha256_file(path) for path in paths.values()} if destination is not None else {}
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
        if destination is None:
            from backtest_and_analysis.time_series.evaluator import _prepare_bar_evaluation

            _prepare_bar_evaluation(labeler, **options)
            if factors is None and "factor_value" not in bars:
                raise ValueError("Supply factors_csv or a factor_value Bar column")
            for instrument, group in bars.groupby("instrument_id"):
                session_calendar = calendars.get(instrument) if isinstance(calendars, dict) else calendars
                if session_calendar is None:
                    raise ValueError(f"Calendar is missing for instrument {instrument}")
                for timestamp in (group.timestamp_utc.min(), group.timestamp_utc.max()):
                    session_calendar.session_at(timestamp, "ETH")
            return {"analysis_mode": options["analysis_mode"], "inputs": {key: str(path) for key, path in paths.items()},
                    "horizons": list(labeler.horizons), "view": labeler.view}, {}
        result = evaluate_bars(bars, factors, labeler, **options)
    if hashes != {str(path): sha256_file(path) for path in paths.values()}:
        raise ValueError('Input CSV changed during evaluation')
    return result, hashes


def main() -> int:
    from backtest_and_analysis.time_series.artifacts import (
        LegacyEvaluationOutputError, validate_evaluation_output_dir,
    )

    destination = Path(sys.argv[2])
    try:
        validate_evaluation_output_dir(destination)
        config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
        payload = evaluate(config, destination)
    except Exception as exc:
        if not isinstance(exc, LegacyEvaluationOutputError):
            destination.joinpath('summary.json').unlink(missing_ok=True)
        precondition = isinstance(exc, (ValueError, TypeError, KeyError, FileNotFoundError))
        exit_code = getattr(exc, 'exit_code', 2 if precondition else 3)
        details = {'exception_type': type(exc).__name__, **getattr(exc, 'details', {})}
        payload = dict(ok=False, exit_code=exit_code,
            error=dict(code=getattr(exc, 'code', 'time_series_precondition_failed' if precondition
                                    else 'time_series_execution_failed'), message=str(exc),
                       details=details))
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return payload['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
