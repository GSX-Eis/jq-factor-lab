"""Handle Bar and factor-value storage commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.agent_harness import runner
from tools.agent_harness._cli_compat import _cli_bindings
from tools.agent_harness.runner import HarnessResult


def _handle_bar_storage(args: argparse.Namespace) -> HarnessResult:
    import duckdb

    from utils.bar_builder import audit_bars, build_bars

    cli = _cli_bindings(__name__, runner)
    action, started = args.command, cli._utc_now()
    output = cli._output_dir(action, cli._resolve_args_output_root(args.output_root), args.run_id)
    common = dict(action=action, mode='local', output_dir=output, run_id=args.run_id,
                  command=[action, '--db', args.db], started_at=started, artifacts=[])
    try:
        path = Path(args.db).expanduser().resolve()
        if not path.is_file() and action != 'import-bars':
            raise ValueError(f'Database does not exist: {path}')
        if args.dry_run:
            metrics = dict(database=str(path), write=not getattr(args, 'check_only', False))
        else:
            with duckdb.connect(str(path), read_only=getattr(args, 'check_only', False)) as connection:
                instruments = args.instruments.split(',') if args.instruments else None
                if action == 'audit-bars':
                    metrics = audit_bars(connection, instruments=instruments, start=args.start,
                                         end=args.end, bucket_strategy=args.bucket_strategy)
                elif action == 'build-bars':
                    metrics = build_bars(connection, source=args.source, targets=args.targets.split(','),
                        instruments=instruments, start=args.start, end=args.end,
                        bucket_strategy=args.bucket_strategy, check_only=args.check_only)
                elif action == 'build-roll-schedule':
                    from utils.futures_roll import build_roll_schedule
                    metrics = build_roll_schedule(connection, product_roots=args.product_roots.split(','),
                        start=args.start, end=args.end, rule=args.rule, confirm_days=args.confirm_days,
                        forward_only=args.forward_only)
                else:
                    import pandas as pd

                    from backtest_and_analysis.time_series.csv_reader import (
                        read_bar_csv,
                    )
                    from data_pipeline.loaders.market_bar_importer import (
                        import_market_bars,
                    )
                    bars = read_bar_csv(args.bars)
                    metadata = pd.DataFrame(json.loads(Path(args.metadata).read_text(encoding='utf-8')))
                    count = import_market_bars(connection, bars, metadata,
                        bucket_strategy=args.bucket_strategy,
                        derive=tuple(args.derive.split(',')) if args.derive else (),
                        legacy_compatibility=False)
                    metrics = dict(imported_bars=count)
        result = cli._success_result(**common, metrics=metrics)
    except (ValueError, LookupError, OSError, duckdb.Error) as exc:
        result = cli._failure_result(**common, reason='bar_storage_precondition_failed',
                                 exit_code=cli.EXIT_PRECONDITION, metrics={'message': str(exc)})
    return cli._complete_result(result, json_out=args.json_out)


def _parse_strict_utc_csv_column(series: Any, name: str) -> Any:
    import pandas as pd
    pattern = r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:Z|\+00:00|\+0000)'
    text = series.astype('string')
    if not text.str.fullmatch(pattern).fillna(False).all():
        raise ValueError(f'{name} must explicitly specify UTC (Z, +00:00 or +0000)')
    return pd.to_datetime(text, format='ISO8601', utc=True).astype('datetime64[ns, UTC]')


def _format_factor_timestamps(series: Any) -> Any:
    microseconds = series.dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    nanoseconds = series.dt.nanosecond
    if nanoseconds.ne(0).any():
        microseconds = microseconds + nanoseconds.astype(str).str.zfill(3)
    return microseconds + 'Z'


def _handle_factor_values_store(args: argparse.Namespace) -> HarnessResult:
    import datetime as dt
    import uuid

    import duckdb
    import pandas as pd
    import pyarrow as pa

    from backtest_and_analysis.time_series.contracts import sha256_file
    from backtest_and_analysis.time_series.csv_reader import read_factor_csv
    from utils.code_version import resolve_code_version
    from utils.market_factor_store import (
        FactorValueConflictError,
        MarketFactorValueStore,
        TABLE_COLUMNS,
        build_identity_record,
        compute_factor_identity_hash,
    )
    from utils.session_calendar import SessionCalendar
    from utils.time_axis import BarSchedule, Frequency

    cli = _cli_bindings(__name__, runner)
    action, started = args.command, cli._utc_now()
    output = cli._output_dir(action, cli._resolve_args_output_root(args.output_root), args.run_id)
    store_path = Path(args.factor_values_store).expanduser().resolve()
    command = [action, '--factor-values-store', str(store_path)]
    common = dict(action=action, output_dir=output, run_id=args.run_id,
                  command=command, started_at=started, artifacts=[])

    if args.fake_result:
        return cli._fake_result(
            action=action, fake_result=args.fake_result, output_dir=output,
            run_id=args.run_id, started_at=started, command=command, json_out=args.json_out,
        )

    if args.dry_run:
        plan_metrics: dict[str, Any] = {
            'dry_run': True,
            'factor_values_store': str(store_path),
        }
        if action == 'write-factor-values':
            plan_metrics['input'] = str(Path(args.input).expanduser().resolve())
            if args.factor_id:
                plan_metrics['factor_id'] = args.factor_id
            if args.frequency:
                plan_metrics['frequency'] = args.frequency
        return cli._complete_result(
            cli._success_result(**common, mode='dry_run', reason='dry_run', metrics=plan_metrics),
            json_out=args.json_out,
        )

    try:
        store = MarketFactorValueStore(store_path)
        if action == 'verify-factor-values':
            if not store_path.is_dir():
                raise FileNotFoundError(f'Factor values store does not exist: {store_path}')
            report = store.verify_store()
            if report['ok']:
                result = cli._success_result(**common, mode='local', metrics=report)
            else:
                result = cli._failure_result(
                    **common, mode='local', reason='factor_store_verification_failed',
                    exit_code=cli.EXIT_QUALITY_GATE, metrics=report,
                )
            return cli._complete_result(result, json_out=args.json_out)

        if action == 'read-factor-values':
            if not store_path.is_dir():
                raise FileNotFoundError(f'Factor values store does not exist: {store_path}')
            instruments = [item.strip() for item in args.instruments.split(',') if item.strip()] if args.instruments else None
            metrics: dict[str, Any] = {'factor_values_store': str(store_path)}
            artifacts: list[dict[str, Any]] = []
            if args.list_identities:
                identities_df = store.list_identities(factor_id=args.factor_id, frequency=args.frequency)
                metrics['identity_count'] = int(len(identities_df))
                identities_out = identities_df.copy()
                identities_out['created_at'] = _format_factor_timestamps(identities_out['created_at'])
                metrics['identities'] = identities_out.to_dict('records')
            has_row_filters = any(
                getattr(args, name) is not None
                for name in ('factor_id', 'factor_identity_hash', 'frequency', 'session_view',
                             'instruments', 'start_utc', 'end_utc', 'session_start', 'session_end', 'out')
            )
            if has_row_filters or not args.list_identities:
                frame = store.read_factor_values(
                    factor_id=args.factor_id,
                    factor_identity_hash=args.factor_identity_hash,
                    frequency=args.frequency,
                    session_view=args.session_view,
                    instruments=instruments,
                    start_utc=args.start_utc,
                    end_utc=args.end_utc,
                    session_start=args.session_start,
                    session_end=args.session_end,
                )
                plan = store.last_read_plan
                metrics['row_count'] = int(len(frame))
                if plan is not None:
                    metrics['resolved_factor_ids'] = list(plan.resolved_factor_ids)
                    metrics['resolved_frequencies'] = list(plan.resolved_frequencies)
                    metrics['scanned_shards'] = [str(p.relative_to(store.parquet_root)) for p in plan.shard_paths]
                if args.limit and args.limit > 0:
                    preview = frame.head(args.limit).copy()
                    preview['timestamp_utc'] = _format_factor_timestamps(preview['timestamp_utc'])
                    preview['created_at'] = _format_factor_timestamps(preview['created_at'])
                    metrics['preview'] = preview.to_dict('records')
                if args.out:
                    out_path = Path(args.out).expanduser().resolve()
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if out_path.suffix.lower() == '.parquet':
                        frame.to_parquet(out_path, index=False)
                    else:
                        export_df = frame.copy()
                        export_df['timestamp_utc'] = _format_factor_timestamps(export_df['timestamp_utc'])
                        export_df['created_at'] = _format_factor_timestamps(export_df['created_at'])
                        export_df.to_csv(out_path, index=False)
                    artifacts.append({'artifact_type': 'factor_values_export', 'path': str(out_path)})
            result = cli._success_result(
                action=action, mode='local', output_dir=output, run_id=args.run_id,
                command=command, started_at=started, metrics=metrics, artifacts=artifacts,
            )
            return cli._complete_result(result, json_out=args.json_out)

        # write-factor-values
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f'Input file does not exist: {input_path}')

        cal_obj: SessionCalendar | None = None
        if args.calendar:
            cal_obj = SessionCalendar(
                args.calendar,
                args.calendar_start,
                args.calendar_end,
                rth_reference_calendar=args.rth_reference_calendar,
            )
        if cal_obj is None:
            raise ValueError('--calendar is required for write-factor-values to verify session_id')

        if input_path.suffix.lower() == '.parquet':
            raw_df = pd.read_parquet(input_path)
            is_full_schema = set(TABLE_COLUMNS).issubset(raw_df.columns)
        else:
            header_cols = list(pd.read_csv(input_path, nrows=0).columns)
            is_full_schema = set(TABLE_COLUMNS).issubset(header_cols)
            if is_full_schema:
                raw_df = pd.read_csv(input_path, dtype={'factor_id': 'string', 'factor_identity_hash': 'string',
                                                      'frequency': 'string', 'session_view': 'string',
                                                      'instrument_id': 'string', 'timestamp_utc': 'string',
                                                      'session_id': 'string', 'source_build_id': 'string',
                                                      'calendar_hash': 'string', 'code_version': 'string',
                                                      'created_at': 'string', 'batch_id': 'string'})
                raw_df['timestamp_utc'] = cli._parse_strict_utc_csv_column(raw_df['timestamp_utc'], 'timestamp_utc')
                raw_df['created_at'] = cli._parse_strict_utc_csv_column(raw_df['created_at'], 'created_at')
            else:
                raw_df = read_factor_csv(input_path)

        identity_rec = None
        if is_full_schema:
            batch_df = raw_df
        else:
            if not args.factor_id or not args.frequency or not args.analysis_mode or not args.source_build_id:
                raise ValueError(
                    '--factor-id, --frequency, --analysis-mode, and --source-build-id are required '
                    'when input does not contain full market_factor_values schema'
                )
            if cal_obj is None:
                raise ValueError('--calendar is required when input does not contain full market_factor_values schema')
            parsed_freq = Frequency.parse(args.frequency)
            if not parsed_freq.is_valid or parsed_freq.canonical is None:
                raise ValueError(f'Invalid frequency: {args.frequency!r}')
            canonical_freq = parsed_freq.canonical
            strategy = BarSchedule(
                cal_obj, canonical_freq, view=args.session_view, bucket_strategy=args.bucket_strategy
            ).bucket_strategy
            params = json.loads(args.params_json) if args.params_json else {}
            code_ver, code_ver_src = resolve_code_version()
            source_sha = sha256_file(input_path)
            fhash, payload = compute_factor_identity_hash(
                factor_id=args.factor_id,
                factor_class=args.factor_class,
                source_sha256=source_sha,
                params=params,
                frequency=canonical_freq,
                session_view=args.session_view,
                analysis_mode=args.analysis_mode,
                bucket_strategy=strategy,
                calendar_hash=cal_obj.calendar_hash,
                source_build_id=args.source_build_id,
                data_version=args.data_version,
                rth_reference_calendar=cal_obj.rth_reference_calendar,
                continuous_adjust=args.adjust,
                code_version=code_ver,
            )
            now_dt = dt.datetime.now(dt.timezone.utc)
            now_iso = now_dt.isoformat()
            identity_rec = build_identity_record(
                factor_identity_hash=fhash,
                payload=payload,
                code_version_source=code_ver_src,
                created_at=now_iso,
            )
            sessions = []
            for ts in raw_df['timestamp_utc']:
                owner = cal_obj.session_at(ts, args.session_view)
                if owner is None:
                    raise ValueError(f'timestamp_utc {ts} is outside any open {args.session_view} session')
                sessions.append(owner.session_date)
            batch_df = raw_df.assign(
                factor_id=args.factor_id,
                factor_identity_hash=fhash,
                frequency=canonical_freq,
                session_view=args.session_view,
                session_id=sessions,
                source_build_id=args.source_build_id,
                calendar_hash=cal_obj.calendar_hash,
                code_version=code_ver,
                created_at=pd.Timestamp(now_dt),
                batch_id=uuid.uuid4().hex,
            ).loc[:, list(TABLE_COLUMNS)]

        receipt = store.append_factor_values(
            batch_df,
            identity_record=identity_rec,
            calendar=cal_obj,
        )
        result = cli._success_result(**common, mode='local', metrics=receipt.to_dict())
    except FactorValueConflictError as exc:
        result = cli._failure_result(
            **common,
            mode='local',
            reason='factor_value_conflict',
            exit_code=cli.EXIT_QUALITY_GATE,
            metrics={
                'message': str(exc),
                'factor_id': exc.factor_id,
                'factor_identity_hash': exc.factor_identity_hash,
                'conflict_count': exc.conflict_count,
                'examples': exc.examples,
            },
        )
    except (duckdb.Error, pa.lib.ArrowException, RuntimeError) as exc:
        result = cli._failure_result(
            **common,
            mode='local',
            reason='factor_store_execution_failed',
            exit_code=cli.EXIT_EXECUTION,
            metrics={'message': str(exc)},
        )
    except (ValueError, TypeError, LookupError, OSError, json.JSONDecodeError) as exc:
        result = cli._failure_result(
            **common,
            mode='local',
            reason='factor_store_precondition_failed',
            exit_code=cli.EXIT_PRECONDITION,
            metrics={'message': str(exc)},
        )
    return cli._complete_result(result, json_out=args.json_out)
