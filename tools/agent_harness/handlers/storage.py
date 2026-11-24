"""Handle Bar and factor-value storage commands."""
from __future__ import annotations

import argparse
import json
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tools.agent_harness import runner
from tools.agent_harness._cli_compat import _cli_bindings
from tools.agent_harness.runner import HarnessResult

if TYPE_CHECKING:
    from utils.session_calendar import SessionCalendar


def _handle_bar_storage(args: argparse.Namespace) -> HarnessResult:
    import duckdb

    from utils.bar_builder import audit_bars, build_bars
    from utils.market_db import MarketDbProfileError, open_market_db
    from utils.market_db.validation import require_market_db_profile

    cli = _cli_bindings(__name__, runner)
    action, started = args.command, cli._utc_now()
    output = cli._output_dir(action, cli._resolve_args_output_root(args.output_root), args.run_id)
    common = dict(action=action, mode='local', output_dir=output, run_id=args.run_id,
                  command=[action, '--db', args.db], started_at=started, artifacts=[])
    if action == 'import-bars' and args.manifest:
        common['command'].extend(['--manifest', args.manifest])
    try:
        path = Path(args.db).expanduser().resolve()
        uses_profile = action in ('import-bars', 'build-bars')
        if not uses_profile and not path.is_file():
            raise ValueError(f'Database does not exist: {path}')
        if action == 'import-bars' and args.manifest:
            metrics = _import_bar_manifest(path, args)
        elif args.dry_run and not uses_profile:
            metrics = dict(database=str(path), write=True)
        else:
            check_only = getattr(args, 'check_only', False)
            with ExitStack() as stack:
                if uses_profile:
                    connection, _, _ = stack.enter_context(open_market_db(path, read_only=args.dry_run or check_only))
                else:
                    connection = stack.enter_context(duckdb.connect(str(path)))
                if uses_profile and args.dry_run and not check_only:
                    require_market_db_profile(connection, write=True)
                instruments = args.instruments.split(',') if args.instruments else None
                if action == 'audit-bars':
                    metrics = audit_bars(connection, instruments=instruments, start=args.start,
                                         end=args.end, bucket_strategy=args.bucket_strategy)
                elif action == 'build-bars':
                    metrics = build_bars(connection, source=args.source, targets=args.targets.split(','),
                        instruments=instruments, start=args.start, end=args.end,
                        bucket_strategy=args.bucket_strategy, check_only=args.check_only or args.dry_run)
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
                        legacy_compatibility=False, check_only=args.dry_run)
                    metrics = dict(imported_bars=count)
        if uses_profile and args.dry_run:
            metrics.update(database=str(path), write=not getattr(args, 'check_only', False))
        result = cli._success_result(**common, metrics=metrics)
    except (ValueError, LookupError, OSError, duckdb.Error) as exc:
        reason = exc.code if isinstance(exc, MarketDbProfileError) else 'bar_storage_precondition_failed'
        result = cli._failure_result(**common, reason=reason,
                                 exit_code=cli.EXIT_PRECONDITION, metrics={'message': str(exc)})
    return cli._complete_result(result, json_out=args.json_out)


def _import_bar_manifest(path: Path, args: argparse.Namespace) -> dict:
    from data_pipeline.loaders._transactions import import_transaction
    from data_pipeline.loaders.market_bar_importer import import_market_bars
    from slots import build_slot_from_config
    from slots.declarative.slot import DeclarativeBarFeed
    from utils.bars.builder import audit_bars
    from utils.market_db import open_market_db
    from utils.market_db.validation import (
        require_market_db_profile, validate_market_db_calendar, validate_market_db_provider,
    )
    from utils.slots import load_slot_config

    manifest = Path(args.manifest).expanduser().resolve()
    config = load_slot_config(manifest)
    provider = config.bar_feed.provider
    with open_market_db(path) as (connection, profile, _):
        require_market_db_profile(connection, write=True)
        validate_market_db_provider(profile, provider, domain=config.market_domain)
    slot = build_slot_from_config(config, base_dir=manifest.parent)
    feed = slot.bar_feed
    if not isinstance(feed, DeclarativeBarFeed):
        raise ValueError('--manifest requires a declarative Bar source')
    source_profile = slot.market_db_profile
    domain = source_profile.market_domain if source_profile is not None else config.market_domain
    validate_market_db_provider(profile, provider, domain=domain)
    strategy = args.bucket_strategy or feed.bucket_strategy
    derive = tuple(args.derive.split(',')) if args.derive else ()
    bars = feed.load(bucket_strategy=strategy)
    if bars.empty:
        raise ValueError('Manifest contains no bars to import')
    instruments = sorted(bars.instrument_id.unique())
    metadata = feed.metadata.loc[feed.metadata.instrument_id.isin(instruments)]
    with open_market_db(path, read_only=args.dry_run) as (connection, profile, _):
        validate_market_db_provider(profile, provider, domain=domain)
        validate_market_db_calendar(profile, slot.calendar.calendar_id, slot.calendar.timezone)
        transaction = nullcontext() if args.dry_run else import_transaction(connection)
        with transaction:
            count = import_market_bars(
                connection, bars, metadata, bucket_strategy=strategy, derive=derive,
                legacy_compatibility=False, slot_provider=provider, check_only=args.dry_run,
                _transaction_open=True,
            )
            if args.dry_run:
                return dict(imported_bars=count)
            audit = audit_bars(connection, instruments=instruments, _transaction_open=True)
            placeholders = ','.join('?' for _ in instruments)
            quality = connection.execute(
                f'''SELECT frequency, status, count(*), sum(n_expected_bars), sum(n_actual_bars),
                           sum(incomplete_buckets)
                    FROM bar_quality WHERE instrument_id IN ({placeholders})
                    GROUP BY frequency, status ORDER BY frequency, status''', instruments,
            ).fetchall()
    fields = ('frequency', 'status', 'sessions', 'expected_bars', 'actual_bars', 'incomplete_buckets')
    return dict(imported_bars=count, **audit, quality=[dict(zip(fields, row)) for row in quality])


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


def _factor_value_calendar(args: argparse.Namespace) -> tuple[SessionCalendar, dict[str, Any]]:
    from utils.session_calendar import SessionCalendar, VERIFIED_XSHG

    fingerprint = args.slot_data_fingerprint
    if fingerprint is not None and fingerprint.lstrip().startswith('{'):
        fingerprint = json.loads(fingerprint)
    slot = dict(slot_id=args.slot_id, slot_config_hash=args.slot_config_hash,
                adjustment_policy=args.adjustment_policy, slot_data_fingerprint=fingerprint)
    if not args.slot_config:
        if args.slot_db:
            raise ValueError('--slot-db requires --slot-config')
        if not args.calendar:
            raise ValueError('--calendar or --slot-config is required to verify session_id')
        calendar = SessionCalendar(args.calendar, args.calendar_start, args.calendar_end,
                                   rth_reference_calendar=args.rth_reference_calendar)
        return calendar, slot

    import duckdb

    from slots import build_slot_from_config
    from slots.declarative.slot import _calendar
    from utils.slots import load_slot_config

    if any((args.calendar, args.calendar_start, args.calendar_end, args.rth_reference_calendar)):
        raise ValueError('--slot-config supplies the calendar. Omit the explicit calendar options.')
    config = load_slot_config(Path(args.slot_config).expanduser())
    for field in ('slot_id', 'slot_config_hash'):
        value = getattr(config, field)
        if slot[field] is not None and slot[field] != value:
            raise ValueError(f'--{field.replace("_", "-")} disagrees with --slot-config')
        slot[field] = value
    provider = config.calendar.provider
    if provider == 'jq_ashare_local' and not args.slot_db:
        options = config.calendar.options
        if options.keys() - {'start', 'end', 'calendar_id', 'timezone'}:
            raise ValueError('Native calendar options may only declare start, end, calendar_id and timezone')
        if options.get('calendar_id', VERIFIED_XSHG) not in ('XSHG', VERIFIED_XSHG):
            raise ValueError('jq_ashare_local requires the XSHG calendar')
        calendar = SessionCalendar.from_pmc(**{key: options[key] for key in ('start', 'end') if key in options})
        if options.get('timezone', calendar.timezone) != calendar.timezone:
            raise ValueError('jq_ashare_local calendar timezone disagrees with its declaration')
        default_adjustment = 'dynamic_pit'
    elif provider in ('jq_ashare_local', 'cn_futures', 'global_futures'):
        if not args.slot_db:
            raise ValueError(f'--slot-db is required to resolve the {provider} calendar from instrument metadata')
        database = Path(args.slot_db).expanduser()
        if not database.is_file():
            raise FileNotFoundError(f'Slot database does not exist: {database}')
        with duckdb.connect(str(database), read_only=True) as connection:
            calendar = build_slot_from_config(config, connection=connection).calendar
        default_adjustment = ('dynamic_pit' if provider == 'jq_ashare_local'
                              else config.bar_feed.options.get('adjustment', 'none'))
    else:
        if config.bar_feed.provider == 'duckdb_table':
            calendar = build_slot_from_config(config, base_dir=Path(args.slot_config).expanduser().resolve().parent).calendar
        else:
            calendar, _ = _calendar(config)
        default_adjustment = 'none'
    slot['adjustment_policy'] = args.adjustment_policy or args.adjust or default_adjustment
    return calendar, slot


def _handle_factor_values_store(args: argparse.Namespace) -> HarnessResult:
    import csv
    import datetime as dt
    import hashlib
    import uuid

    import duckdb
    import pandas as pd
    import pyarrow as pa

    from backtest_and_analysis.time_series.contracts import sha256_file, validate_factor_frame
    from utils.code_version import resolve_code_version
    from utils.market_factor_store import (
        FactorValueConflictError,
        MARKET_FACTOR_IDENTITY_SCHEMA_VERSION,
        MarketFactorValueStore,
        TABLE_COLUMNS,
        build_identity_record,
        compute_factor_identity_hash,
    )
    from utils.market_factor.identity import _normalize_slot_fields
    from utils.market_factor.schema import _canonical_json
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
            if args.slot_config:
                plan_metrics['slot_config'] = args.slot_config
        if args.slot_id:
            plan_metrics['slot_id'] = args.slot_id
        return cli._complete_result(
            cli._success_result(**common, mode='dry_run', reason='dry_run', metrics=plan_metrics),
            json_out=args.json_out,
        )

    try:
        store = MarketFactorValueStore(store_path)
        filters = dict(factor_id=args.factor_id, frequency=args.frequency, slot_id=args.slot_id)
        if action != 'write-factor-values':
            filters['factor_identity_hash'] = args.factor_identity_hash
        if action == 'verify-factor-values':
            if not store_path.is_dir():
                raise FileNotFoundError(f'Factor values store does not exist: {store_path}')
            report = store.verify_store(**filters)
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
                identities_df = store.list_identities(**filters)
                metrics['identity_count'] = int(len(identities_df))
                identities_out = identities_df.copy()
                identities_out['created_at'] = _format_factor_timestamps(identities_out['created_at'])
                identities_out = identities_out.astype(object).where(identities_out.notna(), None)
                metrics['identities'] = identities_out.to_dict('records')
            has_row_filters = any(
                getattr(args, name) is not None
                for name in ('factor_id', 'factor_identity_hash', 'frequency', 'slot_id', 'session_view',
                             'instruments', 'start_utc', 'end_utc', 'session_start', 'session_end', 'out')
            )
            if has_row_filters or not args.list_identities:
                frame = store.read_factor_values(
                    **filters,
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
                    metrics['preview'] = preview.astype(object).where(preview.notna(), None).to_dict('records')
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

        cal_obj, slot = _factor_value_calendar(args)

        if input_path.suffix.lower() == '.parquet':
            raw_df = pd.read_parquet(input_path)
            is_full_schema = set(TABLE_COLUMNS).issubset(raw_df.columns)
        else:
            with input_path.open(encoding='utf-8-sig', newline='') as source:
                header_cols = next((row for row in csv.reader(source) if any(field.strip() for field in row)), [])
            if len(set(header_cols)) != len(header_cols):
                raise ValueError('CSV column names must be unique')
            is_full_schema = set(TABLE_COLUMNS).issubset(header_cols)
            aliases = {'timestamp': 'timestamp_utc', 'symbol': 'instrument_id'}
            if not is_full_schema and any(alias in header_cols and name in header_cols for alias, name in aliases.items()):
                raise ValueError('CSV contains both an alias and its canonical column')
            text_columns = (set(TABLE_COLUMNS) - {'factor_value'}) | set(aliases) | {'source', 'data_version'}
            raw_df = pd.read_csv(input_path, dtype=dict.fromkeys(text_columns, 'string'), float_precision='round_trip')
            if not is_full_schema:
                raw_df = raw_df.rename(columns=aliases)
                if raw_df.empty and 'factor_value' in raw_df:
                    raw_df['factor_value'] = raw_df['factor_value'].astype('float64')
            if 'timestamp_utc' in raw_df:
                raw_df['timestamp_utc'] = cli._parse_strict_utc_csv_column(raw_df['timestamp_utc'], 'timestamp_utc')
            if is_full_schema:
                raw_df['created_at'] = cli._parse_strict_utc_csv_column(raw_df['created_at'], 'created_at')

        identity_rec = None
        if is_full_schema:
            batch_df = raw_df
            if not batch_df.empty:
                records = store.list_identities(factor_identity_hash=batch_df['factor_identity_hash'].iloc[0])
                if records.empty:
                    raise ValueError('Full-schema input requires a registered factor identity')
                identity_rec = records.iloc[0].to_dict()
                identity_rec['created_at'] = identity_rec['created_at'].isoformat()
                if identity_rec['source_schema_version'] != MARKET_FACTOR_IDENTITY_SCHEMA_VERSION:
                    raise ValueError('Historical v1 identities are read-only; supply a v2 identity')
                slot['adjustment_policy'] = args.adjustment_policy or args.adjust
                registered = {field: identity_rec[field] for field in slot}
                supplied = registered | {field: value for field, value in slot.items() if value is not None}
                if _normalize_slot_fields(**supplied) != registered:
                    raise ValueError('Requested slot metadata does not match the registered factor identity')
                if args.factor_source_sha256 is not None:
                    factor_class = json.loads(identity_rec['identity_payload_json'])['factor_class']
                    if factor_class['source_sha256'] != [args.factor_source_sha256]:
                        raise ValueError('--factor-source-sha256 does not match the registered factor definition')
        else:
            raw_df = validate_factor_frame(raw_df)
            if not args.factor_id or not args.frequency or not args.analysis_mode:
                raise ValueError(
                    '--factor-id, --frequency and --analysis-mode are required '
                    'when input does not contain full market_factor_values schema'
                )
            explicit_slot = any(value is not None for value in slot.values())
            if explicit_slot:
                slot['adjustment_policy'] = slot['adjustment_policy'] or args.adjust
                if any(value is None for value in slot.values()):
                    raise ValueError(
                        'Provide --slot-config or both --slot-id and --slot-config-hash, '
                        'with --slot-data-fingerprint and an adjustment policy'
                    )
                if args.factor_source_sha256 is None:
                    raise ValueError('Slot imports require --factor-source-sha256 with a stable factor definition hash')
            elif args.source_build_id is None:
                raise ValueError('Legacy writes require --source-build-id. Slot writes require --slot-data-fingerprint.')
            parsed_freq = Frequency.parse(args.frequency)
            if not parsed_freq.is_valid or parsed_freq.canonical is None:
                raise ValueError(f'Invalid frequency: {args.frequency!r}')
            canonical_freq = parsed_freq.canonical
            strategy = BarSchedule(
                cal_obj, canonical_freq, view=args.session_view,
                bucket_strategy=args.bucket_strategy or ('session' if parsed_freq.kind == 'session' else None),
            ).bucket_strategy
            params = json.loads(args.params_json) if args.params_json else {}
            code_ver, code_ver_src = resolve_code_version()
            source_sha = args.factor_source_sha256
            if source_sha is None:
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
                continuous_adjust=None if explicit_slot else args.adjust,
                code_version=code_ver,
                **slot,
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
                owner = cal_obj.session_at(ts, args.session_view, frequency=parsed_freq)
                if owner is None:
                    raise ValueError(f'timestamp_utc {ts} is outside the {canonical_freq} {args.session_view} calendar')
                sessions.append(owner.session_date)
            if 'session_id' in raw_df and not raw_df['session_id'].astype('string').fillna('').eq(sessions).all():
                raise ValueError('session_id mismatch with the supplied calendar')
            source_build_id = args.source_build_id
            if source_build_id is None:
                fingerprint = payload['slot_data_fingerprint']
                source_build_id = (fingerprint if isinstance(fingerprint, str)
                                   else hashlib.sha256(_canonical_json(fingerprint).encode()).hexdigest())
            batch_df = raw_df.assign(
                factor_id=args.factor_id,
                factor_identity_hash=fhash,
                frequency=canonical_freq,
                session_view=args.session_view,
                session_id=sessions,
                source_build_id=source_build_id,
                calendar_hash=cal_obj.calendar_hash,
                code_version=code_ver,
                created_at=pd.Timestamp(now_dt),
                batch_id=uuid.uuid4().hex,
            ).loc[:, list(TABLE_COLUMNS)]

        receipt = store.append_factor_values(
            batch_df,
            identity_record=None if is_full_schema else identity_rec,
            calendar=cal_obj,
        )
        result = cli._success_result(**common, mode='local', metrics={
            **receipt.to_dict(), 'identity_record': identity_rec,
        })
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
    except (ValueError, TypeError, LookupError, OSError, json.JSONDecodeError, csv.Error) as exc:
        result = cli._failure_result(
            **common,
            mode='local',
            reason='factor_store_precondition_failed',
            exit_code=cli.EXIT_PRECONDITION,
            metrics={'message': str(exc)},
        )
    return cli._complete_result(result, json_out=args.json_out)
