"""Bind Stage 2 market inputs to the historical A-share FactorBase panel contract."""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

import duckdb
import pandas as pd

from slots.jq_ashare_local.aux import AShareAuxFeatures
from slots.jq_ashare_local.feed import AShareBarFeed
from slots.jq_ashare_local.microstructure import AShareMicrostructure
from slots.jq_ashare_local.panel import ASharePanelReader, LEGACY_CALENDAR_ID, require_ashare_panel_profile
from slots.jq_ashare_local.universe import AShareUniverse
from utils.bars.catalog import table_exists
from utils.data_store import connect_input, digest
from utils.market_db import MarketDbProfile, MarketDbProfileError
from utils.market_db.errors import MARKET_DB_PROFILE_MISSING
from utils.market_db.storage import _require_same_hash, is_file_backed_market_db
from utils.market_db.validation import (
    calendar_key, market_db_calendar_bounds, require_market_db_profile, validate_market_db_provider,
)
from utils.pit import normalize_price_money
from utils.session_calendar import SessionCalendar
from utils.slots import MarketSlot, SlotRegistry
from utils.time_axis import Frequency


if TYPE_CHECKING:
    from utils.market_factor import FactorWriteReceipt, MarketFactorValueStore


def save_factor_values(factor, store: "MarketFactorValueStore", **options: Any) -> "FactorWriteReceipt":
    """Persist factor output in the Slot-backed factor value store."""
    from utils.market_factor import save_factor_values as persist

    return persist(factor, store, **options)


@lru_cache(maxsize=64)
def _ashare_calendar(calendar_id: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> SessionCalendar:
    return SessionCalendar(calendar_id, start, end)


class AShareFactorCalendar:
    """Delay calendar construction for factors that only consume legacy panels."""

    def __init__(self, start: str, end: str, override=None, *,
                 load_profile: Callable[[], MarketDbProfile | None]):
        self.start, self.end, self.override = start, end, override
        self.load_profile = load_profile

    @cached_property
    def default(self) -> SessionCalendar:
        if isinstance(self.override, SessionCalendar):
            return self.override
        profile = self.load_profile()
        if profile is not None:
            validate_market_db_provider(profile, 'jq_ashare_local', domain='cn_equity')
            left, right = market_db_calendar_bounds(profile, self.start, self.end)
            return _ashare_calendar(calendar_key(profile.calendar_id, market_domain='cn_equity'), left, right)
        return _ashare_calendar(LEGACY_CALENDAR_ID, self.start, self.end)

    @property
    def calendar_hash(self) -> str:
        return self.default.calendar_hash

    def validate_coverage(self, start, end):
        return self.default.validate_coverage(start, end)

    def sessions(self, start=None, end=None):
        return self.default.sessions(start, end)

    def session_at(self, timestamp, view='ETH', *, frequency=None):
        return self.default.session_at(timestamp, view, frequency=frequency)

    def bar_grid(self, frequency, start=None, end=None, *, view='ETH', bucket_strategy=None):
        return self.default.bar_grid(frequency, start, end, view=view, bucket_strategy=bucket_strategy)

    def for_instrument(self, metadata: dict) -> SessionCalendar:
        from data_pipeline.loaders.market_bar_importer import instrument_calendar

        calendar = self.override
        if isinstance(calendar, dict):
            calendar = calendar.get(metadata['instrument_id'])
        if calendar is not None:
            return calendar
        if self.override is not None:
            raise ValueError('session_calendar must supply a calendar for every instrument')
        return instrument_calendar(metadata, self.start, self.end)


class AShareFactorFeed:
    def __init__(self, connection, adapter, path, config, view, *, bindings: dict | None = None,
                 profile: MarketDbProfile | None = None):
        self.connection, self.adapter, self.path = connection, adapter, path
        self.config, self.view = config, view
        self.bindings, self.profile = bindings, profile
        self._verified_input: tuple[duckdb.DuckDBPyConnection, MarketDbProfile] | None = None

    def _check_input_profile(self, connection: duckdb.DuckDBPyConnection) -> MarketDbProfile:
        # A read-only adapter can be reused across panels or reopened on a different file.
        verified = self._verified_input
        if verified is not None and verified[0] is connection:
            return verified[1]
        profile = require_ashare_panel_profile(connection, expected=self.profile)
        self._verified_input = connection, profile
        return profile

    @contextmanager
    def _reader(self) -> Iterator[ASharePanelReader]:
        with ExitStack() as stack:
            connection = self.connection
            profile = self.profile
            if connection is None:
                if self.adapter is not None:
                    connection = self.adapter._get_connection()
                    if self.adapter.duckdb_path:
                        profile = self._check_input_profile(connection)
                elif self.path:
                    connection = stack.enter_context(connect_input(self.path, bindings=self.bindings))
                    profile = require_ashare_panel_profile(connection, expected=profile)
            with ASharePanelReader(connection, legacy_snapshot=True, config=self.reader_config,
                                   market_db_profile=profile) as reader:
                yield reader

    def calendar_profile(self) -> MarketDbProfile | None:
        if self.profile is not None or not self.path:
            return self.profile
        with self._reader() as reader:
            return reader.market_db_profile

    def factor_write_provenance(self, frequency: Frequency, instruments: list[str]) -> tuple[dict, dict]:
        """Return physical catalog entries and per-session build IDs for the bound feed."""
        with self._reader() as reader:
            catalog = reader.catalog
            entries = {}
            for instrument in instruments:
                try:
                    entries[instrument] = catalog.resolve(frequency.canonical, instrument)
                except LookupError:
                    continue
            connection = reader.connection
            builds = {}
            if connection is not None and table_exists(connection, 'bar_build_log'):
                rows = connection.execute(
                    'SELECT instrument_id, session_id, build_id FROM bar_build_log '
                    'WHERE frequency = ? AND instrument_id IN (SELECT unnest(?))',
                    [frequency.canonical, instruments],
                ).fetchall()
                builds = {(instrument, str(session)): build for instrument, session, build in rows}
            return entries, builds

    @cached_property
    def reader_config(self) -> dict:
        return (self.adapter.data_config if self.adapter is not None else self.config()).as_dict()

    def instrument_metadata(self, instruments: Sequence[str]) -> pd.DataFrame:
        with self._reader() as reader:
            if table_exists(reader.connection, 'instrument_metadata'):
                return reader.get_instrument_metadata(instruments)
            profile = require_market_db_profile(reader.connection)
            validate_market_db_provider(profile, 'jq_ashare_local', domain='cn_equity')
            bound = ASharePanelReader(reader.connection, config=self.reader_config, market_db_profile=profile)
            return bound.legacy_instrument_metadata(instruments)

    def read_execution_bars(self, instruments, frequency: Frequency, *, start_utc, end_utc):
        with self._reader() as reader:
            return reader.read(instruments, frequency.canonical, start_utc, end_utc, view=self.view)

    def read_signal_bars(self, instruments, frequency: Frequency, *, start_utc, end_utc, as_of_utc):
        # The historical Bar interface consumes the source's versioned prices as supplied.
        return self.read_execution_bars(instruments, frequency, start_utc=start_utc, end_utc=end_utc)

    def read_factor_panel(self, factor, fields, date_field, code_field, table_name,
                          additional_condition) -> pd.DataFrame:
        if not isinstance(fields, (list, tuple)):
            raise ValueError('fields must be a list or tuple of column names')
        selected = list(fields)
        if factor.pit_enabled and factor.pit_factor_column not in selected:
            selected.append(factor.pit_factor_column)
        adapter = factor._duckdb_adapter or self.adapter
        config = adapter.data_config if adapter is not None else factor._get_local_data_config()
        standard = date_field == 'date' and code_field == 'stock_code' and not additional_condition
        normalize = adapter is not None and standard
        if normalize and factor.data_frequency in ('daily', '1d') and 'money' in selected and 'paused' not in selected:
            selected.append('paused')
        check_profile = self._check_input_profile if adapter is not None and adapter.duckdb_path else None

        frame = ASharePanelReader.read_panel(
            factor.all_stocks, factor.data_frequency, factor.start_date, factor.end_date, selected,
            config=config, connection=factor.db_connection or self.connection, adapter=adapter,
            date_field=date_field, code_field=code_field, additional_condition=additional_condition,
            market_db_profile=self.profile, check_profile=check_profile,
        )
        if normalize:
            frequency = factor._frequency_route.frequency
            money_frequency = 'daily' if frequency.kind == 'session' else '1m' if frequency.minutes == 1 else '5m'
            frame = normalize_price_money(frame, source='local', frequency=money_frequency)
            if 'paused' not in fields and 'paused' in frame:
                frame = frame.drop(columns='paused')
        return frame

    @staticmethod
    def adjust_factor_panel(frame: pd.DataFrame, factor_column: str) -> pd.DataFrame:
        from utils.pit import apply_pit_adjustment

        return apply_pit_adjustment(frame, factor_column)


class AShareFactorAux:
    def __init__(self, feed, calendar):
        self.feed, self.calendar = feed, calendar

    def read_aux_features(self, grid, *, features):
        with self.feed._reader() as reader:
            return AShareAuxFeatures(
                reader.connection, self.calendar.default, market_db_profile=reader.market_db_profile,
            ).read_aux_features(grid, features=features)

    def read_factor_panel(self, factor, fields, date_field, code_field, table_name,
                          additional_condition) -> pd.DataFrame:
        adapter = factor._duckdb_adapter or self.feed.adapter
        if adapter is not None and adapter.duckdb_path and factor.all_stocks:
            self.feed._check_input_profile(adapter._get_connection())
        if adapter is not None and date_field == 'date' and code_field == 'stock_code' and not additional_condition:
            arguments = dict(stocks=factor.all_stocks, start_date=factor.start_date,
                             end_date=factor.end_date, fields=list(fields))
            if table_name == 'factor_data':
                return adapter.fetch_factor_data(**arguments)
            return adapter.fetch_table_data(table_name=table_name, **arguments)
        connection = adapter._get_connection() if adapter is not None else factor.db_connection or self.feed.connection
        config = adapter.data_config if adapter is not None else factor._get_local_data_config()
        return _read_aux_panel(factor, connection, config, fields, date_field, code_field, table_name,
                               additional_condition)


def _read_aux_panel(self, connection, config, fields, date_field, code_field, table_name, additional_condition):
    if not isinstance(connection, duckdb.DuckDBPyConnection):
        raise TypeError("Supply a DuckDB connection, duckdb_path, Parquet directory, or data binding")

    def quote(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    dataset = config.dataset(
        config.factor_table if table_name == 'factor_data' else table_name)
    physical_table = dataset.table
    if date_field == 'date':
        date_field = dataset.date_column
    if code_field == 'stock_code':
        code_field = dataset.stock_column
    if not isinstance(fields, (list, tuple)):
        raise ValueError('fields must be a list or tuple of column names')
    excluded = {date_field, code_field, 'date', 'stock_code', 'code'}
    selected = list(dict.fromkeys(f for f in fields if f not in excluded))
    from utils.data_access_config import Dataset
    from utils.data_store import table_name as quote_table
    selected = list(Dataset(physical_table, date_field, code_field).fields(selected))
    if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
        raise ValueError('start_date must not be after end_date')
    if not self.all_stocks:
        frame = pd.DataFrame(columns=['date', 'code', *selected])
        frame['date'] = pd.to_datetime(frame['date'])
        return frame
    columns = [f'{quote(date_field)} AS date', f'{quote(code_field)} AS code'] + [quote(f) for f in selected]
    placeholders = ', '.join('?' for _ in self.all_stocks)
    end_exclusive = (pd.Timestamp(self.end_date) + pd.Timedelta(days=1)).to_pydatetime()
    sql = f"""SELECT {', '.join(columns)} FROM {quote_table(physical_table)}
        WHERE CAST({quote(date_field)} AS TIMESTAMP) >= ?
          AND CAST({quote(date_field)} AS TIMESTAMP) < ?
          AND {quote(code_field)} IN ({placeholders})
          {additional_condition}
        ORDER BY {quote(code_field)}, {quote(date_field)}"""
    frame = connection.execute(sql, [pd.Timestamp(self.start_date).to_pydatetime(), end_exclusive, *self.all_stocks]).fetchdf()
    frame['date'] = pd.to_datetime(frame['date'])
    return frame


class AShareFactorUniverse:
    def __init__(self, feed, calendar, native=None):
        self.feed, self.calendar, self.native = feed, calendar, native

    def instrument_metadata(self, instruments):
        return self.feed.instrument_metadata(instruments)

    def instrument_spec(self, instrument_id):
        if self.native is not None:
            return self.native.instrument_spec(instrument_id)
        with self.feed._reader() as reader:
            return AShareUniverse(reader.connection, self.calendar.default).instrument_spec(instrument_id)

    def resolve_universe(self, universe_id, *, as_of_utc):
        if self.native is not None:
            return self.native.resolve_universe(universe_id, as_of_utc=as_of_utc)
        with self.feed._reader() as reader:
            return AShareUniverse(reader.connection, self.calendar.default).resolve_universe(
                universe_id, as_of_utc=as_of_utc)

    def resolve_active_contract(self, instrument_id, *, as_of_utc):
        if self.native is not None:
            return self.native.resolve_active_contract(instrument_id, as_of_utc=as_of_utc)
        return instrument_id

    def period_pools(self, factor, manager_factory):
        if not factor.stock_filters:
            raise ValueError("使用动态股票池时必须提供stock_filters")
        factor.dynamic_pool_manager = manager_factory(
            factor.db_connection, fundamental_table=factor.fundamental_table,
            price_table=factor.price_table, db_backend=factor.db_backend, duckdb_path=factor.duckdb_path,
            parquet_dir=factor.parquet_dir, use_duckdb_cache=factor.use_duckdb_cache,
            duckdb_cache_size_mb=factor.duckdb_cache_size_mb, data_bindings=factor.data_bindings,
            source_policy=factor.source_policy, data_config_path=factor.data_config_path,
        )
        manager = factor.dynamic_pool_manager
        connection = getattr(manager, '_duckdb_conn', None)
        if connection is not None and is_file_backed_market_db(connection):
            require_ashare_panel_profile(connection, expected=self.feed.profile)
        if not manager.validate_filters(factor.stock_filters):
            raise ValueError("股票筛选条件或数据源验证失败，请检查字段、指数条件和数据源")
        return manager.get_stock_pools_for_period(
            factor.start_date, factor.end_date, factor.stock_filters, factor.max_stocks_per_month,
            factor.sort_by, factor.ascending, rebalance_period_months=factor.rebalance_period_months,
            rebalance_schedule=factor.rebalance_schedule, schedule=factor.pool_schedule,
        )


class AShareFactorMicrostructure:
    execution_profile = AShareMicrostructure(None, None).execution_profile

    def __init__(self, feed, calendar):
        self.feed, self.calendar = feed, calendar

    def tradability_mask(self, grid, *, instrument_id):
        with self.feed._reader() as reader:
            return AShareMicrostructure(reader.connection, self.calendar.default).tradability_mask(
                grid, instrument_id=instrument_id)


def resolve_slot(slot: MarketSlot | None, slot_id: str | None,
                 registry: SlotRegistry | None) -> MarketSlot | None:
    if slot is not None and slot_id is not None:
        raise ValueError('Use only slot or slot_id')
    if slot is not None and not isinstance(slot, MarketSlot):
        raise TypeError('slot must be a MarketSlot')
    if slot_id is not None:
        if registry is None:
            raise ValueError('slot_id requires a slot_registry')
        return registry.get(slot_id)
    return slot


def uses_ashare_panel(slot: MarketSlot | None) -> bool:
    return slot is None or (
        isinstance(slot.bar_feed, (AShareBarFeed, AShareFactorFeed))
        and (slot.market_db_profile is None or slot.market_db_profile.market_domain == 'cn_equity')
    )


def bind_ashare_slot(factor, native: MarketSlot | None) -> MarketSlot:
    from functools import partial
    from utils.data_access_config import load_data_access_config

    if native is not None and isinstance(native.bar_feed, AShareFactorFeed):
        return native
    profile = native.market_db_profile if native is not None else None
    connection = factor.db_connection
    if connection is not None:
        try:
            actual = require_market_db_profile(connection)
        except MarketDbProfileError as exc:
            if profile is not None or exc.code != MARKET_DB_PROFILE_MISSING:
                raise
        else:
            if profile is not None:
                _require_same_hash(actual, profile, 'Factor database disagrees with its Slot')
            if native is not None or factor.session_calendar is None:
                validate_market_db_provider(actual, 'jq_ashare_local', domain='cn_equity')
            profile = actual
    override = factor.session_calendar
    if override is None and native is not None:
        override = native.calendar
    feed = AShareFactorFeed(factor.db_connection, factor._duckdb_adapter, factor.duckdb_path,
                           partial(load_data_access_config, factor.data_config_path), factor.session_view,
                           bindings=factor.data_bindings, profile=profile)
    calendar = AShareFactorCalendar(factor.start_date, factor.end_date, override, load_profile=feed.calendar_profile)
    universe = AShareFactorUniverse(feed, calendar, native.universe if native else None)
    aux = native.aux_features if native is not None else None
    if aux is None or (type(aux) is AShareAuxFeatures
                       and aux.connection is native.bar_feed.connection and aux.calendar is native.calendar):
        aux = AShareFactorAux(feed, calendar)
    if native is not None:
        return replace(native, calendar=calendar, bar_feed=feed, universe=universe, aux_features=aux)
    return MarketSlot(
        'jq_ashare_local', calendar, feed, universe, aux, AShareFactorMicrostructure(feed, calendar),
        market_db_profile=profile,
    )


def adjust_panel(factor, frame: pd.DataFrame) -> pd.DataFrame:
    if not factor.pit_enabled or frame.empty:
        return frame
    slot = getattr(factor, 'slot', None)
    feed = slot.bar_feed if slot is not None else AShareFactorFeed
    adjust = getattr(feed, 'adjust_factor_panel', None)
    # Other slots provide already-adjusted signal Bars at their declared anchor.
    return frame if adjust is None else adjust(frame, factor.pit_factor_column)


def _instrument_config(spec) -> dict:
    return {key: value.isoformat() if key in ('listing_date', 'expiry_date') and value else value
            for key, value in asdict(spec).items()}


def config_identity(slot: MarketSlot) -> str | None:
    """Use declared identity, or stable configuration of the bundled adapter."""
    declared = getattr(slot, 'slot_config_hash', None)
    if declared is not None:
        return declared

    feed = slot.bar_feed
    profile = asdict(slot.microstructure.execution_profile)
    if isinstance(feed, AShareFactorFeed):
        if type(slot.aux_features) is not AShareFactorAux:
            return None
        return digest({'provider': 'jq_ashare_local', 'execution_profile': profile})

    from slots.declarative.slot import (
        DeclarativeAuxFeatures, DeclarativeBarFeed, DeclarativeMicrostructure, DeclarativeUniverse,
    )
    from slots.futures.slot import _FuturesData
    from utils.slots.schema import _config_value
    if type(feed) is DeclarativeBarFeed:
        if (type(slot.universe) is not DeclarativeUniverse
                or type(slot.aux_features) is not DeclarativeAuxFeatures
                or type(slot.microstructure) is not DeclarativeMicrostructure):
            return None

        def source_config(source):
            if source is None:
                return None
            return {'provider': source.provider.provider, 'base_dir': str(source.base_dir),
                    'options': _config_value(source.provider.options, freeze=False)}

        return digest({
            'provider': 'declarative', 'execution_profile': profile,
            'bar_feed': source_config(feed.source), 'universe': source_config(slot.universe.membership),
            'aux': source_config(slot.aux_features.source), 'micro': source_config(slot.microstructure.source),
            'specs': {key: _instrument_config(spec) for key, spec in slot.universe.specs.items()},
            'universes': dict(slot.universe.static),
        })
    if type(feed) is _FuturesData:
        if not all(component is feed for component in (slot.universe, slot.aux_features, slot.microstructure)):
            return None
        from dataclasses import is_dataclass

        if not is_dataclass(feed.policy):
            return None
        return digest({'provider': 'futures', 'root': feed.root, 'view': feed.view,
                       'adjustment': feed.adjustment, 'exclude_delivery_month': feed.exclude_delivery_month,
                       'policy': type(feed.policy).__qualname__, 'parameters': asdict(feed.policy),
                       'specs': {key: _instrument_config(spec) for key, spec in feed.specs.items()},
                       'execution_profile': profile})
    return None
