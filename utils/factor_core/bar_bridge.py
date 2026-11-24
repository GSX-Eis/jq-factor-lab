"""Frequency routing and the Session-based Bar interface for FactorBase."""
import hashlib
from typing import Optional

import numpy as np
import pandas as pd

from utils.market_db.identity import slot_market_db_identity
from utils.data_revision import capture_factor_inputs, carry_factor_inputs, frame_digest, window_factor_input_fingerprint


def initialize_frequency_route(
    self,
    data_frequency: str,
    analysis_mode: Optional[str],
    session_view: str,
    output_frequency: Optional[str],
    label_config: Optional[dict],
    data_version: Optional[str],
    session_calendar,
    source_revision: Optional[str],
    *,
    slot=None,
    legacy_panel: bool = False,
) -> None:
    from utils.time_axis.frequency import FactorFrequencyRoute, Frequency

    if isinstance(data_frequency, Frequency):
        data_frequency = data_frequency.raw
    self.data_frequency = data_frequency
    self.analysis_mode = analysis_mode
    self.session_view = session_view
    self.output_frequency = output_frequency or data_frequency
    self.label_config = dict(label_config or {})
    self.data_version = data_version
    self.session_calendar = session_calendar
    self.source_revision = source_revision
    route = FactorFrequencyRoute.resolve(data_frequency, analysis_mode)
    if slot is not None and (not legacy_panel or not route.frequency.is_valid):
        route = FactorFrequencyRoute.resolve(data_frequency, analysis_mode, slot=slot)
        if analysis_mode is None:
            route = FactorFrequencyRoute(route.frequency, False, False)
    self._frequency_route = route
    if analysis_mode is not None and analysis_mode not in ('time_series', 'cross_sectional'):
        raise ValueError('analysis_mode must be time_series or cross_sectional')
    if self._frequency_route.explicit_interface and analysis_mode is None:
        raise ValueError('analysis_mode must be explicit for the new Bar interface')
    if session_view not in ('ETH', 'RTH'):
        raise ValueError('session_view must be ETH or RTH')
    if not self._frequency_route.bar_grid and slot is not None and (
            Frequency.parse(self.output_frequency).canonical != self._frequency_route.frequency.canonical):
        raise ValueError('output_frequency without analysis_mode must match data_frequency')
    if self._frequency_route.bar_grid:
        if _session_rollup(self):
            if analysis_mode != 'cross_sectional':
                raise ValueError('daily Session output requires cross_sectional analysis_mode')
        elif Frequency.parse(self.output_frequency).canonical != self._frequency_route.frequency.canonical:
            raise ValueError('Bar output_frequency must match data_frequency or be daily/1d')


def _session_rollup(self) -> bool:
    return (self._frequency_route.frequency.kind == 'intraday'
            and self.output_frequency in ('daily', '1d'))


def _bucket_strategy(feed, frequency, instrument):
    strategy = getattr(feed, 'bucket_strategy', None)
    # Native futures publish the bucket choice in the physical catalog.
    if getattr(getattr(feed, 'calendar', None), 'is_domestic_futures', False) and strategy is None:
        catalog = getattr(feed, 'catalog', None)
        if catalog is not None:
            contracts = list(feed.specs) if instrument == feed.root else [instrument]
            strategies = set()
            for contract in contracts:
                try:
                    entry = catalog.resolve(frequency.canonical, contract, asset_class='futures')
                except LookupError:
                    continue
                strategies.add(entry.bucket_strategy)
            if len(strategies) != 1:
                raise ValueError(f'Expected one bucket strategy for {instrument} at {frequency.canonical}')
            strategy = strategies.pop()
    return strategy


def _metadata_for(slot, instrument, calendar, as_of) -> dict:
    from dataclasses import asdict

    try:
        spec = slot.universe.instrument_spec(instrument)
    except (KeyError, ValueError):
        contract = slot.universe.resolve_active_contract(instrument, as_of_utc=as_of)
        spec = slot.universe.instrument_spec(contract)
    return {**asdict(spec), 'instrument_id': instrument, 'point_value': spec.contract_multiplier,
            'contract_expiry': spec.expiry_date, 'timezone': getattr(calendar, 'timezone', None),
            'calendar_id': getattr(calendar, 'calendar_id', None),
            'rth_reference_calendar': getattr(calendar, 'rth_reference_calendar', None)}


_BAR_KEYS = ['timestamp_utc', 'instrument_id']
_AVAILABLE = 'signal_available_at'


def _raw_coverage_attrs(coverage: dict) -> dict:
    bounds = [pair for axis in coverage.values() for pair in axis.values()]
    return dict(coverage, raw_coverage_start_utc=min((a for a, _ in bounds if pd.notna(a)), default=pd.NaT),
                raw_coverage_end_utc=max((b for _, b in bounds if pd.notna(b)), default=pd.NaT))


def _merge_raw_coverage(coverage: dict, attrs: dict) -> None:
    for name in ('raw_coverage_by_instrument', 'raw_coverage_by_key'):
        axis = coverage.setdefault(name, {})
        for key, (start, end) in attrs[name].items():
            previous_start, previous_end = axis.get(key, (pd.NaT, pd.NaT))
            axis[key] = (min((a for a in (previous_start, start) if pd.notna(a)), default=pd.NaT),
                         max((b for b in (previous_end, end) if pd.notna(b)), default=pd.NaT))


def _read_slot_bars(self, *, execution: bool = False, expected=None, as_of=None):
    """Read the requested slot and align it to its expected session clock."""
    from backtest_and_analysis.time_series.contracts import BAR_COLUMNS, PRICE_COLUMNS, validate_bar_frame

    active = getattr(self, '_slot_input_window', None)
    if expected is None and active is not None:
        expected, as_of = active
    slot = self.slot
    frequency = self._frequency_route.frequency
    view = self.session_view
    provider_metadata = getattr(slot.universe, 'instrument_metadata', None)
    records = (provider_metadata(self.all_stocks).set_index('instrument_id', drop=False).to_dict('index')
               if provider_metadata is not None and self.all_stocks else {})
    parts, grids, calendars, metadata = [], [], {}, []
    coverage = {'raw_coverage_by_instrument': {}, 'raw_coverage_by_key': {}}
    instruments = (dict.fromkeys(self.all_stocks) if expected is None else expected.instrument_id.unique())
    for instrument in instruments:
        record = records.get(instrument)
        calendar = slot.calendar.for_instrument(record) if record is not None else slot.calendar
        calendars[instrument] = calendar
        strategy = _bucket_strategy(slot.bar_feed, frequency, instrument)
        scheduled = (calendar.bar_grid(frequency, self.start_date, self.end_date, view=view,
                                       bucket_strategy=strategy) if expected is None
                     else expected.loc[expected.instrument_id.eq(instrument), ['timestamp_utc', 'session_id']])
        if view == 'RTH' and record is not None:
            bounds_grid = calendar.bar_grid(frequency, self.start_date, self.end_date,
                                            bucket_strategy=strategy)
        else:
            bounds_grid = scheduled
        grid = scheduled.assign(instrument_id=instrument)
        grids.append(grid)
        if bounds_grid.empty:
            if record is not None:
                metadata.append(record)
            continue
        start, end = bounds_grid.timestamp_utc.iloc[[0, -1]]
        cutoff = as_of
        if cutoff is None:
            session = calendar.session_at(end, 'ETH' if record is not None else view, frequency=frequency)
            cutoff = max(end, session.segments[-1][1])
        metadata.append(record if record is not None else _metadata_for(slot, instrument, calendar, cutoff))
        if execution:
            part = slot.bar_feed.read_execution_bars([instrument], frequency, start_utc=start, end_utc=end)
        else:
            part = slot.bar_feed.read_signal_bars([instrument], frequency, start_utc=start, end_utc=end,
                                                 as_of_utc=cutoff)
        for name, value in [('frequency', frequency.canonical), ('source', slot.slot_id),
                            ('data_version', self.data_version or 'unversioned')]:
            if name not in part:
                part = part.assign(**{name: value})
        raw_start, raw_end = part.timestamp_utc.min(), part.timestamp_utc.max()
        if not part.empty:
            coverage['raw_coverage_by_instrument'][instrument] = (raw_start, raw_end)
            coverage['raw_coverage_by_key'][(instrument, frequency.canonical)] = (raw_start, raw_end)
        for name in coverage:
            coverage[name].update(part.attrs.get(name, {}))
        if not part.instrument_id.eq(instrument).all():
            raise ValueError('MarketSlot returned bars for an unrequested instrument')
        if view == 'RTH' and record is None:
            part = part.loc[part.timestamp_utc.isin(grid.timestamp_utc)]
        parts.append(part.assign(timestamp_utc=part.timestamp_utc.astype('datetime64[ns, UTC]')))
    bars = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=BAR_COLUMNS).astype({'timestamp_utc': 'datetime64[ns, UTC]',
                                     **dict.fromkeys((*PRICE_COLUMNS, 'volume'), float)})
    bars = validate_bar_frame(bars)
    if not bars.frequency.eq(frequency.canonical).all():
        raise ValueError('market Bars do not match the requested frequency')
    if (not execution and self.data_version is not None
            and not bars.data_version.eq(self.data_version).all()):
        raise ValueError('market Bars do not match the requested data_version')
    keys = ['timestamp_utc', 'instrument_id']
    if grids:
        grid = pd.concat(grids, ignore_index=True)
    else:
        grid = bars[keys].iloc[:0].assign(session_id=pd.Series(dtype=str))
    bars = bars.drop(columns='session_id', errors='ignore').merge(
        grid, on=keys, how='left', validate='one_to_one')
    if bars.session_id.isna().any():
        raise ValueError('bars contain timestamps outside the requested Session grid')
    aligned = grid.merge(bars.drop(columns='session_id'), on=keys, how='left', validate='one_to_one',
                         indicator=True)
    aligned['observed'] = aligned.pop('_merge').eq('both')
    aligned['frequency'] = frequency.canonical
    hashes = {instrument: calendar.calendar_hash for instrument, calendar in calendars.items()}
    bars.attrs = dict(_raw_coverage_attrs(coverage), session_view=view, calendar_hash=hashes,
                      instrument_metadata=metadata)
    aligned.attrs = bars.attrs.copy()
    return bars, aligned.sort_values(['instrument_id', 'timestamp_utc']).reset_index(drop=True), calendars


def _remember_market_bars(self, bars, grid, calendars) -> None:
    metadata = pd.DataFrame(bars.attrs['instrument_metadata'])
    self.instrument_metadata = metadata
    self.market_calendars = calendars
    self.calendar_hash = bars.attrs['calendar_hash']
    self.rth_reference_calendar = dict(zip(metadata.get('instrument_id', []),
                                           metadata.get('rth_reference_calendar', [])))
    self.market_bars, self.market_bar_grid = bars, grid


def _fetch_market_bars(self) -> pd.DataFrame:
    if not self._frequency_route.bar_grid:
        raise ValueError('market Bars require an explicit Bar analysis_mode')
    if self._ashare_panel:
        _remember_market_bars(self, *_read_slot_bars(self))
    else:
        causal_slot_values(self, self._slot_clock(), remember=True, calculate=False)
    return self.market_bar_grid


def _panel_dates(self, frame):
    if self._frequency_route.frequency.kind != 'intraday':
        return pd.to_datetime(frame.session_id)
    if self._ashare_panel:
        return frame.timestamp_utc.dt.tz_convert(self.slot.calendar.default.timezone).dt.tz_localize(None)
    return frame.timestamp_utc


def read_slot_panel(self, fields, table_name: str, price: bool) -> pd.DataFrame:
    """Expose slot reads through the established date/code factor panel keys."""
    if not isinstance(fields, (list, tuple)):
        raise ValueError('fields must be a list or tuple of column names')
    selected = list(dict.fromkeys(field for field in fields if field not in {'date', 'code', 'stock_code'}))
    window = getattr(self, '_slot_input_window', None)
    if price:
        frame, _, _ = _read_slot_bars(self)
    else:
        keys = ['instrument_id', 'timestamp_utc', 'session_id']
        grid = window[0] if window is not None else self._slot_clock()
        if self._ashare_panel and self._frequency_route.frequency.kind == 'intraday':
            grid = grid.assign(timestamp_utc=grid[_AVAILABLE])
        qualified = [field if table_name == 'factor_data' else f'{table_name}.{field}' for field in selected]
        frame = self.slot.read_aux_features(grid[keys], features=qualified).rename(
            columns=dict(zip(qualified, selected)))
    if self.use_dynamic_pool and self.dynamic_pool_mode in ('raw', 'both'):
        frame = frame.loc[_pool_mask(self, frame)]
    frame = frame.assign(date=_panel_dates(self, frame), code=frame.instrument_id)
    columns = ['date', 'code', *selected]
    if frame.empty:
        return frame.reindex(columns=columns)
    return frame[columns].sort_values(['code', 'date'], kind='stable', ignore_index=True)


def _pool_mask(self, frame: pd.DataFrame) -> np.ndarray:
    members = {day: set(self.get_stock_pool_for_date(day)) for day in frame.session_id.unique()}
    return np.array([instrument in members[day] for instrument, day in
                     zip(frame.instrument_id, frame.session_id)], dtype=bool)


def session_rolling(self, data: pd.DataFrame, column: str, window: int, **kwargs) -> pd.Series:
    """Roll the expected grid returned by _fetch_market_bars, preserving gaps."""
    from backtest_and_analysis.time_series.factor_bridge import session_rolling
    return session_rolling(data, column, window, **kwargs)


def _window_cache_key(self, raw: pd.DataFrame, targets: pd.DataFrame | None, declared: dict | None) -> str | None:
    if not self.cache_dir:
        return None
    identity = raw.copy(deep=False)
    if targets is not None:
        identity.attrs['signal_keys'] = list(targets.itertuples(index=False, name=None))
    return self._get_cache_key(identity, captured_inputs={'declared_params': declared})


def _bar_window_values(self, raw, force_recalculate, targets=None):
    from backtest_and_analysis.time_series.factor_bridge import bar_factor_frame

    pool_mask = None
    if not self._ashare_panel and self.use_dynamic_pool:
        pool_mask = _pool_mask(self, raw)
    calculation_grid = raw if pool_mask is None or self.dynamic_pool_mode == 'factor' else raw.loc[pool_mask]
    self.last_raw_data = calculation_grid
    declared = capture_factor_inputs(self)
    key = _window_cache_key(self, raw, targets, declared)
    if key and not force_recalculate:
        cached = self._load_from_cache(key)
        if cached is not None:
            values = bar_factor_frame(cached)
            if targets is not None:
                values = targets.merge(values, on=_BAR_KEYS, validate='one_to_one')
            return carry_factor_inputs(values, declared)
    expected = raw[_BAR_KEYS]
    values = (expected.iloc[:0].assign(factor_value=pd.Series(dtype=float))
              if calculation_grid.empty else bar_factor_frame(self.calculate_factor(calculation_grid)))
    checked = values.merge(expected, on=_BAR_KEYS, how='left', indicator=True)
    if (checked._merge.eq('left_only') & checked.factor_value.notna()).any():
        raise ValueError('factor output contains keys outside the Session grid')
    values = expected.merge(values, on=_BAR_KEYS, how='left', validate='one_to_one')
    values.loc[~raw.observed.to_numpy(), 'factor_value'] = np.nan
    if pool_mask is not None and self.dynamic_pool_mode in ('factor', 'both'):
        values.loc[~pool_mask, 'factor_value'] = np.nan
    if targets is not None:
        values = targets.merge(values, on=_BAR_KEYS, validate='one_to_one')
    if key:
        self._save_to_cache(_wide_bar_values(values), key)
    return carry_factor_inputs(values, declared)


def _wide_bar_values(values):
    wide = values.pivot(index='timestamp_utc', columns='instrument_id', values='factor_value')
    return wide.rename_axis(index='date')


def _calculate_panel_window(self, targets, force_recalculate, input_hash) -> pd.DataFrame:
    raw = self._get_raw_data_for_factor()
    if input_hash is not None:
        input_hash.update(bytes.fromhex(frame_digest(raw)))
    self.last_raw_data = raw
    result = targets[_BAR_KEYS].copy()
    empty = result.iloc[:0].assign(factor_value=pd.Series(dtype=float))
    if raw.empty:
        return empty

    declared = capture_factor_inputs(self)
    key = _window_cache_key(self, raw, result, declared)
    cached = self._load_from_cache(key) if key and not force_recalculate else None
    if cached is None:
        if self.pit_enabled:
            raw = self._apply_pit_adjustment(raw)
        values = self._normalize_factor_dataframe(self.calculate_factor(raw), raw)
        if values.empty:
            return empty
    else:
        values = cached
    target_dates = _panel_dates(self, targets)
    panel = values.reindex(index=pd.DatetimeIndex(target_dates.unique()))
    result['factor_value'] = [
        panel.at[day, instrument] if instrument in panel.columns else np.nan
        for day, instrument in zip(target_dates, targets.instrument_id)
    ]
    if key and cached is None:
        selected = result.assign(date=target_dates).pivot(index='date', columns='instrument_id', values='factor_value')
        self._save_to_cache(selected, key)
    return carry_factor_inputs(result, declared)


def causal_slot_values(self, clock, *, targets=None, force_recalculate=False,
                       remember=False, calculate=True) -> pd.DataFrame:
    """Re-read the history as known at each emitted Bar's availability."""
    targets = clock if targets is None else targets
    bar = self._frequency_route.bar_grid
    values, diagonals, coverage = [], [], {}
    input_hash = hashlib.sha256() if calculate and self.slot.slot_data_fingerprint in (None, slot_market_db_identity(self.slot)) else None
    for cutoff, target in targets.groupby(_AVAILABLE, sort=True):
        window = clock.loc[clock[_AVAILABLE].le(cutoff)]
        self._slot_input_window = (window, cutoff)
        try:
            if not bar:
                values.append(_calculate_panel_window(self, target, force_recalculate, input_hash))
            else:
                window_bars, raw, calendars = _read_slot_bars(self)
                take = target[_BAR_KEYS]
                if remember:
                    diagonals.append(take.merge(raw, on=_BAR_KEYS, validate='one_to_one'))
                    _merge_raw_coverage(coverage, window_bars.attrs)
                if calculate:
                    if input_hash is not None:
                        input_hash.update(bytes.fromhex(frame_digest(raw)))
                    _remember_market_bars(self, window_bars, raw, calendars)
                    values.append(_bar_window_values(self, raw, force_recalculate, targets=take))
        finally:
            del self._slot_input_window
    if remember:
        if diagonals:
            grid = pd.concat(diagonals, ignore_index=True).sort_values(['instrument_id', 'timestamp_utc'])
            bars = grid.loc[grid.observed].drop(columns='observed').reset_index(drop=True)
            bars.attrs = {**window_bars.attrs, **_raw_coverage_attrs(coverage)}
            grid.attrs = bars.attrs.copy()
        else:
            bars, grid, calendars = _read_slot_bars(self, expected=clock)
        _remember_market_bars(self, bars, grid.reset_index(drop=True), calendars)

    result = (pd.concat(values, ignore_index=True) if values else
              targets.iloc[:0][_BAR_KEYS].assign(factor_value=pd.Series(dtype=float)))
    provenance = window_factor_input_fingerprint(values)
    if input_hash is not None:
        provenance['factor_input_sha256'] = input_hash.hexdigest()
    result.attrs['factor_input_fingerprint'] = provenance
    return result


def _get_market_factor_values(self, force_recalculate: bool = False) -> pd.DataFrame:
    from backtest_and_analysis.time_series.factor_bridge import incomplete_bar_keys, mask_incomplete_bars, session_daily_frame
    bar = self._frequency_route.bar_grid
    if self._ashare_panel:
        raw = self._fetch_market_bars()
        digest = frame_digest(raw) if self.slot.slot_data_fingerprint in (None, slot_market_db_identity(self.slot)) else None
        values = _bar_window_values(self, raw, force_recalculate)
        if digest is not None:
            values.attrs['factor_input_fingerprint']['factor_input_sha256'] = digest
    else:
        clock = self._slot_clock()
        values = causal_slot_values(self, clock, force_recalculate=force_recalculate, remember=bar)
    if bar and _session_rollup(self):
        grid = self.market_bar_grid
        panel = session_daily_frame(mask_incomplete_bars(values, incomplete_bar_keys(grid)), grid)
    elif bar:
        panel = _wide_bar_values(values)
        panel.attrs['incomplete_bar_keys'] = incomplete_bar_keys(self.market_bar_grid)
    else:
        keyed = clock.merge(values, on=_BAR_KEYS, how='inner', validate='one_to_one')
        panel = panel_slot_values(self, keyed)
        if self.use_dynamic_pool and self.dynamic_pool_mode in ('factor', 'both'):
            panel = self._apply_dynamic_pool_to_factor_df(panel)
    panel.attrs.update(values.attrs)
    return panel


def panel_slot_values(self, keyed: pd.DataFrame) -> pd.DataFrame:
    if self._frequency_route.bar_grid and not _session_rollup(self):
        return _wide_bar_values(keyed)
    dates = _panel_dates(self, keyed)
    if _session_rollup(self):
        from backtest_and_analysis.time_series.factor_bridge import session_daily_frame
        return session_daily_frame(keyed[[*_BAR_KEYS, 'factor_value']], keyed)
    return keyed.assign(date=dates).pivot(index='date', columns='instrument_id', values='factor_value').rename_axis(
        columns='code')


def get_bar_factor_frame(self, force_recalculate: bool = False) -> pd.DataFrame:
    """Return UTC factor keys with unavailable source-Bar values masked."""
    from backtest_and_analysis.time_series.factor_bridge import bar_factor_frame, mask_incomplete_bars
    if not self._frequency_route.bar_grid or _session_rollup(self):
        raise ValueError('get_bar_factor_frame requires minute Bar output')
    panel = self._get_market_factor_values(force_recalculate)
    values = bar_factor_frame(panel)
    aligned = self.market_bar_grid[_BAR_KEYS].merge(values, on=_BAR_KEYS, how='left', validate='one_to_one')
    return mask_incomplete_bars(aligned, panel.attrs['incomplete_bar_keys']).sort_values(_BAR_KEYS).reset_index(drop=True)


def execution_panel(self, price_type: str, period: int | None = None) -> pd.DataFrame:
    """Compute moves on each instrument's full execution clock, including absent Bars."""
    _, grid, _ = _read_slot_bars(self, execution=True)
    price = grid[price_type]
    if 'is_complete' in grid:
        price = price.where(grid.is_complete.eq(True))
    if self.use_dynamic_pool:
        price = price.where(_pool_mask(self, grid))
    values = grid.assign(factor_value=price)
    if period is not None and not _session_rollup(self):
        future = price.groupby(grid.instrument_id, sort=False).shift(-period)
        values['factor_value'] = future / price - 1
    wide = panel_slot_values(self, values)
    return wide.shift(-period) / wide - 1 if period is not None and _session_rollup(self) else wide


def get_session_daily_pricing(self) -> pd.Series:
    """Return MultiIndex(asset, date) prices bound to the Session calendar."""
    from backtest_and_analysis.time_series.factor_bridge import session_daily_pricing
    if not self._frequency_route.bar_grid:
        raise ValueError('Session pricing requires minute Bar input')
    if self._ashare_panel:
        if not hasattr(self, 'market_bar_grid'):
            self._fetch_market_bars()
        return session_daily_pricing(self.market_bar_grid, self.market_calendars)
    _, grid, calendars = _read_slot_bars(self, execution=True)
    return session_daily_pricing(grid, calendars)


def evaluate_bar_factor(self, **evaluator_options):
    """Label and evaluate using the same read snapshot and explicit mode."""
    from backtest_and_analysis.time_series.evaluator import evaluate_bars
    from backtest_and_analysis.time_series.labels import BarReturnLabeler

    factors = self.get_bar_factor_frame()
    labeler = BarReturnLabeler(self.market_calendars, view=self.session_view,
                               factor_keys_only=not self._ashare_panel, **self.label_config)
    if self._ashare_panel:
        execution = self.market_bars
        bars = execution.drop(columns='factor_value', errors='ignore').merge(
            factors, on=_BAR_KEYS, how='left', validate='many_to_one')
        bars.attrs = dict(execution.attrs)
        return evaluate_bars(bars, labeler=labeler, analysis_mode=self.analysis_mode, **evaluator_options)

    factors = factors.merge(self.market_bars[_BAR_KEYS], on=_BAR_KEYS, validate='one_to_one')
    execution, grid, _ = _read_slot_bars(self, execution=True)
    for name, missing in [('volume', 0), ('source', self.slot_id), ('data_version', 'unversioned')]:
        grid[name] = grid[name].fillna(missing)
    grid['is_complete'] = grid.observed & grid.get('is_complete', grid.observed).eq(True)
    grid.attrs = execution.attrs.copy()
    return evaluate_bars(grid, factors=factors, labeler=labeler,
                         analysis_mode=self.analysis_mode, **evaluator_options)
