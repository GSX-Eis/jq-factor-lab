"""Calendar-only rebalance scheduling; selection never branches on frequency."""
from copy import deepcopy
import json
from numbers import Integral
from pathlib import Path

import pandas as pd
from utils.trading_calendar import CalendarCoverageError

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'stock_pool.json'


def load_pool_policy(path=None):
    policy = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    selection = policy['selection']
    if selection['missing_previous'] not in {'same_day', 'raise'}:
        raise ValueError('selection.missing_previous must be same_day or raise')
    for key in ('fundamental_prefixes', 'fundamental_fields'):
        if not isinstance(selection[key], list) or not all(isinstance(v, str) and v for v in selection[key]):
            raise ValueError(f'selection.{key} must be a list of nonempty strings')
    aliases = selection['state_field_aliases']
    if not isinstance(aliases, dict) or not all(isinstance(k, str) and k and isinstance(v, str) and v for k,v in aliases.items()):
        raise ValueError('selection.state_field_aliases must map field names')
    # Existing custom policy files inherit only the newly introduced source rules.
    if 'source_priority' not in selection:
        selection['source_priority'] = json.loads(DEFAULT_CONFIG.read_text())['selection']['source_priority']
    priorities = selection['source_priority']
    if not isinstance(priorities, dict) or set(priorities) != {'state', 'percentile', 'sort'}:
        raise ValueError('selection.source_priority must define state, percentile and sort')
    for kind, order in priorities.items():
        if (not isinstance(order, list) or not order
                or any(not isinstance(v, str) or v not in {'fundamental', 'price'} for v in order)
                or len(set(order)) != len(order)):
            raise ValueError(f'selection.source_priority.{kind} must list distinct fundamental/price sources')
    return policy


def resolve_schedule(policy, schedule=None, *, legacy='quarterly', months=1):
    if schedule is not None:
        if not isinstance(schedule, dict):
            raise ValueError('schedule must be an object')
        unit = schedule.get('unit')
        if unit not in {'week', 'month'}:
            raise ValueError('schedule.unit must be week or month')
        spec = deepcopy(policy['schedules']['weekly' if unit == 'week' else 'monthly'])
        spec.update(boundary='complete_period', include_previous=True)
        spec.update(schedule)
    else:
        name = 'monthly' if legacy == 'quarterly' else legacy
        if name not in policy['schedules']:
            raise ValueError(f'Unknown rebalance schedule: {legacy!r}')
        spec = deepcopy(policy['schedules'][name])
        if legacy in {'monthly', 'quarterly'}:
            spec['interval'] = months
            spec['anchor'] = '2000-01-01'
    if spec.get('unit') not in ('week', 'month'):
        raise ValueError('schedule.unit must be week or month')
    allowed = {'unit','interval','session','short_period','boundary','include_previous','anchor','months'}
    if set(spec) - allowed:
        raise ValueError(f'Unknown schedule fields: {sorted(set(spec)-allowed)}')
    for key in ('interval', 'session'):
        if isinstance(spec[key], bool) or not isinstance(spec[key], Integral):
            raise ValueError(f'schedule.{key} must be an integer')
    if spec['interval'] < 1 or spec['session'] == 0:
        raise ValueError('interval must be positive and session must be nonzero')
    if spec['short_period'] not in {'last_session','skip','raise'}:
        raise ValueError('Unknown short_period policy')
    if spec['boundary'] not in {'query_window','complete_period'} or not isinstance(spec['include_previous'], bool):
        raise ValueError('Invalid schedule boundary/include_previous')
    if spec['boundary'] == 'query_window' and spec['include_previous']:
        raise ValueError('query_window cannot include a previous period')
    if spec['interval'] > 1 and not spec.get('anchor'):
        raise ValueError('An explicit anchor is required for multi-period schedules')
    if 'anchor' in spec:
        anchor = pd.Timestamp(spec['anchor'])
        if pd.isna(anchor) or anchor.tzinfo is not None or anchor != anchor.normalize():
            raise ValueError('schedule.anchor must be a local date')
    if 'months' in spec:
        selected = spec['months']
        if spec['unit'] != 'month' or not isinstance(selected, list) or not selected or any(
            isinstance(v,bool) or not isinstance(v,int) or not 1 <= v <= 12 for v in selected
        ) or len(set(selected)) != len(selected):
            raise ValueError('schedule.months must contain distinct month numbers')
    return spec


def generate_rebalance_dates(calendar, start, end, spec):
    start, end = calendar.validate_coverage(start, end)
    frequency = 'W-SUN' if spec['unit'] == 'week' else 'M'
    anchor = pd.Period(spec.get('anchor', '2000-01-03'), freq=frequency)
    window = spec['boundary'] == 'query_window'
    days = calendar.sessions(start, end) if window else calendar.sessions()
    groups = {}
    for day in days:
        period = day.to_period(frequency)
        if (period.ordinal-anchor.ordinal) % spec['interval']:
            continue
        if spec.get('months') and period.month not in spec['months']:
            continue
        groups.setdefault(period, []).append(day)
    selected = []
    incomplete = []
    rejected_previous = []
    for period, sessions in groups.items():
        if not window and (period.start_time < calendar.start or period.end_time.normalize() > calendar.end):
            incomplete.append(period)
            continue
        position = spec['session'] - 1 if spec['session'] > 0 else spec['session']
        if -len(sessions) <= position < len(sessions):
            selected.append(sessions[position])
        elif spec['short_period'] == 'last_session':
            selected.append(sessions[-1])
        elif spec['short_period'] == 'raise':
            if period.start_time <= end and period.end_time >= start:
                raise ValueError(f'Period {period} has fewer than {abs(spec["session"])} sessions')
            if period.end_time < start:
                rejected_previous.append(period)
    if any(p.start_time <= end and p.end_time >= start for p in incomplete):
        raise CalendarCoverageError('Calendar must cover each requested complete rebalance period')
    result = [day for day in selected if start <= day <= end]
    if spec['include_previous'] and (not result or result[0] > start):
        previous = [day for day in selected if day < start]
        # Do not carry an older pool across a period whose policy requires failure.
        # A later valid rebalance makes earlier rejected periods irrelevant here.
        for period in reversed(rejected_previous):
            if not previous or period.start_time > previous[-1]:
                raise ValueError(f'Period {period} has fewer than {abs(spec["session"])} sessions')
        if not previous:
            raise CalendarCoverageError('Import calendar coverage for the previous rebalance period')
        result.insert(0, previous[-1])
    return pd.DatetimeIndex(result)
