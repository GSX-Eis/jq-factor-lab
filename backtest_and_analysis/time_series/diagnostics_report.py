"""Full-sample diagnostics on the expected Bar grid for each instrument."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields
from html import escape
from typing import Any

import pandas as pd

from backtest_and_analysis.evaluation import EvaluationEngine
from backtest_and_analysis.evaluation.kernels import CorrelationResult
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.evaluation.types import BootstrapInterval, DirectionalAccuracy, ReturnDistribution

from .contracts import _numeric_column
from .evaluator import EXCLUSION_COLUMNS, EvaluationResult
from .labels import BarReturnLabeler, _grid


def _ic(result: CorrelationResult) -> dict[str, Any]:
    return dict(sample_count=int(result.counts[0]), spearman_ic=float(result.values[0]),
                reason=result.reasons[0])


def evaluate_diagnostics(result: EvaluationResult, labeler: BarReturnLabeler, *,
                         n_quantiles: int, min_samples: int) -> dict[str, Any]:
    """Reuse the published labels and verify the actual holding intervals on every phase."""
    counts = {(row.instrument_id, row.frequency, row.horizon_bars): row
              for row in result.ic_by_horizon.itertuples(index=False) if row.scope == 'per_instrument'}
    horizons = []
    for (instrument, frequency), labels in result.per_bar_labels.groupby(['instrument_id', 'frequency']):
        calendar = labeler.calendar[instrument] if isinstance(labeler.calendar, Mapping) else labeler.calendar
        first, last = labels.signal_timestamp.min(), labels.signal_timestamp.max()
        grid, _, _ = _grid(calendar, frequency, first, last, labeler.view, labeler.bucket_strategy)
        grid = grid[(grid >= first) & (grid <= last)]
        signals = labels.drop_duplicates('signal_timestamp').set_index('signal_timestamp')
        values = signals.factor_value
        signal_valid = values.notna().reindex(grid, fill_value=False).to_numpy()[:, None]
        if pd.api.types.is_integer_dtype(values.dtype):
            dtype = getattr(values.dtype, 'numpy_dtype', values.dtype)
            factor = values.reindex(grid, fill_value=0).to_numpy(dtype=dtype, na_value=0)
        else:
            factor = _numeric_column(values.reindex(grid))
        returns, valid, entries, exits = {}, {}, {}, {}
        for horizon, group in labels.groupby('horizon_bars'):
            h = int(horizon)
            aligned = group.set_index('signal_timestamp').reindex(grid)
            returns[h] = aligned.return_pct.to_numpy(dtype=float)[:, None]
            valid[h] = aligned.status.eq('valid').to_numpy(dtype=bool)[:, None]
            entries[h] = aligned.entry_timestamp.to_numpy(dtype='datetime64[ns]')[:, None]
            exits[h] = aligned.exit_timestamp.to_numpy(dtype='datetime64[ns]')[:, None]
        assets = pd.Index([instrument])
        panel = EvaluationPanel(grid, assets, factor[:, None], returns, valid,
                                pd.MultiIndex.from_product([grid, assets]), pd.Index(['factor', *returns]))
        stats = EvaluationEngine('time_series', 'bar').evaluate(
            panel, n_quantiles=n_quantiles, min_samples=min_samples,
            signal_valid=signal_valid, entry_timestamps=entries, exit_timestamps=exits,
        ).splits['full'].statistics
        for h, horizon in stats.horizons.items():
            row = horizon.instruments[instrument]
            tally = counts[instrument, frequency, h]
            horizons.append(dict(
                instrument_id=instrument, frequency=frequency, horizon_bars=h,
                grid_start_utc=first, grid_end_utc=last,
                counts=dict(total_count=int(tally.total_count), valid_count=int(tally.valid_count),
                            excluded_count=int(tally.excluded_count), exclusion_counts=tally.exclusion_counts),
                ic=_ic(row.spearman_ic), bootstrap=asdict(row.bootstrap), direction=asdict(row.direction),
                return_distribution=asdict(row.returns),
                buckets={q: dict(direction=asdict(row.direction_by_bucket[q]), return_distribution=asdict(dist))
                         for q, dist in row.returns_by_bucket.items()},
                non_overlapping_ic={offset: _ic(ic) for offset, ic in row.non_overlapping_ic.items()},
            ))
    return dict(schema='time_series_diagnostics.v1', return_unit='percent',
                quantile_scope='full_sample_descriptive_only', horizons=horizons)


def _prefixed(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {f'{prefix}_{key}': value for key, value in values.items()}


KEY_COLUMNS = ('instrument_id', 'frequency', 'horizon_bars')
COUNT_COLUMNS = ('total_count', 'valid_count', 'excluded_count',
                 *(f'excluded_{reason}' for reason in EXCLUSION_COLUMNS))
IC_COLUMNS = ('ic_sample_count', 'ic_spearman_ic', 'ic_reason')
BOOTSTRAP_COLUMNS = tuple(f'bootstrap_{field.name}' for field in fields(BootstrapInterval))
DIRECTION_COLUMNS = tuple(f'direction_{field.name}' for field in fields(DirectionalAccuracy))
RETURN_COLUMNS = tuple(f'return_{field.name}' for field in fields(ReturnDistribution))
DIAGNOSTICS_CSV_COLUMNS = (
    *KEY_COLUMNS, 'scope', 'index', *COUNT_COLUMNS, *IC_COLUMNS,
    *BOOTSTRAP_COLUMNS, *DIRECTION_COLUMNS, *RETURN_COLUMNS,
)


def diagnostics_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten the same JSON rows for CSV and Markdown publication."""
    rows = []
    for horizon in report['horizons']:
        key = {name: horizon[name] for name in KEY_COLUMNS}
        count = horizon['counts']
        rows.append(dict(
            **key, scope='overall', **{name: count[name] for name in COUNT_COLUMNS[:3]},
            **_prefixed('excluded', count['exclusion_counts']), **_prefixed('ic', horizon['ic']),
            **_prefixed('bootstrap', horizon['bootstrap']), **_prefixed('direction', horizon['direction']),
            **_prefixed('return', horizon['return_distribution']),
        ))
        for q, bucket in horizon['buckets'].items():
            rows.append(dict(**key, scope='bucket', index=q, **_prefixed('direction', bucket['direction']),
                             **_prefixed('return', bucket['return_distribution'])))
        for offset, ic in horizon['non_overlapping_ic'].items():
            rows.append(dict(**key, scope='offset', index=offset, **_prefixed('ic', ic)))
    return rows


def _markdown_cell(value: Any) -> str:
    if value is None:
        return '—'
    if isinstance(value, float):
        return format(value, '.8g')
    return escape(str(value)).replace('|', '&#124;').replace('\r', ' ').replace('\n', ' ')


def render_diagnostics_markdown(report: Mapping[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        '# Time-series diagnostics report', '',
        f"Schema `{report['schema']}`. Returns are in percent.",
        'Full-sample quantile buckets are descriptive.', '',
    ]
    sections = [
        ('Counts and Spearman IC', ('overall',), (*COUNT_COLUMNS, *IC_COLUMNS)),
        ('Bootstrap Spearman intervals', ('overall',), BOOTSTRAP_COLUMNS),
        ('Direction accuracy', ('overall', 'bucket'), ('scope', 'index', *DIRECTION_COLUMNS)),
        ('Conditional return distribution', ('overall', 'bucket'), ('scope', 'index', *RETURN_COLUMNS)),
        ('Non-overlapping offsets', ('offset',), ('index', *IC_COLUMNS)),
    ]
    for title, scopes, values in sections:
        columns = (*KEY_COLUMNS, *values)
        lines.extend([f'## {title}', '', '| ' + ' | '.join(columns) + ' |',
                      '| ' + ' | '.join('---' for _ in columns) + ' |'])
        for row in rows:
            if row['scope'] in scopes:
                lines.append('| ' + ' | '.join(_markdown_cell(row.get(name)) for name in columns) + ' |')
        lines.append('')
    return '\n'.join(lines)
