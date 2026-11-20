"""Diagnostics on the expected Bar grid for each instrument."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from html import escape
from typing import Any

import pandas as pd

from backtest_and_analysis.evaluation import EvaluationEngine
from backtest_and_analysis.evaluation.types import DirectionalAccuracy, ReturnDistribution

from .evaluation_v2 import _bar_panel, _time_series_rows
from .evaluator import EXCLUSION_COLUMNS, EvaluationResult, _counts
from .labels import BarReturnLabeler


def evaluate_diagnostics(result: EvaluationResult, labeler: BarReturnLabeler, *,
                         n_quantiles: int, min_samples: int,
                         calibration_scenario: str | None = None,
                         legacy_full: bool = False) -> dict[str, Any]:
    """Reuse split diagnostics when present, and preserve the full v1 report contract."""
    advanced = result.evaluation_v2
    if advanced is not None and 'train' in advanced.splits and not legacy_full:
        horizons = []
        groups = result.per_bar_labels.groupby(list(KEY_COLUMNS), sort=False)
        for split, evaluation in advanced.splits.items():
            for row in evaluation['horizons']:
                labels = groups.get_group(tuple(row[name] for name in KEY_COLUMNS))
                clock = labels.signal_available_at
                keep = clock <= evaluation['train_end'] if split == 'train' else clock >= evaluation['oos_start']
                counts = _counts(pd.Categorical(
                    labels.loc[keep, 'exclusion_reason'], categories=EXCLUSION_COLUMNS,
                ).codes)
                horizons.append(dict(
                    **row, split=split,
                    counts={k: counts[k] for k in (*COUNT_COLUMNS[:3], 'exclusion_counts')},
                ))
        return dict(schema='time_series_diagnostics.v2', return_unit='percent',
                    quantile_scope='fitted_on_train', horizons=horizons)

    counts = {(row.instrument_id, row.frequency, row.horizon_bars): row
              for row in result.ic_by_horizon.itertuples(index=False) if row.scope == 'per_instrument'}
    rows = []
    if advanced is not None and 'full' in advanced.splits and not legacy_full:
        rows = advanced.splits['full']['horizons']
    else:
        for (instrument, frequency), labels in result.per_bar_labels.groupby(['instrument_id', 'frequency']):
            panel, expected, signal_valid, _, entries, exits = _bar_panel(labels, labeler, frequency)
            stats = EvaluationEngine('time_series', 'bar').evaluate(
                panel, n_quantiles=n_quantiles, min_samples=min_samples,
                expected_grid=expected, signal_valid=signal_valid, entry_timestamps=entries, exit_timestamps=exits,
                calibration_scenario=calibration_scenario,
            ).splits['full'].statistics
            rows.extend(_time_series_rows(stats, instrument, frequency))
    horizons = []
    for row in sorted(rows, key=lambda row: tuple(row[name] for name in KEY_COLUMNS)):
        tally = counts[tuple(row[name] for name in KEY_COLUMNS)]
        horizons.append(dict(
            **{key: row[key] for key in (*KEY_COLUMNS, 'grid_start_utc', 'grid_end_utc')},
            counts=dict(total_count=int(tally.total_count), valid_count=int(tally.valid_count),
                        excluded_count=int(tally.excluded_count), exclusion_counts=tally.exclusion_counts),
            ic=row['ic'], bootstrap={name: row['bootstrap'][name] for name in BOOTSTRAP_FIELDS},
            **{key: row[key] for key in ('interval_usability', 'direction', 'return_distribution',
                                       'buckets', 'non_overlapping_ic')},
        ))
    return dict(schema='time_series_diagnostics.v1', return_unit='percent',
                quantile_scope='full_sample_descriptive_only', horizons=horizons)


def _prefixed(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {f'{prefix}_{key}': value for key, value in values.items()}


KEY_COLUMNS = ('instrument_id', 'frequency', 'horizon_bars')
COUNT_COLUMNS = ('total_count', 'valid_count', 'excluded_count',
                 *(f'excluded_{reason}' for reason in EXCLUSION_COLUMNS))
IC_COLUMNS = ('ic_sample_count', 'ic_spearman_ic', 'ic_reason')
# The v1 CSV includes only the original interval fields.
BOOTSTRAP_FIELDS = ('grid_size', 'block_length', 'seed', 'n_boot', 'valid_replicates',
                    'min_valid_replicates', 'alpha', 'lower', 'upper', 'reason')
BOOTSTRAP_COLUMNS = tuple(f'bootstrap_{name}' for name in BOOTSTRAP_FIELDS)
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
        purge = {}
        if 'split' in horizon:
            key['split'] = horizon['split']
            purge['split_purged_count'] = horizon['split_purged_count']
        count = horizon['counts']
        rows.append(dict(
            **key, scope='overall', **{name: count[name] for name in COUNT_COLUMNS[:3]},
            **_prefixed('excluded', count['exclusion_counts']), **_prefixed('ic', horizon['ic']),
            **_prefixed('bootstrap', {name: horizon['bootstrap'][name] for name in BOOTSTRAP_FIELDS}),
            **purge, **_prefixed('direction', horizon['direction']),
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
    split = report['schema'] == 'time_series_diagnostics.v2'
    purge = ('split_purged_count',) if split else ()
    lines = [
        '# Time-series diagnostics report', '',
        f"Schema `{report['schema']}`. Returns are in percent.",
        ('Quantile buckets use the frozen training fit. Label counts precede split purging.' if split
         else 'Full-sample quantile buckets are descriptive.'), '',
    ]
    sections = [
        ('Counts and Spearman IC', ('overall',), (*COUNT_COLUMNS, *purge, *IC_COLUMNS)),
        ('Bootstrap Spearman intervals', ('overall',), BOOTSTRAP_COLUMNS),
        ('Direction accuracy', ('overall', 'bucket'), ('scope', 'index', *DIRECTION_COLUMNS)),
        ('Conditional return distribution', ('overall', 'bucket'), ('scope', 'index', *RETURN_COLUMNS)),
        ('Non-overlapping offsets', ('offset',), ('index', *IC_COLUMNS)),
    ]
    for title, scopes, values in sections:
        columns = (*(('split',) if split else ()), *KEY_COLUMNS, *values)
        lines.extend([f'## {title}', '', '| ' + ' | '.join(columns) + ' |',
                      '| ' + ' | '.join('---' for _ in columns) + ' |'])
        for row in rows:
            if row['scope'] in scopes:
                lines.append('| ' + ' | '.join(_markdown_cell(row.get(name)) for name in columns) + ' |')
        lines.append('')
        if values == BOOTSTRAP_COLUMNS:
            for horizon in report['horizons']:
                if (usability := horizon.get('interval_usability')) is None:
                    continue
                instrument, frequency, bars = (_markdown_cell(horizon[name]) for name in KEY_COLUMNS)
                descriptive = 'yes' if usability['descriptive_stats_usable'] else 'no'
                significance = 'yes' if usability['significance_interpretation_usable'] else 'no'
                prefix = f"{horizon['split']} · " if split else ''
                lines.append(
                    f"- {prefix}{instrument} {frequency}, horizon {bars} · "
                    f"Calibration **{usability['calibration_status']}**. "
                    f"Descriptive statistics usable **{descriptive}**. "
                    f"Significance interpretation usable **{significance}**."
                )
            lines.append('')
    return '\n'.join(lines)
