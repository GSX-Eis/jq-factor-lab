"""Reproduce the NERE-828 synthetic NQ 1min acceptance report."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_and_analysis.evaluation import EvaluationEngine
from backtest_and_analysis.evaluation.kernels import spearman
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.time_series import BarReturnLabeler
from backtest_and_analysis.time_series.artifacts import _json_value
from backtest_and_analysis.time_series.contracts import EXCLUSION_PRIORITY
from utils.code_version import resolve_code_version
from utils.session_calendar import SessionCalendar
from utils.time_axis import BarSchedule


CONFIG = {
    'start_date': '2020-01-01',
    'end_date': '2026-12-31',
    'calendar': 'CME Globex Equity',
    'rth_reference_calendar': 'NYSE',
    'frequency': '1min',
    'instrument_id': 'NQ_SYNTHETIC',
    'boundary_policy': 'same_session',
    'entry_offset_bars': 1,
    'horizons': [1, 5, 15, 60],
    'split': {'train_end': '2025-12-31T23:59:59.999999999Z', 'oos_start': '2026-01-01T00:00:00Z'},
    'quantile_scheme': 'fitted_on_train',
    'n_quantiles': 5,
    'standardization': 'zscore',
    'direction_rule': 'sign of purged training horizon-1 Spearman IC, +1 on a tie',
    'synthetic_seed': 791,
    'bootstrap': dict(block_length=240, seed=828, n_boot=99, min_valid_replicates=90, alpha=0.05),
}


def bar_grid(calendar: SessionCalendar, view: str) -> pd.DatetimeIndex:
    schedule = BarSchedule(calendar, '1min', view=view)
    grids = [pd.DatetimeIndex(schedule.for_session(day).start_utc) for day in calendar.sessions()]
    return grids[0].append(grids[1:])


def synthetic_bars(calendar: SessionCalendar) -> pd.DataFrame:
    times = bar_grid(calendar, 'ETH')
    rng = np.random.default_rng(CONFIG['synthetic_seed'])
    draws = rng.integers(-8, 9, len(times), dtype=np.int16)
    change = draws + np.r_[np.int16(0), draws[:-1]]
    # One tick is 0.25 points. The signal is known by the current Bar close.
    close = (50_000 + np.cumsum(change)) / 4
    opening = np.r_[12_500.0, close[:-1]]
    return pd.DataFrame(dict(
        timestamp_utc=times, instrument_id=CONFIG['instrument_id'], frequency='1min',
        open=opening, high=np.maximum(opening, close) + 0.25, low=np.minimum(opening, close) - 0.25,
        close=close, volume=1, factor_value=change.astype(float),
        source='SYNTHETIC', data_version='SYNTHETIC_NERE828_v1',
    ))


def input_identity(bars: pd.DataFrame) -> dict:
    digest = hashlib.sha256()
    digest.update(bars.timestamp_utc.dt.as_unit('ns').astype('int64').to_numpy(dtype='<i8').tobytes())
    columns = ['open', 'high', 'low', 'close', 'volume', 'factor_value']
    for name in columns:
        digest.update(bars[name].to_numpy(dtype='<f8').tobytes())
    return dict(
        source='SYNTHETIC', data_version=bars.data_version.iloc[0],
        real_market_data_used=False, instrument_id=bars.instrument_id.iloc[0], frequency='1min',
        rows=len(bars), first_open_utc=bars.timestamp_utc.iloc[0].isoformat(),
        last_open_utc=bars.timestamp_utc.iloc[-1].isoformat(),
        hash_encoding=['timestamp_utc:int64_le_ns', *(f'{name}:float64_le' for name in columns)],
        sha256=digest.hexdigest(),
    )


def correlation(result) -> dict:
    return dict(sample_count=int(result.counts[0]), spearman_ic=float(result.values[0]),
                reason=result.reasons[0])


def evaluate_view(bars: pd.DataFrame, calendar: SessionCalendar, view: str) -> dict:
    times = bar_grid(calendar, view)
    selected = bars.set_index('timestamp_utc').reindex(times)
    if selected.instrument_id.isna().any():
        raise ValueError(f'Synthetic fixture requires every expected {view} Bar')
    asset = CONFIG['instrument_id']
    frame = pd.DataFrame({'factor': selected.factor_value.to_numpy()},
                         index=pd.MultiIndex.from_product([times, [asset]]))
    del selected

    entry, exits, reasons = {}, {}, {}
    # Keep one long label table at a time. The engine receives one panel with four horizons.
    for h in CONFIG['horizons']:
        labels = BarReturnLabeler(
            calendar, view=view, horizons=h, entry_offset_bars=CONFIG['entry_offset_bars'],
            boundary_policy=CONFIG['boundary_policy'], include_scheduled_exit=True,
        ).label(bars)
        if not times.equals(pd.DatetimeIndex(labels.signal_timestamp)):
            raise ValueError(f'Labeler and expected {view} grid disagree')
        available = pd.DatetimeIndex(labels.signal_available_at)
        frame[h] = labels.return_pct.to_numpy()
        entry[h] = labels.entry_timestamp.to_numpy(dtype='datetime64[ns]')[:, None]
        exits[h] = labels.scheduled_exit_timestamp.to_numpy(dtype='datetime64[ns]')[:, None]
        reasons[h] = pd.Categorical(labels.exclusion_reason, categories=list(EXCLUSION_PRIORITY))
        del labels

    panel = EvaluationPanel.from_long(frame, datetimes=times)
    del frame
    train = available <= pd.Timestamp(CONFIG['split']['train_end'])
    oos = available >= pd.Timestamp(CONFIG['split']['oos_start'])
    first_oos = available[oos][0]
    train_pairs = panel.valid[1][train] & (exits[1][train] <= first_oos.to_datetime64())
    direction_ic = spearman(panel.factor[train], panel.returns[1][train], valid=train_pairs,
                            axis=0, min_samples=3)
    if not np.isfinite(direction_ic.values[0]):
        raise ValueError(f'Cannot determine the training direction for {view}')
    direction = -1 if direction_ic.values[0] < 0 else 1
    result = EvaluationEngine('time_series', 'bar').evaluate(
        panel, split=CONFIG['split'], quantile_scheme=CONFIG['quantile_scheme'],
        signal_available_at=available, entry_timestamps=entry, exit_timestamps=exits,
        expected_grid=np.ones_like(panel.factor, dtype=bool), n_quantiles=CONFIG['n_quantiles'],
        direction=direction, standardization=CONFIG['standardization'], **CONFIG['bootstrap'],
    )

    splits = {}
    for name, rows in [('train', train), ('oos', oos)]:
        split = result.splits[name]
        horizons = {}
        for h, horizon in split.statistics.horizons.items():
            row = horizon.instruments[asset]
            counts = {reason.value: int(count) for reason, count in reasons[h][rows].value_counts().items()}
            valid = int(split.valid[h].sum())
            total = int(rows.sum())
            purge_from_valid = int(panel.valid[h][rows].sum()) - valid
            phases = {phase: correlation(ic) for phase, ic in row.non_overlapping_ic.items()}
            if (valid + sum(counts.values()) + purge_from_valid != total
                    or row.ic.counts[0] != valid
                    or len(phases) != h
                    or sum(phase['sample_count'] for phase in phases.values()) != valid):
                raise ValueError(f'Count conservation failed for {view}/{name}/h={h}')
            horizons[h] = dict(
                counts=dict(total=total, valid=valid, excluded=total - valid,
                            label_exclusions=counts, split_purged_from_valid=purge_from_valid,
                            split_purged_count=split.split_purged_count[h]),
                ic=correlation(row.ic), bootstrap=asdict(row.bootstrap),
                direction=asdict(row.direction),
                direction_by_bucket={q: asdict(x) for q, x in row.direction_by_bucket.items()},
                return_distribution=asdict(row.returns),
                return_distribution_by_bucket={q: asdict(x) for q, x in row.returns_by_bucket.items()},
                non_overlapping_ic=phases,
            )
        signals = available[rows]
        splits[name] = dict(
            first_signal_available_at=signals[0].isoformat(), last_signal_available_at=signals[-1].isoformat(),
            grid_rows=len(signals), calendar_year_counts=signals.year.value_counts().sort_index().to_dict(),
            quantile_scheme=split.quantile_scheme, fit_identity=split.fit_identity,
            intervals_verified=split.statistics.intervals_verified, horizons=horizons,
        )
    if splits['train']['fit_identity'] != splits['oos']['fit_identity']:
        raise ValueError(f'Training and OOS fit identities disagree for {view}')
    return dict(calendar_hash=calendar.calendar_hash, first_oos_signal=first_oos.isoformat(),
                direction_fit=correlation(direction_ic), **splits)


def render_markdown(report: dict) -> str:
    source = report['data_identity']
    lines = [
        '# NQ 1min 时序验收', '', '## 数据来源与身份', '',
        f"数据来源 **{source['source']}**，版本 `{source['data_version']}`，"
        f"输入 {source['rows']:,} 根 Bar。SHA-256 为 `{source['sha256']}`。",
        '该数据仅用于验收。', '', '## 交易日历与训练切分', '',
        'CME Globex Equity 的 ETH 与参照 NYSE 的 RTH。训练期 2020–2025，OOS 为 2026。',
        '收益单位为百分数。置信区间用于合成数据的链路检查。', '',
    ]
    for view, result in report['views'].items():
        lines.extend([f'## {view}', '', f"日历 SHA-256 为 `{result['calendar_hash']}`。", ''])
        for name in ('train', 'oos'):
            split = result[name]
            lines.extend([
                f'### {name}', '',
                f"拟合身份 `{split['fit_identity']['sha256']}`。应有网格 {split['grid_rows']:,} 根。",
                '',
                '| 持有 Bar | 有效 / 候选 | 跨界清除 | Spearman IC | 区间下界 / 上界 | 方向命中 / 分母 | 上涨基准率 | 非重叠偏移数 |',
                '|---:|---:|---:|---:|---:|---:|---:|---:|',
            ])
            for h, row in split['horizons'].items():
                counts, ic, ci, direction = (row[key] for key in ('counts', 'ic', 'bootstrap', 'direction'))
                lines.append(
                    f"| {h} | {counts['valid']:,} / {counts['total']:,} | {counts['split_purged_count']} | "
                    f"{ic['spearman_ic']:.6f} | {ci['lower']:.6f} / {ci['upper']:.6f} | "
                    f"{direction['hits']:,} / {direction['denominator']:,} | {direction['up_baseline']:.4f} | "
                    f"{len(row['non_overlapping_ic'])} |",
                )
            lines.append('')
    lines.extend([
        '## 完整统计', '',
        '同名 JSON 包含逐桶方向命中、收益 P5/P25/P50/P75/P95 与尾部均值、'
        '所有非重叠偏移的样本数与 IC、剔除计数和数据探测记录。', '',
    ])
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text(encoding='utf-8'))
    if probe['status'] != 'no_real_nq_1min_source_found':
        parser.error('Run this synthetic fixture only after the local data probe finds no complete real source')
    output = args.output
    markdown = output.with_suffix('.md')
    if output.exists() or markdown.exists():
        parser.error('Select unused report paths')
    calendar = SessionCalendar(CONFIG['calendar'], CONFIG['start_date'], CONFIG['end_date'],
                               rth_reference_calendar=CONFIG['rth_reference_calendar'])
    bars = synthetic_bars(calendar)
    identity = input_identity(bars)
    version, _ = resolve_code_version(Path.cwd())
    views = {}
    for view in ('ETH', 'RTH'):
        print(f'Evaluating {view} on {len(bars):,} source Bars', flush=True)
        views[view] = evaluate_view(bars, calendar, view)
    report = _json_value(dict(
        schema='nq_1min_acceptance.v1', purpose='synthetic_acceptance_only', code_version=version,
        config=CONFIG, data_identity=identity, data_probe=probe, return_unit='percent', views=views,
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_text = render_markdown(report)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    markdown.write_text(markdown_text, encoding='utf-8')


if __name__ == '__main__':
    main()
