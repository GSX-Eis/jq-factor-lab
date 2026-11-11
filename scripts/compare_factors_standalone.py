#!/usr/bin/env python3
"""
Standalone factor comparison script for cloud vs local alignment.
Usage: python compare_factors_standalone.py --factor FactorName --cloud-csv path/to/cloud.csv --local-csv path/to/local.csv --output outputs/alignment/
"""

import pandas as pd
import numpy as np
import argparse
import os
from scipy import stats
import json


def load_and_prepare_data(cloud_csv, local_csv):
    """Load and prepare cloud and local data."""
    cloud_df = pd.read_csv(cloud_csv)
    local_df = pd.read_csv(local_csv)

    # Standardize column names
    cloud_df.columns = [c.lower() for c in cloud_df.columns]
    local_df.columns = [c.lower() for c in local_df.columns]

    # Rename columns to standard names
    col_mapping = {
        'date': 'date',
        'code': 'code',
        'stock_code': 'code',
        'value': 'value',
        'factor_value': 'value'
    }

    cloud_df = cloud_df.rename(columns={k: v for k, v in col_mapping.items() if k in cloud_df.columns})
    local_df = local_df.rename(columns={k: v for k, v in col_mapping.items() if k in local_df.columns})

    return cloud_df, local_df


def calculate_daily_correlation(cloud_df, local_df):
    """Calculate daily Spearman correlation between cloud and local values."""
    cloud_df['date'] = pd.to_datetime(cloud_df['date'])
    local_df['date'] = pd.to_datetime(local_df['date'])

    merged = pd.merge(
        cloud_df[['date', 'code', 'value']].rename(columns={'value': 'cloud_value'}),
        local_df[['date', 'code', 'value']].rename(columns={'value': 'local_value'}),
        on=['date', 'code'],
        how='inner'
    )

    daily_stats = []
    for date, group in merged.groupby('date'):
        if len(group) < 5:
            continue

        corr, pvalue = stats.spearmanr(group['cloud_value'], group['local_value'])

        daily_stats.append({
            'date': date.strftime('%Y-%m-%d'),
            'spearman': corr,
            'pvalue': pvalue,
            'n_stocks': len(group),
            'cloud_mean': group['cloud_value'].mean(),
            'local_mean': group['local_value'].mean(),
            'cloud_std': group['cloud_value'].std(),
            'local_std': group['local_value'].std(),
            'max_abs_diff': np.max(np.abs(group['cloud_value'] - group['local_value']))
        })

    return pd.DataFrame(daily_stats), merged


def generate_report(factor_name, daily_stats, merged, output_dir):
    """Generate alignment report."""
    os.makedirs(output_dir, exist_ok=True)

    report = {
        'factor_name': factor_name,
        'alignment_status': 'UNKNOWN',
        'summary': {},
        'daily_details': daily_stats.to_dict('records')
    }

    if daily_stats.empty:
        report['alignment_status'] = 'FAIL'
        report['summary'] = {
            'message': 'No overlapping data found between cloud and local'
        }
    else:
        # Calculate summary statistics
        spearman_mean = daily_stats['spearman'].mean()
        spearman_min = daily_stats['spearman'].min()
        max_diff_mean = daily_stats['max_abs_diff'].mean()

        report['summary'] = {
            'total_days': len(daily_stats),
            'date_range': f"{daily_stats['date'].min()} to {daily_stats['date'].max()}",
            'spearman_mean': round(spearman_mean, 6),
            'spearman_min': round(spearman_min, 6),
            'max_abs_diff_mean': round(max_diff_mean, 10)
        }

        # Determine alignment status
        if spearman_mean >= 0.99 and spearman_min >= 0.95:
            report['alignment_status'] = 'PASS'
        elif spearman_mean >= 0.95:
            report['alignment_status'] = 'WARN'
        else:
            report['alignment_status'] = 'FAIL'

    # Save report
    report_path = os.path.join(output_dir, f'{factor_name}_alignment_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Save detailed CSV
    csv_path = os.path.join(output_dir, f'{factor_name}_daily_stats.csv')
    daily_stats.to_csv(csv_path, index=False)

    # Save merged data
    merged_path = os.path.join(output_dir, f'{factor_name}_merged_data.csv')
    merged.to_csv(merged_path, index=False)

    return report, report_path


def main():
    parser = argparse.ArgumentParser(description='Compare cloud and local factor values')
    parser.add_argument('--factor', type=str, required=True, help='Factor name')
    parser.add_argument('--cloud-csv', type=str, required=True, help='Path to cloud CSV file')
    parser.add_argument('--local-csv', type=str, required=True, help='Path to local CSV file')
    parser.add_argument('--output', type=str, default='outputs/alignment/', help='Output directory')

    args = parser.parse_args()

    print(f"Comparing factor: {args.factor}")
    print(f"Cloud CSV: {args.cloud_csv}")
    print(f"Local CSV: {args.local_csv}")

    # Load data
    print("\nLoading data...")
    cloud_df, local_df = load_and_prepare_data(args.cloud_csv, args.local_csv)
    print(f"Cloud: {len(cloud_df)} rows")
    print(f"Local: {len(local_df)} rows")

    # Calculate daily correlation
    print("\nCalculating daily correlation...")
    daily_stats, merged = calculate_daily_correlation(cloud_df, local_df)
    print(f"Overlapping days: {len(daily_stats)}")

    if len(daily_stats) > 0:
        print(f"Mean Spearman: {daily_stats['spearman'].mean():.6f}")
        print(f"Min Spearman: {daily_stats['spearman'].min():.6f}")

    # Generate report
    print("\nGenerating report...")
    report, report_path = generate_report(args.factor, daily_stats, merged, args.output)

    print(f"\n{'='*60}")
    print(f"Alignment Status: {report['alignment_status']}")
    print(f"{'='*60}")
    print(f"Report saved to: {report_path}")

    if report['alignment_status'] == 'PASS':
        print("\n✅ Factor is aligned between cloud and local!")
        return 0
    elif report['alignment_status'] == 'WARN':
        print("\n⚠️  Factor alignment is acceptable but could be improved.")
        return 0
    else:
        print("\n❌ Factor is NOT aligned. Please check the diagnosis report.")
        return 1


if __name__ == '__main__':
    exit(main())
