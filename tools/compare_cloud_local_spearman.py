"""
云端 vs 本地因子值对比脚本
比较 CSV 文件和 DuckDB 数据库中的因子值及 Spearman 相关系数
全量对比所有因子在所有日期上的数据
"""

import duckdb
import pandas as pd
from scipy.stats import spearmanr
import numpy as np
from pathlib import Path
from datetime import datetime


def load_csv_data(csv_path: Path) -> pd.DataFrame:
    """加载 CSV 数据"""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def load_db_factor_data(conn: duckdb.DuckDBPyConnection, date, factor_name: str) -> pd.DataFrame:
    """从 DuckDB 加载指定日期和因子的数据"""
    query = f"""
        SELECT stock_code, factor_value as db_value 
        FROM factor_values 
        WHERE date = '{date}' AND factor_name = '{factor_name}'
    """
    return conn.execute(query).fetchdf()


def compute_spearman_for_factor_date(
    csv_df: pd.DataFrame, 
    conn: duckdb.DuckDBPyConnection, 
    date, 
    factor: str,
    db_factor_suffix: str = '_small_cap_9_rules'
) -> dict | None:
    """计算单个因子在单个日期的 Spearman 相关系数"""
    # CSV 数据
    csv_subset = csv_df[(csv_df['date'] == date) & (csv_df['factor'] == factor)].copy()
    csv_subset = csv_subset[['code', 'value']].rename(
        columns={'code': 'stock_code', 'value': 'csv_value'}
    )
    
    # DuckDB 数据
    db_factor_name = factor + db_factor_suffix
    db_df = load_db_factor_data(conn, date, db_factor_name)
    
    if len(csv_subset) == 0 or len(db_df) == 0:
        return None
    
    # 合并
    merged = csv_subset.merge(db_df, on='stock_code', how='inner')
    
    if len(merged) < 10:
        return None
    
    # 计算 Spearman 相关系数
    corr, pval = spearmanr(merged['csv_value'], merged['db_value'])
    
    # 计算值差异
    merged['diff'] = merged['csv_value'] - merged['db_value']
    max_diff = merged['diff'].abs().max()
    mean_diff = merged['diff'].abs().mean()
    
    # 计算精确匹配和近似匹配
    exact_match = (merged['diff'].abs() < 1e-10).sum()
    approx_match = (merged['diff'].abs() < 1e-6).sum()
    
    return {
        'date': date,
        'factor': factor,
        'csv_count': len(csv_subset),
        'db_count': len(db_df),
        'matched': len(merged),
        'spearman_corr': corr,
        'max_abs_diff': max_diff,
        'mean_abs_diff': mean_diff,
        'exact_match': exact_match,
        'approx_match': approx_match
    }


def run_full_comparison(
    csv_path: str = 'outputs/synthesis/alignment/cloud_factor_dump.csv',
    db_path: str = 'analysis_results.duckdb',
) -> pd.DataFrame:
    """运行全量对比 - 所有因子在所有日期"""
    csv_path = Path(csv_path)
    
    # 加载 CSV
    csv_df = load_csv_data(csv_path)
    csv_dates = sorted(csv_df['date'].unique())
    csv_factors = sorted(csv_df['factor'].unique())
    
    print('=== 数据概览 ===')
    print(f'CSV 日期数: {len(csv_dates)}, 因子数: {len(csv_factors)}')
    print(f'日期范围: {csv_dates[0]} ~ {csv_dates[-1]}')
    print(f'总记录数: {len(csv_df)}')
    
    # 连接 DuckDB
    conn = duckdb.connect(db_path, read_only=True)
    
    results = []
    total_combinations = len(csv_dates) * len(csv_factors)
    processed = 0
    
    print(f'\n开始全量对比: {total_combinations} 个 (日期, 因子) 组合...')
    
    for factor in csv_factors:
        for date in csv_dates:
            result = compute_spearman_for_factor_date(csv_df, conn, date, factor)
            if result:
                results.append(result)
            processed += 1
            if processed % 500 == 0:
                print(f'  进度: {processed}/{total_combinations} ({100*processed/total_combinations:.1f}%)')
    
    conn.close()
    
    print(f'完成! 有效对比: {len(results)} 组')
    return pd.DataFrame(results)


def generate_factor_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    """按因子汇总统计"""
    factor_summary = result_df.groupby('factor').agg({
        'matched': 'sum',
        'exact_match': 'sum',
        'approx_match': 'sum',
        'spearman_corr': ['mean', 'min', 'max', 'std'],
        'max_abs_diff': 'max',
        'mean_abs_diff': 'mean'
    }).reset_index()
    
    # 展平列名
    factor_summary.columns = [
        'factor', 'total_matched', 'total_exact', 'total_approx',
        'mean_spearman', 'min_spearman', 'max_spearman', 'std_spearman',
        'max_abs_diff', 'mean_abs_diff'
    ]
    
    # 计算匹配率
    factor_summary['exact_rate'] = factor_summary['total_exact'] / factor_summary['total_matched']
    factor_summary['approx_rate'] = factor_summary['total_approx'] / factor_summary['total_matched']
    
    # 按 mean_spearman 排序
    factor_summary = factor_summary.sort_values('mean_spearman', ascending=True)
    
    return factor_summary


def print_detailed_report(result_df: pd.DataFrame, factor_summary: pd.DataFrame) -> str:
    """生成详细报告"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    report_lines = []
    report_lines.append(f'\n{"="*80}')
    report_lines.append(f'云端 vs 本地因子值全量对比报告')
    report_lines.append(f'生成时间: {timestamp}')
    report_lines.append(f'{"="*80}')
    
    # 总体统计
    report_lines.append(f'\n=== 总体统计 ===')
    report_lines.append(f'对比组合数 (日期×因子): {len(result_df)}')
    report_lines.append(f'因子数量: {result_df["factor"].nunique()}')
    report_lines.append(f'日期数量: {result_df["date"].nunique()}')
    report_lines.append(f'总匹配记录数: {result_df["matched"].sum():,}')
    report_lines.append(f'总精确匹配数: {result_df["exact_match"].sum():,}')
    report_lines.append(f'总近似匹配数: {result_df["approx_match"].sum():,}')
    
    total_matched = result_df["matched"].sum()
    total_exact = result_df["exact_match"].sum()
    total_approx = result_df["approx_match"].sum()
    
    report_lines.append(f'\n精确匹配率 (|diff|<1e-10): {100*total_exact/total_matched:.2f}%')
    report_lines.append(f'近似匹配率 (|diff|<1e-6): {100*total_approx/total_matched:.2f}%')
    
    # Spearman 统计
    report_lines.append(f'\n=== Spearman 相关系数统计 ===')
    report_lines.append(f'平均值: {result_df["spearman_corr"].mean():.6f}')
    report_lines.append(f'最小值: {result_df["spearman_corr"].min():.6f}')
    report_lines.append(f'最大值: {result_df["spearman_corr"].max():.6f}')
    report_lines.append(f'标准差: {result_df["spearman_corr"].std():.6f}')
    
    # 分类统计
    high_corr = (factor_summary['mean_spearman'] >= 0.99).sum()
    mid_corr = ((factor_summary['mean_spearman'] >= 0.9) & (factor_summary['mean_spearman'] < 0.99)).sum()
    low_corr = (factor_summary['mean_spearman'] < 0.9).sum()
    
    report_lines.append(f'\n=== 因子分类 ===')
    report_lines.append(f'高相关因子 (≥0.99): {high_corr} 个')
    report_lines.append(f'中相关因子 (0.9-0.99): {mid_corr} 个')
    report_lines.append(f'低相关因子 (<0.9): {low_corr} 个')
    
    # 低相关因子详情
    low_corr_factors = factor_summary[factor_summary['mean_spearman'] < 0.99].head(20)
    if len(low_corr_factors) > 0:
        report_lines.append(f'\n=== 需关注因子 (mean_spearman < 0.99) ===')
        for _, row in low_corr_factors.iterrows():
            report_lines.append(
                f"  {row['factor']}: mean={row['mean_spearman']:.4f}, "
                f"min={row['min_spearman']:.4f}, exact_rate={row['exact_rate']:.1%}"
            )
    
    # 完美匹配因子
    perfect_factors = factor_summary[
        (factor_summary['mean_spearman'] >= 0.9999) & 
        (factor_summary['exact_rate'] >= 0.95)
    ]
    report_lines.append(f'\n=== 完美匹配因子 (mean_spearman≥0.9999 且 exact_rate≥95%) ===')
    report_lines.append(f'共 {len(perfect_factors)} 个因子')
    
    report = '\n'.join(report_lines)
    print(report)
    return report


def main():
    """主函数"""
    # 运行全量对比
    result_df = run_full_comparison()
    
    # 生成因子汇总
    factor_summary = generate_factor_summary(result_df)
    
    # 打印详细报告
    report = print_detailed_report(result_df, factor_summary)
    
    # 保存结果
    output_dir = Path('outputs/synthesis/alignment')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_df.to_csv(output_dir / 'full_comparison_details.csv', index=False)
    factor_summary.to_csv(output_dir / 'factor_summary.csv', index=False)
    
    print(f'\n结果已保存到:')
    print(f'  - {output_dir / "full_comparison_details.csv"}')
    print(f'  - {output_dir / "factor_summary.csv"}')
    
    return result_df, factor_summary


if __name__ == '__main__':
    main()
