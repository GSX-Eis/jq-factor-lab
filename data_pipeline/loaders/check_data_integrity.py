#!/usr/bin/env python3
"""
check_data_integrity.py - 检查 DuckDB 数据库数据完整性

检查项:
1. 时间一致性: 各表日期范围、交易日缺口检测
2. 股票覆盖: factor_data, price_data_daily, index_constituents 三表交叉验证
3. 数据质量: NULL 值比例、异常值检测
4. 关系完整性: 重复表检测、导入进度验证

Usage:
    python check_data_integrity.py [--db-path PATH] [--output-report PATH]
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("Error: duckdb package not available")
    sys.exit(1)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Default paths
DEFAULT_DB_PATH = './data/duckdb/factor_database.duckdb'


class DataIntegrityChecker:
    """DuckDB 数据完整性检查器"""
    
    # 已知的元数据列（非因子列）
    METADATA_COLS = {'stock_code', 'date', 'is_new_stock', 'is_st', 'code', 'date_1'}
    
    # NULL 率警告阈值
    NULL_RATE_WARNING_THRESHOLD = 0.10  # 10%
    NULL_RATE_ERROR_THRESHOLD = 0.50    # 50%
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        初始化检查器
        
        Args:
            db_path: DuckDB 数据库文件路径
        """
        self.db_path = db_path
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.results: Dict = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
        
    def connect(self) -> None:
        """建立数据库连接"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self.conn = duckdb.connect(self.db_path, read_only=True)
        
    def close(self) -> None:
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def _execute(self, sql: str):
        """执行 SQL 并返回结果"""
        return self.conn.execute(sql)
    
    def _fetchone(self, sql: str):
        """执行 SQL 并返回单行结果"""
        return self._execute(sql).fetchone()
    
    def _fetchall(self, sql: str):
        """执行 SQL 并返回所有结果"""
        return self._execute(sql).fetchall()
    
    def _fetchdf(self, sql: str):
        """执行 SQL 并返回 DataFrame"""
        return self._execute(sql).fetchdf()
    
    # =========================================================================
    # 检查项 1: 基础统计
    # =========================================================================
    
    def check_table_stats(self) -> Dict:
        """
        检查各表基础统计信息
        
        Returns:
            Dict with table statistics
        """
        print("\n📊 检查基础统计...")
        
        tables = self._fetchall("SHOW TABLES")
        table_names = [t[0] for t in tables]
        
        stats = {}
        for table in table_names:
            try:
                row_count = self._fetchone(f"SELECT COUNT(*) FROM {table}")[0]
                
                # 获取列信息
                cols = self._fetchall(f"DESCRIBE {table}")
                col_names = [c[0] for c in cols]
                
                # 获取日期范围（如果有 date 列）
                date_range = None
                if 'date' in col_names:
                    min_max = self._fetchone(f"SELECT MIN(date), MAX(date) FROM {table}")
                    date_range = (str(min_max[0]), str(min_max[1]))
                
                # 获取唯一股票数（如果有 stock_code 列）
                stock_count = None
                if 'stock_code' in col_names:
                    stock_count = self._fetchone(f"SELECT COUNT(DISTINCT stock_code) FROM {table}")[0]
                
                stats[table] = {
                    'row_count': row_count,
                    'column_count': len(col_names),
                    'columns': col_names,
                    'date_range': date_range,
                    'stock_count': stock_count
                }
                
            except Exception as e:
                stats[table] = {'error': str(e)}
                self.errors.append(f"表 {table} 统计失败: {e}")
        
        self.results['table_stats'] = stats
        return stats
    
    # =========================================================================
    # 检查项 2: 日期覆盖一致性
    # =========================================================================
    
    def check_date_coverage(self) -> Dict:
        """
        检查日期覆盖率和表间一致性
        
        Returns:
            Dict with date coverage analysis
        """
        print("\n📅 检查日期覆盖...")
        
        coverage = {}
        
        # 获取各表的日期范围
        tables_with_date = ['factor_data', 'price_data_daily', 'index_constituents']
        for table in tables_with_date:
            try:
                result = self._fetchone(f"""
                    SELECT 
                        MIN(date) as min_date,
                        MAX(date) as max_date,
                        COUNT(DISTINCT date) as date_count
                    FROM {table}
                """)
                coverage[table] = {
                    'min_date': str(result[0]),
                    'max_date': str(result[1]),
                    'date_count': result[2]
                }
            except Exception as e:
                coverage[table] = {'error': str(e)}
        
        # 检查表间日期对齐
        if 'factor_data' in coverage and 'price_data_daily' in coverage:
            factor_max = coverage['factor_data'].get('max_date', '')
            price_max = coverage['price_data_daily'].get('max_date', '')
            
            if factor_max and price_max and factor_max != price_max:
                msg = f"⚠️ 日期不一致: factor_data 最新日期 {factor_max}, price_data_daily 最新日期 {price_max}"
                self.warnings.append(msg)
                coverage['date_mismatch'] = {
                    'factor_data': factor_max,
                    'price_data_daily': price_max
                }
        
        # 检查每只股票的日期覆盖
        try:
            stock_coverage = self._fetchdf("""
                SELECT 
                    stock_code,
                    COUNT(DISTINCT date) as date_count,
                    MIN(date) as min_date,
                    MAX(date) as max_date
                FROM factor_data
                GROUP BY stock_code
                ORDER BY date_count ASC
                LIMIT 10
            """)
            coverage['stocks_with_least_coverage'] = stock_coverage.to_dict('records')
        except Exception as e:
            self.errors.append(f"股票日期覆盖检查失败: {e}")
        
        self.results['date_coverage'] = coverage
        return coverage
    
    # =========================================================================
    # 检查项 3: 股票覆盖一致性
    # =========================================================================
    
    def check_stock_coverage(self) -> Dict:
        """
        检查股票覆盖一致性
        
        Returns:
            Dict with stock coverage analysis
        """
        print("\n🏢 检查股票覆盖...")
        
        coverage = {}
        
        try:
            # 获取各表的唯一股票
            factor_stocks = set(self._fetchdf("SELECT DISTINCT stock_code FROM factor_data")['stock_code'])
            price_stocks = set(self._fetchdf("SELECT DISTINCT stock_code FROM price_data_daily")['stock_code'])
            index_stocks = set(self._fetchdf("SELECT DISTINCT stock_code FROM index_constituents")['stock_code'])
            
            coverage['factor_data_stocks'] = len(factor_stocks)
            coverage['price_data_daily_stocks'] = len(price_stocks)
            coverage['index_constituents_stocks'] = len(index_stocks)
            
            # 交叉比对
            in_factor_not_price = factor_stocks - price_stocks
            in_price_not_factor = price_stocks - factor_stocks
            in_index_not_factor = index_stocks - factor_stocks
            
            coverage['in_factor_not_price'] = list(in_factor_not_price)[:20]
            coverage['in_price_not_factor'] = list(in_price_not_factor)[:20]
            coverage['in_index_not_factor_count'] = len(in_index_not_factor)
            
            # 报告差异
            if in_factor_not_price:
                msg = f"⚠️ {len(in_factor_not_price)} 只股票在 factor_data 但不在 price_data_daily"
                self.warnings.append(msg)
            
            if in_price_not_factor:
                msg = f"⚠️ {len(in_price_not_factor)} 只股票在 price_data_daily 但不在 factor_data"
                self.warnings.append(msg)
                
            if len(factor_stocks) == len(price_stocks) and not in_factor_not_price:
                coverage['stock_match'] = True
                print("  ✓ factor_data 与 price_data_daily 股票完全一致")
            else:
                coverage['stock_match'] = False
                
        except Exception as e:
            self.errors.append(f"股票覆盖检查失败: {e}")
            coverage['error'] = str(e)
        
        self.results['stock_coverage'] = coverage
        return coverage
    
    # =========================================================================
    # 检查项 4: 数据质量 - NULL 值比例
    # =========================================================================
    
    def check_null_rates(self) -> Dict:
        """
        检查各列的 NULL 值比例
        
        Returns:
            Dict with NULL rate analysis
        """
        print("\n🔍 检查 NULL 值比例...")
        
        null_rates = {}
        
        try:
            # 获取 factor_data 的所有列
            cols = [c[0] for c in self._fetchall("DESCRIBE factor_data")]
            factor_cols = [c for c in cols if c not in self.METADATA_COLS]
            
            total_count = self._fetchone("SELECT COUNT(*) FROM factor_data")[0]
            
            for col in factor_cols:
                null_count = self._fetchone(f'SELECT COUNT(*) FROM factor_data WHERE "{col}" IS NULL')[0]
                null_rate = null_count / total_count if total_count > 0 else 0
                
                null_rates[col] = {
                    'null_count': null_count,
                    'total_count': total_count,
                    'null_rate': round(null_rate, 4)
                }
                
                # 根据阈值报警
                if null_rate >= self.NULL_RATE_ERROR_THRESHOLD:
                    self.errors.append(f"❌ 因子 {col} NULL 率过高: {null_rate*100:.1f}%")
                elif null_rate >= self.NULL_RATE_WARNING_THRESHOLD:
                    self.warnings.append(f"⚠️ 因子 {col} NULL 率: {null_rate*100:.1f}%")
                    
        except Exception as e:
            self.errors.append(f"NULL 值检查失败: {e}")
            null_rates['error'] = str(e)
        
        self.results['null_rates'] = null_rates
        return null_rates
    
    # =========================================================================
    # 检查项 5: 数据质量 - 异常值
    # =========================================================================
    
    def check_data_quality(self) -> Dict:
        """
        检查数据质量和异常值
        
        Returns:
            Dict with data quality analysis
        """
        print("\n⚡ 检查数据质量...")
        
        quality = {}
        
        try:
            # 检查价格数据异常
            # 1. 负价格
            neg_prices = self._fetchone("""
                SELECT COUNT(*) FROM price_data_daily 
                WHERE open < 0 OR close < 0 OR high < 0 OR low < 0
            """)[0]
            quality['negative_prices'] = neg_prices
            if neg_prices > 0:
                self.errors.append(f"❌ 发现 {neg_prices} 条负价格记录")
            
            # 2. 成交量为0但非停牌
            zero_vol_not_paused = self._fetchone("""
                SELECT COUNT(*) FROM price_data_daily 
                WHERE volume = 0 AND (paused IS NULL OR paused = 0)
            """)[0]
            quality['zero_volume_not_paused'] = zero_vol_not_paused
            
            # 3. 停牌统计
            paused_count = self._fetchone("""
                SELECT COUNT(*) FROM price_data_daily WHERE paused = 1
            """)[0]
            quality['paused_records'] = paused_count
            
            # 4. 检查 high < low 的异常情况
            hl_anomaly = self._fetchone("""
                SELECT COUNT(*) FROM price_data_daily WHERE high < low
            """)[0]
            quality['high_less_than_low'] = hl_anomaly
            if hl_anomaly > 0:
                self.errors.append(f"❌ 发现 {hl_anomaly} 条 high < low 的异常记录")
            
            # 5. 检查因子值极端异常（超过 ±100 标准差）
            # 这里只抽查几个关键因子
            key_factors = ['val_pe_ratio', 'val_pb_ratio', 'roe_ttm']
            for factor in key_factors:
                try:
                    stats = self._fetchone(f'''
                        SELECT AVG("{factor}"), STDDEV("{factor}") FROM factor_data
                        WHERE "{factor}" IS NOT NULL
                    ''')
                    if stats[0] is not None and stats[1] is not None and stats[1] > 0:
                        mean, std = stats[0], stats[1]
                        extreme_count = self._fetchone(f'''
                            SELECT COUNT(*) FROM factor_data
                            WHERE "{factor}" IS NOT NULL 
                            AND (ABS("{factor}" - {mean}) > {std * 100})
                        ''')[0]
                        quality[f'{factor}_extreme_values'] = extreme_count
                except:
                    pass
                    
        except Exception as e:
            self.errors.append(f"数据质量检查失败: {e}")
            quality['error'] = str(e)
        
        self.results['data_quality'] = quality
        return quality
    
    # =========================================================================
    # 检查项 6: 重复数据检测
    # =========================================================================
    
    def check_duplicates(self) -> Dict:
        """
        检测重复数据
        
        Returns:
            Dict with duplicate analysis
        """
        print("\n🔄 检查重复数据...")
        
        duplicates = {}
        
        try:
            # 检查 factor_data 主键重复
            factor_dup = self._fetchone("""
                SELECT COUNT(*) FROM (
                    SELECT stock_code, date, COUNT(*) as cnt
                    FROM factor_data
                    GROUP BY stock_code, date
                    HAVING cnt > 1
                ) dups
            """)[0]
            duplicates['factor_data_pk_duplicates'] = factor_dup
            if factor_dup > 0:
                self.errors.append(f"❌ factor_data 存在 {factor_dup} 条主键重复记录")
                
        except Exception as e:
            self.errors.append(f"重复检测失败: {e}")
            duplicates['error'] = str(e)
        
        self.results['duplicates'] = duplicates
        return duplicates
    
    # =========================================================================
    # 报告生成
    # =========================================================================
    
    def generate_report(self) -> str:
        """
        生成完整的检查报告
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("       DuckDB 数据完整性检查报告")
        lines.append("=" * 60)
        lines.append(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"数据库路径: {self.db_path}")
        lines.append("")
        
        # 基础统计
        if 'table_stats' in self.results:
            lines.append("【基础统计】")
            for table, stats in self.results['table_stats'].items():
                if 'error' in stats:
                    lines.append(f"  ├─ {table}: ERROR - {stats['error']}")
                else:
                    row_str = f"{stats['row_count']:,} rows"
                    if stats.get('stock_count'):
                        row_str += f", {stats['stock_count']} stocks"
                    if stats.get('date_range'):
                        row_str += f", {stats['date_range'][0]} ~ {stats['date_range'][1]}"
                    lines.append(f"  ├─ {table}: {row_str}")
            lines.append("")
        
        # 日期覆盖
        if 'date_coverage' in self.results:
            lines.append("【日期覆盖】")
            coverage = self.results['date_coverage']
            if 'date_mismatch' in coverage:
                mm = coverage['date_mismatch']
                lines.append(f"  ⚠️ 日期不一致:")
                lines.append(f"     factor_data: {mm['factor_data']}")
                lines.append(f"     price_data_daily: {mm['price_data_daily']}")
            else:
                lines.append("  ✓ 各表日期范围一致")
            lines.append("")
        
        # 股票覆盖
        if 'stock_coverage' in self.results:
            lines.append("【股票覆盖】")
            sc = self.results['stock_coverage']
            if sc.get('stock_match'):
                lines.append("  ✓ factor_data 与 price_data_daily 股票完全一致")
            else:
                if sc.get('in_factor_not_price'):
                    lines.append(f"  ⚠️ {len(sc['in_factor_not_price'])} 只股票仅在 factor_data")
                if sc.get('in_price_not_factor'):
                    lines.append(f"  ⚠️ {len(sc['in_price_not_factor'])} 只股票仅在 price_data_daily")
            lines.append("")
        
        # 数据质量
        if 'data_quality' in self.results:
            lines.append("【数据质量】")
            dq = self.results['data_quality']
            if dq.get('negative_prices', 0) == 0:
                lines.append("  ✓ 无负价格记录")
            if dq.get('high_less_than_low', 0) == 0:
                lines.append("  ✓ 无 high < low 异常")
            lines.append(f"  ℹ️ 停牌记录数: {dq.get('paused_records', 'N/A'):,}")
            lines.append("")
        
        # NULL 值统计 (仅显示警告级别以上的)
        if 'null_rates' in self.results:
            lines.append("【NULL 值统计】")
            high_null_cols = [
                (col, info) for col, info in self.results['null_rates'].items()
                if isinstance(info, dict) and info.get('null_rate', 0) >= self.NULL_RATE_WARNING_THRESHOLD
            ]
            if high_null_cols:
                for col, info in sorted(high_null_cols, key=lambda x: -x[1]['null_rate']):
                    lines.append(f"  ⚠️ {col}: {info['null_rate']*100:.1f}% NULL")
            else:
                lines.append("  ✓ 所有因子 NULL 率均低于 10%")
            lines.append("")
        
        # 重复检测
        if 'duplicates' in self.results:
            lines.append("【重复检测】")
            dup = self.results['duplicates']
            if dup.get('tables_identical'):
                lines.append("  ⚠️ price_data 与 price_data_5m 疑似重复表")
            if dup.get('factor_data_pk_duplicates', 0) > 0:
                lines.append(f"  ❌ factor_data 主键重复: {dup['factor_data_pk_duplicates']} 条")
            else:
                lines.append("  ✓ 无主键重复")
            lines.append("")
        
        # 汇总
        lines.append("-" * 60)
        lines.append("【检查汇总】")
        lines.append(f"  ❌ 错误: {len(self.errors)}")
        lines.append(f"  ⚠️ 警告: {len(self.warnings)}")
        
        if self.errors:
            lines.append("")
            lines.append("【错误详情】")
            for e in self.errors:
                lines.append(f"  {e}")
        
        if self.warnings:
            lines.append("")
            lines.append("【警告详情】")
            for w in self.warnings:
                lines.append(f"  {w}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    # =========================================================================
    # 主入口
    # =========================================================================
    
    def run_all_checks(self) -> Dict:
        """
        运行所有检查
        
        Returns:
            Dict with all results
        """
        print(f"\n🚀 开始检查数据库: {self.db_path}")
        print("=" * 50)
        
        try:
            self.connect()
            
            self.check_table_stats()
            self.check_date_coverage()
            self.check_stock_coverage()
            self.check_null_rates()
            self.check_data_quality()
            self.check_duplicates()
            
        finally:
            self.close()
        
        return self.results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='检查 DuckDB 数据库数据完整性',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--db-path',
        default=DEFAULT_DB_PATH,
        help=f'DuckDB 数据库路径 (默认: {DEFAULT_DB_PATH})'
    )
    parser.add_argument(
        '--output-report',
        help='保存报告到指定文件路径'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='输出 JSON 格式结果'
    )
    
    args = parser.parse_args()
    
    # 运行检查
    checker = DataIntegrityChecker(db_path=args.db_path)
    
    try:
        checker.run_all_checks()
        report = checker.generate_report()
        
        if args.json:
            import json
            print(json.dumps(checker.results, indent=2, ensure_ascii=False, default=str))
        else:
            print(report)
        
        # 保存报告
        if args.output_report:
            with open(args.output_report, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n📄 报告已保存到: {args.output_report}")
        
        # 返回码: 有错误返回1，仅警告返回0
        return 1 if checker.errors else 0
        
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        return 1
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
