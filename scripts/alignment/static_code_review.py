#!/usr/bin/env python3
"""
Factor Alignment Static Code Review Script

自动执行 12 项对齐检查，返回结构化结果。
替代 agent 手动读取 alignment_checklist.md 并逐项审查。

Usage:
    python static_code_review.py --factor factors/exp/target_factor.py
    python static_code_review.py --factor factors/exp/target_factor.py --verbose
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class CheckResult:
    """单项检查结果"""

    check_id: str
    name: str
    status: Literal["OK", "ISSUE", "WARN", "SKIP"]
    message: str = ""
    line_numbers: list[int] = field(default_factory=list)
    suggestion: str = ""


@dataclass
class ReviewReport:
    """审查报告"""

    factor_file: str
    factor_name: str
    results: list[CheckResult] = field(default_factory=list)
    suggested_logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factor_file": self.factor_file,
            "factor_name": self.factor_name,
            "summary": {
                "total": len(self.results),
                "issues": sum(1 for r in self.results if r.status == "ISSUE"),
                "warnings": sum(1 for r in self.results if r.status == "WARN"),
                "ok": sum(1 for r in self.results if r.status == "OK"),
            },
            "issues": [
                {
                    "check_id": r.check_id,
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "lines": r.line_numbers,
                    "suggestion": r.suggestion,
                }
                for r in self.results
                if r.status in ("ISSUE", "WARN")
            ],
            "suggested_logs": self.suggested_logs,
        }

    def to_summary(self) -> str:
        """生成简洁的文本摘要"""
        lines = [f"=== Phase 1 审查结果: {self.factor_name} ==="]

        issue_count = 0
        for r in self.results:
            if r.status == "ISSUE":
                issue_count += 1
                line_info = f"L{r.line_numbers[0]}" if r.line_numbers else ""
                lines.append(
                    f"[ISSUE-{issue_count}] {r.check_id} {r.name}: {line_info}, {r.message}"
                )
            elif r.status == "WARN":
                line_info = f"L{r.line_numbers[0]}" if r.line_numbers else ""
                lines.append(f"[WARN] {r.check_id} {r.name}: {line_info}, {r.message}")

        if issue_count == 0:
            lines.append("[OK] 12项全部通过，进入 Phase 2")
        elif self.suggested_logs:
            lines.append(f"建议 Phase 2 重点插入日志: {', '.join(self.suggested_logs)}")

        return "\n".join(lines)


class FactorCodeReviewer:
    """因子代码静态审查器"""

    def __init__(self, factor_path: Path, verbose: bool = False):
        self.factor_path = factor_path
        self.verbose = verbose
        self.content = factor_path.read_text(encoding="utf-8")
        self.lines = self.content.splitlines()
        self.factor_name = self._extract_factor_name()

    def _extract_factor_name(self) -> str:
        """从文件中提取因子类名"""
        match = re.search(r"class\s+(\w+Factor)\s*\(", self.content)
        return match.group(1) if match else self.factor_path.stem

    def _find_lines(self, pattern: str, flags: int = 0) -> list[int]:
        """查找匹配模式的行号（1-indexed）"""
        regex = re.compile(pattern, flags)
        return [i + 1 for i, line in enumerate(self.lines) if regex.search(line)]

    def _has_pattern(self, pattern: str, flags: int = 0) -> bool:
        """检查是否存在匹配模式"""
        return bool(re.search(pattern, self.content, flags))

    def check_1_factor_field(self) -> CheckResult:
        """CHECK-1: fields 必须包含 'factor' 触发 PIT 复权"""
        # 查找 _get_raw_data_for_factor 或 fields 定义
        fields_lines = self._find_lines(r"fields\s*=\s*\[")

        if not fields_lines:
            return CheckResult("CHECK-1", "factor字段", "SKIP", "未找到 fields 定义")

        # The adjustment column is required for price reads, not money-flow reads.
        # Parse syntax so comments cannot accidentally exempt an unknown table.
        try:
            nodes = list(ast.walk(ast.parse(self.content)))
        except SyntaxError:
            nodes = []
        fundamental = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "getattr" and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name) and node.args[0].id == "self"
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "fundamental_table"
            for node in nodes
        )
        reads_raw = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_fetch_raw_data" for node in nodes)
        reads_price = any(
            (isinstance(node, ast.Constant) and isinstance(node.value, str)
             and node.value.startswith("price_data"))
            or (isinstance(node, ast.Attribute) and node.attr == "price_table")
            or (isinstance(node, ast.Call)
                and ((isinstance(node.func, ast.Name) and node.func.id in {"get_price", "fetch_price"})
                     or (isinstance(node.func, ast.Attribute) and node.func.attr in {"get_price", "fetch_price"})))
            for node in nodes
        )
        def known_nonprice(value):
            return (
                isinstance(value, ast.Constant) and isinstance(value.value, str)
                and not value.value.startswith("price_data")
            ) or (
                isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == "getattr" and len(value.args) >= 2
                and isinstance(value.args[0], ast.Name) and value.args[0].id == "self"
                and isinstance(value.args[1], ast.Constant)
                and value.args[1].value == "fundamental_table"
            )

        bindings = {}
        for node in nodes:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
                value = node.value if isinstance(node, ast.AnnAssign) else None
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings.setdefault(target.id, []).append(value)
        safe_names = {name for name, values in bindings.items() if all(known_nonprice(value) for value in values)}
        safe_reads = True
        for node in nodes:
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_fetch_raw_data"):
                continue
            table = next((item.value for item in node.keywords if item.arg == "table_name"), ast.Constant("factor_data"))
            if (len(node.args) > 1 or any(item.arg is None for item in node.keywords)
                    or not (known_nonprice(table) or isinstance(table, ast.Name) and table.id in safe_names)):
                safe_reads = False
                break
        if fundamental and reads_raw and not reads_price and safe_reads:
            return CheckResult("CHECK-1", "factor字段", "SKIP", "仅查询财务/资金流表；factor 仅用于价格 PIT 复权")

        # 检查是否包含 "factor"
        has_factor = self._has_pattern(r'fields\s*=\s*\[[^\]]*["\']factor["\']')

        if has_factor:
            return CheckResult("CHECK-1", "factor字段", "OK")
        else:
            return CheckResult(
                "CHECK-1",
                "factor字段",
                "ISSUE",
                "fields 缺少 'factor'，PIT 复权不触发",
                fields_lines,
                "添加 'factor' 到 fields 列表",
            )

    def check_2_avg_price_timing(self) -> CheckResult:
        """CHECK-2: avg_price 必须在 calculate_factor 中计算"""
        avg_price_lines = self._find_lines(
            r"avg_price\s*=\s*.*money.*volume|money\s*/\s*volume"
        )

        if not avg_price_lines:
            return CheckResult("CHECK-2", "avg_price时机", "SKIP", "未使用 avg_price")

        # 检查是否在 _get_raw_data 中
        in_get_raw = self._has_pattern(
            r"def\s+_get_raw_data.*?(?=def\s+\w+|class\s+\w+|\Z)", re.DOTALL
        )
        raw_data_match = re.search(
            r"def\s+_get_raw_data.*?(?=def\s+\w+|class\s+\w+|\Z)",
            self.content,
            re.DOTALL,
        )

        if raw_data_match:
            raw_data_section = raw_data_match.group()
            if re.search(r"avg_price\s*=", raw_data_section):
                return CheckResult(
                    "CHECK-2",
                    "avg_price时机",
                    "ISSUE",
                    "avg_price 在 _get_raw_data 中计算（PIT 之前）",
                    avg_price_lines,
                    "移到 calculate_factor() 中计算",
                )

        return CheckResult("CHECK-2", "avg_price时机", "OK")

    def check_3_money_no_pit(self) -> CheckResult:
        """CHECK-3: money 字段不需要 PIT 调整"""
        # 检查是否对 money 做了 factor 调整
        money_factor_lines = self._find_lines(r"money\s*\*\s*factor|money\s*/\s*factor")

        if money_factor_lines:
            return CheckResult(
                "CHECK-3",
                "money不调整",
                "ISSUE",
                "money 被 factor 调整（不应该）",
                money_factor_lines,
                "money 是实际金额，不需要 PIT 调整",
            )

        return CheckResult("CHECK-3", "money不调整", "OK")

    def check_4_pool_selection_d1(self) -> CheckResult:
        """CHECK-4: 股票池选择 D+1 逻辑"""
        # 查找池选择相关代码
        pool_lines = self._find_lines(r"sorted_keys|monthly_stock_pools|_get_pool")

        if not pool_lines:
            return CheckResult("CHECK-4", "池选择D+1", "SKIP", "未使用动态股票池")

        # 检查是否使用了错误的区间匹配
        wrong_pattern = self._has_pattern(r"if\s+key_dt\s*<=\s*date|key\s*<=\s*date")
        correct_pattern = self._has_pattern(r"Timedelta.*days.*1|>=.*next|>.*date")

        if wrong_pattern and not correct_pattern:
            return CheckResult(
                "CHECK-4",
                "池选择D+1",
                "ISSUE",
                "使用区间匹配 (key <= date)，应改为 D+1",
                pool_lines,
                "找 >= date + 1day 的最近调仓日",
            )

        return CheckResult("CHECK-4", "池选择D+1", "OK")

    def check_5_regression_pool(self) -> CheckResult:
        """CHECK-5: 截面回归：池内拟合β，全量输出"""
        # 检查是否有回归相关代码
        regression_lines = self._find_lines(r"lstsq|LinearRegression|np\.linalg")

        if not regression_lines:
            return CheckResult("CHECK-5", "截面回归", "SKIP", "未使用截面回归")

        # 检查是否有 use_pool_for_regression 或 _get_pool_for_regression
        has_pool_regression = self._has_pattern(
            r"use_pool_for_regression|_get_pool_for_regression"
        )

        if not has_pool_regression:
            return CheckResult(
                "CHECK-5",
                "截面回归",
                "WARN",
                "有回归但未见池内拟合逻辑",
                regression_lines,
                "确认是否需要池内拟合β、全量输出残差",
            )

        return CheckResult("CHECK-5", "截面回归", "OK")

    def check_6_benchmark_by_date(self) -> CheckResult:
        """CHECK-6: 截面 benchmark 按日期用对应池"""
        benchmark_lines = self._find_lines(r"benchmark|mean\(axis=1\)|median\(")

        if not benchmark_lines:
            return CheckResult(
                "CHECK-6", "截面benchmark", "SKIP", "未使用截面 benchmark"
            )

        # 检查是否在循环中按日期获取池
        has_date_loop = self._has_pattern(r"for.*date.*in|for.*output_date")
        has_pool_in_loop = self._has_pattern(r"_get_pool.*date|pool.*=.*get.*date")

        if benchmark_lines and not (has_date_loop and has_pool_in_loop):
            return CheckResult(
                "CHECK-6",
                "截面benchmark",
                "WARN",
                "使用 benchmark 但未见按日期获取池",
                benchmark_lines,
                "每个输出日期 T，用 T 日池算 benchmark",
            )

        return CheckResult("CHECK-6", "截面benchmark", "OK")

    def check_7_stable_sort(self) -> CheckResult:
        """CHECK-7: 稳定排序"""
        sort_lines = self._find_lines(r"sort_values|\.rank\(")

        if not sort_lines:
            return CheckResult("CHECK-7", "稳定排序", "SKIP", "未使用排序")

        # 检查是否使用了稳定排序
        has_stable = self._has_pattern(
            r"kind\s*=\s*['\"]stable['\"]|method\s*=\s*['\"]first['\"]"
        )
        has_secondary_key = self._has_pattern(r"sort_values\s*\(\s*\[")

        if not (has_stable or has_secondary_key):
            return CheckResult(
                "CHECK-7",
                "稳定排序",
                "WARN",
                "排序未指定 stable 或次级键",
                sort_lines,
                "添加 kind='mergesort' 或 method='first' (JQ Python3.6不支持kind='stable')",
            )

        return CheckResult("CHECK-7", "稳定排序", "OK")

    def check_8_nan_handling(self) -> CheckResult:
        """CHECK-8: NaN 处理一致性"""
        nan_lines = self._find_lines(r"fillna|dropna|isna|isnull")

        if not nan_lines:
            return CheckResult(
                "CHECK-8",
                "NaN处理",
                "WARN",
                "未见显式 NaN 处理",
                suggestion="添加 fillna/dropna 处理",
            )

        # 检查是否有 reindex + fillna 的危险模式
        dangerous = self._has_pattern(r"reindex.*fillna\(0")

        if dangerous:
            return CheckResult(
                "CHECK-8",
                "NaN处理",
                "WARN",
                "reindex + fillna(0) 可能引入额外 0 值",
                nan_lines,
                "改用 dropna(how='all').fillna(0)",
            )

        return CheckResult("CHECK-8", "NaN处理", "OK")

    def check_9_recursive_filter(self) -> CheckResult:
        """CHECK-9: 递归滤波器使用滚动窗口"""
        # 检查是否有 IIR/递归滤波器特征
        iir_patterns = [
            r"ewm\(",
            r"alpha\s*=.*1\s*-\s*alpha",
            r"for.*date.*in.*all_dates",
        ]

        iir_lines = []
        for pattern in iir_patterns:
            iir_lines.extend(self._find_lines(pattern))

        if not iir_lines:
            return CheckResult("CHECK-9", "递归滤波器", "SKIP", "未使用递归滤波器")

        # 检查是否使用滚动窗口
        has_rolling_window = self._has_pattern(r"window_size|rolling_window|\[-\d+:\]")

        if not has_rolling_window:
            return CheckResult(
                "CHECK-9",
                "递归滤波器",
                "WARN",
                "可能一次算所有日期（初始状态不同）",
                iir_lines,
                "改用滚动窗口，每次从固定起点开始",
            )

        return CheckResult("CHECK-9", "递归滤波器", "OK")

    def check_10_plan_window_dynamic(self) -> CheckResult:
        """CHECK-10: plan_window 动态读取"""
        plan_window_lines = self._find_lines(r"plan_window|get_plan_window")

        if not plan_window_lines:
            return CheckResult("CHECK-10", "plan_window", "SKIP", "未使用 plan_window")

        # 检查是否硬编码
        hardcoded = self._has_pattern(r"return\s+\d{3,4}\s*$", re.MULTILINE)

        if hardcoded:
            return CheckResult(
                "CHECK-10",
                "plan_window",
                "ISSUE",
                "plan_window 硬编码",
                plan_window_lines,
                "从 REQ 动态读取 min_bars * (1 + slack_ratio)",
            )

        return CheckResult("CHECK-10", "plan_window", "OK")

    def check_11_req_sync(self) -> CheckResult:
        """CHECK-11: REQ 配置同步"""
        # 这个检查需要读取外部文件
        local_req_path = Path("config/factor_requirements.json")
        cloud_dumper_path = Path("jq_strategies/cloud_factor_dumper.py")

        if not local_req_path.exists() or not cloud_dumper_path.exists():
            return CheckResult("CHECK-11", "REQ同步", "SKIP", "配置文件不存在")

        try:
            local_req = json.loads(local_req_path.read_text())
            cloud_content = cloud_dumper_path.read_text()

            # 提取云端 REQ - 使用括号匹配而非正则
            req_start = cloud_content.find("REQ = {")
            if req_start == -1:
                req_start = cloud_content.find("REQ= {")
            if req_start == -1:
                req_start = cloud_content.find("REQ ={")

            if req_start == -1:
                return CheckResult("CHECK-11", "REQ同步", "WARN", "云端 REQ 定义未找到")

            # 找到 REQ = { 后的 { 位置
            brace_start = cloud_content.find("{", req_start)
            if brace_start == -1:
                return CheckResult("CHECK-11", "REQ同步", "WARN", "REQ 格式异常")

            # 括号匹配找到完整的字典
            depth = 0
            brace_end = brace_start
            for i, char in enumerate(cloud_content[brace_start:], start=brace_start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        brace_end = i
                        break

            req_str = cloud_content[brace_start : brace_end + 1]
            cloud_req = json.loads(req_str)

            local_factors = set(local_req.get("factors", {}).keys())
            cloud_factors = set(cloud_req.get("factors", {}).keys())

            only_local = local_factors - cloud_factors
            only_cloud = cloud_factors - local_factors

            if only_local or only_cloud:
                msg_parts = []
                if only_local:
                    # 只显示前3个，避免输出过长
                    sample = list(only_local)[:3]
                    suffix = f"...等{len(only_local)}个" if len(only_local) > 3 else ""
                    msg_parts.append(f"仅本地: {sample}{suffix}")
                if only_cloud:
                    sample = list(only_cloud)[:3]
                    suffix = f"...等{len(only_cloud)}个" if len(only_cloud) > 3 else ""
                    msg_parts.append(f"仅云端: {sample}{suffix}")
                return CheckResult(
                    "CHECK-11",
                    "REQ同步",
                    "ISSUE",
                    "; ".join(msg_parts),
                    suggestion="同步 factor_requirements.json 到 cloud_factor_dumper.py",
                )

        except json.JSONDecodeError as e:
            return CheckResult("CHECK-11", "REQ同步", "WARN", f"JSON解析错误: {e}")
        except Exception as e:
            return CheckResult("CHECK-11", "REQ同步", "WARN", f"检查异常: {e}")

        return CheckResult("CHECK-11", "REQ同步", "OK")

    def check_12_field_mapping(self) -> CheckResult:
        """CHECK-12: 云端字段名映射"""
        # 检查是否使用了需要映射的字段
        cloud_fields = ["val_market_cap", "roe_ttm", "val_pe_ratio", "val_pb_ratio"]
        used_fields = []

        for field in cloud_fields:
            if field in self.content:
                used_fields.append(field)

        if not used_fields:
            return CheckResult("CHECK-12", "字段映射", "SKIP", "未使用需映射字段")

        # 检查是否有映射逻辑
        has_mapping = self._has_pattern(r"field_mapping|mapping\s*=\s*\{")

        if not has_mapping:
            return CheckResult(
                "CHECK-12",
                "字段映射",
                "WARN",
                f"使用 {used_fields} 但未见映射逻辑",
                suggestion="在 calculate_factor 开头添加字段映射",
            )

        return CheckResult("CHECK-12", "字段映射", "OK")

    def run_all_checks(self) -> ReviewReport:
        """执行所有检查"""
        report = ReviewReport(
            factor_file=str(self.factor_path),
            factor_name=self.factor_name,
        )

        checks = [
            self.check_1_factor_field,
            self.check_2_avg_price_timing,
            self.check_3_money_no_pit,
            self.check_4_pool_selection_d1,
            self.check_5_regression_pool,
            self.check_6_benchmark_by_date,
            self.check_7_stable_sort,
            self.check_8_nan_handling,
            self.check_9_recursive_filter,
            self.check_10_plan_window_dynamic,
            self.check_11_req_sync,
            self.check_12_field_mapping,
        ]

        for check_func in checks:
            result = check_func()
            report.results.append(result)

        # 根据 ISSUE 推荐日志
        log_suggestions = {
            "CHECK-4": "[POOL_SELECT_LOCAL]",
            "CHECK-5": "[REGRESSION]",
            "CHECK-1": "[FACTOR_INPUT]",
            "CHECK-2": "[FACTOR_INPUT]",
        }

        for r in report.results:
            if r.status == "ISSUE" and r.check_id in log_suggestions:
                log = log_suggestions[r.check_id]
                if log not in report.suggested_logs:
                    report.suggested_logs.append(log)

        return report


def main():
    parser = argparse.ArgumentParser(description="Factor Alignment Static Code Review")
    parser.add_argument("--factor", "-f", required=True, help="Factor file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    factor_path = Path(args.factor)
    if not factor_path.exists():
        print(f"Error: Factor file not found: {factor_path}", file=sys.stderr)
        sys.exit(1)

    reviewer = FactorCodeReviewer(factor_path, verbose=args.verbose)
    report = reviewer.run_all_checks()

    if args.json:
        output = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    else:
        output = report.to_summary()

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Report saved to: {args.output}")
    else:
        print(output)

    # 返回码：有 ISSUE 返回 1
    issues = sum(1 for r in report.results if r.status == "ISSUE")
    sys.exit(1 if issues > 0 else 0)


if __name__ == "__main__":
    main()
