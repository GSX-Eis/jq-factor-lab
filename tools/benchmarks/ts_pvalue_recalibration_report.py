"""Publish the frozen JSON and a Markdown table for every NERE-1279 threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.benchmarks.ts_pvalue_checkpoint import BINDING_KEYS
from utils.local_outputs import write_json


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _percent(value: float) -> str:
    return f"{100 * value:.3f}%"


def _interval(values: list[float]) -> str:
    return " – ".join(map(_percent, values))


def _difference_interval(values: list[float]) -> str:
    return " 至 ".join(f"{100 * value:+.3f}" for value in values) + " 个百分点"


def _mark(passes: bool) -> str:
    return "通过" if passes else "未通过"


def markdown(report: dict) -> str:
    diag, pilot, decision = report["diagnostic"], report["pilot"], report["decision"]
    lines = [
        "# NERE-1279 时序 bootstrap P 值确认性重校准", "",
        f"**最终判定 {_mark(decision['passes'])}。**", "",
        f"第 0 步合并拒绝率 {_percent(diag['rejection_rate'])}"
        f"（{diag['rejected']:,}/900,000），达到 5.25% 路线 A 阈值。", "",
        f"第 0 步登记 `{diag['preregistration_commit']}`，计算源码 `{diag['computation_commit']}`。",
        f"路线 A 登记 `{report['preregistration_commit']}`。",
        f"主矩阵计算源码 `{report['warp_computation_commit']}`，"
        f"嵌套计算源码 `{report['computation_commit']}`。",
        "在 Git 中用 `git merge-base --is-ancestor <登记> <计算源码>` 核验先后顺序。"
        "原始计数提交也包含两个登记提交。", "",
        f"本机试算估计主矩阵 {pilot['warp']['estimated_wall_seconds'] / 3600:.2f} 小时，"
        f"嵌套 {pilot['nested']['estimated_wall_seconds'] / 3600:.2f} 小时。"
        f"T=4096 试算 {pilot['probe_4096']['wall_seconds']:.1f} 秒，"
        f"外推 {pilot['probe_4096']['estimated_wall_seconds'] / 3600:.2f} 小时。", "",
        f"第 0 步、本次主矩阵、嵌套实际计算分别为 {diag['seconds']:.1f} 秒、"
        f"{report['warp_seconds']:.1f} 秒、{report['nested_seconds']:.1f} 秒。", "",
        "## 零效应与正负对照逐格结果", "",
        "零效应的 99% 区间同时计入评估和临界值误差，硬上限用单侧 99% 上界。"
        "Holm 对 162 个零效应阈值统一校正。正负对照报告有方向的功效"
        "及同时考虑临界值和评估不确定性的区间。", "",
        "| 场景 | T | h | α | 拒绝率 | 99% 区间 | 单侧上界 | Holm P | 同向功效 | 逐格判定 |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for cell in report["cells"]:
        for row in cell["thresholds"]:
            power = cell["sign"] != 0
            lines.append(
                f"| {cell['scenario']} | {cell['grid_size']} | {cell['horizon']} | {row['alpha']:g} | "
                f"{_percent(row['rejection_rate'])} | {_interval(row['power_ci99'] if power else row['ci99'])} | "
                + (f"— | — | {_percent(row['power'])}" if power
                   else f"{_percent(row['upper99'])} | {row['holm_p']:.6g} | —")
                + f" | {_mark(row['passes'])} |"
            )
    lines += [
        "", "## 合并偏差", "",
        "| T | α | 全部零效应拒绝率 | 全部判定 | 排除失败特征后的拒绝率 | 排除后判定 |",
        "| ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for full, retained in zip(decision["full_pooled"], decision["retained_pooled"], strict=True):
        lines.append(
            f"| {full['grid_size']} | {full['alpha']:g} | {_percent(full['rejection_rate'])} | "
            f"{_mark(full['passes'])} | {_percent(retained['rejection_rate'])} | {_mark(retained['passes'])} |"
        )
    lines += [
        "", "## 生产 P 值嵌套抽查", "",
        "每格模拟 4,000 个样本，每个样本使用生产 PCG64 圆形块随机流和 9,999 次重抽样计算 +1 双侧 P 值。"
        "百分点差值的 99% 区间计入嵌套与 warp-speed 的抽样误差。", "",
        "| 场景 | T | h | 嵌套拒绝率 | Warp-speed 拒绝率 | 差值 99% 区间 | 逐格判定 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    nested = report["nested"]
    for row in nested["cells"]:
        lines.append(
            f"| {row['scenario']} | {row['grid_size']} | {row['horizon']} | "
            f"{_percent(row['rejection_rate'])} | {_percent(row['warp_rate'])} | "
            f"{_difference_interval(row['ci99'])} | {_mark(row['passes'])} |"
        )
    lines += [
        "", f"合并差值区间 {_difference_interval(nested['pooled']['ci99'])}。嵌套判定 {_mark(nested['passes'])}。", "",
        "## 适用范围", "",
    ]
    if decision["excluded_features"]:
        lines.append("按预登记退路评估排除 " + "、".join(decision["excluded_features"]) + "。")
    if decision["exclude_512_alpha_005"]:
        lines.append("按预登记收窄 T=512，仅保留 α=0.01 和 0.0025。")
    if decision["passes"]:
        lines.append("通过范围限于登记的离散 T、h、α、保留的合成特征及单品种完整网格。")
    else:
        if decision["retained_failures"] and not decision["exclude_512_alpha_005"]:
            pairs = "、".join(f"T={size}、α={alpha:g}" for size, alpha in decision["retained_failures"])
            lines.append(f"排除失败特征后仍有未通过配置（{pairs}），超出预登记退路。")
        lines.append("v3 P 值仍用于诊断。时序择优门禁继续关闭。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    pilot, warp, result = [_read(args.workspace / f"{stage}.json") for stage in ("pilot", "warp", "nested")]
    for key in BINDING_KEYS:
        if key not in pilot or not pilot[key] == warp.get(key) == result.get(key):
            raise ValueError(f"Inconsistent {key} in the experiment parts")
    diagnostic = _read(args.report_dir / "step0_diagnostic.json")
    report = result | dict(diagnostic=diagnostic, pilot=pilot, cells=warp["cells"],
                           warp_computation_commit=warp["computation_commit"],
                           warp_seconds=warp["seconds"], nested_seconds=result["seconds"])
    write_json(args.report_dir / "ts_recalibration.json", report)
    (args.report_dir / "ts_recalibration.md").write_text(markdown(report))


if __name__ == "__main__":
    main()
