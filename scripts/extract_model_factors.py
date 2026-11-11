#!/usr/bin/env python3
"""Extract all factors used across model bundles in results/."""

import json
from pathlib import Path
from collections import defaultdict


def extract_factors_from_models() -> dict:
    """Scan all model bundles and extract factor usage."""
    results_dir = Path("results")
    
    # Track factor usage: factor_name -> list of experiments using it
    factor_usage: dict[str, set[str]] = defaultdict(set)
    # Track per-experiment factors
    exp_factors: dict[str, set[str]] = {}
    
    for model_file in sorted(results_dir.glob("lasso_model_bundle_*.json")):
        exp_name = model_file.stem.replace("lasso_model_bundle_", "")
        exp_factors[exp_name] = set()
        
        with open(model_file) as f:
            data = json.load(f)
        
        # Extract factors from all rolling models
        for model in data.get("index", {}).get("models", []):
            features = model.get("feature_names", [])
            for feat in features:
                # Remove suffix like "_small_cap_9_rules"
                base_name = feat.replace("_small_cap_9_rules", "")
                factor_usage[base_name].add(exp_name)
                exp_factors[exp_name].add(base_name)
    
    return factor_usage, exp_factors


def main():
    factor_usage, exp_factors = extract_factors_from_models()
    
    # Sort by usage count (descending)
    sorted_factors = sorted(factor_usage.items(), key=lambda x: -len(x[1]))
    total_exps = len(exp_factors)
    
    print("=" * 80)
    print("因子使用频率统计")
    print("=" * 80)
    print(f"\n总唯一因子数: {len(sorted_factors)}")
    print(f"总实验数: {total_exps}")
    
    # 频率分布统计
    freq_buckets = {
        "高频 (≥15 exp)": [],
        "中高频 (10-14 exp)": [],
        "中频 (5-9 exp)": [],
        "低频 (2-4 exp)": [],
        "极低频 (1 exp)": [],
    }
    
    for factor, exps in sorted_factors:
        count = len(exps)
        if count >= 15:
            freq_buckets["高频 (≥15 exp)"].append((factor, count))
        elif count >= 10:
            freq_buckets["中高频 (10-14 exp)"].append((factor, count))
        elif count >= 5:
            freq_buckets["中频 (5-9 exp)"].append((factor, count))
        elif count >= 2:
            freq_buckets["低频 (2-4 exp)"].append((factor, count))
        else:
            freq_buckets["极低频 (1 exp)"].append((factor, count))
    
    print("\n" + "=" * 80)
    print("频率分布")
    print("=" * 80)
    for bucket, factors in freq_buckets.items():
        print(f"\n{bucket}: {len(factors)} 个因子")
    
    # 详细列表
    for bucket, factors in freq_buckets.items():
        print("\n" + "-" * 80)
        print(f"{bucket} ({len(factors)} 个)")
        print("-" * 80)
        for factor, count in factors:
            pct = count / total_exps * 100
            print(f"  {factor:55s} | {count:2d} exp ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
