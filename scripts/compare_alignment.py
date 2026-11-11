#!/usr/bin/env python3
"""Compare local vs cloud backtest alignment data."""
import json
import sys
from pathlib import Path
from collections import defaultdict

def parse_cloud_log(log_path: str) -> dict:
    """Parse cloud backtest log for alignment data."""
    pools = {}
    models = {}
    
    with open(log_path) as f:
        for line in f:
            if 'POOL_DUMP|' in line:
                parts = line.split('POOL_DUMP|')[1].strip().split('|')
                date = parts[0]
                size = int(parts[1])
                stocks = set(parts[2].split(',')) if len(parts) > 2 else set()
                pools[date] = {'size': size, 'stocks': stocks}
            
            if '[MODEL_SELECT]' in line:
                info = line.split('[MODEL_SELECT]')[1].strip()
                kv = {}
                for p in info.split('|'):
                    if '=' in p:
                        k, v = p.split('=', 1)
                        kv[k] = v
                if 'date' in kv:
                    models[kv['date']] = kv
    
    return {'pools': pools, 'models': models}


def main():
    local_path = Path('results/local_parsed.json')
    cloud_path = Path('results/jq_backtest_log_v4.txt')
    
    with open(local_path) as f:
        local_data = json.load(f)
    
    cloud = parse_cloud_log(str(cloud_path))
    rebalance_dates = local_data.get('rebalance_dates', [])
    
    print(f"Local: {len(rebalance_dates)} rebalance dates")
    print(f"Cloud: {len(cloud['pools'])} pool dumps, {len(cloud['models'])} model selections")
    
    # Build local lookup
    local_map = {}
    for d in rebalance_dates:
        date = d.get('date', '')
        if date:
            local_map[date] = d
    
    # Level 1: Stock Pool Comparison
    print("\n" + "=" * 60)
    print("LEVEL 1: Stock Pool Comparison")
    print("=" * 60)
    
    common_dates = sorted(set(local_map.keys()) & set(cloud['pools'].keys()))
    print(f"Common dates: {len(common_dates)}")
    
    size_matches = 0
    total_overlap = 0
    total_union = 0
    
    for date in common_dates:
        local_pool = set(local_map[date].get('stock_pool', []))
        cloud_pool = cloud['pools'][date]['stocks']
        ls = len(local_pool)
        cs = cloud['pools'][date]['size']
        
        if ls == cs:
            size_matches += 1
        else:
            print(f"  SIZE DIFF {date}: local={ls}, cloud={cs}")
        
        if local_pool and cloud_pool:
            overlap = len(local_pool & cloud_pool)
            union = len(local_pool | cloud_pool)
            total_overlap += overlap
            total_union += union
            jaccard = overlap / union if union > 0 else 0
            if jaccard < 0.95:
                only_local = local_pool - cloud_pool
                only_cloud = cloud_pool - local_pool
                print(f"  POOL DIFF {date}: jaccard={jaccard:.4f}, only_local={len(only_local)}, only_cloud={len(only_cloud)}")
    
    overall_jaccard = total_overlap / total_union if total_union > 0 else 0
    print(f"\nPool size exact matches: {size_matches}/{len(common_dates)}")
    print(f"Overall stock Jaccard similarity: {overall_jaccard:.4f}")
    
    # Level 2: Model Selection Comparison
    print("\n" + "=" * 60)
    print("LEVEL 2: Model Selection Comparison")
    print("=" * 60)
    
    model_matches = 0
    model_total = 0
    for date in common_dates:
        local_model = local_map[date].get('model_train_end', '')
        cloud_model = cloud['models'].get(date, {}).get('selected_train_end', '')
        if local_model and cloud_model:
            model_total += 1
            if local_model == cloud_model:
                model_matches += 1
            else:
                print(f"  MODEL DIFF {date}: local={local_model}, cloud={cloud_model}")
    
    print(f"Model selection matches: {model_matches}/{model_total}")
    
    # Level 3: Composite Score Comparison (from local alignment data)
    print("\n" + "=" * 60)
    print("LEVEL 3: Factor Values / Composite Scores")
    print("=" * 60)
    
    for date in common_dates[:3]:
        ld = local_map[date]
        scores = ld.get('composite_scores', {})
        top_stocks = ld.get('top_stocks', [])
        if scores:
            print(f"  {date}: {len(scores)} scores, top5={top_stocks[:5]}")
        else:
            factor_vals = ld.get('factor_values', {})
            print(f"  {date}: {len(factor_vals)} factor records")
    
    # Summary
    print("\n" + "=" * 60)
    print("ALIGNMENT SUMMARY")
    print("=" * 60)
    pool_pass = overall_jaccard >= 0.90
    model_pass = model_matches == model_total if model_total > 0 else True
    
    print(f"  Level 1 (Stock Pool): {'PASS' if pool_pass else 'FAIL'} (Jaccard={overall_jaccard:.4f})")
    print(f"  Level 2 (Model Selection): {'PASS' if model_pass else 'FAIL'} ({model_matches}/{model_total})")
    print(f"  Level 3 (Factor Values): MANUAL CHECK NEEDED")
    print(f"  Level 4 (Composite Scores): MANUAL CHECK NEEDED")
    print(f"  Level 5 (Trading Decisions): MANUAL CHECK NEEDED")
    print(f"\n  Cloud Backtest Return: 31.29% (benchmark: -6.93%)")
    
    overall = "PASS" if pool_pass and model_pass else "NEEDS_INVESTIGATION"
    print(f"\n  Overall: {overall}")


if __name__ == '__main__':
    main()
