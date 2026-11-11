#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloud Diagnostic: PIT Adjustment Verification on JoinQuant

Upload this to JoinQuant research environment and run to verify
that cloud factor values use PIT-adjusted prices.

Expected output: Shows intermediate PIT values for 003017.XSHE
confirming that get_price(fq='pre') returns PIT-adjusted close/volume.
"""

import numpy as np
import pandas as pd

# Test stock and dates
TEST_STOCK = '003017.XSHE'
FACTOR_DATE = '2024-02-06'
WINDOW_60D = 90  # fetch extra for safety


def compute_pvdc_60d(vol_data, close_data, window=60):
    """Compute PVDC factor for the last date in the series."""
    n = len(vol_data)
    if n < window + 1:
        return float('nan')
    v_delta = np.diff(vol_data[n - window - 1:n])
    c_delta = np.diff(close_data[n - window - 1:n])
    mask = ~(np.isnan(v_delta) | np.isnan(c_delta))
    if mask.sum() < int(window * 0.7):
        return float('nan')
    vd, cd = v_delta[mask], c_delta[mask]
    cov_val = np.sum((vd - vd.mean()) * (cd - cd.mean())) / (len(vd) - 1)
    return -cov_val


print("=" * 70)
print("PIT Adjustment Verification - Cloud Side")
print("=" * 70)

# Scenario 1: Raw prices (fq=None)
df_raw = get_price(
    TEST_STOCK,
    end_date=FACTOR_DATE,
    count=WINDOW_60D,
    fields=['close', 'volume', 'money'],
    frequency='daily',
    fq=None,
    panel=False,
)
print(f"\nScenario 1: Raw prices (fq=None)")
print(f"  Rows: {len(df_raw)}")
print(f"  Close range: [{df_raw['close'].min():.2f}, {df_raw['close'].max():.2f}]")
print(f"  Volume range: [{df_raw['volume'].min():.0f}, {df_raw['volume'].max():.0f}]")

# Scenario 2: PIT-adjusted prices (fq='pre')
df_pit = get_price(
    TEST_STOCK,
    end_date=FACTOR_DATE,
    count=WINDOW_60D,
    fields=['close', 'volume', 'money'],
    frequency='daily',
    fq='pre',
    panel=False,
)
print(f"\nScenario 2: PIT-adjusted (fq='pre')")
print(f"  Rows: {len(df_pit)}")
print(f"  Close range: [{df_pit['close'].min():.2f}, {df_pit['close'].max():.2f}]")
print(f"  Volume range: [{df_pit['volume'].min():.0f}, {df_pit['volume'].max():.0f}]")

# Scenario 3: Get factor values for manual PIT
df_factor = get_price(
    TEST_STOCK,
    end_date=FACTOR_DATE,
    count=WINDOW_60D,
    fields=['factor'],
    frequency='daily',
    fq='pre',
    panel=False,
)
print(f"\nFactor (PIT) values:")
unique_factors = df_factor['factor'].unique()
print(f"  Unique values: {sorted(unique_factors)}")
for f in sorted(unique_factors):
    dates_with_f = df_factor[df_factor['factor'] == f].index
    print(f"  factor={f:.6f}: {dates_with_f[0]} to {dates_with_f[-1]} ({len(dates_with_f)} days)")

# Compute PVDC_60D under each scenario
pvdc_raw = compute_pvdc_60d(
    df_raw['volume'].values.astype(np.float64),
    df_raw['close'].values.astype(np.float64),
    60,
)
pvdc_pit = compute_pvdc_60d(
    df_pit['volume'].values.astype(np.float64),
    df_pit['close'].values.astype(np.float64),
    60,
)

# Scenario 4: Manual PIT using raw + factor
factor_end = df_factor['factor'].iloc[-1]
factor_ratio = df_factor['factor'].values / factor_end
close_manual_pit = df_raw['close'].values * factor_ratio
volume_manual_pit = df_raw['volume'].values / factor_ratio
pvdc_manual_pit = compute_pvdc_60d(
    volume_manual_pit.astype(np.float64),
    close_manual_pit.astype(np.float64),
    60,
)

print(f"\n{'─' * 70}")
print(f"PVDC_60D for {TEST_STOCK} at {FACTOR_DATE}:")
print(f"{'─' * 70}")
print(f"  Raw (fq=None):           {pvdc_raw:>15.2f}")
print(f"  PIT (fq='pre'):          {pvdc_pit:>15.2f}")
print(f"  Manual PIT (raw×ratio):  {pvdc_manual_pit:>15.2f}")
print(f"  Ratio PIT/Raw:           {pvdc_pit/pvdc_raw:>15.4f}")
print()

# Reference values from alignment
print("Reference from alignment data:")
print(f"  JQ  (cloud) PVDC_60D:    {-252443.70:>15.2f}")
print(f"  Local       PVDC_60D:    {-30013.24:>15.2f}")
print()

# Determine which scenario the cloud uses
diff_raw = abs(pvdc_raw - (-252443.70))
diff_pit = abs(pvdc_pit - (-252443.70))
diff_manual = abs(pvdc_manual_pit - (-252443.70))

print("Which scenario matches the cloud alignment value?")
print(f"  |Raw - JQ| = {diff_raw:.2f}")
print(f"  |PIT - JQ| = {diff_pit:.2f}")
print(f"  |Manual PIT - JQ| = {diff_manual:.2f}")
best = min([(diff_raw, 'RAW'), (diff_pit, 'PIT'), (diff_manual, 'MANUAL_PIT')], key=lambda x: x[0])
print(f"  → Cloud value closest to: {best[1]} (diff={best[0]:.2f})")

print(f"\n{'=' * 70}")
print("CONCLUSION")
print(f"{'=' * 70}")
if best[1] in ('PIT', 'MANUAL_PIT'):
    print("  Cloud uses PIT-adjusted prices (fq='pre' or manual PIT).")
    print("  Local uses RAW prices (no PIT adjustment).")
    print("  This explains the value difference with preserved Spearman correlation.")
else:
    print("  Cloud also uses RAW prices. Investigate other causes.")
