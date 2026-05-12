# -*- coding: utf-8 -*-
"""
Build CSI500 (zz500) factor panel + risk filters from cached kline.
MVP: price/volume factors only (skip fundamentals).
"""
import os, sys, pickle, time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

for p in ['HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(p, None)
os.environ['NO_PROXY'] = '*'

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from quant.factors import compute_all_factors, FACTOR_REGISTRY

KLINE = Path('data/cache/kline')
OUT = Path('data/cache')
SD = '20210101'
ED = '20260430'

# 1. load constituents
cons = pd.read_parquet(OUT / 'zz500_constituents.parquet')
codes = sorted(cons['code'].astype(str).str.zfill(6).tolist())
print(f"[panel] {len(codes)} CSI500 constituents")

# 2. read kline per stock + compute factors
per_stock = {}
miss = 0
for i, code in enumerate(codes, 1):
    f = KLINE / f"{code}_{SD}_{ED}_qfq.parquet"
    if not f.exists():
        miss += 1
        continue
    kl = pd.read_parquet(f)
    if kl.empty or len(kl) < 200:
        miss += 1
        continue
    kl['date'] = pd.to_datetime(kl['date'])
    kl = kl.sort_values('date').set_index('date')
    per_stock[code] = compute_all_factors(kl)
    if i % 50 == 0 or i == len(codes):
        print(f"  [{i}/{len(codes)}] valid={len(per_stock)} miss={miss}")

# 3. assemble panel
print("\n[panel] assembling factor panel...")
panel = {}
for fname in FACTOR_REGISTRY.keys():
    panel[fname] = pd.DataFrame({
        c: df[fname] for c, df in per_stock.items() if fname in df.columns
    }).sort_index()
    print(f"  {fname}: {panel[fname].shape}")

# Save factor panel
fp_out = OUT / 'factor_panel_zz500.pkl'
with open(fp_out, 'wb') as fout:
    pickle.dump(panel, fout)
print(f"[saved] {fp_out}")

# 4. build risk filters: liquidity_amount_60 only (skip ROE, no fundamentals)
print("\n[risk] building liquidity filter...")
liq_dict = {}
for code in per_stock.keys():
    f = KLINE / f"{code}_{SD}_{ED}_qfq.parquet"
    kl = pd.read_parquet(f)
    kl['date'] = pd.to_datetime(kl['date'])
    kl = kl.sort_values('date').set_index('date')
    if 'amount' in kl.columns:
        liq_dict[code] = kl['amount'].rolling(60, min_periods=20).mean()

target_idx = panel['low_vol_60'].index
liq_df = pd.DataFrame(liq_dict).sort_index()
liq_aligned = liq_df.reindex(liq_df.index.union(target_idx)).sort_index().ffill().reindex(target_idx)
print(f"  liquidity_amount_60: {liq_aligned.shape}, last day cov: {liq_aligned.iloc[-1].notna().sum()}/{len(per_stock)}")

risk = {
    'liquidity_amount_60': liq_aligned,
    # no roe_yoy in MVP
}
rf_out = OUT / 'risk_filters_zz500.pkl'
with open(rf_out, 'wb') as fout:
    pickle.dump(risk, fout)
print(f"[saved] {rf_out}")

# Diagnostic
print("\n=== diagnostic snapshot 2025-12-31 ===")
date = pd.Timestamp('2025-12-31')
for fn in ['momentum_60', 'reversal_5', 'low_vol_60']:
    if date in panel[fn].index:
        s = panel[fn].loc[date].dropna()
        print(f"  {fn}: n={len(s)}, mean={s.mean():.4f}, std={s.std():.4f}")

print("\n[DONE]")
