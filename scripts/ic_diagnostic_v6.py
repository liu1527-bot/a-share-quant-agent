# -*- coding: utf-8 -*-
"""
V6 prep: cross-window IC diagnostic for 9 factors.
- Compute monthly IC (rank-IC) for each factor in each window
- Aggregate to mean IC and IR per window
- Heatmap + recommendation
"""
import os, sys, pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

for p in ['HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(p, None)
os.environ['NO_PROXY'] = '*'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
PANEL = pickle.load(open(ROOT / 'data' / 'cache' / 'factor_panel_hs300.pkl', 'rb'))
KLINE = ROOT / 'data' / 'cache' / 'kline'
OUT = ROOT / 'reports' / 'v6_ic'
OUT.mkdir(parents=True, exist_ok=True)

# Windows
WINDOWS = [
    ('W1_2021H1_top', '2021-01-04', '2021-06-30'),
    ('W2_growth_crash', '2021-07-01', '2022-12-31'),
    ('W3_shock_down', '2023-01-01', '2024-08-31'),
    ('W4_924_rebound', '2024-09-01', '2025-03-31'),
    ('W5_slow_bull', '2025-04-01', '2025-12-31'),
    ('W6_2026_rally', '2026-01-01', '2026-04-30'),
]

FACTORS = list(PANEL.keys())
print(f"[Setup] panel factors: {FACTORS}")
print(f"[Setup] panel range: {next(iter(PANEL.values())).index.min().date()} to {next(iter(PANEL.values())).index.max().date()}")


def build_price_matrix():
    """Load close prices for all 280 stocks (using 2021 cache + 2026 cache)."""
    codes = next(iter(PANEL.values())).columns.tolist()
    print(f"[Price] loading {len(codes)} stocks...")
    frames = []
    for tk in codes:
        # Combine all available kline segments for this ticker
        for fn in [
            f"{tk}_20180101_20201231_qfq.parquet",
            f"{tk}_20210101_20251231_qfq.parquet",
            f"{tk}_20250901_20260430_qfq.parquet",
        ]:
            p = KLINE / fn
            if p.exists():
                df = pd.read_parquet(p)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')[['close']].rename(columns={'close': tk})
                frames.append(df)
    if not frames:
        raise RuntimeError("no kline files found!")
    price = pd.concat(frames, axis=1)
    # Merge duplicate columns (cumulative + 2026 overlap) — newer pandas needs T+groupby+T
    price = price.T.groupby(level=0).last().T
    price = price.sort_index()
    print(f"[Price] matrix shape={price.shape}, range {price.index.min().date()} to {price.index.max().date()}")
    return price


PRICE = build_price_matrix()


def get_rebalance_dates(start, end, freq='ME'):
    s = pd.Series(1, index=PRICE.index)
    s = s[(s.index >= start) & (s.index <= end)]
    return s.resample(freq).last().dropna().index


def compute_ic_one_period(factor_df, t_now, t_next):
    """Rank-IC = Spearman corr between factor at t_now and ret t_now->t_next."""
    valid = factor_df.index[factor_df.index <= t_now]
    if len(valid) == 0:
        return np.nan
    f = factor_df.loc[valid[-1]]
    if t_now not in PRICE.index or t_next not in PRICE.index:
        # find nearest
        before = PRICE.index[PRICE.index <= t_now]
        after = PRICE.index[(PRICE.index > t_now) & (PRICE.index <= t_next)]
        if len(before) == 0 or len(after) == 0:
            return np.nan
        t_now = before[-1]
        t_next = after[-1]
    p0 = PRICE.loc[t_now]
    p1 = PRICE.loc[t_next]
    ret = (p1 / p0 - 1)
    # align
    common = f.dropna().index.intersection(ret.dropna().index)
    if len(common) < 30:
        return np.nan
    return f.loc[common].rank().corr(ret.loc[common].rank())


def diagnostic():
    results = []
    for wname, wstart, wend in WINDOWS:
        rebal = get_rebalance_dates(wstart, wend)
        if len(rebal) < 2:
            print(f"[skip] {wname}: not enough rebalance dates")
            continue
        for fname, fdf in PANEL.items():
            ics = []
            for i in range(len(rebal) - 1):
                ic = compute_ic_one_period(fdf, rebal[i], rebal[i + 1])
                if not np.isnan(ic):
                    ics.append(ic)
            if len(ics) < 2:
                results.append({'window': wname, 'factor': fname, 'n': 0, 'mean_ic': np.nan, 'std_ic': np.nan, 'ir': np.nan})
                continue
            arr = np.array(ics)
            mean = arr.mean()
            std = arr.std(ddof=1)
            ir = mean / std if std > 0 else np.nan
            results.append({
                'window': wname,
                'factor': fname,
                'n': len(ics),
                'mean_ic': mean,
                'std_ic': std,
                'ir': ir,
            })
    df = pd.DataFrame(results)
    df.to_csv(OUT / 'cross_window_ic.csv', index=False, encoding='utf-8-sig')
    print(f"\n[Saved] {OUT / 'cross_window_ic.csv'}")

    # Pivot for heatmap
    pivot_ic = df.pivot(index='factor', columns='window', values='mean_ic')
    pivot_ir = df.pivot(index='factor', columns='window', values='ir')
    pivot_ic.to_csv(OUT / 'pivot_mean_ic.csv', encoding='utf-8-sig')
    pivot_ir.to_csv(OUT / 'pivot_ir.csv', encoding='utf-8-sig')

    # Order factors
    factor_order = ['value_pb', 'value_pe', 'quality_roe',
                    'reversal_5', 'low_vol_60', 'amount_ratio_5_20',
                    'momentum_60', 'momentum_120_5', 'ma_pos_20']
    factor_order = [f for f in factor_order if f in pivot_ic.index]
    pivot_ic = pivot_ic.loc[factor_order]
    pivot_ir = pivot_ir.loc[factor_order]

    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    for ax, pv, title, vmax in [
        (axes[0], pivot_ic, 'Mean IC by Window', 0.15),
        (axes[1], pivot_ir, 'IR by Window (Mean/Std)', 1.5),
    ]:
        im = ax.imshow(pv.values, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels(pv.columns, rotation=30, ha='right', fontsize=9)
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels(pv.index, fontsize=10)
        ax.set_title(title, fontsize=12)
        for i in range(len(pv.index)):
            for j in range(len(pv.columns)):
                v = pv.iloc[i, j]
                if pd.isna(v):
                    txt = '-'
                else:
                    txt = f"{v:+.2f}"
                ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                        color='black' if abs(v) < vmax * 0.6 else 'white' if abs(v) > vmax * 0.8 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.suptitle('V6 Cross-Window IC/IR Diagnostic - 9 Factors x 6 Regimes', fontsize=13)
    plt.tight_layout()
    fig_path = OUT / 'ic_heatmap.png'
    plt.savefig(fig_path, dpi=130, bbox_inches='tight')
    print(f"[Saved] {fig_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Mean IC by Window (positive = factor works that way)")
    print("=" * 70)
    print(pivot_ic.round(3).to_string())
    print("\n" + "=" * 70)
    print("IR by Window (>0.5 strong, <0.2 noise)")
    print("=" * 70)
    print(pivot_ir.round(2).to_string())

    # Spotlight W6
    if 'W6_2026_rally' in pivot_ic.columns:
        print("\n" + "=" * 70)
        print("SPOTLIGHT: W6 (2026 rally) - which factors deliver alpha?")
        print("=" * 70)
        w6 = pd.DataFrame({
            'mean_ic': pivot_ic['W6_2026_rally'],
            'ir': pivot_ir['W6_2026_rally'],
        }).sort_values('ir', ascending=False)
        print(w6.round(3).to_string())


if __name__ == '__main__':
    diagnostic()
