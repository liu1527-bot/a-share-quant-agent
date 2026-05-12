# -*- coding: utf-8 -*-
"""
V9 threshold sensitivity scan.

Test thresholds: 5%, 7%, 10%, 12%, 15% on the same 4 walk-forward windows.
Decision rule (committed upfront):
  Robust if:  win_rate >= 3/4  AND  avg_excess >= +1pp  AND  worst_window >= -3pp
"""
import os, sys, pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
for p in ['HTTP_PROXY', 'HTTPS_PROXY']: os.environ.pop(p, None)
os.environ['NO_PROXY'] = '*'
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sys.stdout.reconfigure(encoding='utf-8')

from quant import config
from quant.strategy import generate_holdings
from quant.backtest import holdings_to_weights, compute_portfolio_returns

PANEL = pickle.load(open('data/cache/factor_panel_hs300.pkl', 'rb'))
RISK = pickle.load(open('data/cache/risk_filters_hs300.pkl', 'rb'))
KLINE = Path('data/cache/kline')
OUT = Path('reports/v9_threshold_scan'); OUT.mkdir(parents=True, exist_ok=True)

import glob
bench_frames = []
for f in glob.glob('data/cache/benchmark_000300_*.parquet'):
    b = pd.read_parquet(f)
    b['date'] = pd.to_datetime(b['date'])
    bench_frames.append(b.set_index('date')['close'])
BENCH = pd.concat(bench_frames).sort_index()
BENCH = BENCH[~BENCH.index.duplicated(keep='last')]

V5_WEIGHTS = {'value_pb': 0.30, 'value_pe': 0.25, 'reversal_5': 0.20,
              'low_vol_60': 0.15, 'momentum_120_5': 0.10}

WINDOWS = [
    ('W3_shock_down',   '2023-01-01', '2024-08-31'),
    ('W4_924_rebound',  '2024-09-01', '2025-03-31'),
    ('W5_slow_bull',    '2025-04-01', '2025-12-31'),
    ('W6_2026_rally',   '2026-01-01', '2026-04-30'),
]
THRESHOLDS = [0.05, 0.07, 0.10, 0.12, 0.15]


def build_price_from_cache(codes):
    frames = []
    for tk in codes:
        for fn in [f"{tk}_20180101_20201231_qfq.parquet",
                   f"{tk}_20210101_20251231_qfq.parquet",
                   f"{tk}_20250901_20260430_qfq.parquet"]:
            p = KLINE / fn
            if p.exists():
                df = pd.read_parquet(p)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')[['close']].rename(columns={'close': tk})
                frames.append(df)
    if not frames: return pd.DataFrame()
    price = pd.concat(frames, axis=1).T.groupby(level=0).last().T
    return price.sort_index()


def run_v5(start, end):
    cf = (config.REBALANCE_FREQ, config.TOP_N, config.MAX_PER_INDUSTRY, config.FACTOR_WEIGHTS)
    config.REBALANCE_FREQ='ME'; config.TOP_N=30; config.MAX_PER_INDUSTRY=3
    config.FACTOR_WEIGHTS = V5_WEIGHTS
    try:
        h = generate_holdings(PANEL, start_date=start, end_date=end, top_n=30, risk_panel=RISK)
        if not h: return None
        codes = sorted(set().union(*[set(d.index) for d in h.values()]))
        price = build_price_from_cache(codes)
        price = price.loc[(price.index >= start) & (price.index <= end)].dropna(how='all')
        wdf = holdings_to_weights(h, price.index, codes)
        return compute_portfolio_returns(wdf, price, cost_per_side=0.0015).dropna()
    finally:
        (config.REBALANCE_FREQ, config.TOP_N, config.MAX_PER_INDUSTRY, config.FACTOR_WEIGHTS) = cf


def apply_hushbull(v5_perf, threshold):
    if v5_perf is None: return None, 0
    df = v5_perf[['net_ret']].copy()
    df.columns = ['v5_ret']
    df['ym'] = df.index.to_period('M')
    b_m = BENCH.resample('ME').last().pct_change().dropna()
    b_m_ym = b_m.copy(); b_m_ym.index = b_m_ym.index.to_period('M')
    df['v9_ret'] = df['v5_ret']
    n_trig = 0
    for ym, group in df.groupby('ym'):
        prior = ym - 1
        if prior in b_m_ym.index and b_m_ym.loc[prior] > threshold:
            df.loc[df['ym'] == ym, 'v9_ret'] = 0.0
            n_trig += 1
    df['v9_nav'] = (1 + df['v9_ret']).cumprod()
    out = pd.DataFrame({'net_ret': df['v9_ret'], 'nav': df['v9_nav']}, index=df.index)
    return out, n_trig


def total_return(perf):
    if perf is None or len(perf) == 0: return np.nan
    return perf['nav'].iloc[-1] - 1


def main():
    # cache V5 perf per window (run once)
    print("Running V5 baseline for each window...")
    v5_cache = {}
    v5_totals = {}
    for wname, ws, we in WINDOWS:
        v5 = run_v5(ws, we)
        v5_cache[wname] = (v5, ws, we)
        v5_totals[wname] = total_return(v5)
        print(f"  V5 [{wname}] = {v5_totals[wname]*100:+.2f}%")

    # scan thresholds
    rows = []
    for thr in THRESHOLDS:
        row = {'threshold_pct': thr*100}
        excesses = []
        wins = 0
        total_trig = 0
        for wname, (v5, ws, we) in v5_cache.items():
            v9, ntr = apply_hushbull(v5, thr)
            v9_total = total_return(v9)
            row[f'{wname}_V9'] = v9_total
            row[f'{wname}_excess'] = v9_total - v5_totals[wname]
            row[f'{wname}_trig'] = ntr
            excesses.append(v9_total - v5_totals[wname])
            if v9_total > v5_totals[wname]: wins += 1
            total_trig += ntr
        row['avg_excess'] = np.mean(excesses)
        row['worst_excess'] = np.min(excesses)
        row['win_rate'] = wins/len(WINDOWS)
        row['total_triggers'] = total_trig
        # robust check
        row['robust'] = (
            row['win_rate'] >= 0.75 and
            row['avg_excess'] >= 0.01 and
            row['worst_excess'] >= -0.03
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT/'v9_threshold_scan.csv', index=False, encoding='utf-8-sig')

    print("\n" + "="*100)
    print("V9 THRESHOLD SCAN RESULTS")
    print("="*100)
    print(f"{'Thr%':>5} {'avg_exc':>9} {'worst':>9} {'win':>5} {'trig':>5} {'robust':>7}  per-window excess (pp)")
    for _, r in df.iterrows():
        per_win = '  '.join([f"{r[f'{w[0]}_excess']*100:+6.2f}" for w in WINDOWS])
        rob = 'YES' if r['robust'] else 'no'
        print(f"{r['threshold_pct']:>4.0f}% {r['avg_excess']*100:>+8.2f} {r['worst_excess']*100:>+8.2f} "
              f"{r['win_rate']*4:>3.0f}/4 {r['total_triggers']:>4.0f}  {rob:>6}  {per_win}")

    # plot heatmap-style
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    ax = axes[0]
    win_names = [w[0] for w in WINDOWS]
    excess_matrix = np.array([[df.loc[i, f'{w}_excess']*100 for w in win_names] for i in df.index])
    im = ax.imshow(excess_matrix, aspect='auto', cmap='RdYlGn', vmin=-10, vmax=10)
    ax.set_xticks(range(len(win_names))); ax.set_xticklabels(win_names, rotation=15, ha='right')
    ax.set_yticks(range(len(df))); ax.set_yticklabels([f"{int(t*100)}%" for t in THRESHOLDS])
    ax.set_xlabel('Window'); ax.set_ylabel('Threshold')
    ax.set_title('V9 Excess vs V5 (pp) by Threshold and Window')
    plt.colorbar(im, ax=ax, label='Excess (pp)')
    for i in range(len(df)):
        for j in range(len(win_names)):
            ax.text(j, i, f"{excess_matrix[i,j]:+.1f}", ha='center', va='center',
                    color='black', fontsize=10, fontweight='bold')

    ax2 = axes[1]
    x = np.arange(len(df))
    avg = df['avg_excess']*100
    worst = df['worst_excess']*100
    ax2.bar(x-0.2, avg, 0.4, label='Avg excess (pp)', color='steelblue')
    ax2.bar(x+0.2, worst, 0.4, label='Worst window (pp)', color='salmon')
    ax2.set_xticks(x); ax2.set_xticklabels([f"{int(t*100)}%" for t in THRESHOLDS])
    ax2.set_xlabel('Threshold'); ax2.set_ylabel('Excess (pp)')
    ax2.set_title('V9 Avg vs Worst across Thresholds')
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.axhline(1, color='green', lw=0.5, linestyle='--', label='Robust target +1pp')
    ax2.axhline(-3, color='red', lw=0.5, linestyle='--', label='Robust floor -3pp')
    ax2.legend(); ax2.grid(alpha=0.3, axis='y')
    for i, (a, w) in enumerate(zip(avg, worst)):
        ax2.annotate(f"{a:+.2f}", (i-0.2, a), textcoords='offset points',
                     xytext=(0, 3 if a>=0 else -10), ha='center', fontsize=9)
        ax2.annotate(f"{w:+.2f}", (i+0.2, w), textcoords='offset points',
                     xytext=(0, 3 if w>=0 else -10), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT/'v9_threshold_scan.png', dpi=130, bbox_inches='tight')
    print(f"\n[Saved] {OUT/'v9_threshold_scan.csv'}")
    print(f"[Saved] {OUT/'v9_threshold_scan.png'}")

    robust_rows = df[df['robust']]
    if len(robust_rows) > 0:
        print(f"\n[VERDICT] Robust thresholds: {robust_rows['threshold_pct'].tolist()}")
    else:
        print("\n[VERDICT] No threshold passes the robust test. V9 is fragile / single-point lucky.")


if __name__ == '__main__':
    main()
