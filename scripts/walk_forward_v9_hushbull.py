# -*- coding: utf-8 -*-
"""
V9_HushBull walk-forward.

Rule: Each rebalance month t, look at the COMPLETED prior calendar month's
HS300 return r_{t-1}. If r_{t-1} > +5%, V9 stays in cash for month t (zero
return). Otherwise run V5 as normal.

This uses only information available at decision time (no look-ahead).
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
OUT = Path('reports/walk_forward_v9'); OUT.mkdir(parents=True, exist_ok=True)

# load full HS300 benchmark history
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

BULL_THRESHOLD = 0.05  # +5% prior month -> hold cash this month


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


def apply_hushbull(v5_perf, threshold=0.05):
    """
    For each calendar month in v5_perf, check the PRIOR completed month's
    HS300 return. If > threshold, zero out V5 returns for the current month.
    Returns: v9_perf, list of bull-trigger months, prior-month returns.
    """
    if v5_perf is None: return None, [], pd.Series(dtype=float)
    df = v5_perf[['net_ret']].copy()
    df.columns = ['v5_ret']
    df['ym'] = df.index.to_period('M')

    # benchmark monthly returns over a wide range
    b_m = BENCH.resample('ME').last().pct_change().dropna()
    b_m_ym = b_m.copy()
    b_m_ym.index = b_m_ym.index.to_period('M')

    df['v9_ret'] = df['v5_ret']
    triggers = []
    prior_rets = {}
    for ym, group in df.groupby('ym'):
        prior = ym - 1
        if prior in b_m_ym.index:
            r = b_m_ym.loc[prior]
            prior_rets[ym] = r
            if r > threshold:
                df.loc[df['ym'] == ym, 'v9_ret'] = 0.0
                triggers.append((str(ym), r))

    df['v9_nav'] = (1 + df['v9_ret']).cumprod()
    out = pd.DataFrame({'net_ret': df['v9_ret'], 'nav': df['v9_nav']}, index=df.index)
    return out, triggers, pd.Series(prior_rets)


def metrics(perf):
    if perf is None or len(perf) == 0:
        return {'total':np.nan,'ann':np.nan,'sharpe':np.nan,'mdd':np.nan}
    nav = perf['nav']; ret = perf['net_ret']
    days = max((nav.index[-1]-nav.index[0]).days, 1)
    total = nav.iloc[-1]-1
    ann = nav.iloc[-1]**(365/days)-1
    vol = ret.std()*np.sqrt(252)
    sharpe = (ann-0.025)/vol if vol>0 else np.nan
    peak = nav.cummax(); mdd = (nav/peak-1).min()
    return {'total':total,'ann':ann,'sharpe':sharpe,'mdd':mdd}


def bench_total(s, e):
    s,e = pd.to_datetime(s), pd.to_datetime(e)
    b = BENCH.loc[(BENCH.index>=s)&(BENCH.index<=e)]
    return b.iloc[-1]/b.iloc[0]-1 if len(b) else np.nan


def main():
    rows = []
    all_triggers = []
    for wname, ws, we in WINDOWS:
        print(f"\n{'='*60}\n[{wname}] {ws} -> {we}\n{'='*60}")
        v5 = run_v5(ws, we); m5 = metrics(v5)
        v9, triggers, _ = apply_hushbull(v5, threshold=BULL_THRESHOLD); m9 = metrics(v9)
        bh = bench_total(ws, we)

        n_trigs = len(triggers)
        trig_str = ', '.join([f"{ym}({r*100:+.1f}%)" for ym, r in triggers]) or 'NONE'
        print(f"  V5         total={m5['total']*100:+6.2f}% sharpe={m5['sharpe']:+.2f} mdd={m5['mdd']*100:+6.2f}%")
        print(f"  V9_HushBull total={m9['total']*100:+6.2f}% sharpe={m9['sharpe']:+.2f} mdd={m9['mdd']*100:+6.2f}%")
        print(f"  HS300 {bh*100:+6.2f}%   Triggers ({n_trigs}): {trig_str}")
        rows.append({
            'window': wname, 'V5': m5['total'], 'V9': m9['total'], 'HS300': bh,
            'V9_vs_V5': m9['total']-m5['total'],
            'V9_beats_V5': m9['total']>m5['total'],
            'n_triggers': n_trigs,
            'triggers': trig_str,
        })
        for ym, r in triggers:
            all_triggers.append({'window': wname, 'month': ym, 'prior_hs300': r})

    df = pd.DataFrame(rows)
    df.to_csv(OUT/'walk_forward_v9.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(all_triggers).to_csv(OUT/'v9_triggers.csv', index=False, encoding='utf-8-sig')

    print("\n" + "="*70 + "\nSUMMARY\n" + "="*70)
    for _, r in df.iterrows():
        f = 'WIN' if r['V9_beats_V5'] else 'LOSE'
        print(f"  {r['window']:20s} V5={r['V5']*100:+6.2f}% V9={r['V9']*100:+6.2f}% "
              f"(vs V5: {r['V9_vs_V5']*100:+.2f}pp) trig={r['n_triggers']} [{f}]")
    wins = df['V9_beats_V5'].sum()
    avg = df['V9_vs_V5'].mean()*100
    total_trigs = df['n_triggers'].sum()
    print(f"\n  V9 beats V5: {wins}/{len(df)}  avg excess: {avg:+.2f}pp  total triggers: {total_trigs}")

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    x = np.arange(len(df)); w = 0.27
    ax = axes[0]
    ax.bar(x-w, df['HS300']*100, w, label='HS300', color='#a0a0a0')
    ax.bar(x,   df['V5']*100, w, label='V5', color='#1f77b4')
    ax.bar(x+w, df['V9']*100, w, label='V9_HushBull', color='#9467bd')
    ax.set_xticks(x); ax.set_xticklabels(df['window'], rotation=15, ha='right')
    ax.set_ylabel('Total Return (%)')
    ax.set_title(f'V9 HushBull (skip month after HS300 monthly +{BULL_THRESHOLD*100:.0f}%)  -  V9 beats V5: {wins}/{len(df)}')
    ax.axhline(0, color='gray', lw=0.5); ax.legend(); ax.grid(alpha=0.3, axis='y')
    for i, n in enumerate(df['n_triggers']):
        ax.annotate(f"trig={n}", (i, max(df.loc[i,'V5'], df.loc[i,'V9'])*100),
                    textcoords='offset points', xytext=(0, 5), ha='center', fontsize=9)

    ax2 = axes[1]
    excess = df['V9_vs_V5']*100
    colors = ['#9467bd' if v>0 else '#d62728' for v in excess]
    ax2.bar(x, excess, color=colors)
    ax2.set_xticks(x); ax2.set_xticklabels(df['window'], rotation=15, ha='right')
    ax2.set_ylabel('V9 - V5 (pp)')
    ax2.set_title(f'V9 Excess over V5  -  Avg: {avg:+.2f}pp,  Total Triggers: {total_trigs}')
    ax2.axhline(0, color='gray', lw=0.5); ax2.grid(alpha=0.3, axis='y')
    for i, v in enumerate(excess):
        ax2.annotate(f"{v:+.2f}pp", (i, v), textcoords='offset points',
                     xytext=(0, 3 if v>=0 else -10), ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT/'walk_forward_v9.png', dpi=130, bbox_inches='tight')
    print(f"\n[Saved] {OUT/'walk_forward_v9.csv'}")
    print(f"[Saved] {OUT/'walk_forward_v9.png'}")
    print(f"[Saved] {OUT/'v9_triggers.csv'}")


if __name__ == '__main__':
    main()
