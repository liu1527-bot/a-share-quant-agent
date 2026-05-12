# -*- coding: utf-8 -*-
"""
V5+V8 risk-parity (L2) walk-forward.
Each rebalance date t: weight_i = (1/vol_i) / sum(1/vol_j),
vol_i = 60-day rolling annualized vol of strategy i ending at t-1.
Monthly rebalance, aligned with strategy rebalance.
Compare: 50/50 equal-weight vs Risk-Parity vs V5 alone.
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

PANEL_HS300 = pickle.load(open('data/cache/factor_panel_hs300.pkl', 'rb'))
RISK_HS300 = pickle.load(open('data/cache/risk_filters_hs300.pkl', 'rb'))
PANEL_ZZ500 = pickle.load(open('data/cache/factor_panel_zz500.pkl', 'rb'))
RISK_ZZ500 = pickle.load(open('data/cache/risk_filters_zz500.pkl', 'rb'))
OUT = Path('reports/walk_forward_v8_rp'); OUT.mkdir(parents=True, exist_ok=True)
KLINE = Path('data/cache/kline')

BENCH_HS300 = pd.read_parquet('data/cache/benchmark_000300_20180102_20201231.parquet')
BENCH_HS300['date'] = pd.to_datetime(BENCH_HS300['date'])
BENCH_HS300 = BENCH_HS300.set_index('date')['close'].sort_index()
BENCH_ZZ500 = pd.read_parquet('data/cache/benchmark_000905_20210101_20260430.parquet')
BENCH_ZZ500['date'] = pd.to_datetime(BENCH_ZZ500['date'])
BENCH_ZZ500 = BENCH_ZZ500.set_index('date')['close'].sort_index()


def build_price_hs300(codes):
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


def build_price_zz500(codes):
    frames = []
    for tk in codes:
        p = KLINE / f"{tk}_20210101_20260430_qfq.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')[['close']].rename(columns={'close': tk})
            frames.append(df)
    if not frames: return pd.DataFrame()
    price = pd.concat(frames, axis=1).T.groupby(level=0).last().T
    return price.sort_index()


WINDOWS = [
    ('W3_shock_down',   '2023-01-01', '2024-08-31'),
    ('W4_924_rebound',  '2024-09-01', '2025-03-31'),
    ('W5_slow_bull',    '2025-04-01', '2025-12-31'),
    ('W6_2026_rally',   '2026-01-01', '2026-04-30'),
]

V5_WEIGHTS = {'value_pb': 0.30, 'value_pe': 0.25, 'reversal_5': 0.20,
              'low_vol_60': 0.15, 'momentum_120_5': 0.10}
V8_WEIGHTS = {'momentum_60': 0.30, 'momentum_120_5': 0.25,
              'reversal_5': 0.25, 'low_vol_60': 0.20}


def run_v5(start, end):
    cf = (config.REBALANCE_FREQ, config.TOP_N, config.MAX_PER_INDUSTRY, config.FACTOR_WEIGHTS)
    config.REBALANCE_FREQ='ME'; config.TOP_N=30; config.MAX_PER_INDUSTRY=3
    config.FACTOR_WEIGHTS = V5_WEIGHTS
    try:
        h = generate_holdings(PANEL_HS300, start_date=start, end_date=end, top_n=30, risk_panel=RISK_HS300)
        if not h: return None
        codes = sorted(set().union(*[set(d.index) for d in h.values()]))
        price = build_price_hs300(codes)
        price = price.loc[(price.index >= start) & (price.index <= end)].dropna(how='all')
        wdf = holdings_to_weights(h, price.index, codes)
        return compute_portfolio_returns(wdf, price, cost_per_side=0.0015).dropna()
    finally:
        (config.REBALANCE_FREQ, config.TOP_N, config.MAX_PER_INDUSTRY, config.FACTOR_WEIGHTS) = cf


def run_v8(start, end):
    cf = (config.REBALANCE_FREQ, config.TOP_N, config.MAX_PER_INDUSTRY, config.FACTOR_WEIGHTS)
    config.REBALANCE_FREQ='ME'; config.TOP_N=30; config.MAX_PER_INDUSTRY=None
    config.FACTOR_WEIGHTS = V8_WEIGHTS
    try:
        h = generate_holdings(PANEL_ZZ500, start_date=start, end_date=end, top_n=30, risk_panel=RISK_ZZ500)
        if not h: return None
        codes = sorted(set().union(*[set(d.index) for d in h.values()]))
        price = build_price_zz500(codes)
        price = price.loc[(price.index >= start) & (price.index <= end)].dropna(how='all')
        wdf = holdings_to_weights(h, price.index, codes)
        return compute_portfolio_returns(wdf, price, cost_per_side=0.0015).dropna()
    finally:
        (config.REBALANCE_FREQ, config.TOP_N, config.MAX_PER_INDUSTRY, config.FACTOR_WEIGHTS) = cf


def combine_5050(v5, v8):
    if v5 is None or v8 is None: return None
    df = pd.DataFrame({'v5': v5['net_ret'], 'v8': v8['net_ret']}).dropna()
    df['net_ret'] = 0.5*df['v5'] + 0.5*df['v8']
    df['nav'] = (1+df['net_ret']).cumprod()
    return df


def combine_riskparity(v5, v8, vol_window=60, rebal_freq='ME'):
    """
    Risk-parity weights, recomputed at each month-end.
    Use t-1 closing as cutoff for vol calc (no look-ahead).
    """
    if v5 is None or v8 is None: return None
    df = pd.DataFrame({'v5': v5['net_ret'], 'v8': v8['net_ret']}).dropna()
    if len(df) < vol_window + 5: return None

    # rebalance dates: month-ends within df.index
    s = pd.Series(1, index=df.index)
    rebal_dates = s.resample(rebal_freq).last().dropna().index
    rebal_dates = rebal_dates.intersection(df.index)

    # weight series, one row per day, ffill from latest rebal
    w_v5 = pd.Series(np.nan, index=df.index, name='w_v5')
    w_v8 = pd.Series(np.nan, index=df.index, name='w_v8')

    for d in rebal_dates:
        # use returns strictly BEFORE d to avoid look-ahead
        hist = df.loc[df.index < d].tail(vol_window)
        if len(hist) < 20:
            # warmup: equal weight
            w_v5.loc[d] = 0.5; w_v8.loc[d] = 0.5
            continue
        vol5 = hist['v5'].std()
        vol8 = hist['v8'].std()
        if vol5 <= 0 or vol8 <= 0:
            w_v5.loc[d] = 0.5; w_v8.loc[d] = 0.5
        else:
            inv5 = 1.0/vol5; inv8 = 1.0/vol8
            w_v5.loc[d] = inv5/(inv5+inv8)
            w_v8.loc[d] = inv8/(inv5+inv8)

    # before first rebal: equal weight
    if len(rebal_dates) > 0:
        first = rebal_dates[0]
        w_v5.loc[df.index < first] = 0.5
        w_v8.loc[df.index < first] = 0.5

    w_v5 = w_v5.ffill(); w_v8 = w_v8.ffill()
    df['w_v5'] = w_v5; df['w_v8'] = w_v8
    df['net_ret'] = df['w_v5']*df['v5'] + df['w_v8']*df['v8']
    df['nav'] = (1+df['net_ret']).cumprod()
    return df


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


def bench_total(bench, s, e):
    s,e = pd.to_datetime(s), pd.to_datetime(e)
    b = bench.loc[(bench.index>=s)&(bench.index<=e)]
    return b.iloc[-1]/b.iloc[0]-1 if len(b) else np.nan


def main():
    rows = []
    for wname, ws, we in WINDOWS:
        print(f"\n{'='*60}\n[{wname}] {ws} -> {we}\n{'='*60}")
        v5 = run_v5(ws, we); m5 = metrics(v5)
        v8 = run_v8(ws, we); m8 = metrics(v8)
        eq = combine_5050(v5, v8); meq = metrics(eq)
        rp = combine_riskparity(v5, v8, vol_window=60, rebal_freq='ME'); mrp = metrics(rp)

        # average risk-parity weight over the window
        avg_w_v5 = rp['w_v5'].mean() if rp is not None else np.nan
        avg_w_v8 = rp['w_v8'].mean() if rp is not None else np.nan

        bh = bench_total(BENCH_HS300, ws, we)
        bz = bench_total(BENCH_ZZ500, ws, we)
        print(f"  V5         total={m5['total']*100:+6.2f}% sharpe={m5['sharpe']:+.2f}")
        print(f"  V8         total={m8['total']*100:+6.2f}% sharpe={m8['sharpe']:+.2f}")
        print(f"  EQ 50/50   total={meq['total']*100:+6.2f}% sharpe={meq['sharpe']:+.2f}")
        print(f"  RiskParity total={mrp['total']*100:+6.2f}% sharpe={mrp['sharpe']:+.2f}  (avg w_v5={avg_w_v5:.2f}, w_v8={avg_w_v8:.2f})")
        print(f"  HS300 {bh*100:+6.2f}%   CSI500 {bz*100:+6.2f}%")
        rows.append({
            'window': wname,
            'V5': m5['total'], 'V8': m8['total'],
            'EQ': meq['total'], 'RP': mrp['total'],
            'HS300': bh, 'CSI500': bz,
            'avg_w_v5': avg_w_v5, 'avg_w_v8': avg_w_v8,
            'RP_vs_V5': mrp['total']-m5['total'],
            'RP_vs_EQ': mrp['total']-meq['total'],
            'RP_beats_V5': mrp['total']>m5['total'],
            'RP_beats_EQ': mrp['total']>meq['total'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT/'walk_forward_v8_rp.csv', index=False, encoding='utf-8-sig')

    print("\n" + "="*70 + "\nSUMMARY\n" + "="*70)
    for _, r in df.iterrows():
        f1 = 'WIN' if r['RP_beats_V5'] else 'LOSE'
        f2 = 'WIN' if r['RP_beats_EQ'] else 'LOSE'
        print(f"  {r['window']:20s} V5={r['V5']*100:+6.2f}% V8={r['V8']*100:+6.2f}% "
              f"EQ={r['EQ']*100:+6.2f}% RP={r['RP']*100:+6.2f}% "
              f"(w_v5={r['avg_w_v5']:.2f}) [vsV5:{f1} vsEQ:{f2}]")
    print(f"\n  RP beats V5: {df['RP_beats_V5'].sum()}/{len(df)}  avg excess: {df['RP_vs_V5'].mean()*100:+.2f}pp")
    print(f"  RP beats EQ: {df['RP_beats_EQ'].sum()}/{len(df)}  avg excess: {df['RP_vs_EQ'].mean()*100:+.2f}pp")

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    x = np.arange(len(df)); w = 0.18
    ax = axes[0]
    ax.bar(x-2*w, df['V5']*100, w, label='V5', color='#1f77b4')
    ax.bar(x-w,   df['V8']*100, w, label='V8', color='#ff7f0e')
    ax.bar(x,     df['EQ']*100, w, label='EQ 50/50', color='#2ca02c')
    ax.bar(x+w,   df['RP']*100, w, label='RiskParity', color='#9467bd')
    ax.bar(x+2*w, df['HS300']*100, w, label='HS300', color='#a0a0a0')
    ax.set_xticks(x); ax.set_xticklabels(df['window'], rotation=15, ha='right')
    ax.set_ylabel('Total Return (%)')
    ax.set_title(f"Walk-Forward: RiskParity vs Equal-Weight  (RP beats V5: {df['RP_beats_V5'].sum()}/{len(df)})")
    ax.axhline(0, color='gray', lw=0.8); ax.legend(loc='best', fontsize=9); ax.grid(alpha=0.3, axis='y')

    ax2 = axes[1]
    rp_vs_v5 = df['RP_vs_V5']*100
    rp_vs_eq = df['RP_vs_EQ']*100
    ax2.bar(x-0.2, rp_vs_v5, 0.4, label='RP - V5', color=['#9467bd' if v>0 else '#d62728' for v in rp_vs_v5])
    ax2.bar(x+0.2, rp_vs_eq, 0.4, label='RP - EQ', color=['#2ca02c' if v>0 else '#d62728' for v in rp_vs_eq])
    ax2.set_xticks(x); ax2.set_xticklabels(df['window'], rotation=15, ha='right')
    ax2.set_ylabel('Excess (pp)')
    ax2.set_title(f"RP excess  -  vs V5 avg {df['RP_vs_V5'].mean()*100:+.2f}pp  /  vs EQ avg {df['RP_vs_EQ'].mean()*100:+.2f}pp")
    ax2.axhline(0, color='gray', lw=0.8); ax2.legend(loc='best'); ax2.grid(alpha=0.3, axis='y')
    for i, v in enumerate(rp_vs_v5):
        ax2.annotate(f"{v:+.1f}", (i-0.2, v), textcoords='offset points',
                     xytext=(0, 3 if v>=0 else -10), ha='center', fontsize=9)
    for i, v in enumerate(rp_vs_eq):
        ax2.annotate(f"{v:+.1f}", (i+0.2, v), textcoords='offset points',
                     xytext=(0, 3 if v>=0 else -10), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT/'walk_forward_v8_rp.png', dpi=130, bbox_inches='tight')
    print(f"\n[Saved] {OUT/'walk_forward_v8_rp.csv'}")
    print(f"[Saved] {OUT/'walk_forward_v8_rp.png'}")


if __name__ == '__main__':
    main()
