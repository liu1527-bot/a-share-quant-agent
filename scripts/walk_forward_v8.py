# -*- coding: utf-8 -*-
"""
V8 walk-forward on CSI500 (zz500): momentum-style strategy.
3 OOS windows (MVP).
Compare V8 alone, V5 alone (HS300), V5+V8 50/50 combo, vs HS300 / CSI500.
"""
import os, sys, pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

for p in ['HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(p, None)
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
OUT = Path('reports/walk_forward_v8'); OUT.mkdir(parents=True, exist_ok=True)
KLINE = Path('data/cache/kline')

# benchmarks
BENCH_HS300 = pd.read_parquet('data/cache/benchmark_000300_20180102_20201231.parquet')
BENCH_HS300['date'] = pd.to_datetime(BENCH_HS300['date'])
BENCH_HS300 = BENCH_HS300.set_index('date')['close'].sort_index()

BENCH_ZZ500 = pd.read_parquet('data/cache/benchmark_000905_20210101_20260430.parquet')
BENCH_ZZ500['date'] = pd.to_datetime(BENCH_ZZ500['date'])
BENCH_ZZ500 = BENCH_ZZ500.set_index('date')['close'].sort_index()


def build_price_hs300(codes):
    """V5 HS300 cache: 3 fragments."""
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
    price = pd.concat(frames, axis=1)
    price = price.T.groupby(level=0).last().T
    return price.sort_index()


def build_price_zz500(codes):
    """CSI500 cache: single fragment 2021-2026."""
    frames = []
    for tk in codes:
        p = KLINE / f"{tk}_20210101_20260430_qfq.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')[['close']].rename(columns={'close': tk})
            frames.append(df)
    if not frames: return pd.DataFrame()
    price = pd.concat(frames, axis=1)
    price = price.T.groupby(level=0).last().T
    return price.sort_index()


# 3 OOS windows (MVP - skipped W2 since CSI500 only has data from 2021)
WINDOWS = [
    ('W3_shock_down',   '2023-01-01', '2024-08-31'),
    ('W4_924_rebound',  '2024-09-01', '2025-03-31'),
    ('W5_slow_bull',    '2025-04-01', '2025-12-31'),
    ('W6_2026_rally',   '2026-01-01', '2026-04-30'),
]

V5_WEIGHTS = {
    'value_pb': 0.30, 'value_pe': 0.25, 'reversal_5': 0.20,
    'low_vol_60': 0.15, 'momentum_120_5': 0.10,
}
# V8 momentum-style (no fundamentals - MVP)
V8_WEIGHTS = {
    'momentum_60': 0.30, 'momentum_120_5': 0.25,
    'reversal_5': 0.25, 'low_vol_60': 0.20,
}


def run_v5(start, end):
    cf = (config.REBALANCE_FREQ, config.TOP_N, config.MAX_PER_INDUSTRY, config.FACTOR_WEIGHTS)
    config.REBALANCE_FREQ = 'ME'
    config.TOP_N = 30
    config.MAX_PER_INDUSTRY = 3
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
    config.REBALANCE_FREQ = 'ME'
    config.TOP_N = 30
    config.MAX_PER_INDUSTRY = None  # no industry cap (CSI500 not in industry_map)
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


def combine_5050(perf_v5, perf_v8):
    """Daily 50/50 V5+V8 portfolio (rebalanced daily, simple)."""
    if perf_v5 is None or perf_v8 is None: return None
    df = pd.DataFrame({'v5': perf_v5['net_ret'], 'v8': perf_v8['net_ret']}).dropna()
    df['net_ret'] = 0.5 * df['v5'] + 0.5 * df['v8']
    df['nav'] = (1 + df['net_ret']).cumprod()
    return df


def metrics(perf):
    if perf is None or len(perf) == 0:
        return {'total': np.nan, 'ann': np.nan, 'sharpe': np.nan, 'mdd': np.nan}
    nav = perf['nav']; ret = perf['net_ret']
    days = max((nav.index[-1] - nav.index[0]).days, 1)
    total = nav.iloc[-1] - 1
    ann = nav.iloc[-1] ** (365 / days) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ann - 0.025) / vol if vol > 0 else np.nan
    peak = nav.cummax()
    mdd = (nav / peak - 1).min()
    return {'total': total, 'ann': ann, 'sharpe': sharpe, 'mdd': mdd}


def bench_metrics(bench, start, end):
    s = pd.to_datetime(start); e = pd.to_datetime(end)
    b = bench.loc[(bench.index >= s) & (bench.index <= e)]
    if len(b) == 0: return {'total': np.nan}
    return {'total': b.iloc[-1] / b.iloc[0] - 1}


def main():
    rows = []
    for wname, ws, we in WINDOWS:
        print(f"\n{'='*60}\n[{wname}] {ws} -> {we}\n{'='*60}")
        v5 = run_v5(ws, we); m5 = metrics(v5)
        print(f"  V5(HS300)        total={m5['total']*100:+6.2f}% sharpe={m5['sharpe']:+.2f} mdd={m5['mdd']*100:+6.2f}%")
        v8 = run_v8(ws, we); m8 = metrics(v8)
        print(f"  V8(CSI500-mom)   total={m8['total']*100:+6.2f}% sharpe={m8['sharpe']:+.2f} mdd={m8['mdd']*100:+6.2f}%")
        combo = combine_5050(v5, v8); mc = metrics(combo)
        print(f"  V5+V8 50/50      total={mc['total']*100:+6.2f}% sharpe={mc['sharpe']:+.2f} mdd={mc['mdd']*100:+6.2f}%")
        bh = bench_metrics(BENCH_HS300, ws, we)
        bz = bench_metrics(BENCH_ZZ500, ws, we)
        print(f"  HS300 {bh['total']*100:+6.2f}%   CSI500 {bz['total']*100:+6.2f}%")
        rows.append({
            'window': wname, 'start': ws, 'end': we,
            'V5_total': m5['total'], 'V5_sharpe': m5['sharpe'], 'V5_mdd': m5['mdd'],
            'V8_total': m8['total'], 'V8_sharpe': m8['sharpe'], 'V8_mdd': m8['mdd'],
            'COMBO_total': mc['total'], 'COMBO_sharpe': mc['sharpe'], 'COMBO_mdd': mc['mdd'],
            'HS300_total': bh['total'], 'CSI500_total': bz['total'],
            'COMBO_vs_V5': mc['total'] - m5['total'],
            'V8_vs_CSI500': m8['total'] - bz['total'],
            'COMBO_beats_V5': mc['total'] > m5['total'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'walk_forward_v8.csv', index=False, encoding='utf-8-sig')
    print(f"\n[Saved] {OUT/'walk_forward_v8.csv'}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for _, r in df.iterrows():
        f = 'WIN' if r['COMBO_beats_V5'] else 'LOSE'
        print(f"  {r['window']:25s} V5={r['V5_total']*100:+6.2f}% V8={r['V8_total']*100:+6.2f}% "
              f"COMBO={r['COMBO_total']*100:+6.2f}% (vs V5: {r['COMBO_vs_V5']*100:+.2f}pp) [{f}]")
    wins = df['COMBO_beats_V5'].sum()
    avg = df['COMBO_vs_V5'].mean() * 100
    print(f"\n  COMBO beats V5: {wins}/{len(df)}  avg excess: {avg:+.2f}pp")
    v8_alpha_avg = df['V8_vs_CSI500'].mean() * 100
    print(f"  V8 alpha vs CSI500: avg {v8_alpha_avg:+.2f}pp")

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    x = np.arange(len(df)); w = 0.18
    ax = axes[0]
    ax.bar(x - 2*w, df['HS300_total']*100, w, label='HS300', color='#a0a0a0')
    ax.bar(x - w,   df['CSI500_total']*100, w, label='CSI500', color='#808080')
    ax.bar(x,       df['V5_total']*100,    w, label='V5 (HS300 value)', color='#1f77b4')
    ax.bar(x + w,   df['V8_total']*100,    w, label='V8 (CSI500 momentum)', color='#ff7f0e')
    ax.bar(x + 2*w, df['COMBO_total']*100, w, label='V5+V8 50/50', color='#2ca02c')
    ax.set_xticks(x); ax.set_xticklabels(df['window'], rotation=15, ha='right')
    ax.set_ylabel('Total Return (%)')
    ax.set_title(f'Walk-Forward V5 + V8 Combo: COMBO beats V5 in {wins}/{len(df)}')
    ax.axhline(0, color='gray', linewidth=0.8); ax.legend(loc='best', fontsize=9); ax.grid(alpha=0.3, axis='y')

    ax2 = axes[1]
    excess = df['COMBO_vs_V5']*100
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in excess]
    ax2.bar(x, excess, color=colors)
    ax2.set_xticks(x); ax2.set_xticklabels(df['window'], rotation=15, ha='right')
    ax2.set_ylabel('COMBO - V5 (pp)')
    ax2.set_title(f'Combo Excess over V5  -  Avg: {avg:+.2f}pp')
    ax2.axhline(0, color='gray', linewidth=0.8); ax2.grid(alpha=0.3, axis='y')
    for i, v in enumerate(excess):
        ax2.annotate(f"{v:+.2f}pp", (i, v), textcoords='offset points',
                     xytext=(0, 3 if v >= 0 else -10), ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / 'walk_forward_v8.png', dpi=130, bbox_inches='tight')
    print(f"[Saved] {OUT/'walk_forward_v8.png'}")


if __name__ == '__main__':
    main()
