# -*- coding: utf-8 -*-
"""
Walk-forward validation: V5 baseline vs WINNER (top=20, no_limit, ME) vs HS300.
Test if WINNER beats V5 robustly across 5 OOS windows.
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

PANEL = pickle.load(open('data/cache/factor_panel_hs300.pkl', 'rb'))
RISK = pickle.load(open('data/cache/risk_filters_hs300.pkl', 'rb'))
OUT = Path('reports/walk_forward_winner'); OUT.mkdir(parents=True, exist_ok=True)
KLINE = Path('data/cache/kline')

# Load HS300 benchmark
BENCH = pd.read_parquet('data/cache/benchmark_000300_20180102_20201231.parquet')
BENCH['date'] = pd.to_datetime(BENCH['date'])
BENCH = BENCH.set_index('date')['close'].sort_index()


def build_price_from_cache(codes):
    frames = []
    for tk in codes:
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
        return pd.DataFrame()
    price = pd.concat(frames, axis=1)
    price = price.T.groupby(level=0).last().T
    return price.sort_index()


WINDOWS = [
    ('W2_growth_crash', '2021-07-01', '2022-12-31'),
    ('W3_shock_down',   '2023-01-01', '2024-08-31'),
    ('W4_924_rebound',  '2024-09-01', '2025-03-31'),
    ('W5_slow_bull',    '2025-04-01', '2025-12-31'),
    ('W6_2026_rally',   '2026-01-01', '2026-04-30'),
]

V5_WEIGHTS = {
    'value_pb': 0.30, 'value_pe': 0.25, 'reversal_5': 0.20,
    'low_vol_60': 0.15, 'momentum_120_5': 0.10,
}

CONFIGS = {
    # Baseline V5: top=30, max_per_industry=3
    'V5':     {'freq': 'ME', 'top_n': 30, 'max_ind': 3},
    # WINNER:   top=20, no_limit
    'WINNER': {'freq': 'ME', 'top_n': 20, 'max_ind': None},
}


def run_strategy(cfg, start, end):
    orig_freq = config.REBALANCE_FREQ
    orig_top = config.TOP_N
    orig_max = config.MAX_PER_INDUSTRY
    orig_w = config.FACTOR_WEIGHTS
    config.REBALANCE_FREQ = cfg['freq']
    config.TOP_N = cfg['top_n']
    config.MAX_PER_INDUSTRY = cfg['max_ind']
    config.FACTOR_WEIGHTS = V5_WEIGHTS
    try:
        holdings = generate_holdings(PANEL, start_date=start, end_date=end,
                                     top_n=cfg['top_n'], risk_panel=RISK)
        if len(holdings) == 0:
            return None
        all_codes = set()
        for h in holdings.values():
            all_codes.update(h.index)
        price = build_price_from_cache(sorted(all_codes))
        price = price.loc[(price.index >= start) & (price.index <= end)]
        price = price.dropna(how='all')
        weights_df = holdings_to_weights(holdings, price.index, sorted(all_codes))
        perf = compute_portfolio_returns(weights_df, price, cost_per_side=0.0015)
        return perf.dropna()
    finally:
        config.REBALANCE_FREQ = orig_freq
        config.TOP_N = orig_top
        config.MAX_PER_INDUSTRY = orig_max
        config.FACTOR_WEIGHTS = orig_w


def metrics(perf, start, end):
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


def hs300_metrics(start, end):
    s = pd.to_datetime(start); e = pd.to_datetime(end)
    b = BENCH.loc[(BENCH.index >= s) & (BENCH.index <= e)]
    if len(b) == 0:
        return {'total': np.nan, 'ann': np.nan}
    total = b.iloc[-1] / b.iloc[0] - 1
    days = max((b.index[-1] - b.index[0]).days, 1)
    ann = (b.iloc[-1] / b.iloc[0]) ** (365 / days) - 1
    return {'total': total, 'ann': ann}


def main():
    rows = []
    for wname, wstart, wend in WINDOWS:
        print(f"\n{'='*60}\n[{wname}] {wstart} -> {wend}\n{'='*60}")
        out = {'window': wname, 'start': wstart, 'end': wend}

        for cname, cfg in CONFIGS.items():
            perf = run_strategy(cfg, wstart, wend)
            m = metrics(perf, wstart, wend)
            print(f"  {cname:8s} total={m['total']*100:+6.2f}%  ann={m['ann']*100:+6.2f}%  "
                  f"sharpe={m['sharpe']:+.2f}  mdd={m['mdd']*100:+6.2f}%")
            out[f'{cname}_total'] = m['total']
            out[f'{cname}_ann'] = m['ann']
            out[f'{cname}_sharpe'] = m['sharpe']
            out[f'{cname}_mdd'] = m['mdd']

        # HS300
        h = hs300_metrics(wstart, wend)
        print(f"  HS300    total={h['total']*100:+6.2f}%  ann={h['ann']*100:+6.2f}%")
        out['HS300_total'] = h['total']
        out['HS300_ann'] = h['ann']

        # Excess
        out['WINNER_vs_V5_total'] = out['WINNER_total'] - out['V5_total']
        out['V5_vs_HS300_total'] = out['V5_total'] - out['HS300_total']
        out['WINNER_vs_HS300_total'] = out['WINNER_total'] - out['HS300_total']
        out['WINNER_beats_V5'] = out['WINNER_total'] > out['V5_total']
        out['WINNER_beats_HS300'] = out['WINNER_total'] > out['HS300_total']

        rows.append(out)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'walk_forward_winner.csv', index=False, encoding='utf-8-sig')
    print(f"\n[Saved] {OUT / 'walk_forward_winner.csv'}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: WINNER vs V5 across 5 windows")
    print("="*70)
    for _, r in df.iterrows():
        flag = 'WIN' if r['WINNER_beats_V5'] else 'LOSE'
        h_flag = 'beat HS300' if r['WINNER_beats_HS300'] else 'lose HS300'
        print(f"  {r['window']:25s} V5={r['V5_total']*100:+6.2f}% WINNER={r['WINNER_total']*100:+6.2f}% "
              f"HS300={r['HS300_total']*100:+6.2f}%  [{flag}, {h_flag}]")

    wins_v5 = df['WINNER_beats_V5'].sum()
    wins_h = df['WINNER_beats_HS300'].sum()
    avg_excess_v5 = df['WINNER_vs_V5_total'].mean() * 100
    avg_excess_h = df['WINNER_vs_HS300_total'].mean() * 100
    print(f"\n  WINNER beats V5    : {wins_v5}/{len(df)}  avg excess = {avg_excess_v5:+.2f}pp")
    print(f"  WINNER beats HS300 : {wins_h}/{len(df)}  avg excess = {avg_excess_h:+.2f}pp")

    # Plots
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    x = np.arange(len(df)); w = 0.27
    ax = axes[0]
    ax.bar(x - w, df['HS300_total']*100, w, label='HS300', color='#8c8c8c')
    ax.bar(x,     df['V5_total']*100,    w, label='V5 baseline', color='#1f77b4')
    ax.bar(x + w, df['WINNER_total']*100, w, label='WINNER (top=20, no_limit)', color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels(df['window'], rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Total Return (%)')
    ax.set_title(f'Walk-Forward: WINNER beats V5 in {wins_v5}/{len(df)} windows, beats HS300 in {wins_h}/{len(df)}', fontsize=12)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.legend(loc='best')
    ax.grid(alpha=0.3, axis='y')
    for i in range(len(df)):
        for off, col, val in [(-w, '#8c8c8c', df['HS300_total'].iloc[i]*100),
                              (0, '#1f77b4', df['V5_total'].iloc[i]*100),
                              (w, '#2ca02c', df['WINNER_total'].iloc[i]*100)]:
            ax.annotate(f"{val:+.1f}", (i+off, val), textcoords='offset points',
                        xytext=(0, 3 if val >= 0 else -10), ha='center', fontsize=8, color=col)

    ax2 = axes[1]
    excess = df['WINNER_vs_V5_total']*100
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in excess]
    ax2.bar(x, excess, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['window'], rotation=15, ha='right', fontsize=10)
    ax2.set_ylabel('WINNER - V5 (pp)')
    ax2.set_title(f'Per-Window Excess (WINNER - V5)  -  Avg: {avg_excess_v5:+.2f}pp', fontsize=12)
    ax2.axhline(0, color='gray', linewidth=0.8)
    ax2.grid(alpha=0.3, axis='y')
    for i, v in enumerate(excess):
        ax2.annotate(f"{v:+.2f}pp", (i, v), textcoords='offset points',
                     xytext=(0, 3 if v >= 0 else -10), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT / 'walk_forward_winner.png', dpi=130, bbox_inches='tight')
    print(f"[Saved] {OUT / 'walk_forward_winner.png'}")


if __name__ == '__main__':
    main()
