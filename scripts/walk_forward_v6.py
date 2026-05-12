# -*- coding: utf-8 -*-
"""
Walk-forward V6 vs V5 across 5 windows.
For each window:
  - Run V5 (current weights) and V6 (new weights) using existing pipeline
  - Compare total return, ann return, sharpe, max DD vs HS300
  - Output: walk_forward_v6.csv + bar chart
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

from quant.strategy import generate_holdings
from quant.backtest import build_price_matrix, holdings_to_weights, compute_portfolio_returns

ROOT = Path('.')
PANEL = pickle.load(open('data/cache/factor_panel_hs300.pkl', 'rb'))
RISK = pickle.load(open('data/cache/risk_filters_hs300.pkl', 'rb'))
OUT = ROOT / 'reports' / 'walk_forward_v6'
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = [
    ('W2_growth_crash', '2021-07-01', '2022-12-31'),
    ('W3_shock_down', '2023-01-01', '2024-08-31'),
    ('W4_924_rebound', '2024-09-01', '2025-03-31'),
    ('W5_slow_bull', '2025-04-01', '2025-12-31'),
    ('W6_2026_rally', '2026-01-01', '2026-04-30'),
]

V5_WEIGHTS = {
    'value_pb': 0.30,
    'value_pe': 0.25,
    'reversal_5': 0.20,
    'low_vol_60': 0.15,
    'momentum_120_5': 0.10,
}

V6_WEIGHTS = {
    'value_pb': 0.20,
    'value_pe': 0.25,
    'reversal_5': 0.25,
    'low_vol_60': 0.10,
    'quality_roe': 0.20,
}


def run_strategy(weights, start, end, label):
    """Run strategy with given factor weights for one window."""
    # Patch config.FACTOR_WEIGHTS for this run
    from quant import config
    orig = config.FACTOR_WEIGHTS
    config.FACTOR_WEIGHTS = weights
    try:
        # generate_holdings honors config dates, so we override with start/end
        holdings = generate_holdings(
            PANEL,
            start_date=start,
            end_date=end,
            risk_panel=RISK,
        )
        if len(holdings) == 0:
            return None
        all_codes = set()
        for h in holdings.values():
            all_codes.update(h.index)
        price = build_price_matrix(codes=sorted(all_codes), start_date=start, end_date=end)
        price = price.dropna(how='all')
        weights_df = holdings_to_weights(holdings, price.index, sorted(all_codes))
        perf = compute_portfolio_returns(weights_df, price, cost_per_side=0.0015)
        perf = perf.dropna()
        return perf
    finally:
        config.FACTOR_WEIGHTS = orig


def calc_metrics(perf, label):
    if perf is None or len(perf) == 0:
        return {'label': label, 'total': np.nan, 'ann': np.nan, 'sharpe': np.nan, 'mdd': np.nan, 'periods': 0}
    nav = perf['nav']
    ret = perf['net_ret']
    days = max((nav.index[-1] - nav.index[0]).days, 1)
    total = nav.iloc[-1] - 1
    ann = nav.iloc[-1] ** (365 / days) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ann - 0.025) / vol if vol > 0 else np.nan
    peak = nav.cummax()
    mdd = (nav / peak - 1).min()
    return {'label': label, 'total': total, 'ann': ann, 'sharpe': sharpe, 'mdd': mdd, 'periods': len(nav)}


def main():
    rows = []
    for wname, wstart, wend in WINDOWS:
        print(f"\n{'='*60}\n[{wname}] {wstart} -> {wend}\n{'='*60}")
        perf_v5 = run_strategy(V5_WEIGHTS, wstart, wend, 'V5')
        m_v5 = calc_metrics(perf_v5, 'V5')
        print(f"  V5: total={m_v5['total']*100:+.2f}% ann={m_v5['ann']*100:+.2f}% sharpe={m_v5['sharpe']:.2f} mdd={m_v5['mdd']*100:.2f}%")

        perf_v6 = run_strategy(V6_WEIGHTS, wstart, wend, 'V6')
        m_v6 = calc_metrics(perf_v6, 'V6')
        print(f"  V6: total={m_v6['total']*100:+.2f}% ann={m_v6['ann']*100:+.2f}% sharpe={m_v6['sharpe']:.2f} mdd={m_v6['mdd']*100:.2f}%")

        rows.append({
            'window': wname,
            'start': wstart,
            'end': wend,
            'V5_total': m_v5['total'],
            'V6_total': m_v6['total'],
            'V5_ann': m_v5['ann'],
            'V6_ann': m_v6['ann'],
            'V5_sharpe': m_v5['sharpe'],
            'V6_sharpe': m_v6['sharpe'],
            'V5_mdd': m_v5['mdd'],
            'V6_mdd': m_v6['mdd'],
            'excess_total': m_v6['total'] - m_v5['total'],
            'excess_ann': m_v6['ann'] - m_v5['ann'],
            'V6_wins': m_v6['total'] > m_v5['total'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'walk_forward_v6.csv', index=False, encoding='utf-8-sig')
    print(f"\n[Saved] {OUT / 'walk_forward_v6.csv'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: V6 vs V5 across 5 windows")
    print("=" * 70)
    summary = df[['window', 'V5_total', 'V6_total', 'excess_total', 'V6_wins']].copy()
    summary['V5_total'] = (summary['V5_total'] * 100).round(2).astype(str) + '%'
    summary['V6_total'] = (summary['V6_total'] * 100).round(2).astype(str) + '%'
    summary['excess_total'] = (summary['excess_total'] * 100).round(2).astype(str) + 'pp'
    print(summary.to_string(index=False))

    wins = df['V6_wins'].sum()
    print(f"\nV6 wins: {wins}/{len(df)} windows")
    avg_excess = df['excess_total'].mean() * 100
    print(f"Avg excess: {avg_excess:+.2f}pp")

    # Bar chart
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    x = np.arange(len(df))
    w = 0.35
    ax = axes[0]
    ax.bar(x - w/2, df['V5_total'] * 100, w, label='V5', color='#1f77b4')
    ax.bar(x + w/2, df['V6_total'] * 100, w, label='V6', color='#d62728')
    ax.set_xticks(x)
    ax.set_xticklabels(df['window'], rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Total Return (%)')
    ax.set_title(f'Walk-Forward V6 vs V5 - V6 Wins: {wins}/{len(df)} windows', fontsize=13)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    for i, (v5, v6) in enumerate(zip(df['V5_total'] * 100, df['V6_total'] * 100)):
        ax.annotate(f"{v5:+.1f}", (x[i] - w/2, v5), textcoords='offset points',
                    xytext=(0, 3 if v5 >= 0 else -10), ha='center', fontsize=8, color='#1f77b4')
        ax.annotate(f"{v6:+.1f}", (x[i] + w/2, v6), textcoords='offset points',
                    xytext=(0, 3 if v6 >= 0 else -10), ha='center', fontsize=8, color='#d62728')

    ax2 = axes[1]
    excess = df['excess_total'] * 100
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in excess]
    ax2.bar(x, excess, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['window'], rotation=20, ha='right', fontsize=10)
    ax2.set_ylabel('V6 - V5 Excess (pp)')
    ax2.set_title('Per-Window Excess (V6 - V5)', fontsize=13)
    ax2.axhline(0, color='gray', linewidth=0.8)
    ax2.grid(alpha=0.3, axis='y')
    for i, v in enumerate(excess):
        ax2.annotate(f"{v:+.2f}pp", (x[i], v), textcoords='offset points',
                     xytext=(0, 3 if v >= 0 else -10), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT / 'walk_forward_v6.png', dpi=130, bbox_inches='tight')
    print(f"[Saved] {OUT / 'walk_forward_v6.png'}")


if __name__ == '__main__':
    main()
