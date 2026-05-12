# -*- coding: utf-8 -*-
"""
4 Month Attribution Analysis (2026-01 ~ 2026-04)
- For each rebalance period, compute per-stock return contribution
- Aggregate by industry
- Compare with HS300 monthly returns
- Output charts: NAV curve, monthly excess, top winners/losers
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Try Chinese font; fall back to English labels
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass
from pathlib import Path

# Avoid encoding issues on win-gbk
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / 'data' / 'paper'
KLINE = ROOT / 'data' / 'cache' / 'kline'
OUT = ROOT / 'data' / 'paper' / 'attribution'
OUT.mkdir(exist_ok=True, parents=True)

PERIODS = [
    ('2025-12-31', '2026-01-30'),
    ('2026-01-30', '2026-02-27'),
    ('2026-02-27', '2026-03-31'),
    ('2026-03-31', '2026-04-30'),
]


def load_close(ticker, start, end):
    """Load close price from cached kline. Try multiple file patterns."""
    sd = start.replace('-', '')
    ed = end.replace('-', '')
    # Try the long file first (covers full period)
    candidates = [
        f"{ticker}_20250901_20260430_qfq.parquet",
    ]
    for fn in candidates:
        p = KLINE / fn
        if p.exists():
            df = pd.read_parquet(p)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            return df
    return None


def get_price_on_or_before(df, date):
    """Get close price on or just before target date."""
    if df is None:
        return None
    target = pd.Timestamp(date)
    sub = df[df.index <= target]
    if len(sub) == 0:
        return None
    return float(sub.iloc[-1]['close']), sub.index[-1]


def attribution_one_period(start, end, snapshot_csv):
    """For positions held at `start` (= snapshot_csv on `start`),
    compute per-stock return from start->end."""
    pos = pd.read_csv(snapshot_csv, encoding='utf-8-sig')
    # snapshot is the holding right after rebalance on `start`
    rows = []
    for _, r in pos.iterrows():
        tk = str(r['ticker']).zfill(6)
        df = load_close(tk, start, end)
        if df is None:
            print(f"  [skip] {tk} no data")
            continue
        p0 = get_price_on_or_before(df, start)
        p1 = get_price_on_or_before(df, end)
        if p0 is None or p1 is None:
            print(f"  [skip] {tk} no price")
            continue
        ret = p1[0] / p0[0] - 1
        # cost_value column was rebalanced equal-weight ~33333 per name
        weight = float(r['cost_value']) / float(pos['cost_value'].sum())
        contrib = weight * ret
        rows.append({
            'ticker': tk,
            'name': r['name'],
            'industry': r['industry'],
            'p0': p0[0], 'p1': p1[0],
            'return': ret,
            'weight': weight,
            'contribution': contrib,
        })
    return pd.DataFrame(rows)


def main():
    # 1. Per-period attribution
    all_attr = {}
    for start, end in PERIODS:
        snap = PAPER / 'snapshots' / f'{start}.csv'
        if not snap.exists():
            print(f"missing snapshot {snap}")
            continue
        print(f"\n=== Period {start} -> {end} ===")
        df = attribution_one_period(start, end, snap)
        all_attr[end] = df
        df.to_csv(OUT / f'attribution_{end}.csv', index=False, encoding='utf-8-sig')
        port_ret = df['contribution'].sum()
        print(f"Portfolio gross return (price-only): {port_ret*100:.2f}%")
        # top 5 winners/losers
        df_sorted = df.sort_values('contribution', ascending=False)
        print("Top 5 contributors:")
        print(df_sorted[['name', 'industry', 'return', 'contribution']].head(5).to_string(index=False))
        print("Top 5 detractors:")
        print(df_sorted[['name', 'industry', 'return', 'contribution']].tail(5).to_string(index=False))
        # Industry agg
        ind = df.groupby('industry').agg(
            n=('ticker', 'count'),
            weight=('weight', 'sum'),
            avg_ret=('return', 'mean'),
            contrib=('contribution', 'sum'),
        ).sort_values('contrib', ascending=False)
        print("\nIndustry aggregation:")
        print(ind.to_string())

    # 2. NAV chart vs HS300
    nav = pd.read_csv(PAPER / 'nav_history.csv', encoding='utf-8-sig')
    nav['date'] = pd.to_datetime(nav['date'])

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    ax = axes[0]
    ax.plot(nav['date'], nav['nav'], 'o-', label='Strategy V5', linewidth=2.2, color='#d62728')
    ax.plot(nav['date'], nav['benchmark_nav'], 's-', label='HS300', linewidth=2.2, color='#1f77b4')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title('Paper Trading: NAV vs HS300 (2025-12-31 to 2026-04-30)', fontsize=13)
    ax.set_ylabel('NAV (start = 1.0)')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    # annotate end values
    for i, row in nav.iterrows():
        ax.annotate(f"{row['nav']:.4f}", (row['date'], row['nav']),
                    textcoords='offset points', xytext=(0, 8), fontsize=9, ha='center', color='#d62728')
        ax.annotate(f"{row['benchmark_nav']:.4f}", (row['date'], row['benchmark_nav']),
                    textcoords='offset points', xytext=(0, -14), fontsize=9, ha='center', color='#1f77b4')

    # bottom: monthly excess
    monthly_strat = nav['nav'].pct_change().dropna() * 100
    monthly_bench = nav['benchmark_nav'].pct_change().dropna() * 100
    excess = (monthly_strat - monthly_bench).values
    labels = nav['date'].iloc[1:].dt.strftime('%Y-%m').values
    x = np.arange(len(labels))
    width = 0.28
    ax2 = axes[1]
    ax2.bar(x - width, monthly_strat.values, width, label='Strategy', color='#d62728')
    ax2.bar(x, monthly_bench.values, width, label='HS300', color='#1f77b4')
    ax2.bar(x + width, excess, width, label='Excess', color='#2ca02c')
    ax2.axhline(0, color='gray', linewidth=0.8)
    ax2.set_title('Monthly Returns (%)', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    for i, v in enumerate(excess):
        ax2.annotate(f"{v:+.2f}", (x[i] + width, v),
                     textcoords='offset points', xytext=(0, 3 if v >= 0 else -10),
                     fontsize=8, ha='center', color='#2ca02c')

    plt.tight_layout()
    chart = OUT / 'nav_vs_hs300.png'
    plt.savefig(chart, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {chart}")

    # 3. Industry stack chart for last (April) period
    apr = all_attr.get('2026-04-30')
    if apr is not None and len(apr):
        ind = apr.groupby('industry').agg(
            weight=('weight', 'sum'),
            contrib=('contribution', 'sum'),
            avg_ret=('return', 'mean'),
        ).sort_values('contrib')
        fig2, ax3 = plt.subplots(figsize=(10, 6))
        colors = ['#d62728' if v < 0 else '#2ca02c' for v in ind['contrib']]
        ax3.barh(ind.index, ind['contrib'] * 100, color=colors)
        ax3.set_xlabel('Contribution to portfolio return (%)')
        ax3.set_title('April 2026 - Industry Contribution to Portfolio (sorted)', fontsize=13)
        ax3.axvline(0, color='gray', linewidth=0.8)
        ax3.grid(alpha=0.3, axis='x')
        for i, (idx, v) in enumerate(zip(ind.index, ind['contrib'] * 100)):
            ax3.annotate(f"{v:+.2f}%  (avg {ind.loc[idx, 'avg_ret']*100:+.1f}%)",
                         (v, i), xytext=(5 if v >= 0 else -5, 0),
                         textcoords='offset points',
                         va='center', ha='left' if v >= 0 else 'right', fontsize=9)
        plt.tight_layout()
        chart2 = OUT / 'april_industry.png'
        plt.savefig(chart2, dpi=130, bbox_inches='tight')
        print(f"Saved: {chart2}")

    print("\nDone. Files in", OUT)


if __name__ == '__main__':
    main()
