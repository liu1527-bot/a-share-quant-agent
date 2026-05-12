# -*- coding: utf-8 -*-
"""
V5 alpha attribution (MVP):
  Part 1 - Factor contribution decomposition (which factor brings the score)
  Part 2 - Market regime attribution (when does V5 win)

Span: 2021-01-01 to 2026-04-30 (full V5 backtest period).
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
from quant.strategy import generate_holdings, normalize_cross_section, filter_stocks, apply_risk_filters
from quant.backtest import holdings_to_weights, compute_portfolio_returns

# data
PANEL = pickle.load(open('data/cache/factor_panel_hs300.pkl', 'rb'))
RISK = pickle.load(open('data/cache/risk_filters_hs300.pkl', 'rb'))
KLINE = Path('data/cache/kline')
OUT = Path('reports/v5_attribution'); OUT.mkdir(parents=True, exist_ok=True)

BENCH = pd.read_parquet('data/cache/benchmark_000300_20180102_20201231.parquet')
BENCH['date'] = pd.to_datetime(BENCH['date'])
BENCH = BENCH.set_index('date')['close'].sort_index()

# also try to load the post-2021 benchmark file (compare with one used elsewhere)
# We use the existing one's whole span; if missing post-2021 we top up:
import glob
extra_bench_files = glob.glob('data/cache/benchmark_000300_*.parquet')
bench_frames = []
for f in extra_bench_files:
    b = pd.read_parquet(f)
    b['date'] = pd.to_datetime(b['date'])
    bench_frames.append(b.set_index('date')['close'])
BENCH = pd.concat(bench_frames).sort_index()
BENCH = BENCH[~BENCH.index.duplicated(keep='last')]

V5_WEIGHTS = {'value_pb': 0.30, 'value_pe': 0.25, 'reversal_5': 0.20,
              'low_vol_60': 0.15, 'momentum_120_5': 0.10}

START, END = '2021-01-01', '2026-04-30'


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


# =========================================================
# PART 1: Factor Contribution Decomposition
# =========================================================
def factor_contribution_per_period():
    """
    For each rebalance date, decompose the average score of selected
    holdings into per-factor contribution.
    """
    cf = (config.REBALANCE_FREQ, config.TOP_N, config.MAX_PER_INDUSTRY, config.FACTOR_WEIGHTS)
    config.REBALANCE_FREQ = 'ME'; config.TOP_N = 30; config.MAX_PER_INDUSTRY = 3
    config.FACTOR_WEIGHTS = V5_WEIGHTS
    try:
        holdings = generate_holdings(PANEL, start_date=START, end_date=END,
                                     top_n=30, risk_panel=RISK)
    finally:
        (config.REBALANCE_FREQ, config.TOP_N, config.MAX_PER_INDUSTRY, config.FACTOR_WEIGHTS) = cf

    # for each rebalance date, recompute the normalized factor snapshot
    # and grab the average normalized value of selected holdings per factor
    rows = []
    for date in sorted(holdings.keys()):
        held = holdings[date].index
        snap = {}
        for name, df in PANEL.items():
            valid_dates = df.index[df.index <= date]
            if len(valid_dates) == 0: continue
            snap[name] = df.loc[valid_dates[-1]]
        raw_snap = pd.DataFrame(snap)
        cleaned = filter_stocks(raw_snap)
        # apply same risk filters
        low_vol_snap = snap.get('low_vol_60')
        kept = apply_risk_filters(cleaned.index, date, RISK, low_vol_snap)
        cleaned = cleaned.loc[kept]
        # normalize cross-section (rank by default)
        normalized = normalize_cross_section(cleaned)
        held_in = held.intersection(normalized.index)
        if len(held_in) == 0: continue
        avg_z = normalized.loc[held_in].mean()  # per-factor avg z of holdings
        row = {'date': date}
        # Marginal contribution = weight * avg_z(factor across holdings)
        for f, w in V5_WEIGHTS.items():
            row[f] = w * avg_z.get(f, 0)
        row['total_score'] = sum(row[f] for f in V5_WEIGHTS)
        rows.append(row)

    df = pd.DataFrame(rows).set_index('date')
    return df, holdings


def part1_factor_contribution(contrib):
    """Summary stats for factor contribution."""
    factors = list(V5_WEIGHTS.keys())

    # avg contribution
    avg = contrib[factors].mean()
    # share of total
    share = avg / avg.abs().sum() * 100
    # hit rate (positive contribution months)
    hit = (contrib[factors] > 0).mean() * 100
    # std (consistency)
    std = contrib[factors].std()

    summary = pd.DataFrame({
        'weight': pd.Series(V5_WEIGHTS),
        'avg_contrib': avg.round(4),
        'share_pct': share.round(1),
        'hit_rate_pct': hit.round(1),
        'std': std.round(4),
        'consistency': (avg/std).round(2),  # info-ratio of contribution
    }).loc[factors]

    summary.to_csv(OUT/'part1_factor_contribution_summary.csv', encoding='utf-8-sig')
    contrib.round(4).to_csv(OUT/'part1_factor_contribution_monthly.csv', encoding='utf-8-sig')

    print("\n" + "="*70)
    print("PART 1: FACTOR CONTRIBUTION SUMMARY")
    print("="*70)
    print(summary.to_string())

    # plot 1: stacked bar of monthly contribution
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    ax = axes[0]
    contrib_plot = contrib[factors]
    bottom_pos = np.zeros(len(contrib_plot))
    bottom_neg = np.zeros(len(contrib_plot))
    colors = {'value_pb':'#1f77b4','value_pe':'#2ca02c','reversal_5':'#ff7f0e',
              'low_vol_60':'#9467bd','momentum_120_5':'#d62728'}
    x = np.arange(len(contrib_plot))
    for f in factors:
        vals = contrib_plot[f].values
        pos = np.where(vals > 0, vals, 0)
        neg = np.where(vals < 0, vals, 0)
        ax.bar(x, pos, bottom=bottom_pos, color=colors[f], label=f, width=0.9)
        ax.bar(x, neg, bottom=bottom_neg, color=colors[f], width=0.9)
        bottom_pos += pos
        bottom_neg += neg
    # show ~12 ticks
    tick_idx = np.linspace(0, len(contrib_plot)-1, 10).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([contrib_plot.index[i].strftime('%Y-%m') for i in tick_idx], rotation=30)
    ax.set_ylabel('Score Contribution')
    ax.set_title('V5 Monthly Score Contribution by Factor')
    ax.axhline(0, color='black', lw=0.5)
    ax.legend(loc='upper left', fontsize=9, ncol=5)
    ax.grid(alpha=0.3, axis='y')

    # plot 2: avg contribution + hit rate side by side
    ax2 = axes[1]
    ax2b = ax2.twinx()
    bar_x = np.arange(len(factors))
    ax2.bar(bar_x - 0.2, summary['avg_contrib'], 0.4, color='steelblue', label='Avg Contrib (left)')
    ax2b.bar(bar_x + 0.2, summary['hit_rate_pct'], 0.4, color='orange', label='Hit Rate % (right)')
    ax2.set_xticks(bar_x); ax2.set_xticklabels(factors, rotation=15)
    ax2.set_ylabel('Avg Contribution', color='steelblue')
    ax2b.set_ylabel('Hit Rate %', color='orange')
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_title('Factor Contribution: Magnitude vs Consistency')
    ax2b.axhline(50, color='gray', lw=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUT/'part1_factor_contribution.png', dpi=130, bbox_inches='tight')
    print(f"[Saved] {OUT/'part1_factor_contribution.png'}")
    return summary


# =========================================================
# PART 2: Market Regime Attribution
# =========================================================
def part2_regime_attribution(holdings):
    """Run V5 backtest, classify each month by HS300 monthly return,
    compute V5 alpha per regime."""
    # rebuild V5 perf
    all_codes = sorted(set().union(*[set(h.index) for h in holdings.values()]))
    price = build_price_from_cache(all_codes)
    price = price.loc[(price.index >= START) & (price.index <= END)].dropna(how='all')
    wdf = holdings_to_weights(holdings, price.index, all_codes)
    perf = compute_portfolio_returns(wdf, price, cost_per_side=0.0015).dropna()

    # monthly returns
    v5_m = perf['nav'].resample('ME').last().pct_change().dropna()
    bench = BENCH.loc[(BENCH.index >= START) & (BENCH.index <= END)]
    bench_m = bench.resample('ME').last().pct_change().dropna()

    # align
    df = pd.DataFrame({'v5': v5_m, 'hs300': bench_m}).dropna()
    df['alpha'] = df['v5'] - df['hs300']

    # regime classification
    def classify(r):
        if r > 0.05: return 'A_Bull (>+5%)'
        elif r > 0: return 'B_Rebound (0~+5%)'
        elif r > -0.03: return 'C_Sideways (-3%~0)'
        else: return 'D_Bear (<-3%)'
    df['regime'] = df['hs300'].apply(classify)

    # group stats
    agg = df.groupby('regime').agg(
        months=('alpha', 'count'),
        v5_avg=('v5', 'mean'),
        hs300_avg=('hs300', 'mean'),
        alpha_avg=('alpha', 'mean'),
        alpha_std=('alpha', 'std'),
        win_rate=('alpha', lambda s: (s > 0).mean()),
        alpha_total=('alpha', 'sum'),
    ).round(4)
    agg['v5_avg_pct'] = (agg['v5_avg']*100).round(2)
    agg['hs300_avg_pct'] = (agg['hs300_avg']*100).round(2)
    agg['alpha_avg_pct'] = (agg['alpha_avg']*100).round(2)
    agg['alpha_total_pct'] = (agg['alpha_total']*100).round(2)
    agg['win_rate_pct'] = (agg['win_rate']*100).round(1)
    cols = ['months','v5_avg_pct','hs300_avg_pct','alpha_avg_pct','win_rate_pct','alpha_total_pct']
    agg = agg[cols]

    agg.to_csv(OUT/'part2_regime_attribution.csv', encoding='utf-8-sig')
    df.round(4).to_csv(OUT/'part2_monthly_alpha_by_regime.csv', encoding='utf-8-sig')

    print("\n" + "="*70)
    print("PART 2: V5 ALPHA BY MARKET REGIME")
    print("="*70)
    print(agg.to_string())

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    regimes = sorted(agg.index)
    x = np.arange(len(regimes))
    ax.bar(x-0.2, agg.loc[regimes,'hs300_avg_pct'], 0.4, label='HS300', color='gray')
    ax.bar(x+0.2, agg.loc[regimes,'v5_avg_pct'], 0.4, label='V5', color='#1f77b4')
    ax.set_xticks(x); ax.set_xticklabels(regimes, rotation=15)
    ax.set_ylabel('Avg Monthly Return (%)')
    ax.set_title('V5 vs HS300 Monthly Return by Regime')
    ax.axhline(0, color='black', lw=0.5)
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    for i, r in enumerate(regimes):
        n = int(agg.loc[r,'months'])
        ax.annotate(f"n={n}", (i, max(agg.loc[r,'v5_avg_pct'], agg.loc[r,'hs300_avg_pct'])),
                    textcoords='offset points', xytext=(0, 5), ha='center', fontsize=9)

    ax2 = axes[1]
    alpha = agg.loc[regimes,'alpha_avg_pct']
    win = agg.loc[regimes,'win_rate_pct']
    colors_a = ['#2ca02c' if v>0 else '#d62728' for v in alpha]
    ax2.bar(x-0.2, alpha, 0.4, color=colors_a, label='Alpha (left)')
    ax2b = ax2.twinx()
    ax2b.bar(x+0.2, win, 0.4, color='orange', alpha=0.7, label='Win Rate % (right)')
    ax2.set_xticks(x); ax2.set_xticklabels(regimes, rotation=15)
    ax2.set_ylabel('Avg Monthly Alpha (%)', color='steelblue')
    ax2b.set_ylabel('Win Rate %', color='orange')
    ax2.set_title('V5 Alpha & Win Rate by Regime')
    ax2.axhline(0, color='black', lw=0.5)
    ax2b.axhline(50, color='gray', lw=0.5, linestyle='--')
    for i, v in enumerate(alpha):
        ax2.annotate(f"{v:+.2f}%", (i-0.2, v), textcoords='offset points',
                     xytext=(0, 4 if v>=0 else -12), ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT/'part2_regime_attribution.png', dpi=130, bbox_inches='tight')
    print(f"[Saved] {OUT/'part2_regime_attribution.png'}")
    return agg, df


def main():
    print("Generating V5 holdings & factor contribution...")
    contrib, holdings = factor_contribution_per_period()
    summary = part1_factor_contribution(contrib)
    print("\nGenerating V5 perf & regime attribution...")
    agg, monthly = part2_regime_attribution(holdings)
    print("\nDone. Reports saved under reports/v5_attribution/")


if __name__ == '__main__':
    main()
