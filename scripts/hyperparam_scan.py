# -*- coding: utf-8 -*-
"""
Hyperparameter scan for V5: rebalance freq / top_n / max_per_industry.
Sequential 1-D scan to limit cost.
"""
import os, sys, pickle, time
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
OUT = Path('reports/hyperparam'); OUT.mkdir(parents=True, exist_ok=True)
KLINE = Path('data/cache/kline')


def build_price_from_cache(codes):
    """Use existing parquet cache pieces 2018-2020 + 2021-2025 + 2026."""
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
    price = price.T.groupby(level=0).last().T  # merge dup cols
    return price.sort_index()

START = '2021-01-04'
END = '2026-04-30'

V5_WEIGHTS = {
    'value_pb': 0.30,
    'value_pe': 0.25,
    'reversal_5': 0.20,
    'low_vol_60': 0.15,
    'momentum_120_5': 0.10,
}


def run_one(freq, top_n, max_ind, label):
    """Run V5 with given hyperparams, full period."""
    orig_freq = config.REBALANCE_FREQ
    orig_top = config.TOP_N
    orig_max = config.MAX_PER_INDUSTRY
    orig_w = config.FACTOR_WEIGHTS
    config.REBALANCE_FREQ = freq
    config.TOP_N = top_n
    config.MAX_PER_INDUSTRY = max_ind
    config.FACTOR_WEIGHTS = V5_WEIGHTS
    t0 = time.time()
    try:
        holdings = generate_holdings(PANEL, start_date=START, end_date=END, top_n=top_n, risk_panel=RISK)
        if len(holdings) == 0:
            return None
        all_codes = set()
        for h in holdings.values():
            all_codes.update(h.index)
        # turnover
        turnovers = []
        prev = None
        for d in sorted(holdings.keys()):
            cur = set(holdings[d].index)
            if prev is not None:
                turnovers.append(len(cur - prev) / len(cur))
            prev = cur
        avg_turnover = np.mean(turnovers) if turnovers else 0
        price = build_price_from_cache(sorted(all_codes))
        price = price.loc[(price.index >= START) & (price.index <= END)]
        price = price.dropna(how='all')
        weights_df = holdings_to_weights(holdings, price.index, sorted(all_codes))
        perf = compute_portfolio_returns(weights_df, price, cost_per_side=0.0015)
        perf = perf.dropna()
        nav = perf['nav']
        ret = perf['net_ret']
        days = max((nav.index[-1] - nav.index[0]).days, 1)
        total = nav.iloc[-1] - 1
        ann = nav.iloc[-1] ** (365 / days) - 1
        vol = ret.std() * np.sqrt(252)
        sharpe = (ann - 0.025) / vol if vol > 0 else np.nan
        peak = nav.cummax()
        mdd = (nav / peak - 1).min()
        elapsed = time.time() - t0
        return {
            'label': label,
            'freq': freq, 'top_n': top_n, 'max_ind': max_ind if max_ind else 'no',
            'periods': len(holdings),
            'stocks': len(all_codes),
            'total': total, 'ann': ann, 'sharpe': sharpe, 'mdd': mdd,
            'turnover': avg_turnover,
            'time_s': elapsed,
        }
    finally:
        config.REBALANCE_FREQ = orig_freq
        config.TOP_N = orig_top
        config.MAX_PER_INDUSTRY = orig_max
        config.FACTOR_WEIGHTS = orig_w


def fmt(r):
    return (f"  total={r['total']*100:+6.2f}%  ann={r['ann']*100:+6.2f}%  "
            f"sharpe={r['sharpe']:+.2f}  mdd={r['mdd']*100:+6.2f}%  "
            f"turnover={r['turnover']*100:.1f}%  time={r['time_s']:.0f}s")


def main():
    all_results = []

    # Stage 1: scan freq
    print("\n" + "=" * 70)
    print("Stage 1: Scan rebalance frequency (TOP_N=30, MAX=3)")
    print("=" * 70)
    stage1 = []
    for freq, name in [('W-FRI', 'weekly'), ('2W-FRI', 'biweekly'), ('ME', 'monthly_baseline'), ('QE', 'quarterly')]:
        print(f"\n[{name}] freq={freq}")
        r = run_one(freq, 30, 3, f'freq={name}')
        if r:
            stage1.append(r)
            all_results.append(r)
            print(fmt(r))
    s1 = pd.DataFrame(stage1).sort_values('sharpe', ascending=False)
    print("\n--- Stage 1 ranked by Sharpe ---")
    print(s1[['label', 'total', 'ann', 'sharpe', 'mdd', 'turnover']].round(3).to_string(index=False))
    best_freq = s1.iloc[0]['freq']
    print(f"\n>>> Best freq: {best_freq}")

    # Stage 2: scan top_n
    print("\n" + "=" * 70)
    print(f"Stage 2: Scan TOP_N (freq={best_freq}, MAX=3)")
    print("=" * 70)
    stage2 = []
    for top_n in [15, 20, 30, 40, 50]:
        print(f"\n[top_n={top_n}]")
        r = run_one(best_freq, top_n, 3, f'top_n={top_n}')
        if r:
            stage2.append(r)
            all_results.append(r)
            print(fmt(r))
    s2 = pd.DataFrame(stage2).sort_values('sharpe', ascending=False)
    print("\n--- Stage 2 ranked by Sharpe ---")
    print(s2[['label', 'total', 'ann', 'sharpe', 'mdd', 'turnover']].round(3).to_string(index=False))
    best_top = int(s2.iloc[0]['top_n'])
    print(f"\n>>> Best top_n: {best_top}")

    # Stage 3: scan max_per_industry
    print("\n" + "=" * 70)
    print(f"Stage 3: Scan MAX_PER_INDUSTRY (freq={best_freq}, top_n={best_top})")
    print("=" * 70)
    stage3 = []
    for mx, name in [(2, 'max=2'), (3, 'max=3'), (5, 'max=5'), (None, 'no_limit')]:
        print(f"\n[{name}]")
        r = run_one(best_freq, best_top, mx, name)
        if r:
            stage3.append(r)
            all_results.append(r)
            print(fmt(r))
    s3 = pd.DataFrame(stage3).sort_values('sharpe', ascending=False)
    print("\n--- Stage 3 ranked by Sharpe ---")
    print(s3[['label', 'total', 'ann', 'sharpe', 'mdd', 'turnover']].round(3).to_string(index=False))

    # Save all
    df = pd.DataFrame(all_results)
    df.to_csv(OUT / 'hyperparam_scan.csv', index=False, encoding='utf-8-sig')
    print(f"\n[Saved] {OUT / 'hyperparam_scan.csv'}")

    # Final winner
    print("\n" + "=" * 70)
    print("WINNER (best by Sharpe across all stages)")
    print("=" * 70)
    best = df.sort_values('sharpe', ascending=False).iloc[0]
    print(best.to_string())

    # Visualize Stage 1 + 2 + 3
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for i, (s, title) in enumerate([(s1, 'Stage 1: Rebalance Frequency'),
                                     (s2, 'Stage 2: Top N'),
                                     (s3, 'Stage 3: Max per Industry')]):
        ax = axes[i][0]
        x = np.arange(len(s))
        ax.bar(x, s['ann'] * 100, color='#1f77b4')
        ax.set_xticks(x)
        ax.set_xticklabels(s['label'], rotation=20, ha='right', fontsize=9)
        ax.set_title(f'{title} - Annualized Return (%)', fontsize=11)
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.grid(alpha=0.3, axis='y')
        for j, v in enumerate(s['ann'] * 100):
            ax.annotate(f"{v:+.1f}%", (j, v), textcoords='offset points',
                        xytext=(0, 3 if v >= 0 else -10), ha='center', fontsize=9)

        ax2 = axes[i][1]
        ax2.bar(x, s['sharpe'], color='#d62728')
        ax2.set_xticks(x)
        ax2.set_xticklabels(s['label'], rotation=20, ha='right', fontsize=9)
        ax2.set_title(f'{title} - Sharpe', fontsize=11)
        ax2.axhline(0, color='gray', linewidth=0.8)
        ax2.grid(alpha=0.3, axis='y')
        for j, v in enumerate(s['sharpe']):
            ax2.annotate(f"{v:+.2f}", (j, v), textcoords='offset points',
                         xytext=(0, 3 if v >= 0 else -10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / 'hyperparam_scan.png', dpi=130, bbox_inches='tight')
    print(f"[Saved] {OUT / 'hyperparam_scan.png'}")


if __name__ == '__main__':
    main()
