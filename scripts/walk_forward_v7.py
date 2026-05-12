# -*- coding: utf-8 -*-
"""
V7 walk-forward: V5 + HS300 timing layer.
Timing rule:
  - HS300 close > MA20 AND MA20 > MA60   -> 100% V5
  - else                                 -> 50% V5 + 50% cash
  - 14-day cooldown after each switch (avoid whipsaw)
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
OUT = Path('reports/walk_forward_v7'); OUT.mkdir(parents=True, exist_ok=True)
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

DEFENSIVE_POS = 0.5     # 防御仓位 50%
COOLDOWN_DAYS = 14      # 信号切换冷却期
MA_SHORT = 20
MA_LONG = 60


def make_timing_signal(start, end):
    """Return DataFrame index=date, col='target_pos' in {1.0, DEFENSIVE_POS}"""
    # We need extra history before `start` to compute MA60 at start
    pad = pd.Timedelta(days=140)
    s = pd.to_datetime(start) - pad
    e = pd.to_datetime(end)
    b = BENCH.loc[(BENCH.index >= s) & (BENCH.index <= e)].copy()
    ma_s = b.rolling(MA_SHORT).mean()
    ma_l = b.rolling(MA_LONG).mean()
    raw_full = ((b > ma_s) & (ma_s > ma_l)).astype(float)  # 1=full
    # cooldown filter: only switch when last switch was >= COOLDOWN_DAYS ago
    sig = raw_full.copy()
    last_switch_idx = -10**9
    cur = raw_full.iloc[0]
    for i, d in enumerate(sig.index):
        target = raw_full.iloc[i]
        if target != cur and (i - last_switch_idx) >= COOLDOWN_DAYS:
            cur = target
            last_switch_idx = i
        sig.iloc[i] = cur
    # map: 1=full -> 1.0; 0=defensive -> DEFENSIVE_POS
    target_pos = sig.map(lambda x: 1.0 if x == 1.0 else DEFENSIVE_POS)
    target_pos = target_pos.loc[target_pos.index >= pd.to_datetime(start)]
    return target_pos


def run_v5(start, end):
    orig_freq = config.REBALANCE_FREQ
    orig_top = config.TOP_N
    orig_max = config.MAX_PER_INDUSTRY
    orig_w = config.FACTOR_WEIGHTS
    config.REBALANCE_FREQ = 'ME'
    config.TOP_N = 30
    config.MAX_PER_INDUSTRY = 3
    config.FACTOR_WEIGHTS = V5_WEIGHTS
    try:
        holdings = generate_holdings(PANEL, start_date=start, end_date=end,
                                     top_n=30, risk_panel=RISK)
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


def apply_timing(perf_v5, target_pos_series):
    """V7 = V5 daily return * target_pos (lagged 1 day) + cash interest * (1-pos).
    Cash assumed 0 return."""
    df = perf_v5.copy()
    # Use V5 net return; multiply by lagged target_pos to avoid look-ahead
    pos = target_pos_series.reindex(df.index).ffill().fillna(1.0)
    pos_lag = pos.shift(1).fillna(1.0)
    # Switch cost: 0.05% one-side trading cost when target_pos changes
    pos_change = pos_lag.diff().abs().fillna(0)
    switch_cost = pos_change * 0.0005  # 0.05% per switch (rebalance to V5 portfolio)
    v7_ret = df['net_ret'] * pos_lag - switch_cost
    v7_nav = (1 + v7_ret).cumprod()
    out = pd.DataFrame({
        'v5_nav': df['nav'],
        'v5_ret': df['net_ret'],
        'pos': pos_lag,
        'v7_ret': v7_ret,
        'nav': v7_nav,
        'net_ret': v7_ret,
    })
    return out


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
        perf_v5 = run_v5(wstart, wend)
        m_v5 = metrics(perf_v5)
        print(f"  V5  total={m_v5['total']*100:+6.2f}% ann={m_v5['ann']*100:+6.2f}% sharpe={m_v5['sharpe']:+.2f} mdd={m_v5['mdd']*100:+6.2f}%")

        timing = make_timing_signal(wstart, wend)
        perf_v7 = apply_timing(perf_v5, timing)
        m_v7 = metrics(perf_v7)
        # 计算 timing 统计
        days_full = (perf_v7['pos'] == 1.0).sum()
        days_def = (perf_v7['pos'] == DEFENSIVE_POS).sum()
        switches = (perf_v7['pos'].diff().abs() > 0).sum()
        pct_full = days_full / max(len(perf_v7), 1) * 100
        print(f"  V7  total={m_v7['total']*100:+6.2f}% ann={m_v7['ann']*100:+6.2f}% sharpe={m_v7['sharpe']:+.2f} mdd={m_v7['mdd']*100:+6.2f}%")
        print(f"      timing: full {days_full}d ({pct_full:.0f}%), defensive {days_def}d, switches={switches}")

        h = hs300_metrics(wstart, wend)
        print(f"  HS300 total={h['total']*100:+6.2f}% ann={h['ann']*100:+6.2f}%")

        rows.append({
            'window': wname, 'start': wstart, 'end': wend,
            'V5_total': m_v5['total'], 'V5_ann': m_v5['ann'],
            'V5_sharpe': m_v5['sharpe'], 'V5_mdd': m_v5['mdd'],
            'V7_total': m_v7['total'], 'V7_ann': m_v7['ann'],
            'V7_sharpe': m_v7['sharpe'], 'V7_mdd': m_v7['mdd'],
            'HS300_total': h['total'], 'HS300_ann': h['ann'],
            'V7_vs_V5': m_v7['total'] - m_v5['total'],
            'V7_vs_HS300': m_v7['total'] - h['total'],
            'V7_beats_V5': m_v7['total'] > m_v5['total'],
            'V7_beats_HS300': m_v7['total'] > h['total'],
            'pct_full_pos': pct_full,
            'n_switches': switches,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'walk_forward_v7.csv', index=False, encoding='utf-8-sig')
    print(f"\n[Saved] {OUT / 'walk_forward_v7.csv'}")

    print("\n" + "="*70)
    print("SUMMARY: V7 (V5 + MA20/60 timing) vs V5")
    print("="*70)
    for _, r in df.iterrows():
        f1 = 'WIN' if r['V7_beats_V5'] else 'LOSE'
        f2 = 'beat HS300' if r['V7_beats_HS300'] else 'lose HS300'
        print(f"  {r['window']:25s} V5={r['V5_total']*100:+6.2f}%  V7={r['V7_total']*100:+6.2f}%  "
              f"HS300={r['HS300_total']*100:+6.2f}%  [{f1}, {f2}]  full_pos={r['pct_full_pos']:.0f}% switch={r['n_switches']}")
    wins_v5 = df['V7_beats_V5'].sum()
    wins_h = df['V7_beats_HS300'].sum()
    avg_v5 = df['V7_vs_V5'].mean() * 100
    avg_h = df['V7_vs_HS300'].mean() * 100
    print(f"\n  V7 beats V5    : {wins_v5}/{len(df)}  avg excess = {avg_v5:+.2f}pp")
    print(f"  V7 beats HS300 : {wins_h}/{len(df)}  avg excess = {avg_h:+.2f}pp")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    x = np.arange(len(df)); w = 0.27
    ax = axes[0]
    ax.bar(x - w, df['HS300_total']*100, w, label='HS300', color='#8c8c8c')
    ax.bar(x,     df['V5_total']*100,    w, label='V5 baseline', color='#1f77b4')
    ax.bar(x + w, df['V7_total']*100,    w, label='V7 (V5 + MA20/60 timing)', color='#ff7f0e')
    ax.set_xticks(x); ax.set_xticklabels(df['window'], rotation=15, ha='right')
    ax.set_ylabel('Total Return (%)')
    ax.set_title(f'Walk-Forward: V7 beats V5 in {wins_v5}/{len(df)} windows, beats HS300 in {wins_h}/{len(df)}')
    ax.axhline(0, color='gray', linewidth=0.8); ax.legend(); ax.grid(alpha=0.3, axis='y')
    for i in range(len(df)):
        for off, col, val in [(-w, '#8c8c8c', df['HS300_total'].iloc[i]*100),
                              (0,   '#1f77b4', df['V5_total'].iloc[i]*100),
                              (w,   '#ff7f0e', df['V7_total'].iloc[i]*100)]:
            ax.annotate(f"{val:+.1f}", (i+off, val), textcoords='offset points',
                        xytext=(0, 3 if val >= 0 else -10), ha='center', fontsize=8, color=col)

    ax2 = axes[1]
    excess = df['V7_vs_V5']*100
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in excess]
    ax2.bar(x, excess, color=colors)
    ax2.set_xticks(x); ax2.set_xticklabels(df['window'], rotation=15, ha='right')
    ax2.set_ylabel('V7 - V5 (pp)')
    ax2.set_title(f'Per-Window Excess (V7 - V5)  -  Avg: {avg_v5:+.2f}pp')
    ax2.axhline(0, color='gray', linewidth=0.8); ax2.grid(alpha=0.3, axis='y')
    for i, v in enumerate(excess):
        ax2.annotate(f"{v:+.2f}pp", (i, v), textcoords='offset points',
                     xytext=(0, 3 if v >= 0 else -10), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT / 'walk_forward_v7.png', dpi=130, bbox_inches='tight')
    print(f"[Saved] {OUT / 'walk_forward_v7.png'}")


if __name__ == '__main__':
    main()
