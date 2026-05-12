"""
Monthly Paper-Trading Workflow (V5 Production)
==============================================

One-shot end-to-end run for month-end:
  1. Update NAV using current panel data (mark-to-market)
  2. Rebalance: pick new Top-30 from latest factor panel
  3. Generate human-readable monthly report (Markdown + CSV)
  4. Print the next-month holding list with rebalancing diff

Usage:
  PY=/c/Users/Administrator/AppData/Roaming/Accio/pre-install/python/python.exe
  $PY scripts/monthly_workflow.py 2026-04-30      # specify month-end date
  $PY scripts/monthly_workflow.py                  # default = today

Output files (under reports/paper_trading/):
  YYYY-MM_holdings.csv          - next-month holding (code/name/industry/score/factor breakdown)
  YYYY-MM_rebalance.md          - human-readable monthly report
  YYYY-MM_diff.csv              - sells / buys / keeps vs prior month

NOTE: paper_trade.py is the underlying engine; this wrapper coordinates
it with reporting/Markdown output for monthly review.
"""
import os
import sys
import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime

for p in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(p, None)
os.environ['NO_PROXY'] = '*'
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

REPORT_DIR = Path('reports/paper_trading')
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def get_factor_breakdown(panel, ticker, asof_date):
    """For a given stock + date, return dict of all factor raw values + ranks."""
    out = {}
    for fname, fdf in panel.items():
        if fname.startswith('_'):
            continue
        try:
            sub = fdf[fdf.index <= asof_date]
            if len(sub) == 0 or ticker not in fdf.columns:
                out[fname] = np.nan
                continue
            val = sub[ticker].dropna()
            if len(val) == 0:
                out[fname] = np.nan
                continue
            out[fname] = float(val.iloc[-1])
        except Exception:
            out[fname] = np.nan
    return out


def write_monthly_report(month_str, asof_date, prev_holdings, new_holdings,
                          nav_df, factor_panel):
    """Write reports/paper_trading/YYYY-MM_rebalance.md."""
    md_path = REPORT_DIR / f'{month_str}_rebalance.md'
    csv_path = REPORT_DIR / f'{month_str}_holdings.csv'
    diff_path = REPORT_DIR / f'{month_str}_diff.csv'

    prev_set = set(h['ticker'] for h in prev_holdings) if prev_holdings else set()
    new_set = set(h['ticker'] for h in new_holdings)
    sells = prev_set - new_set
    buys = new_set - prev_set
    keeps = prev_set & new_set
    turnover = len(sells) / max(len(prev_set), 1) * 100

    # Build holdings dataframe with factor breakdown
    rows = []
    for h in new_holdings:
        action = 'KEEP' if h['ticker'] in keeps else 'NEW_BUY'
        breakdown = get_factor_breakdown(factor_panel, h['ticker'], asof_date)
        rows.append({
            'ticker':   h['ticker'],
            'name':     h.get('name', ''),
            'industry': h.get('industry', ''),
            'shares':   h.get('shares', 0),
            'cost_price': h.get('cost_price', 0),
            'cost_value': h.get('cost_value', 0),
            'score':    h.get('score', np.nan),
            'action':   action,
            **{f'f_{k}': v for k, v in breakdown.items()},
        })
    df_new = pd.DataFrame(rows).sort_values('score', ascending=False)
    df_new.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # Diff log
    diff_rows = []
    if prev_holdings:
        prev_dict = {h['ticker']: h for h in prev_holdings}
        for t in sorted(sells):
            h = prev_dict[t]
            diff_rows.append({'action': 'SELL', 'ticker': t, 'name': h.get('name', ''),
                              'industry': h.get('industry', '')})
    for t in sorted(buys):
        h = next(x for x in new_holdings if x['ticker'] == t)
        diff_rows.append({'action': 'BUY', 'ticker': t, 'name': h.get('name', ''),
                          'industry': h.get('industry', ''), 'score': h.get('score')})
    if diff_rows:
        pd.DataFrame(diff_rows).to_csv(diff_path, index=False, encoding='utf-8-sig')

    # ----- Markdown report -----
    nav_df = nav_df.sort_index()
    last = nav_df.iloc[-1]
    cum_ret = (last['nav'] - 1) * 100
    bench_cum = (last['benchmark_nav'] - 1) * 100
    excess_cum = cum_ret - bench_cum

    # Last month return
    if len(nav_df) >= 2:
        prev_row = nav_df.iloc[-2]
        m_ret = (last['nav'] / prev_row['nav'] - 1) * 100
        m_bench = (last['benchmark_nav'] / prev_row['benchmark_nav'] - 1) * 100
        m_excess = m_ret - m_bench
    else:
        m_ret = m_bench = m_excess = float('nan')

    industry_counts = pd.Series(
        [h.get('industry', '') for h in new_holdings]
    ).value_counts()

    lines = [
        f'# {month_str} Paper Trading Monthly Report (V5 Production)',
        '',
        f'**As-of**: {asof_date.date()}  ',
        f'**Run at**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  ',
        f'**Model**: V5 Production (frozen 2026-05-12)  ',
        '',
        '## NAV Summary',
        '',
        f'| Metric | Value |',
        f'|---|---:|',
        f'| Latest NAV | {last["nav"]:.4f} |',
        f'| Latest Value | RMB {last["value"]:,.0f} |',
        f'| Cash | RMB {last["cash"]:,.0f} |',
        f'| Benchmark NAV (HS300) | {last["benchmark_nav"]:.4f} |',
        f'| **Cumulative Return** | **{cum_ret:+.2f}%** |',
        f'| **Cumulative Benchmark** | **{bench_cum:+.2f}%** |',
        f'| **Cumulative Excess** | **{excess_cum:+.2f}pp** |',
        f'| Last Month Portfolio | {m_ret:+.2f}% |',
        f'| Last Month Benchmark | {m_bench:+.2f}% |',
        f'| **Last Month Excess** | **{m_excess:+.2f}pp** |',
        '',
        '## Rebalance Action',
        '',
        f'- **Keep**: {len(keeps)}  /  **Sell**: {len(sells)}  /  **Buy**: {len(buys)}',
        f'- **Turnover**: {turnover:.1f}%',
        '',
    ]

    if sells or buys:
        lines.append('### Sells')
        if sells:
            prev_dict = {h['ticker']: h for h in prev_holdings}
            for t in sorted(sells):
                h = prev_dict[t]
                lines.append(f'- {t} {h.get("name","")} ({h.get("industry","")})')
        else:
            lines.append('- (none)')
        lines.append('')
        lines.append('### Buys')
        if buys:
            for t in sorted(buys):
                h = next(x for x in new_holdings if x['ticker'] == t)
                lines.append(f'- {t} {h.get("name","")} ({h.get("industry","")})  '
                             f'score={h.get("score",0):.4f}')
        else:
            lines.append('- (none)')
        lines.append('')

    lines += [
        '## Industry Distribution',
        '',
    ]
    for ind, n in industry_counts.items():
        lines.append(f'- {ind}: {n}')
    lines.append('')

    lines += [
        '## Top 30 Holdings (by score)',
        '',
        '| # | Ticker | Name | Industry | Score | Action |',
        '|---|---|---|---|---:|---|',
    ]
    for i, (_, row) in enumerate(df_new.iterrows(), 1):
        lines.append(
            f'| {i} | {row["ticker"]} | {row["name"]} | {row["industry"]} | '
            f'{row["score"]:.4f} | {row["action"]} |'
        )
    lines += [
        '',
        '## Files',
        '',
        f'- Holdings CSV: `{csv_path.as_posix()}`',
        f'- Diff CSV: `{diff_path.as_posix() if diff_rows else "(no changes)"}`',
        f'- Snapshot: `data/paper/snapshots/{asof_date.date()}.csv`',
        '',
    ]

    md_path.write_text('\n'.join(lines), encoding='utf-8')
    return md_path, csv_path


def main():
    asof_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y-%m-%d')
    asof = pd.Timestamp(asof_str)
    month_str = asof.strftime('%Y-%m')

    print('=' * 70)
    print(f'V5 Production - Monthly Workflow')
    print(f'As-of date: {asof.date()}')
    print('=' * 70)

    # Load existing state to capture pre-rebalance holdings
    pos_file = Path('data/paper/positions.json')
    prev_holdings = []
    if pos_file.exists():
        prev_state = json.loads(pos_file.read_text(encoding='utf-8'))
        prev_holdings = prev_state.get('holdings', [])
        prev_rebal = prev_state.get('last_rebalance')
        print(f'Prior rebalance: {prev_rebal} ({len(prev_holdings)} holdings)')
    else:
        print('No prior state -> first run requires init separately.')
        print('  Use: python scripts/paper_trade.py init YYYY-MM-DD')
        return

    # Step 1: Update NAV at as-of
    print('\n[1/3] Updating NAV...')
    from scripts.paper_trade import cmd_update_nav
    cmd_update_nav(asof_str)

    # Step 2: Rebalance
    print('\n[2/3] Rebalancing...')
    from scripts.paper_trade import cmd_rebalance
    cmd_rebalance(asof_str)

    # Reload state to capture new holdings
    new_state = json.loads(pos_file.read_text(encoding='utf-8'))
    new_holdings = new_state['holdings']

    # Step 3: Generate Markdown report
    print('\n[3/3] Writing monthly report...')
    nav_df = pd.read_csv('data/paper/nav_history.csv',
                          parse_dates=['date']).set_index('date')
    factor_panel = pickle.load(open('data/cache/factor_panel_hs300.pkl', 'rb'))

    md_path, csv_path = write_monthly_report(
        month_str, asof, prev_holdings, new_holdings, nav_df, factor_panel
    )

    print(f'\n[DONE] Reports written:')
    print(f'  {md_path}')
    print(f'  {csv_path}')
    print()
    print('=' * 70)
    print(f'Next month plan: hold above 30 names until {(asof + pd.offsets.MonthEnd(1)).date()}')
    print('=' * 70)


if __name__ == '__main__':
    main()
