"""
Plot cumulative NAV vs benchmark for paper trading.
Output: reports/paper_trading/cumulative.png
"""
import sys, os
sys.path.insert(0, '.')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

REPORT_DIR = Path('reports/paper_trading')
REPORT_DIR.mkdir(parents=True, exist_ok=True)

nav = pd.read_csv('data/paper/nav_history.csv', parse_dates=['date']).set_index('date').sort_index()
nav['excess_pp'] = (nav['nav'] - nav['benchmark_nav']) * 100

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1]})

# Top: NAV vs benchmark
axes[0].plot(nav.index, (nav['nav'] - 1) * 100, marker='o', lw=2,
             label='V5 Portfolio', color='#2c7be5')
axes[0].plot(nav.index, (nav['benchmark_nav'] - 1) * 100, marker='s', lw=2,
             label='HS300 Benchmark', color='#999999')
axes[0].axhline(0, color='black', lw=0.5)
axes[0].set_ylabel('Cumulative Return (%)')
axes[0].set_title(f'V5 Production Paper Trading - Inception {nav.index[0].date()}')
axes[0].legend(loc='best')
axes[0].grid(alpha=0.3)

# Bottom: cumulative excess
colors = ['#28a745' if v >= 0 else '#dc3545' for v in nav['excess_pp']]
axes[1].bar(nav.index, nav['excess_pp'], width=20, color=colors, alpha=0.7,
            label='Cumulative Excess')
axes[1].axhline(0, color='black', lw=0.5)
axes[1].set_ylabel('Excess (pp)')
axes[1].set_xlabel('Date')
axes[1].grid(alpha=0.3, axis='y')

# Annotate values
for d, e in zip(nav.index, nav['excess_pp']):
    axes[1].annotate(f'{e:+.1f}', xy=(d, e),
                      ha='center', va='bottom' if e >= 0 else 'top',
                      fontsize=9)

plt.tight_layout()
out = REPORT_DIR / 'cumulative.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f'[Saved] {out}')

# Also output a per-month bar chart
fig2, ax = plt.subplots(figsize=(10, 5))
nav['port_ret'] = nav['nav'].pct_change() * 100
nav['bench_ret'] = nav['benchmark_nav'].pct_change() * 100
nav['m_excess'] = nav['port_ret'] - nav['bench_ret']
mdf = nav.dropna(subset=['m_excess']).copy()
xs = range(len(mdf))
w = 0.4
ax.bar([x - w/2 for x in xs], mdf['port_ret'], width=w, label='V5 Portfolio', color='#2c7be5')
ax.bar([x + w/2 for x in xs], mdf['bench_ret'], width=w, label='HS300', color='#999999')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(list(xs))
ax.set_xticklabels([d.strftime('%Y-%m') for d in mdf.index], rotation=0)
ax.set_ylabel('Monthly Return (%)')
ax.set_title('Monthly Returns: V5 vs HS300')
ax.legend()
ax.grid(alpha=0.3, axis='y')
for i, (p, b, e) in enumerate(zip(mdf['port_ret'], mdf['bench_ret'], mdf['m_excess'])):
    ax.annotate(f'{e:+.1f}pp', xy=(i, max(p, b) + 0.3),
                ha='center', va='bottom', fontsize=9,
                color='#28a745' if e >= 0 else '#dc3545')
plt.tight_layout()
out2 = REPORT_DIR / 'monthly_returns.png'
plt.savefig(out2, dpi=110, bbox_inches='tight')
print(f'[Saved] {out2}')
