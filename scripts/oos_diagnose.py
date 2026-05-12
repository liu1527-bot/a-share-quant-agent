"""OOS 诊断: 截断单日回报到 ±20% 重新算指标."""
import os, sys
for p in ['HTTP_PROXY','HTTPS_PROXY']: os.environ.pop(p, None)
os.environ['NO_PROXY']='*'
sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')

import pickle, pandas as pd, numpy as np
from quant import config
config.BACKTEST_START='2018-01-01'
config.BACKTEST_END='2020-12-31'

with open('data/cache/factor_panel_hs300_oos2018.pkl','rb') as f:
    panel = pickle.load(f)
from quant.strategy import generate_holdings
from quant.backtest import build_price_matrix, holdings_to_weights, compute_portfolio_returns, get_benchmark_nav
holdings = generate_holdings(panel)

all_codes = set()
for h in holdings.values(): all_codes.update(h.index)
price = build_price_matrix(codes=list(all_codes))
price = price.dropna(how='all')
weights = holdings_to_weights(holdings, price.index, list(all_codes))

perf = compute_portfolio_returns(weights, price, cost_per_side=0.0015)
perf = perf.dropna()
ret = perf['nav'].pct_change()

# 截断 ±20%
ret_clip = ret.clip(-0.20, 0.20)
nav_clip = (1 + ret_clip.fillna(0)).cumprod()

print('========== V4 OOS 2018-2020 (clip +/- 20pct) ==========')
print(f'  end nav: {nav_clip.iloc[-1]:.4f}')
total = (nav_clip.iloc[-1]-1)*100
print(f'  total: {total:.2f}%')
days = (nav_clip.index[-1]-nav_clip.index[0]).days
ann = (nav_clip.iloc[-1]**(365/days)-1)*100
print(f'  ann: {ann:.2f}%')

vol = ret_clip.std() * np.sqrt(252) * 100
print(f'  ann vol: {vol:.2f}%')
print(f'  sharpe (rf=2.5pct): {(ann/100-0.025)/(vol/100):.2f}')

peak = nav_clip.cummax()
dd = (nav_clip/peak - 1)
print(f'  max dd: {dd.min()*100:.2f}%')
print(f'  win rate: {(ret_clip>0).sum()/len(ret_clip)*100:.1f}%')

# benchmark
bench = get_benchmark_nav(perf.index.min().strftime('%Y-%m-%d'),
                           perf.index.max().strftime('%Y-%m-%d'))
bench_ret = bench.pct_change()
bench_total = (bench.iloc[-1]/bench.iloc[0]-1)*100
bench_ann = (bench.iloc[-1]/bench.iloc[0])**(365/days)*100 - 100
print(f'  --- bench HS300 ---')
print(f'  total: {bench_total:.2f}%, ann: {bench_ann:.2f}%')
print(f'  --- excess ---')
print(f'  ann excess: {ann-bench_ann:.2f}%')
te = (ret_clip - bench_ret.reindex(ret_clip.index).fillna(0)).std() * np.sqrt(252) * 100
print(f'  TE: {te:.2f}%, IR: {(ann-bench_ann)/te:.2f}')

# 持仓数量分析
print()
print('========== 持仓数量分析 ==========')
print(f'调仓期数: {len(holdings)}')
sizes = [len(h) for h in holdings.values()]
print(f'每期持仓数: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f}')

# 看 2020-04-30 那天发生了什么
print()
print('========== 2020-04-30 异动诊断 ==========')
date_problem = pd.Timestamp('2020-04-30')
prev_dates = price.index[price.index < date_problem]
prev_date = prev_dates[-1] if len(prev_dates) else None
print(f'前一交易日: {prev_date}')
if prev_date is not None:
    p_change = price.loc[date_problem] / price.loc[prev_date] - 1
    big = p_change.abs().sort_values(ascending=False).head(8)
    print(f'2020-04-30 涨跌幅 TOP8 (绝对值):')
    print(big)
