"""V5 风控版回测: 主线 panel + risk_panel 三层过滤."""
import os, sys, pickle
for p in ['HTTP_PROXY','HTTPS_PROXY']: os.environ.pop(p, None)
os.environ['NO_PROXY']='*'
sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime
from quant.strategy import generate_holdings
from quant.backtest import build_price_matrix, holdings_to_weights, compute_portfolio_returns

print('[V5 风控回测] 加载 panel + risk panel ...')
panel = pickle.load(open('data/cache/factor_panel_hs300.pkl','rb'))
risk_panel = pickle.load(open('data/cache/risk_filters_hs300.pkl','rb'))
print(f'  panel: {len(panel)} 因子')
print(f'  risk_panel: {list(risk_panel.keys())}')

# 选股 (启用风控)
holdings = generate_holdings(panel, risk_panel=risk_panel)

# 收集所有股票 (跨期)
all_codes = set()
for h in holdings.values(): all_codes.update(h.index)
print(f'\n[回测] 共需 {len(all_codes)} 只股票价格')

price = build_price_matrix(codes=sorted(all_codes))
price = price.dropna(how='all')

weights = holdings_to_weights(holdings, price.index, sorted(all_codes))
perf = compute_portfolio_returns(weights, price, cost_per_side=0.0015)
perf = perf.dropna()

# 保存
out = Path('reports/v5_risk'); out.mkdir(parents=True, exist_ok=True)
perf.to_csv(out/'daily_returns.csv', encoding='utf-8-sig')

# 末期持仓
last_date = max(holdings.keys())
last_holding = holdings[last_date]
last_holding.to_csv(out/'holding.csv', encoding='utf-8-sig')

# 指标
nav = perf['nav']
ret = perf['net_ret']
days = (nav.index[-1]-nav.index[0]).days
total = nav.iloc[-1]-1
ann = nav.iloc[-1]**(365/days)-1
vol = ret.std()*np.sqrt(252)
sharpe = (ann-0.025)/vol
peak = nav.cummax()
dd = (nav/peak-1).min()

print(f'\n=== V5 风控版整体表现 ===')
print(f'  调仓期数: {len(holdings)}')
print(f'  持仓股票总数: {len(all_codes)}')
print(f'  总收益:   {total*100:.2f}%')
print(f'  年化收益: {ann*100:.2f}%')
print(f'  年化波动: {vol*100:.2f}%')
print(f'  夏普:     {sharpe:.2f}')
print(f'  最大回撤: {dd*100:.2f}%')
print(f'\n[OK] saved: reports/v5_risk/')
