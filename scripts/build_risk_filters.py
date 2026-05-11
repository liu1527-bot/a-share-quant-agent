"""
为 V5 风控版构建过滤因子: 流动性 + 波动率 (panel 已有 low_vol_60 直接用) + ROE 同比

输出: data/cache/risk_filters_hs300.pkl
  {
    'liquidity_amount_60': DataFrame(date×ticker, 60日均成交额),
    'roe_yoy': DataFrame(date×ticker, ROE 同比变化率)
  }
"""
import os, sys, pickle, glob
for p in ['HTTP_PROXY','HTTPS_PROXY']: os.environ.pop(p, None)
os.environ['NO_PROXY']='*'
sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
from pathlib import Path

# 读 panel 拿股票列表
panel = pickle.load(open('data/cache/factor_panel_hs300.pkl','rb'))
tickers = list(panel['low_vol_60'].columns)
target_index = panel['low_vol_60'].index
print(f'[risk filters] 目标 {len(tickers)} 只股票, {len(target_index)} 个日期')

# === 1. 流动性: 60 日均成交额 ===
print('[risk filters] 计算 liquidity_amount_60...')
liq_data = {}
for code in tickers:
    files = sorted(glob.glob(f'data/cache/kline/{code}_*_qfq.parquet'))
    if not files:
        continue
    # 拼接所有 kline 文件 (覆盖 2018-2025)
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df['date'] = pd.to_datetime(df['date'])
        dfs.append(df[['date','amount']])
    if not dfs:
        continue
    full = pd.concat(dfs, ignore_index=True).drop_duplicates('date').sort_values('date').set_index('date')
    # 60 日均成交额 (单位: 千)
    liq60 = full['amount'].rolling(60, min_periods=20).mean()
    liq_data[code] = liq60

liq_df = pd.DataFrame(liq_data)
# 对齐到 panel 的日期
liq_aligned = liq_df.reindex(liq_df.index.union(target_index)).sort_index().ffill().reindex(target_index)
print(f'  shape: {liq_aligned.shape}, 末日覆盖率: {liq_aligned.iloc[-1].notna().sum()}/{len(tickers)}')

# === 2. ROE 同比: 从 fundamentals 的 quality_roe 算 ===
print('[risk filters] 计算 roe_yoy...')
fp = pickle.load(open('data/cache/fundamentals_panel_hs300.pkl','rb'))
roe = fp['quality_roe']
print(f'  raw quality_roe shape: {roe.shape}')

# ROE 同比变化: 在 quarterly 频率上 shift(4) 算 1 年差, 然后 reindex 到日频 ffill
roe_q_yoy = roe - roe.shift(4)  # quarter-on-quarter year-over-year
print(f'  quarterly roe_yoy shape: {roe_q_yoy.shape}')

# reindex 到日频
combined = roe_q_yoy.index.union(target_index).sort_values()
roe_yoy = roe_q_yoy.reindex(combined).ffill().reindex(target_index)
print(f'  日频 roe_yoy shape: {roe_yoy.shape}, 末日覆盖率: {roe_yoy.iloc[-1].notna().sum()}/{len(tickers)}')

# === 保存 ===
out = {
    'liquidity_amount_60': liq_aligned,
    'roe_yoy': roe_yoy,
}
Path('data/cache').mkdir(exist_ok=True, parents=True)
with open('data/cache/risk_filters_hs300.pkl','wb') as f:
    pickle.dump(out, f)
print(f'\n[OK] saved: data/cache/risk_filters_hs300.pkl')

# 抽样
print('\n=== 抽样 2024-12-31 ===')
date = pd.Timestamp('2024-12-31')
if date in liq_aligned.index:
    s = liq_aligned.loc[date].dropna().sort_values()
    print(f'流动性 (60日均成交额, 千) 分布:')
    print(f'  min={s.min():.0f}  Q10={s.quantile(0.1):.0f}  median={s.median():.0f}  Q90={s.quantile(0.9):.0f}  max={s.max():.0f}')
    print(f'  最低 5 只: {s.head().to_dict()}')
if date in roe_yoy.index:
    s = roe_yoy.loc[date].dropna().sort_values()
    print(f'\nROE 同比变化分布:')
    print(f'  min={s.min():.2f}  Q10={s.quantile(0.1):.2f}  median={s.median():.2f}  Q90={s.quantile(0.9):.2f}  max={s.max():.2f}')
    print(f'  下滑最严重 5 只 (ROE_yoy<0): {s.head().to_dict()}')
