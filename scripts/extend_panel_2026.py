"""
增量延伸 panel 到 2026-04-30
============================

目的: 把现有 factor_panel_hs300.pkl (止于 2025-12-31)
      延伸到 2026-04-30, 用于 paper trading 重放.

策略:
  1. 拉每只股票的 [2025-09-01, 2026-04-30] K 线 (向前多拉 4 个月,
     避免动量/低波因子在新窗口起点 NaN)
  2. 重新计算 [2025-09-01, 2026-04-30] 的因子矩阵
  3. 用 union+sort 合并到旧 panel (2026 部分追加, 2025 部分若有重叠以旧版为准)
  4. 基本面因子: 用 ffill 把 2025 末值延伸到 2026 (Q1 财报要 4 月底才出全)
  5. risk_panel 同样 extend

输出:
  data/cache/factor_panel_hs300.pkl    (覆盖, 含 2026 数据)
  data/cache/risk_filters_hs300.pkl    (覆盖, 含 2026 数据)
"""
import os, sys, pickle, time
for p in ['HTTP_PROXY','HTTPS_PROXY']: os.environ.pop(p, None)
os.environ['NO_PROXY'] = '*'
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
from pathlib import Path
import akshare as ak

from quant.data_loader import get_stock_pool, get_stock_kline
from quant.factors import compute_all_factors, FACTOR_REGISTRY

CACHE_DIR = Path('data/cache')
EXTEND_START = '2025-09-01'  # 多拉 4 月用于因子热身
EXTEND_END   = '2026-04-30'

print('=' * 70)
print(f'  Extend panel: {EXTEND_START} -> {EXTEND_END}')
print('=' * 70)

# ============= 1. 加载现有 panel =============
panel_old = pickle.load(open(CACHE_DIR / 'factor_panel_hs300.pkl', 'rb'))
risk_old  = pickle.load(open(CACHE_DIR / 'risk_filters_hs300.pkl', 'rb'))
print(f'\n[load] panel keys: {list(panel_old.keys())}')
print(f'[load] panel old idx: {next(iter(panel_old.values())).index.max()}')
print(f'[load] risk_panel keys: {list(risk_old.keys())}')

pool = get_stock_pool()
codes = pool['code'].tolist()
print(f'[load] pool: {len(codes)} stocks')

# ============= 2. 拉每只股票的扩展窗口 K 线 =============
print(f'\n[1/3] Pulling kline {EXTEND_START} -> {EXTEND_END} for {len(codes)} stocks...')
per_stock = {}
fail_codes = []
t0 = time.time()
for i, code in enumerate(codes, 1):
    try:
        kl = get_stock_kline(code, EXTEND_START, EXTEND_END)
        if kl.empty or len(kl) < 60:  # 至少 60 天才能算因子
            fail_codes.append(code); continue
        per_stock[code] = compute_all_factors(kl)
    except Exception as ex:
        fail_codes.append(code)
    if i % 50 == 0 or i == len(codes):
        elapsed = time.time() - t0
        eta = elapsed / i * (len(codes) - i)
        print(f'  {i}/{len(codes)}  ok={len(per_stock)}  fail={len(fail_codes)}  '
              f'elapsed={elapsed:.0f}s  eta={eta:.0f}s')

print(f'\n[1/3] done. Success: {len(per_stock)}/{len(codes)}; fails: {len(fail_codes)}')
if fail_codes[:5]:
    print(f'  Failed examples: {fail_codes[:5]}')

# ============= 3. 重组为 panel 格式 =============
print(f'\n[2/3] Building new panel slice...')
panel_new = {}
for fname in FACTOR_REGISTRY.keys():
    df = pd.DataFrame({code: per_stock[code][fname] for code in per_stock if fname in per_stock[code].columns})
    df = df.sort_index()
    # 只保留 2026-01-01 之后的部分 (2025 用旧 panel 的)
    df_2026 = df[df.index >= '2026-01-01']
    panel_new[fname] = df_2026
    print(f'  {fname}: 2026 slice shape = {df_2026.shape}')

# ============= 4. 合并旧 + 新 =============
print(f'\n[3/3] Merging old + new panel...')
panel_merged = {}
for fname, old_df in panel_old.items():
    if fname in panel_new and not panel_new[fname].empty:
        new_df = panel_new[fname]
        common_cols = old_df.columns.intersection(new_df.columns)
        old_part = old_df[common_cols]
        new_part = new_df[common_cols]
        merged = pd.concat([old_part, new_part]).sort_index()
        # 去重 (以新数据为准, 万一日期重叠)
        merged = merged[~merged.index.duplicated(keep='last')]
        panel_merged[fname] = merged
        print(f'  {fname}: {old_df.shape} + {new_df.shape} -> {merged.shape}')
    else:
        # 基本面因子: 用 ffill 延伸
        if old_df.shape[0] > 0:
            last_date = old_df.index.max()
            # 拼接一个空的 2026 部分, ffill
            new_idx = pd.date_range('2026-01-02', '2026-04-30', freq='B')
            ext = pd.DataFrame(index=new_idx, columns=old_df.columns)
            merged = pd.concat([old_df, ext]).sort_index()
            merged = merged[~merged.index.duplicated(keep='last')]
            merged = merged.ffill()
            panel_merged[fname] = merged
            print(f'  {fname}: ffill-extended to 2026-04-30, shape = {merged.shape}')
        else:
            panel_merged[fname] = old_df

# 保存
pickle.dump(panel_merged, open(CACHE_DIR / 'factor_panel_hs300.pkl', 'wb'))
print(f'\n[save] factor_panel_hs300.pkl saved.')
print(f'  range now: {next(iter(panel_merged.values())).index.min()} -> '
      f'{next(iter(panel_merged.values())).index.max()}')
print(f'  trading days: {len(next(iter(panel_merged.values())))}')

# ============= 5. risk_panel 同样 extend =============
print(f'\n[risk] Extending risk_filters...')
risk_new = {}
for rname, old_df in risk_old.items():
    if rname == 'liquidity_amount_60':
        # 从 K 线重算 60d amount mean
        df = pd.DataFrame()
        for code in per_stock:
            try:
                kl = get_stock_kline(code, EXTEND_START, EXTEND_END)
                kl['date'] = pd.to_datetime(kl['date'])
                kl = kl.set_index('date').sort_index()
                amt60 = kl['amount'].rolling(60, min_periods=30).mean()
                df[code] = amt60
            except Exception:
                continue
        df_2026 = df[df.index >= '2026-01-01']
        common_cols = old_df.columns.intersection(df_2026.columns)
        merged = pd.concat([old_df[common_cols], df_2026[common_cols]]).sort_index()
        merged = merged[~merged.index.duplicated(keep='last')]
        risk_new[rname] = merged
        print(f'  {rname}: -> {merged.shape}')
    elif rname == 'roe_yoy':
        # 同 quality_roe, 用 ffill 延伸 (Q1 财报 4 月底才出全, 暂用 2025 末值)
        last_date = old_df.index.max()
        new_idx = pd.date_range('2026-01-02', '2026-04-30', freq='B')
        ext = pd.DataFrame(index=new_idx, columns=old_df.columns)
        merged = pd.concat([old_df, ext]).sort_index()
        merged = merged[~merged.index.duplicated(keep='last')]
        merged = merged.ffill()
        risk_new[rname] = merged
        print(f'  {rname}: ffill-extended -> {merged.shape}')
    else:
        risk_new[rname] = old_df

pickle.dump(risk_new, open(CACHE_DIR / 'risk_filters_hs300.pkl', 'wb'))
print(f'\n[save] risk_filters_hs300.pkl saved.')

print('\n' + '=' * 70)
print('  DONE. Now you can run: paper_trade.py rebalance / update_nav for 2026 dates')
print('=' * 70)
