"""V4 walk-forward 验证: 直接对 V4 全周期 nav 做时间切片, 不重跑数据."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# V4 IS (2021-2025) 全周期 daily_returns
PERF = pd.read_csv('reports/20260511_202758/daily_returns.csv')
PERF.columns = [c.replace('\ufeff','') for c in PERF.columns]
PERF['date'] = pd.to_datetime(PERF['date'])
PERF = PERF.set_index('date')

# 同期基准 (HS300) - 用 oos 报告里的 benchmark 取不到, 重新拉
import sys, os
for p in ['HTTP_PROXY','HTTPS_PROXY']: os.environ.pop(p, None)
os.environ['NO_PROXY']='*'
sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
from quant.backtest import get_benchmark_nav

print('[walk-forward] 拉 HS300 基准...')
BENCH = get_benchmark_nav('2021-01-01', '2025-12-31')
print(f'  {len(BENCH)} 个交易日')

WINDOWS = [
    ('2021-04-01', '2022-03-31', 'W1: 2021-04~2022-03 (大盘震荡, 价值起涨)'),
    ('2022-04-01', '2023-03-31', 'W2: 2022-04~2023-03 (深度调整 + 反弹)'),
    ('2023-04-01', '2024-03-31', 'W3: 2023-04~2024-03 (慢熊红利)'),
    ('2024-04-01', '2025-04-30', 'W4: 2024-04~2025-04 (924 反转 + 跨年)'),
    ('2025-05-01', '2025-12-31', 'W5: 2025-05~2025-12 (放量后)'),
]

def metrics(net_ret, bench_nav, label):
    """从 net_ret (Series) 计算指标"""
    if len(net_ret) < 20:
        return None
    nav = (1 + net_ret).cumprod()
    days = (net_ret.index[-1] - net_ret.index[0]).days

    total = nav.iloc[-1] - 1
    ann = nav.iloc[-1] ** (365/days) - 1 if days > 0 else 0
    vol = net_ret.std() * np.sqrt(252)
    sharpe = (ann - 0.025) / vol if vol > 0 else 0
    peak = nav.cummax()
    dd = (nav/peak - 1).min()

    # 基准
    bench_slice = bench_nav.loc[net_ret.index[0]:net_ret.index[-1]]
    bench_ret = bench_slice.pct_change().fillna(0)
    bench_total = bench_slice.iloc[-1]/bench_slice.iloc[0] - 1
    bench_ann = (bench_slice.iloc[-1]/bench_slice.iloc[0]) ** (365/days) - 1 if days>0 else 0

    excess_ann = ann - bench_ann
    common_idx = net_ret.index.intersection(bench_ret.index)
    te = (net_ret.loc[common_idx] - bench_ret.loc[common_idx]).std() * np.sqrt(252)
    ir = excess_ann / te if te > 0 else 0
    return dict(window=label, total=total*100, ann=ann*100, vol=vol*100,
                sharpe=sharpe, dd=dd*100, bench_total=bench_total*100,
                bench_ann=bench_ann*100, excess=excess_ann*100, IR=ir,
                n_days=len(net_ret))

results = []
for s, e, label in WINDOWS:
    sub = PERF.loc[s:e, 'net_ret']
    m = metrics(sub, BENCH, label)
    if m:
        results.append(m)
        print(f'\n{label}')
        print(f'  total={m["total"]:.2f}%  ann={m["ann"]:.2f}%  '
              f'sharpe={m["sharpe"]:.2f}  dd={m["dd"]:.2f}%  '
              f'excess={m["excess"]:.2f}%  IR={m["IR"]:.2f}')

print(f'\n{"="*70}\n汇总\n{"="*70}')
df = pd.DataFrame(results)
print(df[['window','total','ann','sharpe','dd','excess','IR']].to_string(index=False))

print(f'\n各指标稳定性 (跨窗口):')
for col, name in [('ann','年化收益'),('sharpe','夏普'),('dd','最大回撤'),
                  ('excess','年化超额'),('IR','信息比率')]:
    m, s = df[col].mean(), df[col].std()
    cv = abs(s/m) if abs(m)>0.01 else float('inf')
    stab = 'STABLE ✓' if cv < 0.5 else ('OK' if cv < 1 else 'UNSTABLE !')
    print(f'  {name:10s}  mean={m:7.2f}  std={s:6.2f}  cv={cv:5.2f}  {stab}')

# 全周期对比
full = metrics(PERF['net_ret'], BENCH, 'FULL: 2021-01~2025-12')
print(f'\n全周期对照:')
print(f'  total={full["total"]:.2f}%  ann={full["ann"]:.2f}%  sharpe={full["sharpe"]:.2f}  '
      f'dd={full["dd"]:.2f}%  excess={full["excess"]:.2f}%  IR={full["IR"]:.2f}')

# 保存
out = Path('reports/walk_forward'); out.mkdir(exist_ok=True, parents=True)
df.to_csv(out/'windows.csv', index=False, encoding='utf-8-sig')

md = ['# V4 Walk-Forward Validation', '',
      f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', '',
      'Source: V4 IS run 2021-2025, sliced into 5 non-overlapping windows.',
      '',
      '## Per-window metrics', '',
      '| Window | Days | Total | Ann | Sharpe | MaxDD | Excess | IR |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in results:
    md.append(f'| {r["window"]} | {r["n_days"]} | {r["total"]:.2f}% | {r["ann"]:.2f}% | '
              f'{r["sharpe"]:.2f} | {r["dd"]:.2f}% | {r["excess"]:.2f}% | {r["IR"]:.2f} |')

md += ['', '## Full-period reference', '',
       f'- Total: {full["total"]:.2f}%  Ann: {full["ann"]:.2f}%  Sharpe: {full["sharpe"]:.2f}',
       f'- MaxDD: {full["dd"]:.2f}%  Excess: {full["excess"]:.2f}%  IR: {full["IR"]:.2f}',
       '', '## Stability (cross-window)', '',
       '| Metric | Mean | Std | CV | Verdict |', '|---|---:|---:|---:|---|']
for col, name in [('ann','Ann Ret'),('sharpe','Sharpe'),('dd','MaxDD'),
                  ('excess','Ann Excess'),('IR','IR')]:
    m, s = df[col].mean(), df[col].std()
    cv = abs(s/m) if abs(m)>0.01 else float('inf')
    stab = 'STABLE' if cv < 0.5 else ('OK' if cv < 1 else 'UNSTABLE')
    md.append(f'| {name} | {m:.2f} | {s:.2f} | {cv:.2f} | {stab} |')

(out/'README.md').write_text('\n'.join(md), encoding='utf-8')
print(f'\n[OK] saved: reports/walk_forward/README.md')
