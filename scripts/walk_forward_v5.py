"""V5 walk-forward 验证 + 与 V4 对比."""
import pandas as pd, numpy as np, os, sys
from pathlib import Path
from datetime import datetime
for p in ['HTTP_PROXY','HTTPS_PROXY']: os.environ.pop(p, None)
os.environ['NO_PROXY']='*'
sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')

V4_PERF = 'reports/20260511_202758/daily_returns.csv'
V5_PERF = 'reports/20260511_215535/daily_returns.csv'

def load_perf(path):
    df = pd.read_csv(path)
    df.columns = [c.replace('\ufeff','') for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')

PERF_V4 = load_perf(V4_PERF)
PERF_V5 = load_perf(V5_PERF)

from quant.backtest import get_benchmark_nav
print('[walk-forward] 拉 HS300 基准...')
BENCH = get_benchmark_nav('2021-01-01','2025-12-31')

WINDOWS = [
    ('2021-04-01','2022-03-31','W1: 2021-04~2022-03 大盘震荡'),
    ('2022-04-01','2023-03-31','W2: 2022-04~2023-03 深调反弹'),
    ('2023-04-01','2024-03-31','W3: 2023-04~2024-03 慢熊红利'),
    ('2024-04-01','2025-04-30','W4: 2024-04~2025-04 924反转'),
    ('2025-05-01','2025-12-31','W5: 2025-05~2025-12 放量后'),
]

def metrics(net_ret, bench_nav):
    if len(net_ret) < 20: return None
    nav = (1+net_ret).cumprod()
    days = (net_ret.index[-1]-net_ret.index[0]).days
    total = nav.iloc[-1]-1
    ann = nav.iloc[-1]**(365/days)-1 if days>0 else 0
    vol = net_ret.std()*np.sqrt(252)
    sharpe = (ann-0.025)/vol if vol>0 else 0
    peak = nav.cummax()
    dd = (nav/peak-1).min()
    bench_slice = bench_nav.loc[net_ret.index[0]:net_ret.index[-1]]
    bench_ret = bench_slice.pct_change().fillna(0)
    bench_ann = (bench_slice.iloc[-1]/bench_slice.iloc[0])**(365/days)-1 if days>0 else 0
    excess_ann = ann - bench_ann
    common_idx = net_ret.index.intersection(bench_ret.index)
    te = (net_ret.loc[common_idx]-bench_ret.loc[common_idx]).std()*np.sqrt(252)
    ir = excess_ann/te if te>0 else 0
    return dict(total=total*100, ann=ann*100, sharpe=sharpe, dd=dd*100,
                excess=excess_ann*100, IR=ir)

print(f'\n{"="*100}')
print(f'{"窗口":40s} {"V4 IR":>8s} {"V5 IR":>8s} {"ΔIR":>7s} | {"V4 超额":>8s} {"V5 超额":>8s} {"V4 回撤":>8s} {"V5 回撤":>8s}')
print('='*100)
rows = []
for s,e,label in WINDOWS:
    m4 = metrics(PERF_V4.loc[s:e,'net_ret'], BENCH)
    m5 = metrics(PERF_V5.loc[s:e,'net_ret'], BENCH)
    if not (m4 and m5): continue
    delta_ir = m5['IR'] - m4['IR']
    flag = '✓' if delta_ir > 0 else ('✗' if delta_ir < -0.1 else '~')
    print(f'{label:40s} {m4["IR"]:>8.2f} {m5["IR"]:>8.2f} {delta_ir:>+7.2f}{flag} | '
          f'{m4["excess"]:>7.2f}% {m5["excess"]:>7.2f}% '
          f'{m4["dd"]:>7.2f}% {m5["dd"]:>7.2f}%')
    rows.append({'window':label,'v4_ir':m4['IR'],'v5_ir':m5['IR'],'delta_ir':delta_ir,
                 'v4_excess':m4['excess'],'v5_excess':m5['excess'],
                 'v4_dd':m4['dd'],'v5_dd':m5['dd'],
                 'v4_ann':m4['ann'],'v5_ann':m5['ann']})

# 全周期
m4f = metrics(PERF_V4['net_ret'], BENCH)
m5f = metrics(PERF_V5['net_ret'], BENCH)
print('-'*100)
print(f'{"全周期 2021-01~2025-12":40s} {m4f["IR"]:>8.2f} {m5f["IR"]:>8.2f} {m5f["IR"]-m4f["IR"]:>+7.2f}  | '
      f'{m4f["excess"]:>7.2f}% {m5f["excess"]:>7.2f}% {m4f["dd"]:>7.2f}% {m5f["dd"]:>7.2f}%')

# 稳定性对比
df = pd.DataFrame(rows)
print(f'\n稳定性 (5 窗口跨期 IR):')
print(f'  V4: mean={df["v4_ir"].mean():.2f}  std={df["v4_ir"].std():.2f}  min={df["v4_ir"].min():.2f}')
print(f'  V5: mean={df["v5_ir"].mean():.2f}  std={df["v5_ir"].std():.2f}  min={df["v5_ir"].min():.2f}')

# 保存
out = Path('reports/walk_forward'); out.mkdir(exist_ok=True, parents=True)
df.to_csv(out/'v4_vs_v5.csv', index=False, encoding='utf-8-sig')
print(f'\n[OK] saved: reports/walk_forward/v4_vs_v5.csv')
