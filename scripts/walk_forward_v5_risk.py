"""V5 风控版 walk-forward 5 窗口对比 V4."""
import os, sys, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime
for p in ['HTTP_PROXY','HTTPS_PROXY']: os.environ.pop(p, None)
os.environ['NO_PROXY']='*'
sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')

V4_PERF = 'reports/20260511_202758/daily_returns.csv'
V5_PERF = 'reports/v5_risk/daily_returns.csv'

def load(path):
    df = pd.read_csv(path)
    df.columns = [c.replace('\ufeff','') for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')

PV4 = load(V4_PERF); PV5 = load(V5_PERF)
from quant.backtest import get_benchmark_nav
print('[walk-forward] 拉 HS300 ...')
BENCH = get_benchmark_nav('2021-01-01','2025-12-31')

WINDOWS = [
    ('2021-04-01','2022-03-31','W1: 2021-04~2022-03 大盘震荡'),
    ('2022-04-01','2023-03-31','W2: 2022-04~2023-03 深调反弹'),
    ('2023-04-01','2024-03-31','W3: 2023-04~2024-03 慢熊红利'),
    ('2024-04-01','2025-04-30','W4: 2024-04~2025-04 924反转'),
    ('2025-05-01','2025-12-31','W5: 2025-05~2025-12 放量后'),
]

def metrics(net_ret, bench_nav):
    if len(net_ret)<20: return None
    nav=(1+net_ret).cumprod()
    days=(net_ret.index[-1]-net_ret.index[0]).days
    total=nav.iloc[-1]-1
    ann=nav.iloc[-1]**(365/days)-1 if days>0 else 0
    vol=net_ret.std()*np.sqrt(252)
    sharpe=(ann-0.025)/vol if vol>0 else 0
    peak=nav.cummax(); dd=(nav/peak-1).min()
    bs=bench_nav.loc[net_ret.index[0]:net_ret.index[-1]]
    br=bs.pct_change().fillna(0)
    bann=(bs.iloc[-1]/bs.iloc[0])**(365/days)-1 if days>0 else 0
    excess=ann-bann
    ci=net_ret.index.intersection(br.index)
    te=(net_ret.loc[ci]-br.loc[ci]).std()*np.sqrt(252)
    ir=excess/te if te>0 else 0
    return dict(total=total*100,ann=ann*100,vol=vol*100,sharpe=sharpe,
                dd=dd*100,excess=excess*100,IR=ir)

print(f'\n{"="*100}')
print(f'{"窗口":40s} {"V4 IR":>7s} {"V5 IR":>7s} {"ΔIR":>7s} | {"V4 DD":>8s} {"V5 DD":>8s} {"ΔDD":>7s}')
print('='*100)
rows=[]
for s,e,label in WINDOWS:
    m4=metrics(PV4.loc[s:e,'net_ret'],BENCH); m5=metrics(PV5.loc[s:e,'net_ret'],BENCH)
    if not (m4 and m5): continue
    di=m5['IR']-m4['IR']; dd=m5['dd']-m4['dd']
    fl='✓' if di>0 else ('✗' if di<-0.1 else '~')
    fd='✓' if dd>0 else ('✗' if dd<-1 else '~')
    print(f'{label:40s} {m4["IR"]:>7.2f} {m5["IR"]:>7.2f} {di:>+7.2f}{fl} | {m4["dd"]:>7.2f}% {m5["dd"]:>7.2f}% {dd:>+7.2f}{fd}')
    rows.append({'window':label,'v4_ir':m4['IR'],'v5_ir':m5['IR'],'delta_ir':di,
                 'v4_dd':m4['dd'],'v5_dd':m5['dd'],'delta_dd':dd,
                 'v4_ann':m4['ann'],'v5_ann':m5['ann'],
                 'v4_excess':m4['excess'],'v5_excess':m5['excess']})

m4f=metrics(PV4['net_ret'],BENCH); m5f=metrics(PV5['net_ret'],BENCH)
print('-'*100)
print(f'{"全周期 2021-01~2025-12":40s} {m4f["IR"]:>7.2f} {m5f["IR"]:>7.2f} {m5f["IR"]-m4f["IR"]:>+7.2f}  | '
      f'{m4f["dd"]:>7.2f}% {m5f["dd"]:>7.2f}% {m5f["dd"]-m4f["dd"]:>+7.2f}')

df=pd.DataFrame(rows)
print(f'\n稳定性 (跨期 IR):')
print(f'  V4: mean={df["v4_ir"].mean():.2f}  std={df["v4_ir"].std():.2f}  min={df["v4_ir"].min():.2f}')
print(f'  V5: mean={df["v5_ir"].mean():.2f}  std={df["v5_ir"].std():.2f}  min={df["v5_ir"].min():.2f}')

print(f'\n稳定性 (跨期 回撤):')
print(f'  V4: mean={df["v4_dd"].mean():.2f}  std={df["v4_dd"].std():.2f}  worst={df["v4_dd"].min():.2f}')
print(f'  V5: mean={df["v5_dd"].mean():.2f}  std={df["v5_dd"].std():.2f}  worst={df["v5_dd"].min():.2f}')

out=Path('reports/walk_forward'); out.mkdir(exist_ok=True,parents=True)
df.to_csv(out/'v4_vs_v5_risk.csv',index=False,encoding='utf-8-sig')
print(f'\n[OK] saved: reports/walk_forward/v4_vs_v5_risk.csv')
