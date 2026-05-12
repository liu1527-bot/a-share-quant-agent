"""Sanity check: HS300 + 5 only persisted holdings 12-31 -> 1-30 returns."""
import os, sys, time
for p in ['HTTP_PROXY','HTTPS_PROXY']: os.environ.pop(p, None)
os.environ['NO_PROXY']='*'
sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, akshare as ak

def fetch(code):
    for i in range(3):
        try:
            df = ak.stock_zh_a_hist_tx(symbol=code, start_date='20251220',
                                       end_date='20260205', adjust='qfq')
            if len(df) > 0: return df
        except Exception: time.sleep(1)
    return None

idx = fetch('sh000300')
if idx is not None:
    idx['date']=pd.to_datetime(idx['date'])
    p0 = float(idx[idx['date']<=pd.Timestamp('2025-12-31')].iloc[-1]['close'])
    p1 = float(idx[idx['date']<=pd.Timestamp('2026-01-30')].iloc[-1]['close'])
    print('HS300 12-31->1-30: %+.2f%%' % ((p1/p0-1)*100))

for code in ['sh601186','sh601800','sh601668','sh600018','sh601169','sh600585','sh601919','sh600028','sh601088','sh601398']:
    df = fetch(code)
    if df is None: continue
    df['date']=pd.to_datetime(df['date'])
    p0 = float(df[df['date']<=pd.Timestamp('2025-12-31')].iloc[-1]['close'])
    p1 = float(df[df['date']<=pd.Timestamp('2026-01-30')].iloc[-1]['close'])
    print('%s 12-31->1-30: %+.2f%%' % (code, (p1/p0-1)*100))
