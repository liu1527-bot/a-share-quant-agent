# -*- coding: utf-8 -*-
"""
CSI500 (中证500) MVP data fetcher.
- Constituents (current snapshot)
- Index benchmark sh000905
- Per-stock kline 2021-01-01 to 2026-04-30
- Resumable: skip if cache exists
"""
import os, sys, time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

for p in ['HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(p, None)
os.environ['NO_PROXY'] = '*'

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from quant.data_loader import get_index_constituents, get_stock_kline, _ak
from quant import config

START = '2021-01-01'
END = '2026-04-30'
SD = START.replace('-', '')
ED = END.replace('-', '')

OUT = Path('data/cache')
KLINE_DIR = OUT / 'kline'
KLINE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_constituents():
    cache = OUT / 'zz500_constituents.parquet'
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"[constituents] cached: {len(df)} stocks")
        return df
    print("[constituents] fetching CSI500 (000905)...")
    df = get_index_constituents('000905')
    if df['code'].duplicated().any():
        df = df.drop_duplicates(subset=['code'], keep='first').reset_index(drop=True)
    df.to_parquet(cache, index=False)
    print(f"[constituents] saved: {len(df)} stocks")
    return df


def fetch_benchmark():
    cache = OUT / f'benchmark_000905_{SD}_{ED}.parquet'
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"[benchmark] cached: {len(df)} rows")
        return df
    print("[benchmark] fetching sh000905...")
    ak = _ak()
    df = ak.stock_zh_a_hist_tx(symbol='sh000905', start_date=SD, end_date=ED, adjust='')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df.to_parquet(cache, index=False)
    print(f"[benchmark] saved: {len(df)} rows  range {df['date'].min()} -> {df['date'].max()}")
    return df


def fetch_klines(codes):
    total = len(codes)
    success = 0
    skipped = 0
    failed = []
    for i, code in enumerate(codes, 1):
        cache = KLINE_DIR / f"{code}_{SD}_{ED}_qfq.parquet"
        if cache.exists():
            skipped += 1
        else:
            try:
                df = get_stock_kline(code, START, END, adjust='qfq')
                if df.empty:
                    failed.append(code)
                else:
                    success += 1
                time.sleep(0.05)
            except Exception as e:
                failed.append(code)
                print(f"  [fail] {code}: {str(e)[:80]}")
        if i % 50 == 0 or i == total:
            print(f"[kline] {i}/{total}  new={success}  cached={skipped}  fail={len(failed)}")
    return success, skipped, failed


def main():
    print("="*60)
    print(f"CSI500 MVP data fetcher  {START} -> {END}")
    print("="*60)
    cons = fetch_constituents()
    fetch_benchmark()
    codes = sorted(cons['code'].tolist())
    print(f"\n[kline] {len(codes)} stocks to process...")
    s, sk, f = fetch_klines(codes)
    print(f"\n=== DONE === new={s}, cached={sk}, fail={len(f)}")
    if f:
        pd.DataFrame({'code': f}).to_csv(OUT / 'zz500_kline_failed.csv', index=False)
        print(f"failures saved -> data/cache/zz500_kline_failed.csv")


if __name__ == '__main__':
    main()
