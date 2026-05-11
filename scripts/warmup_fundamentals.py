"""
预热基本面数据 (PE/PB/ROE)
Usage:
  python scripts/warmup_fundamentals.py
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

# 必须先清掉系统代理
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

import warnings
warnings.filterwarnings('ignore')

from quant.fundamentals import build_fundamentals_panel

if __name__ == "__main__":
    panel = build_fundamentals_panel(refresh=False)
    print("\n[完成] 基本面面板覆盖率:")
    for name, df in panel.items():
        if df.empty:
            print(f"  {name}: 空")
        else:
            print(f"  {name}: {df.shape[0]} 期 x {df.shape[1]} 股")
