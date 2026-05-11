"""
预热基本面数据 (PE/PB/ROE)
Usage:
  python scripts/warmup_fundamentals.py
"""
import sys, os
sys.path.insert(0, '.')

# 仅当 stdout 不是 utf-8 时才重新包装(避免被 monthly_run 调用时关闭已 wrap 的流)
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# 必须先清掉系统代理
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

import warnings
warnings.filterwarnings('ignore')

from quant.fundamentals import build_fundamentals_panel


def main():
    panel = build_fundamentals_panel(refresh=False)
    print("\n[完成] 基本面面板覆盖率:")
    for name, df in panel.items():
        if df.empty:
            print(f"  {name}: 空")
        else:
            print(f"  {name}: {df.shape[0]} 期 x {df.shape[1]} 股")
    return panel


if __name__ == "__main__":
    main()
