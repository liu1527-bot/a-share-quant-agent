"""
每月调仓主入口 (供 GitHub Actions 调用)
=========================================
完整流程:
  1. 预热数据(增量更新K线 + 必要时刷新财务)
  2. 重建因子面板
  3. 跑回测
  4. 生成报告
  5. 输出 latest_holding.md (本月推荐持仓)
"""
import os
import sys
import io
import time
import shutil
from pathlib import Path
from datetime import datetime

# 强制 UTF-8 输出 (幂等,避免被多次调用导致 stream 关闭)
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# 关掉系统代理
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

# 标记开始时间
START = time.time()
TODAY = datetime.now().strftime('%Y%m%d')


def step(name):
    elapsed = time.time() - START
    print(f"\n{'=' * 70}")
    print(f"[{elapsed:6.1f}s] {name}")
    print('=' * 70)


def main():
    print(f"==================================================================")
    print(f"  A股多因子选股 Agent - 月度自动调仓")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"==================================================================")

    # ===== 1. 预热K线数据 =====
    step("Step 1/5: 预热K线数据")
    from scripts.warmup_data import main as warmup_kline
    try:
        warmup_kline()
    except SystemExit:
        pass  # warmup 脚本可能 sys.exit
    except Exception as e:
        print(f"[警告] K线预热部分失败: {e}, 继续...")

    # ===== 2. 检查/拉取基本面数据 =====
    step("Step 2/5: 预热基本面数据")
    from quant import config
    fund_cache = config.CACHE_DIR / f"fundamentals_panel_{config.STOCK_POOL}.pkl"
    if not fund_cache.exists():
        print("[基本面] 缓存不存在,开始拉取(约 15-20 分钟)")
        from scripts.warmup_fundamentals import main as warmup_fund
        try:
            warmup_fund()
        except (SystemExit, AttributeError):
            from quant.fundamentals import build_fundamentals_panel
            build_fundamentals_panel()
    else:
        # 检查时效: 7天以内不刷新
        age_days = (time.time() - fund_cache.stat().st_mtime) / 86400
        if age_days > 30:
            print(f"[基本面] 缓存已 {age_days:.1f} 天,刷新")
            from quant.fundamentals import build_fundamentals_panel
            build_fundamentals_panel(refresh=True)
        else:
            print(f"[基本面] 缓存还新鲜 ({age_days:.1f} 天),复用")

    # ===== 3. 重建因子面板 =====
    step("Step 3/5: 重建因子面板")
    from quant.factor_panel import build_factor_panel
    panel = build_factor_panel(refresh=True)

    # ===== 4. 跑回测 + 报告 =====
    step("Step 4/5: 跑回测")
    from quant.backtest import run_backtest
    bt_result = run_backtest()

    # 打印核心指标
    print("\n[回测结果]")
    for k, v in bt_result['metrics'].items():
        print(f"  {k:12s} : {v}")

    # ===== 5. 生成报告 + 当月持仓 =====
    step("Step 5/5: 生成报告")
    from quant.report import generate_full_report
    report_dir = generate_full_report(bt_result)
    print(f"[报告] {report_dir}")

    # 复制当期 latest_holding 到根目录,方便查看
    src = Path(report_dir) / 'latest_holding.csv'
    if src.exists():
        target_dir = Path('reports') / 'latest'
        target_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy(src, target_dir / 'holding.csv')

        # 生成 Markdown 版本
        import pandas as pd
        df = pd.read_csv(src, index_col=0)
        df = df.drop_duplicates()
        md_lines = [
            f"# {datetime.now().strftime('%Y-%m')} 月度调仓持仓",
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"持仓数: {len(df)}",
            "",
            "## Top 30 持仓",
            "",
            "| 排名 | 股票 | 名称 | 综合得分 | PE倒数 | PB倒数 | ROE | 5日反转 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for code, row in df.head(30).iterrows():
            md_lines.append(
                f"| {int(row['rank'])} | {code} | {row.get('name', '-')} | "
                f"{row['score']:.4f} | {row.get('value_pe', 0):.4f} | "
                f"{row.get('value_pb', 0):.4f} | {row.get('quality_roe', 0):.2f} | "
                f"{row.get('reversal_5', 0):.4f} |"
            )
        md_lines += [
            "",
            "## 回测核心指标",
            "",
        ]
        for k, v in bt_result['metrics'].items():
            md_lines.append(f"- **{k}**: {v}")

        md_content = '\n'.join(md_lines)
        (target_dir / 'README.md').write_text(md_content, encoding='utf-8')
        print(f"[+] 当期持仓: {target_dir / 'README.md'}")

    elapsed = time.time() - START
    print(f"\n{'=' * 70}")
    print(f"  完成! 总耗时 {elapsed:.1f}s")
    print('=' * 70)


if __name__ == "__main__":
    main()
