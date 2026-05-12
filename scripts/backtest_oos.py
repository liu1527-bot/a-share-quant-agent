"""
V4 跨环境回测 (Out-Of-Sample) - 路 A 脉冲版
=============================================
用 2025 年的 hs300 成分股 (含 industry_map) + 2018-2020 三年价格/财务,
跑 V4 行业中性策略,看是否在另一个市场环境也有 alpha。

警告: 有幸存者偏差 (2018 当时存在但 2025 已剔除的股票不会入选,会高估收益)。

输出:
  - reports/oos_2018_2020/README.md
  - reports/oos_2018_2020/holding.csv (最后一期持仓)

关键差异 vs 主回测:
  - BACKTEST_START='2018-01-01', BACKTEST_END='2020-12-31'
  - baidu_period='全部'  (因为 '近五年' 拿不到 2018)
  - sina_start_year='2017'
  - cache_suffix='_oos2018' (隔离缓存)
"""
import os, sys, io, time, shutil
from pathlib import Path
from datetime import datetime

# UTF-8 输出
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# 关代理
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

START = time.time()
SUFFIX = '_oos2018'

def step(name):
    elapsed = time.time() - START
    print(f"\n{'=' * 70}")
    print(f"[{elapsed:6.1f}s] {name}")
    print('=' * 70)


def main():
    print(f"==================================================================")
    print(f"  V4 跨环境回测 (OOS 2018-2020)")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"==================================================================")

    # ===== 0. monkey-patch config 日期 =====
    from quant import config
    OLD_START, OLD_END = config.BACKTEST_START, config.BACKTEST_END
    config.BACKTEST_START = '2018-01-01'
    config.BACKTEST_END = '2020-12-31'
    print(f"[OOS] BACKTEST_START={config.BACKTEST_START}  END={config.BACKTEST_END}")
    print(f"[OOS] cache_suffix={SUFFIX}")

    # ===== 1. 拉基本面 (用 '全部' period + 2017 起的财务) =====
    step("Step 1/4: 拉取/缓存 2018-2020 基本面")
    fund_cache = config.CACHE_DIR / f"fundamentals_panel_{config.STOCK_POOL}{SUFFIX}.pkl"
    from quant.fundamentals import build_fundamentals_panel
    if fund_cache.exists():
        print(f"[OOS] 复用 OOS 基本面缓存: {fund_cache.name}")
    else:
        build_fundamentals_panel(refresh=True,
                                  baidu_period='全部',
                                  sina_start_year='2017',
                                  cache_suffix=SUFFIX)

    # ===== 2. 拉 K 线 (BACKTEST_START/END 已改, 自动按新日期缓存) =====
    step("Step 2/4: 预热 K 线 (2018-01-01 ~ 2020-12-31)")
    from quant.data_loader import get_stock_pool, get_stock_kline
    pool = get_stock_pool()
    codes = pool['code'].tolist()
    print(f"[OOS] {len(codes)} 只股票, 拉 K 线...")
    t0 = time.time()
    ok = 0
    for i, c in enumerate(codes, 1):
        try:
            kl = get_stock_kline(c)
            if not kl.empty:
                ok += 1
        except Exception as e:
            pass
        if i % 50 == 0 or i == len(codes):
            print(f"  进度 {i}/{len(codes)} 成功 {ok}, 用时 {time.time()-t0:.0f}s")

    # ===== 3. 构建因子面板 =====
    step("Step 3/4: 构建因子面板 (OOS)")
    from quant.factor_panel import build_factor_panel
    panel = build_factor_panel(refresh=True, cache_suffix=SUFFIX)

    # ===== 4. 跑回测 =====
    step("Step 4/4: 跑 V4 行业中性回测")
    from quant.strategy import generate_holdings
    from quant.backtest import run_backtest
    holdings = generate_holdings(panel)
    bt = run_backtest(holdings=holdings)

    print("\n[OOS 回测结果]")
    for k, v in bt['metrics'].items():
        print(f"  {k:15s} : {v}")

    # ===== 5. 输出报告 =====
    out_dir = Path('reports') / 'oos_2018_2020'
    out_dir.mkdir(exist_ok=True, parents=True)

    # 最后一期持仓
    last_date = max(holdings.keys())
    last_h = holdings[last_date]
    last_h.to_csv(out_dir / 'holding.csv', encoding='utf-8-sig')

    # NAV 曲线
    nav = bt['nav']
    nav.to_csv(out_dir / 'nav.csv', encoding='utf-8-sig')
    bt['benchmark'].to_csv(out_dir / 'benchmark.csv', encoding='utf-8-sig')

    # README
    md = [
        f"# V4 跨环境回测 (OOS 2018-2020)",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"## 配置",
        f"- 股票池: 2025 年沪深 300 成分股 (280 只, 有幸存者偏差)",
        f"- 回测期: {config.BACKTEST_START} ~ {config.BACKTEST_END}",
        f"- 调仓: 月末, TOP {config.TOP_N}, MAX_PER_INDUSTRY={config.MAX_PER_INDUSTRY}",
        f"- 数据源: 价格 K 线 + 百度估值 '全部' + 新浪财务 '2017'",
        "",
        "## 关键指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
    ]
    for k, v in bt['metrics'].items():
        md.append(f"| {k} | {v} |")
    md.append("")
    md.append("## 持仓最末日 (前 30)")
    md.append("")
    md.append("| 排名 | 股票 | 综合得分 | 行业 |")
    md.append("|---|---|---|---|")
    for i, (code, row) in enumerate(last_h.head(30).iterrows(), 1):
        md.append(f"| {i} | {code} | {row.get('score', 0):.4f} | {row.get('industry', '-')} |")

    (out_dir / 'README.md').write_text('\n'.join(md), encoding='utf-8')

    # 还原 config (虽然进程要退出了)
    config.BACKTEST_START, config.BACKTEST_END = OLD_START, OLD_END

    elapsed = time.time() - START
    print(f"\n{'=' * 70}")
    print(f"  完成! 总耗时 {elapsed:.1f}s")
    print(f"  报告: {out_dir / 'README.md'}")
    print('=' * 70)


if __name__ == "__main__":
    main()
