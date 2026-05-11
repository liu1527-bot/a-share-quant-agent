"""
回测报告生成
==============
产出 4 张图 + 1 份 Markdown 报告:
  1. 净值曲线图 (策略 vs 基准)
  2. 回撤曲线图
  3. 月度收益热力图
  4. 因子权重饼图
  + 报告 Markdown
  + 最新持仓 CSV

输出目录: reports/{YYYYMMDD_HHMMSS}/
"""
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互后端,避免GUI报错
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from . import config


# ========== Matplotlib 中文字体设置 ==========
def setup_chinese_font():
    """Windows 自带中文字体兜底列表。"""
    import matplotlib.font_manager as fm
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name]
            plt.rcParams['axes.unicode_minus'] = False
            return name
    print("[警告] 未找到中文字体,图片中文可能乱码")
    return None

setup_chinese_font()


# ========== 1. 净值曲线 ==========
def plot_nav_curve(nav: pd.Series, benchmark: pd.Series, save_path: Path):
    """策略 vs 基准的累计净值对比图。"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 对齐到同一起点
    bench_aligned = benchmark.reindex(nav.index, method='ffill')
    bench_aligned = bench_aligned / bench_aligned.iloc[0]

    ax.plot(nav.index, nav.values, label='多因子策略', color='#d62728', linewidth=2)
    ax.plot(bench_aligned.index, bench_aligned.values,
            label=f'基准(沪深{config.BENCHMARK})',
            color='#1f77b4', linewidth=1.5, alpha=0.8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    ax.set_title('策略净值曲线对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('净值(初始=1)')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ========== 2. 回撤曲线 ==========
def plot_drawdown(nav: pd.Series, save_path: Path):
    """回撤曲线 - 直观看到最坏时段。"""
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max * 100  # 百分比

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(drawdown.index, drawdown.values, 0,
                     color='red', alpha=0.3, label='回撤')
    ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
    ax.axhline(y=0, color='black', linewidth=0.5)

    # 标注最大回撤位置
    min_idx = drawdown.idxmin()
    min_val = drawdown.min()
    ax.scatter([min_idx], [min_val], color='red', s=100, zorder=5)
    ax.annotate(f'最大回撤: {min_val:.2f}%\n@ {min_idx.date()}',
                xy=(min_idx, min_val), xytext=(20, -30),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->'))

    ax.set_title('策略回撤曲线', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('回撤 (%)')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ========== 3. 月度收益热力图 ==========
def plot_monthly_heatmap(nav: pd.Series, save_path: Path):
    """月度收益热力图 - 看哪些月好、哪些月差。"""
    monthly_ret = nav.resample('ME').last().pct_change().dropna() * 100

    # 整理成 年×月 矩阵
    df = pd.DataFrame({
        'year': monthly_ret.index.year,
        'month': monthly_ret.index.month,
        'ret': monthly_ret.values,
    })
    pivot = df.pivot(index='year', columns='month', values='ret')

    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.6)))

    # 红绿配色: 负=红, 正=绿
    vmax = max(abs(pivot.min().min()), pivot.max().max())
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                    vmin=-vmax, vmax=vmax)

    # 单元格内显示数值
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not pd.isna(val):
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                        color=color, fontsize=10)

    # 加上每年汇总
    yearly_ret = (monthly_ret / 100 + 1).groupby(monthly_ret.index.year).prod() - 1
    yearly_ret_pct = yearly_ret * 100

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{m}月' for m in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{y}  ({yearly_ret_pct.get(y, 0):+.1f}%)'
                         for y in pivot.index])

    ax.set_title('月度收益热力图 (%)  — 行末括号为年度收益',
                  fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label='月收益 (%)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ========== 4. 因子权重饼图 ==========
def plot_factor_weights(save_path: Path):
    """显示当前的因子权重配置。"""
    weights = config.FACTOR_WEIGHTS

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))

    wedges, texts, autotexts = ax.pie(
        weights.values(),
        labels=weights.keys(),
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11},
    )
    for at in autotexts:
        at.set_fontweight('bold')

    ax.set_title('因子权重配置', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ========== 5. Markdown 报告 ==========
def generate_markdown_report(result: dict, output_dir: Path) -> Path:
    """组装最终的 Markdown 报告。"""
    metrics = result['metrics']
    nav = result['nav']
    holdings = result['holdings']

    # 取最新一期持仓
    last_date = max(holdings.keys())
    latest_holding = holdings[last_date]

    # 加上股票名称
    from .data_loader import get_stock_pool
    pool = get_stock_pool().set_index('code')
    latest_holding = latest_holding.join(pool[['name']], how='left')

    # 年度收益
    yearly = nav.resample('YE').last()
    yearly_ret = yearly.pct_change()
    yearly_ret.iloc[0] = yearly.iloc[0] / 1.0 - 1
    bench = result['benchmark']
    bench_yearly = bench.resample('YE').last()
    bench_yearly_ret = bench_yearly.pct_change()
    bench_yearly_ret.iloc[0] = bench_yearly.iloc[0] / 1.0 - 1

    md_lines = [
        f"# 多因子选股策略回测报告",
        f"",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**回测区间**: {nav.index.min().date()} ~ {nav.index.max().date()}  ",
        f"**股票池**: {config.STOCK_POOL.upper()}  ",
        f"**调仓频率**: {config.REBALANCE_FREQ} (月调)  ",
        f"**持仓数量**: Top {config.TOP_N}  ",
        f"**标准化方式**: {config.NORMALIZE_METHOD}  ",
        f"",
        f"---",
        f"",
        f"## 一、核心指标",
        f"",
        f"| 指标 | 数值 |",
        f"|------|------|",
    ]
    for k, v in metrics.items():
        md_lines.append(f"| {k} | **{v}** |")

    md_lines += [
        f"",
        f"---",
        f"",
        f"## 二、净值曲线",
        f"",
        f"![净值曲线](nav_curve.png)",
        f"",
        f"## 三、回撤曲线",
        f"",
        f"![回撤曲线](drawdown.png)",
        f"",
        f"## 四、月度收益热力图",
        f"",
        f"![月度热力图](monthly_heatmap.png)",
        f"",
        f"## 五、因子权重",
        f"",
        f"![因子权重](factor_weights.png)",
        f"",
        f"---",
        f"",
        f"## 六、年度收益对比",
        f"",
        f"| 年份 | 策略 | 基准(沪深300) | 超额 |",
        f"|------|------|------|------|",
    ]
    for year in sorted(set(yearly_ret.index.year) | set(bench_yearly_ret.index.year)):
        s = next((v for d, v in yearly_ret.items() if d.year == year), None)
        b = next((v for d, v in bench_yearly_ret.items() if d.year == year), None)
        if s is None or b is None:
            continue
        excess = s - b
        emoji = "🟢" if excess > 0 else "🔴"
        md_lines.append(
            f"| {year} | {s*100:+.2f}% | {b*100:+.2f}% | "
            f"{emoji} {excess*100:+.2f}% |"
        )

    md_lines += [
        f"",
        f"---",
        f"",
        f"## 七、最新一期持仓 ({last_date.date()})",
        f"",
        f"等权配置, 共 **{len(latest_holding)}** 只, 单只权重 "
        f"**{1.0/len(latest_holding)*100:.2f}%**",
        f"",
        f"| 排名 | 代码 | 名称 | 综合得分 |",
        f"|------|------|------|---------|",
    ]
    for code, row in latest_holding.iterrows():
        md_lines.append(
            f"| {int(row['rank'])} | {code} | {row.get('name','-')} | "
            f"{row['score']:.4f} |"
        )

    md_lines += [
        f"",
        f"---",
        f"",
        f"## 八、报告说明",
        f"",
        f"### 关键指标含义",
        f"- **年化收益率**: 假设按当前复利持续 1 年的收益",
        f"- **夏普比率**: 单位风险换取的超额收益, > 1 优秀, > 0.5 合格",
        f"- **最大回撤**: 净值从峰值到谷底的最大跌幅, 衡量极端风险",
        f"- **信息比率**: 跑赢基准的稳定性, > 0.5 不错, > 1 优秀",
        f"- **卡玛比率**: 年化收益 / 最大回撤, 衡量收益质量",
        f"",
        f"### 回测假设",
        f"- 单边交易成本: 0.15% (含印花税/佣金/滑点)",
        f"- 调仓日选股, **次日**收盘价进场 (避免未来函数)",
        f"- 等权配置, 月内不再平衡",
        f"- 不考虑停牌、涨跌停限制",
        f"",
        f"### 局限性",
        f"- 仅使用量价因子, 无基本面信息",
        f"- 沪深300成分股使用当前名单, 存在轻微幸存者偏差",
        f"- 历史业绩不代表未来表现",
    ]

    md_path = output_dir / "report.md"
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')
    return md_path


# ========== 6. 主入口 ==========
def generate_full_report(result: dict = None,
                         output_dir: Path = None) -> Path:
    """
    生成完整报告(图+md+csv)。

    返回报告文件夹路径。
    """
    if result is None:
        from .backtest import run_backtest
        result = run_backtest()

    # 创建输出目录
    if output_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = config.REPORT_DIR / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[报告] 输出目录: {output_dir}")

    # 生成图表
    print("[报告] 生成净值曲线...")
    plot_nav_curve(result['nav'], result['benchmark'],
                    output_dir / 'nav_curve.png')

    print("[报告] 生成回撤曲线...")
    plot_drawdown(result['nav'], output_dir / 'drawdown.png')

    print("[报告] 生成月度热力图...")
    plot_monthly_heatmap(result['nav'], output_dir / 'monthly_heatmap.png')

    print("[报告] 生成因子权重图...")
    plot_factor_weights(output_dir / 'factor_weights.png')

    # 输出最新持仓 CSV
    last_date = max(result['holdings'].keys())
    latest = result['holdings'][last_date].copy()
    from .data_loader import get_stock_pool
    pool = get_stock_pool().set_index('code')
    latest = latest.join(pool[['name']], how='left')
    latest.to_csv(output_dir / 'latest_holding.csv', encoding='utf-8-sig')
    print(f"[报告] 最新持仓 CSV: latest_holding.csv ({len(latest)} 只)")

    # 输出 Markdown 报告
    md_path = generate_markdown_report(result, output_dir)
    print(f"[报告] Markdown: {md_path.name}")

    # 顺手存一份净值序列
    result['returns'].to_csv(output_dir / 'daily_returns.csv',
                              encoding='utf-8-sig')

    print(f"\n[完成] 报告已生成于: {output_dir}")
    return output_dir


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    output = generate_full_report()
    print(f"\n打开报告: {output / 'report.md'}")
