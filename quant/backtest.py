"""
回测引擎 - 向量化版本
=======================
设计思想:
  - "向量化"指用矩阵运算一次算完所有日期的净值,而不是逐日 for 循环
  - 速度比循环回测快 100~1000 倍
  - 代价: 假设简化(等权、忽略涨跌停、T+1 用次日开盘价)

核心数据流:
  1. 取所有股票每日收盘价 → 价格矩阵 (date × stock)
  2. 算每日收益率 → 收益矩阵
  3. 根据持仓信息构造权重矩阵 (date × stock,行和=1)
  4. 净值 = 权重矩阵 × 收益矩阵 累乘

关键避坑:
  - 调仓日 T 的持仓在 T+1 才生效(避免未来函数)
  - 月内不调权重(等权 → 自然漂移成市值权重)
"""
import numpy as np
import pandas as pd

from . import config
from .data_loader import get_stock_pool, get_stock_kline


# ========== 1. 构建价格矩阵 ==========
def build_price_matrix(codes: list = None,
                       start_date: str = None,
                       end_date: str = None) -> pd.DataFrame:
    """
    把所有股票的收盘价拼成一张大表。
    返回: DataFrame(index=日期, columns=股票代码, value=收盘价)
    """
    if codes is None:
        codes = get_stock_pool()['code'].tolist()
    start_date = start_date or config.BACKTEST_START
    end_date = end_date or config.BACKTEST_END

    print(f"[回测] 构建价格矩阵: {len(codes)} 只股票")
    series_dict = {}
    for code in codes:
        kl = get_stock_kline(code, start_date, end_date)
        if kl.empty:
            continue
        kl = kl.set_index('date')
        series_dict[code] = kl['close']

    price = pd.DataFrame(series_dict).sort_index()
    print(f"[回测] 价格矩阵 shape: {price.shape}")
    return price


# ========== 2. 持仓 → 权重矩阵 ==========
def holdings_to_weights(holdings: dict,
                        price_dates: pd.DatetimeIndex,
                        all_codes: list) -> pd.DataFrame:
    """
    把"调仓日: 持仓清单" 转换为"每日 × 股票"的权重矩阵。

    规则:
      - 调仓日 T 的持仓在 T+1 才开始持有(T+1 收盘价进场)
      - 调仓日之间持仓不变,权重按收益自然漂移
      - 这里简化为: 月内重新计算等权(实务里更常见的做法,避免单股权重过大)

    返回: DataFrame(index=每日, columns=股票, value=权重 0~1)
    """
    rebal_dates = sorted(holdings.keys())
    weights = pd.DataFrame(0.0, index=price_dates, columns=all_codes)

    for i, rebal_date in enumerate(rebal_dates):
        # 这期持仓清单(等权)
        codes_now = list(holdings[rebal_date].index)
        valid_codes = [c for c in codes_now if c in all_codes]
        if not valid_codes:
            continue
        w = 1.0 / len(valid_codes)

        # T+1 生效
        # 找到 rebal_date 在价格矩阵里的位置
        if rebal_date not in price_dates:
            # 找到 ≥ rebal_date 的下一个交易日
            future = price_dates[price_dates > rebal_date]
            if len(future) == 0:
                continue
            effective_start = future[0]
        else:
            idx = price_dates.get_loc(rebal_date)
            if idx + 1 >= len(price_dates):
                continue
            effective_start = price_dates[idx + 1]

        # 持仓终点 = 下一个调仓日(若是最后一期则到末尾)
        if i + 1 < len(rebal_dates):
            next_rebal = rebal_dates[i + 1]
            effective_end = price_dates[price_dates <= next_rebal][-1]
        else:
            effective_end = price_dates[-1]

        # 给这段时间的对应股票赋权重
        mask = (price_dates >= effective_start) & (price_dates <= effective_end)
        for c in valid_codes:
            weights.loc[mask, c] = w

    return weights


# ========== 3. 净值计算 ==========
def compute_portfolio_returns(weights: pd.DataFrame,
                              price: pd.DataFrame,
                              cost_per_side: float = 0.0015) -> pd.DataFrame:
    """
    计算组合每日收益率 + 净值曲线。

    cost_per_side: 单边交易成本(默认 0.15% = 印花税0.05% + 佣金0.025% + 滑点0.075%)
                   买卖各算一次,所以双边成本是 0.3%

    返回:
      DataFrame[ret(日收益), nav(净值), turnover(换手率)]
    """
    # 1. 日收益率矩阵
    daily_ret = price.pct_change()  # 复权后直接 pct_change

    # 2. 用前一日权重 × 当日收益率(避免未来函数)
    weights_lag = weights.shift(1).fillna(0)
    portfolio_ret = (weights_lag * daily_ret).sum(axis=1)

    # 3. 算交易成本: 权重变动绝对值 × cost / 2 (买卖各一次)
    weight_change = (weights - weights.shift(1).fillna(0)).abs().sum(axis=1)
    turnover = weight_change / 2  # 单边换手
    cost = weight_change * cost_per_side  # 总成本

    # 4. 净收益 = 毛收益 - 当日成本
    net_ret = portfolio_ret - cost

    # 5. 累计净值
    nav = (1 + net_ret).cumprod()

    return pd.DataFrame({
        'gross_ret': portfolio_ret,
        'cost': cost,
        'net_ret': net_ret,
        'nav': nav,
        'turnover': turnover,
    })


# ========== 4. 基准 ==========
def get_benchmark_nav(start_date: str = None,
                      end_date: str = None) -> pd.Series:
    """
    取沪深300指数作为基准。
    返回: Series(index=日期, value=归一化净值,起点=1)
    """
    import os
    for k in ['HTTP_PROXY','HTTPS_PROXY']: os.environ.pop(k, None)
    os.environ['NO_PROXY'] = '*'

    import akshare as ak
    sd = (start_date or config.BACKTEST_START).replace('-', '')
    ed = (end_date or config.BACKTEST_END).replace('-', '')

    cache_file = config.CACHE_DIR / f"benchmark_{config.BENCHMARK}_{sd}_{ed}.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
    else:
        print(f"[回测] 拉取基准 {config.BENCHMARK}...")
        df = ak.stock_zh_index_daily(symbol=f"sh{config.BENCHMARK}")
        df['date'] = pd.to_datetime(df['date'])
        df.to_parquet(cache_file, index=False)

    df = df.set_index('date').sort_index()
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    nav = df['close'] / df['close'].iloc[0]
    return nav


# ========== 5. 评估指标 ==========
def performance_metrics(nav: pd.Series,
                        benchmark_nav: pd.Series = None,
                        rf: float = 0.02) -> dict:
    """
    计算回测核心指标。

    nav: 策略净值序列
    benchmark_nav: 基准净值(可选)
    rf: 无风险利率(年化,默认2%)

    返回: dict{指标名: 值}
    """
    daily_ret = nav.pct_change().dropna()
    n_days = len(daily_ret)
    n_years = n_days / 252

    # 累计收益 & 年化收益
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    annual_ret = (nav.iloc[-1] / nav.iloc[0]) ** (1 / n_years) - 1

    # 年化波动
    annual_vol = daily_ret.std() * np.sqrt(252)

    # 夏普比率(年化超额收益 / 年化波动)
    sharpe = (annual_ret - rf) / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # 卡玛比率(年化收益 / 最大回撤)
    calmar = annual_ret / abs(max_dd) if max_dd < 0 else 0

    # 胜率(日)
    win_rate = (daily_ret > 0).mean()

    metrics = {
        '总收益率':   f'{total_ret*100:.2f}%',
        '年化收益率': f'{annual_ret*100:.2f}%',
        '年化波动率': f'{annual_vol*100:.2f}%',
        '夏普比率':   f'{sharpe:.2f}',
        '最大回撤':   f'{max_dd*100:.2f}%',
        '卡玛比率':   f'{calmar:.2f}',
        '日胜率':     f'{win_rate*100:.1f}%',
        '回测天数':   f'{n_days} ({n_years:.1f}年)',
    }

    # 与基准对比
    if benchmark_nav is not None:
        bench_aligned = benchmark_nav.reindex(nav.index, method='ffill')
        bench_total = bench_aligned.iloc[-1] / bench_aligned.iloc[0] - 1
        bench_annual = (bench_aligned.iloc[-1] / bench_aligned.iloc[0]) ** (1/n_years) - 1
        excess = annual_ret - bench_annual

        # 信息比率: 超额收益年化 / 超额收益波动年化
        bench_daily = bench_aligned.pct_change()
        excess_daily = daily_ret - bench_daily
        info_ratio = excess_daily.mean() / excess_daily.std() * np.sqrt(252) if excess_daily.std() > 0 else 0

        metrics['基准总收益']   = f'{bench_total*100:.2f}%'
        metrics['基准年化']     = f'{bench_annual*100:.2f}%'
        metrics['年化超额']     = f'{excess*100:.2f}%'
        metrics['信息比率']     = f'{info_ratio:.2f}'

    return metrics


# ========== 6. 主回测函数 ==========
def run_backtest(holdings: dict = None,
                 cost_per_side: float = 0.0015) -> dict:
    """
    运行完整回测。

    返回: {
        'nav':       策略净值序列,
        'benchmark': 基准净值序列,
        'returns':   每日收益明细 DataFrame,
        'metrics':   性能指标 dict,
        'holdings':  持仓字典(直接传入或重新生成),
    }
    """
    # 1. 取持仓
    if holdings is None:
        from .factor_panel import build_factor_panel
        from .strategy import generate_holdings
        panel = build_factor_panel()
        holdings = generate_holdings(panel)

    # 2. 取所有出现过的股票代码
    all_held_codes = set()
    for h in holdings.values():
        all_held_codes.update(h.index.tolist())
    print(f"[回测] 全期出现过的股票: {len(all_held_codes)} 只")

    # 3. 构建价格矩阵
    price = build_price_matrix(codes=list(all_held_codes))
    price = price.dropna(how='all')

    # 4. 构建权重矩阵
    weights = holdings_to_weights(holdings, price.index, list(all_held_codes))

    # 5. 计算组合表现
    perf = compute_portfolio_returns(weights, price, cost_per_side=cost_per_side)
    perf = perf.dropna()

    # 6. 取基准
    bench = get_benchmark_nav(perf.index.min().strftime('%Y-%m-%d'),
                               perf.index.max().strftime('%Y-%m-%d'))

    # 7. 算指标
    metrics = performance_metrics(perf['nav'], bench)

    return {
        'nav': perf['nav'],
        'benchmark': bench,
        'returns': perf,
        'metrics': metrics,
        'holdings': holdings,
        'weights': weights,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    print("=" * 60)
    print("回测引擎自检")
    print("=" * 60)

    result = run_backtest()

    print("\n" + "=" * 60)
    print("[结果] 回测核心指标")
    print("=" * 60)
    for k, v in result['metrics'].items():
        print(f"  {k:12s} : {v}")

    # 看看年度收益
    nav = result['nav']
    yearly = nav.resample('YE').last()
    yearly_ret = yearly.pct_change()
    yearly_ret.iloc[0] = yearly.iloc[0] / 1.0 - 1  # 第一年从净值1开始
    print(f"\n[年度] 策略年度收益:")
    for d, r in yearly_ret.items():
        print(f"  {d.year} : {r*100:+.2f}%")

    # 基准年度
    bench = result['benchmark']
    bench_yearly = bench.resample('YE').last()
    bench_yearly_ret = bench_yearly.pct_change()
    bench_yearly_ret.iloc[0] = bench_yearly.iloc[0] / 1.0 - 1
    print(f"\n[年度] 基准(沪深300)收益:")
    for d, r in bench_yearly_ret.items():
        print(f"  {d.year} : {r*100:+.2f}%")
