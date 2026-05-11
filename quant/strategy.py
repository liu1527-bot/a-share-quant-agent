"""
多因子选股策略
================
核心流程:
  1. 取调仓日的因子横截面快照
  2. 过滤掉无效股票(因子缺失太多/停牌)
  3. 每个因子做横截面标准化(rank 或 zscore)
  4. 按权重加权合成综合得分
  5. 选 Top N 作为下期持仓

学习重点:
  - 调仓日的生成: 月末/周末交易日
  - 横截面操作 vs 时序操作
  - 因子合成的稳健性处理
"""
import pandas as pd
import numpy as np

from . import config
from .factors import cross_section_zscore, cross_section_rank


# ========== 1. 横截面标准化 ==========
def normalize_cross_section(snapshot: pd.DataFrame,
                            method: str = None) -> pd.DataFrame:
    """
    对因子快照做横截面标准化。

    输入: DataFrame(index=股票, columns=因子)
    输出: 同 shape, 但每列被标准化到统一尺度
    """
    method = method or config.NORMALIZE_METHOD
    fn = cross_section_rank if method == 'rank' else cross_section_zscore
    return snapshot.apply(fn, axis=0)


# ========== 2. 综合打分 ==========
def composite_score(snapshot: pd.DataFrame,
                    weights: dict = None) -> pd.Series:
    """
    把多个标准化后的因子按权重加权,得到综合得分。

    输入:
      snapshot: 已标准化的因子快照 DataFrame(index=股票, columns=因子)
      weights:  {因子名: 权重}, 默认用 config.FACTOR_WEIGHTS

    输出:
      Series(index=股票, value=得分), 已按降序排列
    """
    weights = weights or config.FACTOR_WEIGHTS

    # 只用 weights 里有的因子,且 snapshot 里也存在
    valid_factors = [f for f in weights if f in snapshot.columns]
    if not valid_factors:
        raise ValueError("没有可用因子,请检查因子名称是否匹配")

    # 重新归一化权重(避免某些因子缺失导致权重和≠1)
    w = pd.Series({f: weights[f] for f in valid_factors})
    w = w / w.sum()

    # 加权求和(NaN 不参与)
    score = snapshot[valid_factors].mul(w, axis=1).sum(axis=1, min_count=1)
    return score.sort_values(ascending=False)


# ========== 3. 调仓日生成 ==========
def get_rebalance_dates(trading_days: pd.DatetimeIndex,
                        freq: str = None) -> pd.DatetimeIndex:
    """
    根据频率生成调仓日(取每个周期内的最后一个交易日)。

    freq:
      'M'  月末
      '2W' 双周末
      'W'  周末
    """
    freq = freq or config.REBALANCE_FREQ
    s = pd.Series(1, index=trading_days)
    # resample 取每个周期内最后一个有效日期
    rebal = s.resample(freq).last().dropna().index
    # 过滤,确保都是真实交易日
    return rebal.intersection(trading_days)


# ========== 4. 股票过滤 ==========
def filter_stocks(snapshot: pd.DataFrame,
                  min_factors: int = 4) -> pd.DataFrame:
    """
    清洗候选股票:
      - 过滤掉因子缺失太多的(<min_factors 个有效因子)

    实盘还应该过滤: ST/停牌/涨跌停。简化版先做最基础的。
    """
    valid_count = snapshot.notna().sum(axis=1)
    return snapshot[valid_count >= min_factors]


# ========== 5. 单期选股 ==========
def _apply_industry_neutral(score: pd.Series,
                            top_n: int,
                            max_per_industry: int) -> pd.Series:
    """
    行业中性贪心选股: 按 score 降序遍历,每只股票若所属行业未达上限就选入。
    
    入参 score: 已按降序排好的 Series (index=ticker, value=综合得分)
    返回: 截取后的 Series, 长度 ≤ top_n
    
    Note: 若行业限制太严导致无法凑齐 top_n, 会返回少于 top_n 只。
    """
    from quant.industry_map import get_industry
    selected = []
    industry_counts = {}
    for ticker, sc in score.items():
        ind = get_industry(ticker)
        if industry_counts.get(ind, 0) >= max_per_industry:
            continue  # 该行业已满, 跳过
        selected.append(ticker)
        industry_counts[ind] = industry_counts.get(ind, 0) + 1
        if len(selected) >= top_n:
            break
    return score.loc[selected]


def select_top_n(panel: dict,
                 date: str,
                 top_n: int = None,
                 weights: dict = None,
                 max_per_industry: int = None) -> pd.DataFrame:
    """
    在指定日期选 Top N 股票。

    Args:
        max_per_industry: 单一行业最多多少只 (None=不限制, 默认读 config.MAX_PER_INDUSTRY)

    返回:
      DataFrame(index=股票代码, columns=[score, rank, industry, ...各因子原始值])
    """
    top_n = top_n or config.TOP_N
    if max_per_industry is None:
        max_per_industry = getattr(config, 'MAX_PER_INDUSTRY', None)
    date = pd.to_datetime(date)

    # 1. 取该日的横截面快照(用每个因子≤date的最近一个值)
    snap = {}
    for name, df in panel.items():
        valid_dates = df.index[df.index <= date]
        if len(valid_dates) == 0:
            continue
        snap[name] = df.loc[valid_dates[-1]]
    raw_snap = pd.DataFrame(snap)

    # 2. 过滤
    cleaned = filter_stocks(raw_snap)
    if cleaned.empty:
        return pd.DataFrame()

    # 3. 标准化
    normalized = normalize_cross_section(cleaned)

    # 4. 综合打分
    score = composite_score(normalized, weights)  # 已按降序

    # 5. 选 Top N (可选行业中性)
    if max_per_industry is not None and max_per_industry > 0:
        top = _apply_industry_neutral(score, top_n, max_per_industry)
    else:
        top = score.head(top_n)

    result = pd.DataFrame({'score': top, 'rank': range(1, len(top) + 1)})
    # 附上行业标签
    from quant.industry_map import get_industry
    result['industry'] = [get_industry(t) for t in top.index]
    # 附上原始因子值,便于解释
    result = result.join(raw_snap.loc[top.index].round(4))
    return result


# ========== 6. 全期选股 (生成所有调仓日的持仓) ==========
def generate_holdings(panel: dict,
                      start_date: str = None,
                      end_date: str = None,
                      top_n: int = None) -> dict:
    """
    在所有调仓日生成选股结果。

    返回:
      {调仓日(Timestamp): DataFrame(选股结果)}
    """
    start_date = pd.to_datetime(start_date or config.BACKTEST_START)
    end_date = pd.to_datetime(end_date or config.BACKTEST_END)

    # 取所有可用交易日(用第一个因子的索引作为参考)
    first_factor = next(iter(panel.values()))
    all_dates = first_factor.index
    all_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]

    # 生成调仓日
    rebal_dates = get_rebalance_dates(all_dates)
    print(f"[策略] 调仓日数量: {len(rebal_dates)}, "
          f"频率={config.REBALANCE_FREQ}, Top={top_n or config.TOP_N}")

    holdings = {}
    for d in rebal_dates:
        sel = select_top_n(panel, d, top_n=top_n)
        if not sel.empty:
            holdings[d] = sel

    print(f"[策略] 共生成 {len(holdings)} 期持仓")
    return holdings


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from quant.factor_panel import build_factor_panel
    from quant.data_loader import get_stock_pool

    print("=" * 60)
    print("策略层自检")
    print("=" * 60)

    panel = build_factor_panel()
    pool = get_stock_pool().set_index('code')

    # ===== 测试1: 单期选股(最新一期) =====
    last_date = next(iter(panel.values())).index[-1]
    print(f"\n--- 测试1: {last_date.date()} 选股 Top10 ---")
    result = select_top_n(panel, last_date, top_n=10)
    # 加上股票名称
    result = result.join(pool[['name']], how='left')
    cols = ['name', 'rank', 'score'] + [c for c in result.columns
                                          if c not in ('name', 'rank', 'score')]
    print(result[cols].to_string())

    # ===== 测试2: 全期持仓生成 =====
    print(f"\n--- 测试2: 生成全期持仓 ---")
    holdings = generate_holdings(panel, top_n=config.TOP_N)

    # 看几个调仓日的样本
    sample_dates = list(holdings.keys())
    print(f"\n首期({sample_dates[0].date()})持仓 Top5:")
    print(holdings[sample_dates[0]].head().join(pool[['name']])[
        ['name','rank','score']].to_string())
    print(f"\n末期({sample_dates[-1].date()})持仓 Top5:")
    print(holdings[sample_dates[-1]].head().join(pool[['name']])[
        ['name','rank','score']].to_string())

    # 统计: 持仓变化(换手率粗略观察)
    prev_set = None
    turnovers = []
    for d in sample_dates:
        cur = set(holdings[d].index)
        if prev_set is not None:
            change = len(cur - prev_set) / len(cur)
            turnovers.append(change)
        prev_set = cur
    print(f"\n月均换手率(单边): {np.mean(turnovers)*100:.1f}%")
    print(f"  (= 每月新进股票数 / 持仓总数)")
