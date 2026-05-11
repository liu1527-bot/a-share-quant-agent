"""
单因子有效性评估
=================
回答 3 个问题:
  1. 这个因子能预测下期收益吗?  -> IC 信息系数
  2. 是否随时间稳定?            -> IC_IR 信息比
  3. 用它分组,组间收益有差异吗? -> 5 分组分层回测

设计:
  - 输入: 因子面板 + 价格矩阵
  - 输出: 一个 metrics DataFrame + 分组净值矩阵
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config
from .strategy import get_rebalance_dates


def compute_forward_returns(prices: pd.DataFrame,
                            rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    计算每个调仓日到下个调仓日的收益矩阵。
    返回: DataFrame(index=rebal_date, columns=stock_code)
    """
    fwd = pd.DataFrame(index=rebal_dates[:-1], columns=prices.columns,
                        dtype=float)
    for i in range(len(rebal_dates) - 1):
        d0 = rebal_dates[i]
        d1 = rebal_dates[i + 1]
        # 把 d0,d1 找到价格矩阵最近的交易日
        if d0 not in prices.index or d1 not in prices.index:
            continue
        ret = prices.loc[d1] / prices.loc[d0] - 1
        fwd.loc[d0] = ret
    return fwd


def get_factor_snapshots(panel: dict,
                         rebal_dates: pd.DatetimeIndex) -> dict:
    """
    取每个因子在所有调仓日的横截面快照矩阵。
    返回: {factor_name: DataFrame(index=rebal_date, columns=stock_code)}
    """
    out = {}
    for fname, fdf in panel.items():
        snap = pd.DataFrame(index=rebal_dates, columns=fdf.columns,
                            dtype=float)
        for d in rebal_dates:
            valid = fdf.index[fdf.index <= d]
            if len(valid) > 0:
                snap.loc[d] = fdf.loc[valid[-1]]
        out[fname] = snap
    return out


def compute_ic_series(factor_snap: pd.DataFrame,
                      forward_ret: pd.DataFrame,
                      method: str = 'spearman') -> pd.Series:
    """
    计算每个调仓期的 IC (Spearman 秩相关系数)。
    IC > 0: 因子值大的股票收益更高
    IC < 0: 因子值大的股票收益更低
    """
    common_dates = factor_snap.index.intersection(forward_ret.index)
    ic_list = {}
    for d in common_dates:
        f = factor_snap.loc[d]
        r = forward_ret.loc[d]
        # 对齐两边都有的股票
        df = pd.concat([f, r], axis=1, keys=['f', 'r']).dropna()
        if len(df) < 30:
            continue
        if method == 'spearman':
            ic, _ = spearmanr(df['f'], df['r'])
        else:
            ic = df['f'].corr(df['r'])
        ic_list[d] = ic
    return pd.Series(ic_list, name='IC').sort_index()


def compute_quintile_returns(factor_snap: pd.DataFrame,
                              forward_ret: pd.DataFrame,
                              n_groups: int = 5) -> pd.DataFrame:
    """
    把每期股票按因子值分成 N 组,计算每组的等权收益。
    返回: DataFrame(index=rebal_date, columns=['Q1'..'Q5'])
    """
    common_dates = factor_snap.index.intersection(forward_ret.index)
    rows = []
    for d in common_dates:
        f = factor_snap.loc[d]
        r = forward_ret.loc[d]
        df = pd.concat([f, r], axis=1, keys=['f', 'r']).dropna()
        if len(df) < n_groups * 5:
            continue
        # 按因子值分位
        df['group'] = pd.qcut(df['f'], q=n_groups,
                              labels=[f'Q{i+1}' for i in range(n_groups)],
                              duplicates='drop')
        grp_ret = df.groupby('group', observed=True)['r'].mean()
        rows.append(grp_ret.rename(d))
    return pd.DataFrame(rows)


def evaluate_factor(factor_snap: pd.DataFrame,
                    forward_ret: pd.DataFrame,
                    n_groups: int = 5) -> dict:
    """
    单因子综合评估,返回 metrics dict + 分组收益矩阵。
    """
    ic = compute_ic_series(factor_snap, forward_ret)
    grp = compute_quintile_returns(factor_snap, forward_ret, n_groups)

    if ic.empty or grp.empty:
        return None

    # 关键指标
    ic_mean = ic.mean()
    ic_std = ic.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_win = (ic > 0).sum() / len(ic) if ic_mean > 0 else (ic < 0).sum() / len(ic)

    # 多空收益(Q5 - Q1)
    if grp.shape[1] >= 2:
        long_short = grp.iloc[:, -1] - grp.iloc[:, 0]
        ls_mean = long_short.mean()
        ls_annual = ls_mean * 12  # 月调,年化
    else:
        ls_mean = 0
        ls_annual = 0

    # 分组单调性: Q5 累计净值 / Q1 累计净值
    grp_nav = (1 + grp).cumprod()
    monotonic = grp_nav.iloc[-1].is_monotonic_increasing or \
                grp_nav.iloc[-1].is_monotonic_decreasing

    return {
        'IC_mean': ic_mean,
        'IC_std': ic_std,
        'IC_IR': ic_ir,
        'IC_win_rate': ic_win,
        'periods': len(ic),
        'long_short_mean': ls_mean,
        'long_short_annual': ls_annual,
        'monotonic': monotonic,
        '_ic_series': ic,
        '_quintile_returns': grp,
        '_quintile_nav': grp_nav,
    }


def run_full_evaluation():
    """对所有因子做评估,输出排行榜和详细指标。"""
    from .factor_panel import build_factor_panel
    from .backtest import build_price_matrix

    print("=" * 60)
    print("[评估] 加载数据...")
    print("=" * 60)
    panel = build_factor_panel()

    # 取所有可用日期
    sample = next(iter(panel.values()))
    all_dates = sample.index
    start = pd.to_datetime(config.BACKTEST_START)
    end = pd.to_datetime(config.BACKTEST_END)
    all_dates = all_dates[(all_dates >= start) & (all_dates <= end)]

    rebal_dates = get_rebalance_dates(all_dates)
    print(f"[评估] 调仓日数量: {len(rebal_dates)}")

    # 价格矩阵 (用所有股票)
    all_codes = list(set().union(*[df.columns.tolist() for df in panel.values()]))
    print(f"[评估] 加载 {len(all_codes)} 只股票价格...")
    prices = build_price_matrix(all_codes,
                                start_date=str(start.date()),
                                end_date=str(end.date()))
    print(f"[评估] 价格矩阵 shape: {prices.shape}")

    # 计算 forward return
    fwd_ret = compute_forward_returns(prices, rebal_dates)
    print(f"[评估] forward returns shape: {fwd_ret.shape}")

    # 因子快照
    snaps = get_factor_snapshots(panel, rebal_dates)

    # 评估每个因子
    print(f"\n[评估] 开始评估 {len(snaps)} 个因子...")
    results = {}
    for fname, snap in snaps.items():
        res = evaluate_factor(snap, fwd_ret)
        if res is not None:
            results[fname] = res
            print(f"  [+] {fname:20s} IC={res['IC_mean']:+.4f} "
                  f"IR={res['IC_IR']:+.3f} "
                  f"L-S年化={res['long_short_annual']*100:+.2f}%")

    return results, panel, prices


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    results, panel, prices = run_full_evaluation()

    # 打印排行榜
    print("\n" + "=" * 60)
    print("[排行] 因子有效性排名 (按 |IC_IR| 排序)")
    print("=" * 60)
    summary = pd.DataFrame({
        fname: {
            'IC_mean':       r['IC_mean'],
            'IC_IR':         r['IC_IR'],
            'IC_win_rate':   r['IC_win_rate'],
            'L-S月均':        r['long_short_mean'],
            'L-S年化':        r['long_short_annual'],
            '单调性':         r['monotonic'],
            '期数':           r['periods'],
        }
        for fname, r in results.items()
    }).T
    summary = summary.sort_values('IC_IR', key=lambda x: x.abs(), ascending=False)
    print(summary.round(4).to_string())
