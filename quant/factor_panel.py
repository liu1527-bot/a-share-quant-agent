"""
因子面板构建
==============
把所有股票的所有因子,组装成"3D 数据" (日期 × 股票 × 因子)。
这是策略层的核心输入。

数据结构 (核心):
  factor_panel[factor_name] = DataFrame(index=date, columns=stock_code)

例如取某天的所有股票的动量因子:
  panel['momentum_60'].loc['2024-12-31']  # → Series(index=股票代码)
"""
import time
from pathlib import Path
import pandas as pd

from . import config
from .data_loader import get_stock_pool, get_stock_kline
from .factors import compute_all_factors, FACTOR_REGISTRY


def _merge_fundamentals(panel: dict) -> dict:
    """
    把基本面因子(PE/PB/ROE)合并进面板。
    需要先跑过 scripts/warmup_fundamentals.py。

    基本面因子 (季频/不规则) 会按 价量因子的日期索引 forward-fill。
    """
    try:
        from .fundamentals import build_fundamentals_panel, align_to_daily
    except ImportError:
        print("[基本面] 未找到 fundamentals 模块, 跳过")
        return panel

    fund_cache = config.CACHE_DIR / f"fundamentals_panel_{config.STOCK_POOL}.pkl"
    if not fund_cache.exists():
        print(f"[基本面] 缓存不存在 ({fund_cache.name}),"
              " 请先运行 python scripts/warmup_fundamentals.py")
        return panel

    fund_panel = pd.read_pickle(fund_cache)

    # 取量价因子的日期索引,作为对齐基准
    sample_factor = next(iter(panel.values()))
    daily_index = sample_factor.index

    print(f"[基本面] 合并 PE/PB/ROE → 对齐到 {len(daily_index)} 个交易日...")
    for fname, fdf in fund_panel.items():
        if fdf.empty:
            print(f"  [跳过] {fname}: 空")
            continue
        # 对齐到日频(用 ffill,直到下一次更新)
        aligned = align_to_daily(fdf, daily_index)
        # 只保留量价因子里也有的股票(取交集)
        common_codes = sample_factor.columns.intersection(aligned.columns)
        aligned = aligned[common_codes]
        panel[fname] = aligned
        # 报告覆盖率
        latest = aligned.iloc[-1]
        coverage = latest.notna().sum() / len(common_codes) * 100
        print(f"  [+] {fname}: shape={aligned.shape}, 末日覆盖率={coverage:.0f}%")

    return panel


def build_factor_panel(refresh: bool = False) -> dict:
    """
    构建全股票池的因子面板。

    返回:
        {
          'momentum_60': DataFrame(index=日期, columns=股票代码),
          'low_vol_60':  DataFrame(...),
          ...
        }
    """
    cache_file = config.CACHE_DIR / f"factor_panel_{config.STOCK_POOL}.pkl"
    if not refresh and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 86400:  # 1天有效
            print(f"[缓存] 直接加载因子面板: {cache_file.name}")
            return pd.read_pickle(cache_file)

    pool = get_stock_pool()
    codes = pool['code'].tolist()
    print(f"[因子] 计算 {len(codes)} 只股票的因子...")

    # 收集每只股票的因子矩阵: {code: DataFrame(date, factors)}
    per_stock = {}
    for i, code in enumerate(codes, 1):
        kl = get_stock_kline(code)
        if kl.empty or len(kl) < 200:  # 数据不足跳过
            continue
        per_stock[code] = compute_all_factors(kl)
        if i % 50 == 0 or i == len(codes):
            print(f"  进度 {i}/{len(codes)}, 有效股票 {len(per_stock)}")

    # 重组为 panel: {factor_name: DataFrame(date × stock)}
    print("[因子] 重组为面板格式...")
    panel = {}
    for factor_name in FACTOR_REGISTRY.keys():
        # 横向拼接所有股票的同一因子
        factor_df = pd.DataFrame({
            code: df[factor_name] for code, df in per_stock.items()
            if factor_name in df.columns
        })
        # 确保按日期排序
        factor_df = factor_df.sort_index()
        panel[factor_name] = factor_df

    # 合并基本面因子(PE/PB/ROE)
    panel = _merge_fundamentals(panel)

    # 保存
    pd.to_pickle(panel, cache_file)
    print(f"[因子] 已保存: {cache_file.name}")

    # 打印 shape 总览
    for name, df in panel.items():
        print(f"  {name}: shape = {df.shape}")

    return panel


def get_factor_snapshot(panel: dict, date: str) -> pd.DataFrame:
    """
    取某一天的因子横截面快照。
    返回: DataFrame(index=股票代码, columns=因子名)
    """
    date = pd.to_datetime(date)
    snap = {}
    for name, df in panel.items():
        # 找 ≤ date 的最近一个交易日
        valid_dates = df.index[df.index <= date]
        if len(valid_dates) == 0:
            continue
        snap[name] = df.loc[valid_dates[-1]]
    return pd.DataFrame(snap)


if __name__ == "__main__":
    panel = build_factor_panel(refresh=True)

    print("\n" + "=" * 60)
    print("因子面板自检 - 取 2025-12-31 的横截面快照")
    print("=" * 60)
    snap = get_factor_snapshot(panel, '2025-12-31')
    print(f"\n快照 shape: {snap.shape}  (股票数 × 因子数)")
    print(f"\n前 10 只股票的因子值:")
    print(snap.head(10).round(4).to_string())
    print(f"\n各因子缺失率:")
    print((snap.isna().mean() * 100).round(1).to_string())
