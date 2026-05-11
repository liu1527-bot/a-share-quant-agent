"""
基本面数据加载
================
拉取并缓存:
  - PE/PB 历史 (百度估值接口) - 日频
  - ROE 历史 (新浪财务接口)   - 季频

关键设计:
  1. 都按 point-in-time 处理: 即"那一天能看到的最新值",避免未来函数
  2. 财务数据(季频)用 forward-fill 对齐到日频
  3. 单股拉取失败不影响整体(returns empty)
"""
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

from . import config
from .data_loader import _ak

warnings.filterwarnings('ignore')


# ========== 1. 百度估值数据 (PE/PB 日频) ==========
def get_baidu_valuation(code: str,
                        indicator: str = '市盈率(TTM)',
                        period: str = '近五年') -> pd.DataFrame:
    """
    取个股的估值历史数据 (百度财经)。

    indicator 可选:
      '市盈率(TTM)' / '市盈率(静)' / '市净率' / '市销率' / '总市值'
    period: '近一年' / '近三年' / '近五年' / '全部'

    返回: DataFrame(index=date, columns=['value'])
    """
    safe_ind = indicator.replace('(', '_').replace(')', '_').replace('/', '_')
    cache_file = config.CACHE_DIR / "baidu_val" / f"{code}_{safe_ind}_{period}.parquet"
    cache_file.parent.mkdir(exist_ok=True)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    ak = _ak()
    try:
        df = ak.stock_zh_valuation_baidu(symbol=code,
                                          indicator=indicator,
                                          period=period)
        if df is None or df.empty:
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna().sort_values('date').set_index('date')
        df.to_parquet(cache_file)
        return df
    except Exception as e:
        print(f"  [警告] {code} {indicator} 拉取失败: {str(e)[:80]}")
        return pd.DataFrame()


# ========== 2. 新浪财务指标 (季频) ==========
def get_sina_financial(code: str, start_year: str = '2020') -> pd.DataFrame:
    """
    取个股财务指标 (新浪)。返回包含 ROE、毛利率、增速等所有字段的 DataFrame。

    返回: DataFrame(index=报告期, columns=各财务指标)
    """
    cache_file = config.CACHE_DIR / "sina_fin" / f"{code}_{start_year}.parquet"
    cache_file.parent.mkdir(exist_ok=True)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    ak = _ak()
    try:
        df = ak.stock_financial_analysis_indicator(symbol=code,
                                                     start_year=start_year)
        if df is None or df.empty:
            return pd.DataFrame()
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').set_index('日期')
        # 转 numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.to_parquet(cache_file)
        return df
    except Exception as e:
        print(f"  [警告] {code} 财务拉取失败: {str(e)[:80]}")
        return pd.DataFrame()


# ========== 3. 财报公告日的近似处理 ==========
def add_publish_lag(quarterly_data: pd.DataFrame,
                    lag_days: int = 45) -> pd.DataFrame:
    """
    财务数据是按 报告期 索引的 (3-31, 6-30, 9-30, 12-31)。
    但实际上一季报通常 4 月底才公布、年报次年 4 月底才公布。
    简化处理: 给所有报告期 +lag_days 天作为"可用日期"。

    更精确的做法是用真实公告日,但需要额外接口。
    """
    if quarterly_data.empty:
        return quarterly_data
    df = quarterly_data.copy()
    df.index = df.index + pd.Timedelta(days=lag_days)
    return df


# ========== 4. 拉取批量数据并构建因子面板 ==========
def build_fundamentals_panel(refresh: bool = False) -> dict:
    """
    构建基本面因子面板。

    返回:
      {
        'value_pe':  DataFrame(date × stock),  # 1/PE, 越大越便宜
        'value_pb':  DataFrame(date × stock),
        'quality_roe': DataFrame(date × stock),
      }
    """
    cache_file = config.CACHE_DIR / f"fundamentals_panel_{config.STOCK_POOL}.pkl"
    if not refresh and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 86400 * 7:  # 7天有效
            print(f"[缓存] 加载基本面面板: {cache_file.name}")
            return pd.read_pickle(cache_file)

    from .data_loader import get_stock_pool
    pool = get_stock_pool()
    codes = pool['code'].tolist()
    print(f"[基本面] 拉取 {len(codes)} 只股票的 PE/PB/ROE...")

    pe_dict, pb_dict, roe_dict = {}, {}, {}

    t0 = time.time()
    for i, code in enumerate(codes, 1):
        # PE
        pe_df = get_baidu_valuation(code, '市盈率(TTM)', '近五年')
        if not pe_df.empty:
            # 1/PE,过滤负值(亏损股)
            pe_inv = 1.0 / pe_df['value'].where(pe_df['value'] > 0)
            pe_dict[code] = pe_inv

        # PB
        pb_df = get_baidu_valuation(code, '市净率', '近五年')
        if not pb_df.empty:
            pb_inv = 1.0 / pb_df['value'].where(pb_df['value'] > 0)
            pb_dict[code] = pb_inv

        # ROE (季频 → 加 publish lag → 后面 ffill 对齐)
        fin_df = get_sina_financial(code)
        if not fin_df.empty and '净资产收益率(%)' in fin_df.columns:
            roe = fin_df['净资产收益率(%)'].dropna()
            roe = add_publish_lag(roe.to_frame())['净资产收益率(%)']
            roe_dict[code] = roe

        # 进度
        if i % 20 == 0 or i == len(codes):
            elapsed = time.time() - t0
            speed = i / elapsed
            eta = (len(codes) - i) / speed if speed > 0 else 0
            print(f"  [{i}/{len(codes)}] PE={len(pe_dict)} PB={len(pb_dict)} "
                  f"ROE={len(roe_dict)}, 已用 {elapsed:.0f}s, "
                  f"预计还需 {eta:.0f}s")

        time.sleep(0.05)  # 礼貌间隔

    # 重组成面板 (date × stock 矩阵)
    print("[基本面] 拼装因子面板...")
    pe_panel = pd.DataFrame(pe_dict).sort_index()
    pb_panel = pd.DataFrame(pb_dict).sort_index()
    roe_panel = pd.DataFrame(roe_dict).sort_index()

    panel = {
        'value_pe':    pe_panel,
        'value_pb':    pb_panel,
        'quality_roe': roe_panel,
    }

    pd.to_pickle(panel, cache_file)
    print(f"[基本面] 已保存: {cache_file.name}")
    for name, df in panel.items():
        print(f"  {name}: shape = {df.shape}")

    return panel


def align_to_daily(quarterly_panel: pd.DataFrame,
                   daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    把季频/不规则的财务面板对齐到日频。
    用 forward-fill: 直到下一次更新前都用最新已知值。
    """
    aligned = quarterly_panel.reindex(daily_index, method='ffill')
    return aligned


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    print("=" * 60)
    print("基本面模块自检 - 茅台示例")
    print("=" * 60)

    # 单股测试
    pe = get_baidu_valuation('600519', '市盈率(TTM)', '近五年')
    print(f"\n茅台 PE: {len(pe)} 个数据点")
    print(pe.tail(5).to_string())

    fin = get_sina_financial('600519')
    print(f"\n茅台 财务: {len(fin)} 期, 列数 {len(fin.columns)}")
    print(fin[['净资产收益率(%)', '销售毛利率(%)']].tail(5).to_string())
