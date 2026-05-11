"""
因子库 - 量价因子
==================
所有因子的输入都是单只股票的K线 DataFrame,输出是一个时间序列(索引=日期, 值=因子值)。
设计原则: 纯函数,无副作用,只依赖K线数据。

因子分类:
  - 动量类(Momentum)  : 过去一段时间的涨跌幅
  - 反转类(Reversal)  : 短期反向(超跌反弹)
  - 波动类(Volatility): 风险/稳定性度量
  - 量能类(Volume)    : 成交量/换手活跃度
  - 趋势类(Trend)     : 均线位置
"""
import numpy as np
import pandas as pd


# ========== 标准化K线列 ==========
def _ensure_kline(df: pd.DataFrame) -> pd.DataFrame:
    """统一K线格式,确保有 close/amount 列且按日期排序。"""
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
    return df


# ========== 1. 动量因子 ==========
def momentum(kline: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    N日动量: 过去 window 个交易日的累计收益率
    思想: 强者恒强,涨得好的继续涨

    返回: 与日期对齐的因子值 Series
    """
    df = _ensure_kline(kline)
    return df['close'].pct_change(window)


def momentum_excl_recent(kline: pd.DataFrame,
                         window: int = 60,
                         skip: int = 5) -> pd.Series:
    """
    "12-1" 类动量: 算 window 期收益,但跳过最近 skip 天
    用意: 避开短期反转效应,捕捉中期趋势

    经典学术因子: 过去 12 个月收益率扣除最近 1 个月
    """
    df = _ensure_kline(kline)
    return df['close'].shift(skip) / df['close'].shift(skip + window) - 1


# ========== 2. 反转因子 ==========
def reversal(kline: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    短期反转: 过去 N 日收益率取负
    思想: 超跌反弹,A股短期反转效应显著

    返回值越大 = 过去跌得越狠 = 反弹空间越大
    """
    df = _ensure_kline(kline)
    return -df['close'].pct_change(window)


# ========== 3. 波动率因子 ==========
def volatility(kline: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    日收益率的滚动标准差(年化)
    通常用作 "低波动" 因子: 取负后,波动小的股票得分高
    A股有显著的低波异象(低波股长期跑赢)
    """
    df = _ensure_kline(kline)
    daily_ret = df['close'].pct_change()
    return daily_ret.rolling(window).std() * np.sqrt(252)


def low_volatility(kline: pd.DataFrame, window: int = 60) -> pd.Series:
    """低波因子: 波动率取负,数值越大越"低波"。"""
    return -volatility(kline, window)


# ========== 4. 量能因子 ==========
def amount_ratio(kline: pd.DataFrame, short: int = 5, long: int = 20) -> pd.Series:
    """
    量比: 短期均量 / 长期均量
    > 1 表示近期放量,资金关注度上升
    """
    df = _ensure_kline(kline)
    short_ma = df['amount'].rolling(short).mean()
    long_ma = df['amount'].rolling(long).mean()
    return short_ma / long_ma


# ========== 5. 趋势因子 ==========
def ma_position(kline: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    收盘价相对均线的位置: (close - MA) / MA
    > 0 站上均线(强势), < 0 跌破均线(弱势)
    """
    df = _ensure_kline(kline)
    ma = df['close'].rolling(window).mean()
    return (df['close'] - ma) / ma


# ========== 6. 横截面标准化 ==========
def cross_section_zscore(values: pd.Series) -> pd.Series:
    """
    横截面 z-score 标准化: (x - mean) / std
    用于把不同量纲的因子拉到同一尺度,便于加权合成。

    输入: 同一日期、不同股票的因子值
    输出: 标准化后的值 (大概落在 [-3, 3])
    """
    s = values.dropna()
    if len(s) < 2 or s.std() == 0:
        return pd.Series(0, index=values.index)
    z = (s - s.mean()) / s.std()
    # 极端值截断 (winsorize),避免异常值主导
    z = z.clip(-3, 3)
    return z.reindex(values.index)


def cross_section_rank(values: pd.Series) -> pd.Series:
    """
    横截面百分位排名: 0 ~ 1
    比 z-score 更稳健,不受极端值影响
    """
    return values.rank(pct=True)


# ========== 7. 因子注册表 ==========
# 给每个因子一个名字,后面策略层直接按名字调用
FACTOR_REGISTRY = {
    'momentum_60':      lambda kl: momentum(kl, window=60),
    'momentum_120_5':   lambda kl: momentum_excl_recent(kl, window=120, skip=5),
    'reversal_5':       lambda kl: reversal(kl, window=5),
    'low_vol_60':       lambda kl: low_volatility(kl, window=60),
    'amount_ratio_5_20': lambda kl: amount_ratio(kl, short=5, long=20),
    'ma_pos_20':        lambda kl: ma_position(kl, window=20),
}


def compute_all_factors(kline: pd.DataFrame) -> pd.DataFrame:
    """
    对一只股票的K线计算所有注册因子。
    返回: DataFrame, 索引=日期, 列=因子名
    """
    result = {}
    for name, fn in FACTOR_REGISTRY.items():
        try:
            result[name] = fn(kline)
        except Exception as e:
            print(f"[警告] 因子 {name} 计算失败: {e}")
            result[name] = pd.Series(dtype=float)
    return pd.DataFrame(result)


if __name__ == "__main__":
    # 自检: 用茅台数据算一遍所有因子
    import sys
    sys.path.insert(0, '.')
    from quant.data_loader import get_stock_kline

    print("=" * 60)
    print("因子库自检 - 用 600519(茅台) 数据")
    print("=" * 60)

    kl = get_stock_kline('600519')
    print(f"K线数据: {len(kl)} 个交易日")

    factors = compute_all_factors(kl)
    print(f"\n因子矩阵 shape: {factors.shape}")
    print(f"因子列表: {list(factors.columns)}")

    print("\n最近 5 天的因子值:")
    print(factors.tail(5).to_string())

    print("\n各因子的描述统计:")
    print(factors.describe().round(4).to_string())
