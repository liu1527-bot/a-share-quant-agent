"""
数据加载模块
=============
负责从 AkShare 拉取 A股数据并本地缓存(parquet 格式,读写都很快)。
所有外部数据访问都收口在这里,后面策略/回测都通过这个模块取数。
"""
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from . import config

warnings.filterwarnings('ignore')

# 延迟导入,避免在没装 akshare 时就报错
def _ak():
    import akshare as ak
    return ak


# ========== 1. 股票列表 ==========
def get_stock_list(refresh: bool = False) -> pd.DataFrame:
    """
    获取全A股列表(代码 + 名称)。
    缓存 1 天,避免反复请求。

    返回:
        DataFrame[code, name]  code 形如 '000001'
    """
    cache_file = config.CACHE_DIR / "stock_list.parquet"
    if not refresh and cache_file.exists():
        # 缓存 < 1 天直接复用
        age = time.time() - cache_file.stat().st_mtime
        if age < 86400:
            return pd.read_parquet(cache_file)

    print("[数据] 拉取全A股列表...")
    ak = _ak()
    df = ak.stock_info_a_code_name()
    df.columns = ['code', 'name']

    # 过滤北交所(8开头/4开头)
    if config.EXCLUDE_BJSE:
        df = df[~df['code'].str.startswith(('8', '4'))].reset_index(drop=True)

    df.to_parquet(cache_file, index=False)
    print(f"[数据] 共 {len(df)} 只股票")
    return df


# ========== 2. 指数成分股 ==========
def get_index_constituents(index_code: str = '000300') -> pd.DataFrame:
    """
    获取指数成分股。常用代码:
      000300 = 沪深300
      000905 = 中证500
      000906 = 中证800
    """
    cache_file = config.CACHE_DIR / f"index_{index_code}.parquet"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 86400 * 7:  # 成分股调整频率低,缓存 7 天
            return pd.read_parquet(cache_file)

    print(f"[数据] 拉取指数 {index_code} 成分股...")
    ak = _ak()
    # 用 ak.index_stock_cons (来自中证指数 / akshare 主接口):
    #   返回当前 300 只成分股 + 各自纳入日期
    # 该接口有翻页 bug,会把约 20 只股票输出两次,但 dedup 后即为正确的 300 只快照
    # 注: ak.index_stock_cons_sina 是新浪旧版页面,2025-06 后未更新,缺少最新调入股
    df = ak.index_stock_cons(symbol=index_code)
    df.columns = ['code', 'name', 'date']
    before = len(df)
    df = df.drop_duplicates(subset=['code'], keep='first').reset_index(drop=True)
    if before != len(df):
        print(f"[数据] 指数 {index_code} 翻页 bug 去重: {before} -> {len(df)} 只")
    print(f"[数据] 指数 {index_code} 成分股: {len(df)} 只")
    df.to_parquet(cache_file, index=False)
    return df


def get_stock_pool() -> pd.DataFrame:
    """根据 config.STOCK_POOL 返回当前选股范围。"""
    pool = config.STOCK_POOL
    if pool == 'all':
        df = get_stock_list()
    else:
        mapping = {'hs300': '000300', 'zz500': '000905', 'zz800': '000906'}
        if pool not in mapping:
            raise ValueError(f"未知股票池: {pool}")
        df = get_index_constituents(mapping[pool])
    # 双保险: 防止旧缓存里还有重复
    if df['code'].duplicated().any():
        df = df.drop_duplicates(subset=['code'], keep='first').reset_index(drop=True)
    return df


# ========== 3. 个股历史K线 ==========
def get_stock_kline(code: str,
                    start_date: str = None,
                    end_date: str = None,
                    adjust: str = 'qfq') -> pd.DataFrame:
    """
    获取单只股票日K线(腾讯接口,稳定)。

    参数:
        code: 6位代码,如 '000001'
        start_date: 'YYYY-MM-DD' 或 'YYYYMMDD'
        end_date:   同上
        adjust: 'qfq'前复权 / 'hfq'后复权 / '' 不复权

    返回:
        DataFrame[date, open, close, high, low, amount]
    """
    if not start_date:
        start_date = config.BACKTEST_START
    if not end_date:
        end_date = config.BACKTEST_END

    sd = start_date.replace('-', '')
    ed = end_date.replace('-', '')

    # 加交易所前缀: 6开头=sh,其他=sz
    symbol = ('sh' if code.startswith('6') else 'sz') + code

    cache_file = config.CACHE_DIR / "kline" / f"{code}_{sd}_{ed}_{adjust}.parquet"
    cache_file.parent.mkdir(exist_ok=True)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    ak = _ak()
    try:
        df = ak.stock_zh_a_hist_tx(symbol=symbol,
                                   start_date=sd,
                                   end_date=ed,
                                   adjust=adjust)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df.to_parquet(cache_file, index=False)
        return df
    except Exception as e:
        print(f"[警告] {code} 拉取失败: {e}")
        return pd.DataFrame()


# ========== 4. 实时行情快照(含 PE/PB 等) ==========
def get_spot_snapshot() -> pd.DataFrame:
    """
    全A股实时行情快照,包含 PE/PB/总市值/流通市值等。
    用于因子计算的最新截面数据。

    注意: 东方财富接口在某些网络环境会失败,这里做了重试和降级。
    """
    cache_file = config.CACHE_DIR / "spot_snapshot.parquet"
    # 缓存 4 小时(交易时段内可能想拿最新数据,自行调整)
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 4 * 3600:
            return pd.read_parquet(cache_file)

    print("[数据] 拉取实时行情快照(可能耗时 30~60 秒)...")
    ak = _ak()

    last_err = None
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_spot_em()
            df.to_parquet(cache_file, index=False)
            print(f"[数据] 快照成功,共 {len(df)} 只")
            return df
        except Exception as e:
            last_err = e
            print(f"[重试 {attempt+1}/3] {str(e)[:100]}")
            time.sleep(3)

    raise RuntimeError(f"实时行情拉取失败(已重试3次): {last_err}")


# ========== 5. 简易的多股票批量K线(进度显示) ==========
def batch_get_klines(codes: list,
                     start_date: str = None,
                     end_date: str = None,
                     sleep: float = 0.1) -> dict:
    """
    批量拉取多只股票K线,返回 {code: DataFrame}。
    sleep 是每次请求间隔,避免被限流。
    """
    result = {}
    total = len(codes)
    for i, code in enumerate(codes, 1):
        df = get_stock_kline(code, start_date, end_date)
        if not df.empty:
            result[code] = df
        if i % 50 == 0 or i == total:
            print(f"[批量] 进度 {i}/{total}, 成功 {len(result)} 只")
        time.sleep(sleep)
    return result


if __name__ == "__main__":
    # 直接 python -m quant.data_loader 可以快速验证
    print("=" * 50)
    print("数据加载模块自检")
    print("=" * 50)

    pool = get_stock_pool()
    print(f"\n股票池({config.STOCK_POOL}): {len(pool)} 只")
    print(pool.head(5).to_string())

    print("\n抽样: 平安银行 2024年1月 K线")
    df = get_stock_kline('000001', '2024-01-01', '2024-01-31')
    print(df.to_string())
