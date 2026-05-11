"""
量化选股 Agent - 全局配置
============================
所有可调参数集中在这里，方便后续修改而不用改代码。
"""
import os
from pathlib import Path

# ========== 路径配置 ==========
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORT_DIR = PROJECT_ROOT / "reports"

for d in [DATA_DIR, CACHE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ========== 网络代理处理 ==========
# Windows 环境下系统代理常导致 akshare 超时,这里强制禁用
def disable_proxy():
    for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']:
        os.environ.pop(k, None)
    os.environ['NO_PROXY'] = '*'

disable_proxy()

# ========== 选股配置 ==========
# 股票池: 'all' = 全A股, 'hs300' = 沪深300, 'zz500' = 中证500, 'zz800' = 中证800
# 起步阶段先用 hs300 跑通全流程,验证有效后再切到 'all'
STOCK_POOL = 'hs300'

# 调仓频率: 'ME' = 月末, '2W-FRI' = 双周末, 'W-FRI' = 周末
REBALANCE_FREQ = 'ME'

# 每期持仓数量
TOP_N = 30

# 回测时间范围
BACKTEST_START = '2021-01-01'
BACKTEST_END = '2025-12-31'

# 基准指数
BENCHMARK = '000300'  # 沪深300

# ========== 因子权重 ==========
# 阶段三 (稳健派): 基于因子贡献分析(阶段6)的优化权重
# 关键发现:
#   - value_pb / value_pe 是真Alpha (IC稳定 + 多空收益>20%)
#   - amount_ratio / ma_pos / quality_roe 实测无效或样本不足
#   - momentum 因子在 2021-2025 hs300 内呈负 IC, 但避免反向使用(过拟合)
# 设计原则: 提升真Alpha权重,删除无效因子,保留少量动量做风格平衡
# 总和 = 1.0
FACTOR_WEIGHTS = {
    'value_pb':           0.30,   # ⭐ 真Alpha(IC=0.094, IR=0.314)
    'value_pe':           0.25,   # ⭐ 真Alpha(IC=0.082, IR=0.276)
    'reversal_5':         0.20,   # A股短期反转经典
    'low_vol_60':         0.15,   # 风险控制
    'momentum_120_5':     0.10,   # 少量保留做平衡
    # 已删除: momentum_60(与120_5重复), quality_roe(样本不足),
    #         amount_ratio_5_20(无效), ma_pos_20(无效)
}

# 标准化方式: 'rank' 百分位排名(稳健) / 'zscore' z分数(对正态分布因子更敏感)
NORMALIZE_METHOD = 'rank'

# ========== 股票过滤规则 ==========
# 过滤掉 ST/退市/新股(上市不足 N 天)
FILTER_ST = True
MIN_LIST_DAYS = 365   # 上市满 1 年才纳入
EXCLUDE_BJSE = True   # 排除北交所(数据较少)
