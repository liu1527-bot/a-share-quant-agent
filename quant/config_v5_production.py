"""
V5 生产模型 - FROZEN CONFIG (DO NOT MODIFY)
==============================================

This file captures the EXACT V5 configuration used for paper trading.
- Frozen on:  2026-05-12
- Selected after walk-forward beating 6 variants (V6/V7/V8/V8RP/V9 5%/V9 10%)
- See PRODUCTION_MODEL.md for full justification.

If you want to test changes, copy this file to config_vX_experiment.py
and modify there. NEVER edit this file in-place.
"""
from quant.config import disable_proxy, PROJECT_ROOT, DATA_DIR, CACHE_DIR, REPORT_DIR

disable_proxy()

# ========== Locked V5 Production Settings ==========
MODEL_VERSION = 'V5_PRODUCTION'
MODEL_FROZEN_DATE = '2026-05-12'
MODEL_INCEPTION_DATE = '2025-12-31'  # Paper trading start

STOCK_POOL = 'hs300'
REBALANCE_FREQ = 'ME'        # Month-end
TOP_N = 30
MAX_PER_INDUSTRY = 3         # Max 3 stocks per industry (industry-neutral)
BENCHMARK = '000300'         # HS300

# Backtest range (for reference; production uses live data)
BACKTEST_START = '2021-01-01'
BACKTEST_END = '2025-12-31'

# ========== Locked V5 Factor Weights ==========
# DO NOT change these weights. They are the result of 8+ rounds of testing.
# Each weight reflects the factor's tested alpha contribution under walk-forward OOS.
FACTOR_WEIGHTS = {
    'value_pb':       0.30,   # IC=0.094, IR=0.314, contributed 32.6% of V5 alpha (W3-W6 attribution)
    'value_pe':       0.25,   # IC=0.082, IR=0.276, real value alpha
    'reversal_5':     0.20,   # 100% hit rate (43/43 months positive); the underrated gold
    'low_vol_60':     0.15,   # Risk control
    'momentum_120_5': 0.10,   # Style balance only
}
assert abs(sum(FACTOR_WEIGHTS.values()) - 1.0) < 1e-6, 'Factor weights must sum to 1.0'

# ========== Locked V5 Risk Filters ==========
# Applied BEFORE factor scoring. Removes stocks in the bottom/top tail to avoid blow-ups.
RISK_FILTERS = {
    'liquidity_pct_floor': 0.20,   # Drop bottom 20% by liquidity (60d avg amount)
    'volatility_pct_cap':  0.90,   # Drop top 10% by 60d volatility
    'roe_pct_floor':       0.50,   # Drop bottom 50% by ROE (avoid earnings landmines)
}

# ========== Locked V5 Stock Filters ==========
NORMALIZE_METHOD = 'rank'
FILTER_ST = True
MIN_LIST_DAYS = 365
EXCLUDE_BJSE = True

# ========== Cost Model (paper trading) ==========
TRADING_COSTS = {
    'slippage':       0.001,    # 0.10% one-side
    'commission':     0.00025,  # 0.025%
    'transfer':       0.00001,  # 0.001%
    'stamp':          0.0005,   # 0.05% (sell only)
    'min_commission': 5.0,      # RMB 5 per trade
}

# ========== Operational Settings ==========
PAPER_INITIAL_CAPITAL = 1_000_000   # RMB 100W virtual capital
WEIGHT_SCHEME = 'equal'              # equal-weighted (1/30 each)


def summary():
    """Print model summary card."""
    print('=' * 70)
    print(f'  Production Model: {MODEL_VERSION}')
    print(f'  Frozen on:        {MODEL_FROZEN_DATE}')
    print(f'  Inception:        {MODEL_INCEPTION_DATE}')
    print('=' * 70)
    print(f'Pool: {STOCK_POOL}  Top: {TOP_N}  Freq: {REBALANCE_FREQ}  MaxIndustry: {MAX_PER_INDUSTRY}')
    print(f'Benchmark: {BENCHMARK}')
    print()
    print('Factor weights (locked):')
    for k, v in FACTOR_WEIGHTS.items():
        print(f'  {k:18s}: {v:.2f}')
    print()
    print('Risk filters:')
    for k, v in RISK_FILTERS.items():
        print(f'  {k:24s}: {v}')


if __name__ == '__main__':
    summary()
