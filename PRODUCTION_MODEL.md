# Production Model Card — V5 Multi-Factor

> **Status**: 🟢 LIVE (Paper Trading)
> **Frozen**: 2026-05-12
> **Inception**: 2025-12-31
> **Universe**: HS300 (沪深300)
> **Rebalance**: Monthly (last trading day)
> **Holdings**: Top 30, equal-weighted, max 3 per industry
> **Config file**: [`quant/config_v5_production.py`](quant/config_v5_production.py)

---

## What is V5?

V5 is a **value-tilted multi-factor strategy** for Chinese A-share large caps. It combines five factors with locked weights, applies three risk filters, and re-balances monthly into the 30 highest-scoring HS300 names (subject to a 3-per-industry cap).

```python
FACTOR_WEIGHTS = {
    'value_pb':       0.30,   # 32.6% of historical alpha
    'value_pe':       0.25,
    'reversal_5':     0.20,   # 100% monthly hit rate
    'low_vol_60':     0.15,
    'momentum_120_5': 0.10,
}
RISK_FILTERS = {
    'liquidity_pct_floor': 0.20,   # drop bottom 20% by liquidity
    'volatility_pct_cap':  0.90,   # drop top 10% by volatility
    'roe_pct_floor':       0.50,   # drop bottom 50% by ROE
}
```

---

## Why V5? (Selection Justification)

After 8+ rounds of walk-forward testing on 4 OOS windows (W3 shock-down, W4 924-rebound, W5 slow-bull, W6 2026-rally), V5 is the only configuration where **no variant has reproducibly beaten it under our pre-committed robust standard**:

> **Robust threshold**: win rate ≥ 3/4 AND avg excess ≥ +1pp AND worst window ≥ -3pp

### Variants tested and rejected

| Variant | avg excess vs V5 | win rate | robust | Why rejected |
|---|---:|---:|:---:|---|
| V6 IC-driven weights | -2.87pp | 2/5 | ❌ | IC weighting overfits short windows |
| WINNER (Top 20) | +5.11pp | 2/5 | ❌ | High variance — concentration penalty |
| V7 broad-market timing | -4.45pp | 2/5 | ❌ | Daily timing = noise |
| V8 (CSI500 momentum overlay) | -4.66pp alpha | n/a | ❌ | Momentum decay in CSI500 |
| V5+V8 50/50 | -1.62pp | 3/4 | ❌ | Diluted V5 alpha |
| V5+V8 risk-parity | -1.01pp | 3/4 | ❌ | Volatilities too similar — RP ≈ equal weight |
| V9 HushBull 5% | +1.66pp | 2/4 | ❌ | Only 1 of 4 windows positive (W4 +8.73) |
| V9 HushBull 7% | +1.66pp | 2/4 | ❌ | Same trigger pattern as 5% |
| V9 HushBull 10% (best) | +2.42pp | 2/4 | ❌ | Triggered only 2x; sample size 1 |
| V9 HushBull 12% | +2.18pp | 1/4 | ❌ | Triggered only 1x |
| V9 HushBull 15% | +2.18pp | 1/4 | ❌ | Triggered only 1x |

**Conclusion**: All "improvements" are either single-point lucky or over-fit. V5 is the empirical local optimum on our data.

---

## Known Limitations

1. **Bull-market underperformance**: V5 historical sample (43+ months) shows
   - Sideways months (HS300 -3% to 0%): win rate **94.1%**, avg excess **+2.92pp**
   - Bull months (HS300 > +5%): win rate **14.3%**, avg excess **-3.22pp**, cumulative -22.55pp
   - **Interpretation**: V5 is a value/reversal strategy that lags during broad rallies. Confirmed in real paper trading: 2026-04 (HS300 +8%) cost the portfolio -7.41pp in a single month.

2. **Sample-size ceiling**: Walk-forward only covers 2021-01 → 2026-04 (≈4 OOS windows). Below the 10-window academic minimum. **Conclusions about "best-in-class" are statistically weak.**

3. **No mid-cap exposure**: HS300 universe means we miss CSI500 / CSI1000 alpha. (V8 tried this and failed.)

4. **No timing layer**: We considered HushBull (V9) and bench-trend (V7); both failed robust testing. So V5 stays directional-naive.

---

## Paper Trading Performance (Live since 2025-12-31)

| Month | Portfolio | HS300 | Excess | Notes |
|---|---:|---:|---:|---|
| 2026-01 | +2.43% | +1.65% | **+0.78pp** ✅ | sideways |
| 2026-02 | +1.53% | +0.09% | **+1.44pp** ✅ | sideways |
| 2026-03 | -5.27% | -5.53% | **+0.26pp** ✅ | shock-down (V5 sweet spot) |
| 2026-04 | +0.62% | **+8.03%** | **-7.41pp** 💥 | bull rally (V5 weak spot — as expected) |
| **Cumulative** | **-0.86%** | **+3.83%** | **-4.69pp** | 4 months |

**Verdict**: 3/4 win rate matches pre-trade expectation. Cumulative drag is fully explained by 2026-04 — exactly the failure mode V5 alpha attribution warned about. **No surprises**, but a reminder that V5 needs months of data to demonstrate its edge over a full market cycle.

---

## Operational Workflow

### Monthly (last trading day, after close)
```bash
PYTHON=/c/Users/Administrator/AppData/Roaming/Accio/pre-install/python/python.exe

# 1. Refresh data (kline + fundamentals)
$PYTHON scripts/warmup_data.py

# 2. Rebuild factor panel
$PYTHON scripts/build_zz500_panel.py   # if needed; or rebuild HS300 panel

# 3. Generate next-month holdings
$PYTHON scripts/paper_trade.py rebalance YYYY-MM-DD

# 4. Update NAV
$PYTHON scripts/paper_trade.py update_nav YYYY-MM-DD

# 5. View status
$PYTHON scripts/paper_trade.py status
```

### Files involved
- **State**: `data/paper/positions.json`, `data/paper/nav_history.csv`, `data/paper/trades.csv`
- **Snapshots**: `data/paper/snapshots/YYYY-MM-DD.csv`
- **Factor panel**: `data/cache/factor_panel_hs300.pkl`
- **Risk filters**: `data/cache/risk_filters_hs300.pkl`

---

## Decision Log

| Date | Decision | Reason |
|---|---|---|
| 2026-04-20 | Established Amazon seller research workflow as orthogonal task | (unrelated) |
| 2026-04-26 | V5 selected as walk-forward winner | After V4/V6 testing |
| 2026-05-?? | Built CSI500 panel for V8 testing | Investigate mid-cap overlay |
| 2026-05-?? | V5 alpha attribution: PB-driven, bull-market-weak | Revealed strategic limitation |
| 2026-05-?? | V9 HushBull 5% appeared promising (+1.66pp, 2/4) | Sample too small to commit |
| **2026-05-12** | **V9 threshold scan: NO threshold passes robust standard** | All variants are single-point lucky |
| **2026-05-12** | **V5 frozen as production model** | Empirical local optimum |

---

## Future Improvement Hooks (NOT acted on — see V9 failure)

- **More OOS data**: as paper trading accumulates 12+ months, re-run walk-forward.
- **Conditional timing**: e.g. multi-month confirmation of bull regime instead of 1-month rule.
- **Universe expansion**: only after V5 has 6+ months of confirmed live alpha.
- **Cost optimization**: current ~16bp / month round-trip is acceptable but could be cut.

> ⚠️ **Rule**: Do not modify `config_v5_production.py`. To experiment, copy to `config_vX_experiment.py` and test in a separate walk-forward run. Production stays locked.
