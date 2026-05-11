"""
样本外验证 (Out-of-Sample Validation)
======================================
专业量化研究的标准流程:
  1. 切分数据: 训练期(IS) vs 验证期(OOS)
  2. 仅用IS数据评估因子,得到IS权重
  3. 用同一权重分别在IS/OOS段回测
  4. 对比表现衰减程度

如果 OOS_year_return / IS_year_return > 60%, 算稳健
如果衰减太多, 说明 IS 优化是过拟合
"""
import numpy as np
import pandas as pd

from . import config
from .factor_panel import build_factor_panel
from .factor_eval import (
    get_factor_snapshots, compute_forward_returns, evaluate_factor
)
from .strategy import (
    get_rebalance_dates, normalize_cross_section, composite_score,
    filter_stocks
)
from .backtest import build_price_matrix, run_backtest


# ========== 切分配置 ==========
IS_START = '2021-01-01'
IS_END   = '2023-12-31'
OOS_START = '2024-01-01'
OOS_END   = '2025-12-31'


def evaluate_factors_in_period(panel: dict,
                               prices: pd.DataFrame,
                               start: str, end: str) -> dict:
    """在指定时间段评估所有因子,返回 {factor: metrics}"""
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    sample = next(iter(panel.values()))
    dates = sample.index[(sample.index >= s) & (sample.index <= e)]
    rebal = get_rebalance_dates(dates)
    fwd = compute_forward_returns(prices, rebal)
    snaps = get_factor_snapshots(panel, rebal)

    results = {}
    for fname, snap in snaps.items():
        res = evaluate_factor(snap, fwd)
        if res:
            results[fname] = res
    return results


def derive_robust_weights(eval_results: dict,
                          min_ir: float = 0.15,
                          allow_reverse: bool = False) -> dict:
    """
    基于因子评估结果,生成稳健派权重。
    - 只保留 |IC_IR| > min_ir 的因子
    - allow_reverse=False: 只用正向因子(避免过拟合 reverse 用法)
    - 权重 = |IR| 归一化
    """
    weights = {}
    for fname, r in eval_results.items():
        if abs(r['IC_IR']) < min_ir:
            continue
        # 不允许反向使用
        if not allow_reverse and r['IC_IR'] < 0:
            continue
        # 还要检查多空收益方向是否一致
        if r['long_short_annual'] < 0:
            continue
        weights[fname] = abs(r['IC_IR'])

    total = sum(weights.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in weights.items()}


def backtest_with_weights(panel: dict,
                          prices: pd.DataFrame,
                          weights: dict,
                          start: str, end: str,
                          top_n: int = None) -> dict:
    """用指定权重在指定区间回测,返回 metrics + 净值。"""
    top_n = top_n or config.TOP_N
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)

    sample = next(iter(panel.values()))
    dates = sample.index[(sample.index >= s) & (sample.index <= e)]
    rebal = get_rebalance_dates(dates)

    # 生成持仓
    holdings = {}
    for d in rebal:
        snap = {}
        for name, df in panel.items():
            if name not in weights:
                continue
            valid = df.index[df.index <= d]
            if len(valid) > 0:
                snap[name] = df.loc[valid[-1]]
        raw = pd.DataFrame(snap)
        cleaned = filter_stocks(raw, min_factors=max(2, len(weights) // 2))
        if cleaned.empty:
            continue
        normalized = normalize_cross_section(cleaned)
        score = composite_score(normalized, weights)
        top = score.head(top_n)
        holdings[d] = pd.DataFrame({'score': top, 'rank': range(1, len(top)+1)})

    # 调用 run_backtest 算指标 (它会自己重新加载 price 矩阵)
    if not holdings:
        return None
    res = run_backtest(holdings)
    return res


def main():
    print("=" * 70)
    print("[样本外验证] IS = 2021-2023, OOS = 2024-2025")
    print("=" * 70)

    panel = build_factor_panel()
    sample = next(iter(panel.values()))
    all_codes = list(set().union(*[df.columns.tolist() for df in panel.values()]))
    prices = build_price_matrix(all_codes, start_date=IS_START, end_date=OOS_END)

    # ===== 步骤1: IS 评估 =====
    print("\n[步骤1] 仅用 IS (2021-2023) 数据评估因子")
    print("-" * 70)
    is_eval = evaluate_factors_in_period(panel, prices, IS_START, IS_END)
    is_summary = pd.DataFrame({
        f: {'IC': r['IC_mean'], 'IR': r['IC_IR'],
            'L-S年化': r['long_short_annual']}
        for f, r in is_eval.items()
    }).T.sort_values('IR', key=lambda x: x.abs(), ascending=False)
    print(is_summary.round(4).to_string())

    # ===== 步骤2: 基于 IS 评估生成权重 =====
    print("\n[步骤2] 基于 IS 评估生成稳健权重")
    print("-" * 70)
    is_weights = derive_robust_weights(is_eval, min_ir=0.15)
    if not is_weights:
        print("[警告] IS 期没有足够稳健的因子,降低门槛")
        is_weights = derive_robust_weights(is_eval, min_ir=0.10)

    print(f"  保留 {len(is_weights)} 个因子:")
    for k, v in sorted(is_weights.items(), key=lambda x: -x[1]):
        print(f"    {k:18s}: {v*100:5.1f}%")

    # 同时算一下 OOS 期内的真实因子表现(对照)
    print("\n[对照] OOS 期实际因子表现(仅用于诊断,不参与权重)")
    print("-" * 70)
    oos_eval = evaluate_factors_in_period(panel, prices, OOS_START, OOS_END)
    oos_summary = pd.DataFrame({
        f: {'IC': r['IC_mean'], 'IR': r['IC_IR'],
            'L-S年化': r['long_short_annual']}
        for f, r in oos_eval.items()
    }).T.sort_values('IR', key=lambda x: x.abs(), ascending=False)
    print(oos_summary.round(4).to_string())

    # ===== 步骤3: 同一权重分别在IS/OOS回测 =====
    print("\n[步骤3] 用 IS权重 在 IS / OOS 段回测")
    print("=" * 70)
    print(f"\n--- IS 段回测 (2021-2023) ---")
    is_bt = backtest_with_weights(panel, prices, is_weights, IS_START, IS_END)
    print(f"\n--- OOS 段回测 (2024-2025) ---")
    oos_bt = backtest_with_weights(panel, prices, is_weights, OOS_START, OOS_END)

    # ===== 步骤4: 对比 =====
    print("\n" + "=" * 70)
    print("[结论] IS vs OOS 表现对比")
    print("=" * 70)

    def parse_pct(s):
        """metrics 里的值是 '+12.34%' 这种字符串,转 float"""
        if isinstance(s, str):
            return float(s.replace('%', '').replace(',', ''))
        return float(s)

    keys = ['年化收益率', '夏普比率', '信息比率', '最大回撤', '年化超额']
    rows = []
    for k in keys:
        is_v = is_bt['metrics'].get(k, 'N/A')
        oos_v = oos_bt['metrics'].get(k, 'N/A')
        rows.append((k, is_v, oos_v))

    print(f"\n{'指标':<10s} {'IS (2021-2023)':<20s} {'OOS (2024-2025)':<20s}")
    print("-" * 60)
    for k, isv, oosv in rows:
        print(f"{k:<10s} {str(isv):<20s} {str(oosv):<20s}")

    # 衰减率(用解析后的数字)
    try:
        is_excess = parse_pct(is_bt['metrics']['年化超额'])
        oos_excess = parse_pct(oos_bt['metrics']['年化超额'])
        is_sharpe = parse_pct(is_bt['metrics']['夏普比率'])
        oos_sharpe = parse_pct(oos_bt['metrics']['夏普比率'])
        print(f"\n[超额衰减] IS={is_excess:+.2f}% → OOS={oos_excess:+.2f}%")
        if is_excess > 0:
            decay = (is_excess - oos_excess) / is_excess * 100
            print(f"  衰减率: {decay:+.1f}%")
            if decay < 30:
                print("  [优秀] 衰减<30%, 策略稳健")
            elif decay < 60:
                print("  [良好] 衰减30-60%, 可接受")
            else:
                print("  [警告] 衰减>60%, 存在过拟合")
        print(f"\n[夏普衰减] IS={is_sharpe:+.2f} → OOS={oos_sharpe:+.2f}")

        if oos_excess > 5:
            print(f"\n[最终结论] OOS 仍有 +{oos_excess:.1f}% 年化超额, 策略真实有效 [PASS]")
        elif oos_excess > 0:
            print(f"\n[最终结论] OOS 超额仅 +{oos_excess:.1f}%, 边际有效 [WEAK]")
        else:
            print(f"\n[最终结论] OOS 无超额 ({oos_excess:.1f}%), 策略可能失效 [FAIL]")
    except Exception as e:
        print(f"[衰减分析失败] {e}")

    return is_bt, oos_bt, is_weights


if __name__ == "__main__":
    main()
