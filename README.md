# A 股多因子量化选股 Agent

> 沪深 300 多因子选股策略，基于价值 + 反转 + 低波动因子组合，月度调仓。
> 通过样本外验证，年化超额 +10.17%（vs 沪深300）。

## 📊 策略表现

### 样本外验证（OOS Test）

| 指标 | IS (2021-2023) | OOS (2024-2025) | 衰减 |
|------|---------------|----------------|------|
| 年化收益 | 6.88% | 7.52% | -9.3% (反向衰减) |
| 夏普比率 | 0.20 | 0.36 | -80% (反向衰减) |
| 信息比率 | 0.55 | 0.54 | -1.8% |
| **年化超额** | **+9.53%** | **+10.17%** | **-6.7%** |
| 最大回撤 | -37.51% | -17.24% | 大幅改善 |

---

## 🏗️ 架构

```
project/
├── quant/
│   ├── config.py           # 全局配置 (因子权重/股票池/调仓频率)
│   ├── data_loader.py      # K线/成分股拉取
│   ├── fundamentals.py     # PE/PB/ROE 财务数据拉取
│   ├── factors.py          # 量价因子计算
│   ├── factor_panel.py     # 因子面板构建 (date × stock × factor)
│   ├── strategy.py         # 多因子打分 + 选股
│   ├── backtest.py         # 向量化回测
│   ├── factor_eval.py      # 单因子有效性评估 (IC/IR/分组)
│   ├── oos_validation.py   # 样本外验证
│   └── report.py           # 报告生成 (4图 + Markdown)
├── scripts/
│   ├── warmup_data.py      # K线数据预热
│   ├── warmup_fundamentals.py  # 财务数据预热
│   └── monthly_run.py      # 每月调仓主入口 (供 GitHub Actions 调用)
├── reports/                # 每月报告输出
├── .github/workflows/      # GitHub Actions 自动调仓
└── requirements.txt
```

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 预热数据 (首次约 30 分钟)
python scripts/warmup_data.py
python scripts/warmup_fundamentals.py

# 3. 跑回测
python -m quant.backtest

# 4. 生成报告
python -m quant.report

# 5. 单因子评估
python -m quant.factor_eval

# 6. 样本外验证
python -m quant.oos_validation
```

---

## 📈 因子配置

```python
FACTOR_WEIGHTS = {
    'value_pb':       0.30,   # 1/PB, 真Alpha (IC=0.094, IR=0.314)
    'value_pe':       0.25,   # 1/PE, 真Alpha (IC=0.082, IR=0.276)
    'reversal_5':     0.20,   # 5日反转, A股经典
    'low_vol_60':     0.15,   # 60日低波动
    'momentum_120_5': 0.10,   # 中期动量(扣除最近5日)
}
```

调仓频率：**月末** | 持仓数：**Top 30** | 单边交易成本：**0.15%**

---

## ⚠️ 风险声明

- 本项目仅供学习和研究使用，不构成投资建议
- 历史表现不代表未来收益
- 实盘可能受冲击成本、流动性、政策等多重因素影响
- 请独立判断风险，自负盈亏

---

## 📅 自动化

每月 1 日 09:00 (北京时间) 通过 GitHub Actions 自动调仓，结果发布到 [reports/](reports/) 目录。
