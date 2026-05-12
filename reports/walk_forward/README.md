# V4 Walk-Forward Validation

Generated: 2026-05-11 21:40:15

Source: V4 IS run 2021-2025, sliced into 5 non-overlapping windows.

## Per-window metrics

| Window | Days | Total | Ann | Sharpe | MaxDD | Excess | IR |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1: 2021-04~2022-03 (大盘震荡, 价值起涨) | 243 | 12.52% | 12.56% | 0.47 | -18.99% | 29.98% | 1.73 |
| W2: 2022-04~2023-03 (深度调整 + 反弹) | 243 | 8.72% | 8.75% | 0.30 | -15.89% | 14.03% | 1.11 |
| W3: 2023-04~2024-03 (慢熊红利) | 241 | 15.05% | 15.23% | 0.77 | -14.29% | 28.89% | 2.89 |
| W4: 2024-04~2025-04 (924 反转 + 跨年) | 262 | 1.97% | 1.83% | -0.03 | -15.91% | -2.67% | -0.25 |
| W5: 2025-05~2025-12 (放量后) | 165 | 25.56% | 41.57% | 2.97 | -5.60% | 6.82% | 0.68 |

## Full-period reference

- Total: 80.21%  Ann: 12.52%  Sharpe: 0.54
- MaxDD: -20.66%  Excess: 15.08%  IR: 1.12

## Stability (cross-window)

| Metric | Mean | Std | CV | Verdict |
|---|---:|---:|---:|---|
| Ann Ret | 15.99 | 15.16 | 0.95 | OK |
| Sharpe | 0.90 | 1.20 | 1.33 | UNSTABLE |
| MaxDD | -14.14 | 5.06 | 0.36 | STABLE |
| Ann Excess | 15.41 | 14.11 | 0.92 | OK |
| IR | 1.23 | 1.17 | 0.95 | OK |