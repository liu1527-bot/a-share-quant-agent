"""
数据预热脚本
=============
首次运行: 拉取当前股票池的 5 年历史K线,缓存到 data/cache/
之后所有策略/回测都直接用本地缓存,秒级响应。
"""
import sys
import time
from pathlib import Path

# 让脚本能找到 quant 包
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant import config
from quant.data_loader import get_stock_pool, get_stock_kline


def main():
    print("=" * 60)
    print(f"数据预热: 股票池={config.STOCK_POOL}, "
          f"时间范围={config.BACKTEST_START} ~ {config.BACKTEST_END}")
    print("=" * 60)

    # 1. 取股票池
    pool = get_stock_pool()
    codes = pool['code'].tolist()
    print(f"\n股票池数量: {len(codes)}")

    # 2. 批量拉K线
    success, failed = 0, []
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        df = get_stock_kline(code)
        if df.empty:
            failed.append(code)
        else:
            success += 1

        # 进度显示: 每 20 只打印一次
        if i % 20 == 0 or i == len(codes):
            elapsed = time.time() - t0
            speed = i / elapsed
            eta = (len(codes) - i) / speed if speed > 0 else 0
            print(f"[{i}/{len(codes)}] 成功 {success}, 失败 {len(failed)}, "
                  f"已用 {elapsed:.0f}s, 预计还需 {eta:.0f}s")

        time.sleep(0.05)  # 礼貌性间隔,避免被限流

    # 3. 总结
    print("\n" + "=" * 60)
    print(f"✅ 完成! 成功 {success} / 失败 {len(failed)} / 总计 {len(codes)}")
    print(f"   总耗时: {time.time()-t0:.0f} 秒")
    if failed:
        print(f"   失败代码(前10): {failed[:10]}")
    print(f"   缓存目录: {config.CACHE_DIR / 'kline'}")


if __name__ == "__main__":
    main()
