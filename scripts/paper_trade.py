"""
Paper Trading 脚本 - V5 风控版
==============================

用法:
  python scripts/paper_trade.py init          # 初始化首期持仓 (2025-12-31)
  python scripts/paper_trade.py rebalance     # 月调仓 (用最新数据)
  python scripts/paper_trade.py status        # 查看当前持仓和净值
  python scripts/paper_trade.py update_nav    # 更新到最新 NAV

输出:
  data/paper/positions.json   - 当前持仓
  data/paper/nav_history.csv  - 每日 NAV 历史
  data/paper/trades.csv       - 调仓记录
  data/paper/snapshots/       - 每月 snapshot
"""
import os, sys, json, pickle
for p in ['HTTP_PROXY','HTTPS_PROXY']: os.environ.pop(p, None)
os.environ['NO_PROXY']='*'
sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime

PAPER_DIR = Path('data/paper')
PAPER_DIR.mkdir(parents=True, exist_ok=True)
(PAPER_DIR / 'snapshots').mkdir(exist_ok=True)
INITIAL_CAPITAL = 1_000_000  # 100 万纸面初始资金


def load_state():
    """读当前状态: positions, nav_history, trades."""
    pos_file = PAPER_DIR / 'positions.json'
    if pos_file.exists():
        positions = json.loads(pos_file.read_text(encoding='utf-8'))
    else:
        positions = None
    nav_file = PAPER_DIR / 'nav_history.csv'
    nav = pd.read_csv(nav_file, parse_dates=['date']).set_index('date') if nav_file.exists() else None
    trades_file = PAPER_DIR / 'trades.csv'
    trades = pd.read_csv(trades_file, parse_dates=['date']) if trades_file.exists() else None
    return positions, nav, trades


def save_positions(positions):
    (PAPER_DIR / 'positions.json').write_text(
        json.dumps(positions, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def append_trade_log(date, action, ticker, name, weight, price, score, note=''):
    trades_file = PAPER_DIR / 'trades.csv'
    new_row = pd.DataFrame([{
        'date': date, 'action': action, 'ticker': ticker, 'name': name,
        'weight': round(weight, 4), 'price': round(price, 2),
        'score': round(score, 4) if pd.notna(score) else '', 'note': note
    }])
    if trades_file.exists():
        old = pd.read_csv(trades_file)
        out = pd.concat([old, new_row], ignore_index=True)
    else:
        out = new_row
    out.to_csv(trades_file, index=False, encoding='utf-8-sig')


def cmd_init(rebal_date='2025-12-31'):
    """首期初始化: 在 rebal_date 用 V5 风控选 30 只, 创建 paper trading 状态."""
    from quant.strategy import select_top_n
    from quant.data_loader import get_stock_pool, get_stock_kline

    rebal_date = pd.Timestamp(rebal_date)
    print(f'[init] 首期调仓日: {rebal_date.date()}')

    # 加载 V5 数据
    panel = pickle.load(open('data/cache/factor_panel_hs300.pkl', 'rb'))
    risk_panel = pickle.load(open('data/cache/risk_filters_hs300.pkl', 'rb'))

    sel = select_top_n(panel, rebal_date, top_n=30, risk_panel=risk_panel)
    if sel.empty:
        print('[ERROR] 选股为空'); return
    print(f'[init] 选出 {len(sel)} 只股票')

    pool = get_stock_pool().set_index('code')
    sel = sel.join(pool[['name']], how='left')

    # 等权重 + 取 close 价
    weight_each = 1.0 / len(sel)
    positions = {
        'inception_date': str(rebal_date.date()),
        'last_rebalance': str(rebal_date.date()),
        'initial_capital': INITIAL_CAPITAL,
        'top_n': len(sel),
        'holdings': []
    }

    # 建仓: 每只股票按 close 价买入
    for ticker, row in sel.iterrows():
        try:
            kl = get_stock_kline(ticker, '2025-12-01', '2025-12-31')
            kl['date'] = pd.to_datetime(kl['date'])
            close_at_buy = float(kl[kl['date'] <= rebal_date].iloc[-1]['close'])
        except Exception as ex:
            print(f'  [WARN] {ticker} 无价格: {ex}'); continue
        capital_per_stock = INITIAL_CAPITAL * weight_each
        shares = int(capital_per_stock / close_at_buy / 100) * 100  # A股按手 100 股
        if shares == 0: shares = 100  # 至少买 1 手
        positions['holdings'].append({
            'ticker': ticker,
            'name': row.get('name', ''),
            'industry': row.get('industry', ''),
            'shares': shares,
            'cost_price': round(close_at_buy, 2),
            'cost_value': round(shares * close_at_buy, 2),
            'weight_target': round(weight_each, 4),
            'score': round(float(row['score']), 4),
        })
        append_trade_log(rebal_date.date(), 'BUY', ticker, row.get('name',''),
                          weight_each, close_at_buy, float(row['score']), 'initial position')

    save_positions(positions)
    actual_invested = sum(h['cost_value'] for h in positions['holdings'])
    print(f'\n[init] 持仓创建完成')
    print(f'  目标资金: RMB {INITIAL_CAPITAL:,}')
    print(f'  实际买入: RMB {actual_invested:,.0f} ({actual_invested/INITIAL_CAPITAL*100:.1f}%)')
    print(f'  现金余额: RMB {INITIAL_CAPITAL - actual_invested:,.0f}')

    # 初始化 NAV
    nav_history = pd.DataFrame([{
        'date': rebal_date,
        'nav': 1.0,
        'value': INITIAL_CAPITAL,
        'cash': INITIAL_CAPITAL - actual_invested,
        'invested': actual_invested,
        'benchmark_nav': 1.0,
    }]).set_index('date')
    nav_history.to_csv(PAPER_DIR / 'nav_history.csv', encoding='utf-8-sig')

    # snapshot
    snap = pd.DataFrame(positions['holdings'])
    snap.to_csv(PAPER_DIR / 'snapshots' / f'{rebal_date.date()}.csv',
                index=False, encoding='utf-8-sig')

    print(f'\n[init] 完成! 文件:')
    print(f'  - data/paper/positions.json')
    print(f'  - data/paper/nav_history.csv')
    print(f'  - data/paper/trades.csv')
    print(f'  - data/paper/snapshots/{rebal_date.date()}.csv')


def cmd_status():
    """查看当前持仓 + 净值."""
    positions, nav, trades = load_state()
    if positions is None:
        print('[status] 还未初始化, 请先运行: python scripts/paper_trade.py init')
        return
    print('=' * 70)
    print(f'Paper Trading 状态  ({datetime.now().strftime("%Y-%m-%d %H:%M")})')
    print('=' * 70)
    print(f'起始日: {positions["inception_date"]}  上次调仓: {positions["last_rebalance"]}')
    print(f'初始资金: RMB {positions["initial_capital"]:,}  持仓数: {positions["top_n"]}')
    if nav is not None and len(nav) > 0:
        last = nav.iloc[-1]
        print(f'\n最新净值 ({nav.index[-1].date()}): {last["nav"]:.4f}  (RMB {last["value"]:,.0f})')
        print(f'累计收益: {(last["nav"]-1)*100:+.2f}%')
        if 'benchmark_nav' in last:
            excess = (last["nav"] - last["benchmark_nav"]) * 100
            print(f'同期 HS300: {(last["benchmark_nav"]-1)*100:+.2f}%  超额: {excess:+.2f}%')
    print(f'\n当前持仓 Top 10 (按目标权重):')
    df = pd.DataFrame(positions['holdings']).sort_values('score', ascending=False).head(10)
    print(df[['ticker','name','industry','shares','cost_price','cost_value','score']].to_string(index=False))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'init':
        date = sys.argv[2] if len(sys.argv) > 2 else '2025-12-31'
        cmd_init(date)
    elif cmd == 'status':
        cmd_status()
    elif cmd == 'rebalance':
        print('[rebalance] 月调仓功能 — 下个月需要时实现, 当前先用 init')
    elif cmd == 'update_nav':
        print('[update_nav] 日 NAV 更新功能 — 待实现')
    else:
        print(__doc__)
