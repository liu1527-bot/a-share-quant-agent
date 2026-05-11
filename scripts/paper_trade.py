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

# ============== 摩擦成本模型 (A 股 2025) ==============
COSTS = {
    'slippage':   0.001,    # 0.10% 单边滑点 (大盘股保守估计)
    'commission': 0.00025,  # 0.025% 券商佣金 (万 2.5)
    'transfer':   0.00001,  # 0.001% 过户费
    'stamp':      0.0005,   # 0.05% 印花税 (仅卖方)
    'min_commission': 5.0,  # 单笔最低佣金 5 元
}


def apply_buy_cost(close, shares):
    """买入: 滑点拉高价 + 佣金过户费.
    返回 (实际成交价, 总现金支出, 费用明细)."""
    fill_px = close * (1 + COSTS['slippage'])
    gross = fill_px * shares
    fee_pct = COSTS['commission'] + COSTS['transfer']
    fee = max(gross * fee_pct, COSTS['min_commission'])
    cash_out = gross + fee
    return fill_px, cash_out, fee


def apply_sell_cost(close, shares):
    """卖出: 滑点压低价 + 佣金过户费 + 印花税.
    返回 (实际成交价, 净到手现金, 费用明细)."""
    fill_px = close * (1 - COSTS['slippage'])
    gross = fill_px * shares
    fee_pct = COSTS['commission'] + COSTS['transfer'] + COSTS['stamp']
    fee = max(gross * fee_pct, COSTS['min_commission'])
    cash_in = gross - fee
    return fill_px, cash_in, fee


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

    # 建仓: 每只股票按 close 价买入 (含摩擦成本)
    total_fees = 0.0
    for ticker, row in sel.iterrows():
        try:
            kl = get_stock_kline(ticker, '2025-12-01', '2025-12-31')
            kl['date'] = pd.to_datetime(kl['date'])
            close_at_buy = float(kl[kl['date'] <= rebal_date].iloc[-1]['close'])
        except Exception as ex:
            print(f'  [WARN] {ticker} 无价格: {ex}'); continue
        capital_per_stock = INITIAL_CAPITAL * weight_each
        # 用预估 fill_px (close * 1.001) 算可买股数, 留点 buffer 防超支
        est_fill = close_at_buy * (1 + COSTS['slippage'])
        shares = int(capital_per_stock / est_fill / 100) * 100  # A股按手 100 股
        if shares == 0: shares = 100  # 至少买 1 手
        fill_px, cash_out, fee = apply_buy_cost(close_at_buy, shares)
        total_fees += fee
        positions['holdings'].append({
            'ticker': ticker,
            'name': row.get('name', ''),
            'industry': row.get('industry', ''),
            'shares': shares,
            'cost_price': round(fill_px, 4),       # 已含滑点
            'cost_value': round(cash_out, 2),       # 已含费用
            'weight_target': round(weight_each, 4),
            'score': round(float(row['score']), 4),
        })
        append_trade_log(rebal_date.date(), 'BUY', ticker, row.get('name',''),
                          weight_each, fill_px, float(row['score']),
                          f'initial; fee={fee:.2f}')

    save_positions(positions)
    actual_invested = sum(h['cost_value'] for h in positions['holdings'])
    print(f'\n[init] 持仓创建完成')
    print(f'  目标资金: RMB {INITIAL_CAPITAL:,}')
    print(f'  实际现金支出: RMB {actual_invested:,.0f} ({actual_invested/INITIAL_CAPITAL*100:.1f}%)')
    print(f'  其中买入费用: RMB {total_fees:,.0f} ({total_fees/INITIAL_CAPITAL*10000:.1f} bp)')
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


def _get_close_at(ticker, date, lookback_start=None):
    """取 ticker 在 <= date 的最后一个 close. 失败返回 None."""
    from quant.data_loader import get_stock_kline
    if lookback_start is None:
        lookback_start = (pd.Timestamp(date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    try:
        kl = get_stock_kline(ticker, lookback_start, pd.Timestamp(date).strftime('%Y-%m-%d'))
        kl['date'] = pd.to_datetime(kl['date'])
        sub = kl[kl['date'] <= pd.Timestamp(date)]
        if len(sub) == 0:
            return None
        return float(sub.iloc[-1]['close'])
    except Exception:
        return None


def cmd_rebalance(rebal_date):
    """月调仓: 用最新 panel 选 30 只, 对比当前持仓, 卖差额买差额."""
    from quant.strategy import select_top_n
    from quant.data_loader import get_stock_pool

    rebal_date = pd.Timestamp(rebal_date)
    positions, nav, trades = load_state()
    if positions is None:
        print('[rebalance] 请先 init'); return

    print(f'[rebalance] 调仓日: {rebal_date.date()}')
    print(f'[rebalance] 上次调仓: {positions["last_rebalance"]}')

    # 加载 panel + 选股
    panel = pickle.load(open('data/cache/factor_panel_hs300.pkl', 'rb'))
    risk_panel = pickle.load(open('data/cache/risk_filters_hs300.pkl', 'rb'))
    sel = select_top_n(panel, rebal_date, top_n=30, risk_panel=risk_panel)
    if sel.empty:
        print(f'[ERROR] panel 在 {rebal_date.date()} 选股为空 (检查数据是否覆盖此日期)'); return

    pool = get_stock_pool().set_index('code')
    sel = sel.join(pool[['name']], how='left')
    new_tickers = set(sel.index)
    cur_holdings = {h['ticker']: h for h in positions['holdings']}
    cur_tickers = set(cur_holdings.keys())

    to_sell = cur_tickers - new_tickers
    to_buy = new_tickers - cur_tickers
    keep = cur_tickers & new_tickers

    print(f'\n[rebalance] 调仓动作:')
    print(f'  保留: {len(keep)} 只 | 卖出: {len(to_sell)} 只 | 买入: {len(to_buy)} 只')
    print(f'  换手率: {len(to_sell)/len(cur_tickers)*100:.1f}%')

    # 1. 卖出 (含滑点 + 印花税 + 佣金)
    cash_freed = 0.0
    total_sell_fees = 0.0
    for ticker in to_sell:
        h = cur_holdings[ticker]
        close = _get_close_at(ticker, rebal_date)
        if close is None:
            print(f'  [WARN] {ticker} 卖出取价失败, 用成本价'); close = h['cost_price']
        fill_px, cash_in, fee = apply_sell_cost(close, h['shares'])
        cash_freed += cash_in
        total_sell_fees += fee
        pnl = cash_in - h['cost_value']
        append_trade_log(rebal_date.date(), 'SELL', ticker, h.get('name',''),
                         0, fill_px, np.nan, f'PnL={pnl:+.0f};fee={fee:.0f}')
        print(f'  SELL {ticker} {h.get("name","")}: {h["shares"]} @ {fill_px:.2f} = {cash_in:,.0f} (fee {fee:.0f}; PnL {pnl:+,.0f})')

    # 计算可用现金 (上次现金 + 卖出回笼)
    last_nav = nav.iloc[-1] if nav is not None and len(nav) > 0 else None
    cash_available = float(last_nav['cash']) + cash_freed if last_nav is not None else cash_freed

    # 2. 重建持仓字典 (先把保留的留下)
    new_holdings = []
    for ticker in keep:
        h = cur_holdings[ticker].copy()
        if ticker in sel.index:
            h['score'] = round(float(sel.loc[ticker, 'score']), 4)
        new_holdings.append(h)

    # 3. 买入新股票: 剩余资金均分给 to_buy (含滑点 + 佣金)
    total_buy_fees = 0.0
    if len(to_buy) > 0:
        capital_per_new = cash_available / len(to_buy)
        for ticker in to_buy:
            row = sel.loc[ticker]
            close = _get_close_at(ticker, rebal_date)
            if close is None or close <= 0:
                print(f'  [WARN] {ticker} 买入取价失败, 跳过'); continue
            est_fill = close * (1 + COSTS['slippage'])
            shares = int(capital_per_new / est_fill / 100) * 100
            if shares == 0:
                print(f'  [WARN] {ticker} 资金不足 1 手 ({close:.2f}), 跳过'); continue
            fill_px, cash_out, fee = apply_buy_cost(close, shares)
            total_buy_fees += fee
            new_holdings.append({
                'ticker': ticker,
                'name': row.get('name', ''),
                'industry': row.get('industry', ''),
                'shares': shares,
                'cost_price': round(fill_px, 4),
                'cost_value': round(cash_out, 2),
                'weight_target': round(1.0/len(sel), 4),
                'score': round(float(row['score']), 4),
            })
            append_trade_log(rebal_date.date(), 'BUY', ticker, row.get('name',''),
                             1.0/len(sel), fill_px, float(row['score']),
                             f'rebalance; fee={fee:.0f}')
            print(f'  BUY  {ticker} {row.get("name","")}: {shares} @ {fill_px:.2f} = {cash_out:,.0f} (fee {fee:.0f})')

    print(f'\n[rebalance] 摩擦成本: 卖费 {total_sell_fees:.0f} + 买费 {total_buy_fees:.0f} = {total_sell_fees+total_buy_fees:.0f} ({(total_sell_fees+total_buy_fees)/INITIAL_CAPITAL*10000:.1f} bp)')

    positions['holdings'] = new_holdings
    positions['last_rebalance'] = str(rebal_date.date())
    positions['top_n'] = len(new_holdings)
    save_positions(positions)

    # snapshot
    snap = pd.DataFrame(new_holdings)
    snap.to_csv(PAPER_DIR / 'snapshots' / f'{rebal_date.date()}.csv',
                index=False, encoding='utf-8-sig')
    print(f'\n[rebalance] 完成. snapshot saved.')


def cmd_update_nav(asof_date):
    """拉每只持仓股票当日 close, 算 portfolio value, 写到 nav_history.csv.
    benchmark 用 sh000300 (HS300 指数) 同期 close."""
    from quant.data_loader import get_stock_kline

    asof = pd.Timestamp(asof_date)
    positions, nav, trades = load_state()
    if positions is None:
        print('[update_nav] 请先 init'); return

    print(f'[update_nav] asof: {asof.date()}')

    # 算持仓市值
    total_market_value = 0.0
    missing = []
    for h in positions['holdings']:
        px = _get_close_at(h['ticker'], asof)
        if px is None:
            missing.append(h['ticker']); continue
        total_market_value += h['shares'] * px

    # 现金 = 上一期现金 (paper 不分红, 不交易就不变)
    last_row = nav.iloc[-1] if nav is not None and len(nav) > 0 else None
    cash = float(last_row['cash']) if last_row is not None else 0.0
    portfolio_value = total_market_value + cash
    new_nav = portfolio_value / INITIAL_CAPITAL

    # benchmark: HS300 指数 (用 sh000300 / 或 akshare index_zh_a_hist)
    bench_nav = float(last_row['benchmark_nav']) if last_row is not None else 1.0
    try:
        # 取自 inception 起的 HS300 close
        import akshare as ak
        idx = ak.index_zh_a_hist(symbol='000300', period='daily',
                                 start_date='20251201',
                                 end_date=asof.strftime('%Y%m%d'))
        idx['日期'] = pd.to_datetime(idx['日期'])
        inception = pd.Timestamp(positions['inception_date'])
        idx_start = idx[idx['日期'] <= inception]
        idx_now = idx[idx['日期'] <= asof]
        if len(idx_start) > 0 and len(idx_now) > 0:
            p0 = float(idx_start.iloc[-1]['收盘'])
            p1 = float(idx_now.iloc[-1]['收盘'])
            bench_nav = p1 / p0
    except Exception as ex:
        print(f'  [WARN] benchmark 取数失败, 用前值: {ex}')

    # append
    new_row = pd.DataFrame([{
        'date': asof, 'nav': round(new_nav, 6),
        'value': round(portfolio_value, 2),
        'cash': round(cash, 2), 'invested': round(total_market_value, 2),
        'benchmark_nav': round(bench_nav, 6),
    }]).set_index('date')
    if nav is None:
        out = new_row
    else:
        out = pd.concat([nav[~nav.index.isin([asof])], new_row]).sort_index()
    out.to_csv(PAPER_DIR / 'nav_history.csv', encoding='utf-8-sig')

    pnl_pct = (new_nav - 1) * 100
    bench_pct = (bench_nav - 1) * 100
    excess = pnl_pct - bench_pct
    print(f'\n[update_nav] {asof.date()}:')
    print(f'  Portfolio NAV: {new_nav:.4f}  ({pnl_pct:+.2f}%)')
    print(f'  HS300 NAV:     {bench_nav:.4f}  ({bench_pct:+.2f}%)')
    print(f'  Excess:        {excess:+.2f}pp')
    print(f'  Total Value:   RMB {portfolio_value:,.0f}  (cash {cash:,.0f} + stock {total_market_value:,.0f})')
    if missing:
        print(f'  [WARN] {len(missing)} 只取价失败: {missing[:5]}{"..." if len(missing)>5 else ""}')


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
        date = sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.today().strftime('%Y-%m-%d')
        cmd_rebalance(date)
    elif cmd == 'update_nav':
        date = sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.today().strftime('%Y-%m-%d')
        cmd_update_nav(date)
    else:
        print(__doc__)
