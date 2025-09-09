import pandas as pd

def equity_curve_from_trades(trades, price_df, initial_budget):
    """
    Build an equity curve at candle timestamps using trade log and price series.
    Assumes single-position-at-a-time strategies:
      - BUY opens a position (qty > 0)
      - SELL_* closes it (qty > 0)
    price_df: DataFrame with ['timestamp','close'] at the strategy's timeframe
    trades: list of dicts produced by our strategies (with 'date','action', etc.)
    Returns: DataFrame ['timestamp','equity','cash','btc']
    """
    df = price_df[['timestamp','close']].copy().sort_values('timestamp')
    df['equity'] = float(initial_budget)
    df['cash'] = float(initial_budget)
    df['btc'] = 0.0

    # index trades by timestamp for quick lookups
    tdf = pd.DataFrame(trades)
    tdf = tdf.sort_values('date') if not tdf.empty else tdf

    open_qty = 0.0
    cash = float(initial_budget)

    t_idx = 0
    t_len = 0 if tdf.empty else len(tdf)

    for i, row in df.iterrows():
        ts = row['timestamp']
        price = float(row['close'])

        # apply all trades that happen at or before this timestamp
        while t_idx < t_len and tdf.iloc[t_idx]['date'] <= ts:
            tr = tdf.iloc[t_idx]
            action = tr['action']
            if action == 'BUY':
                qty = float(tr['qty'])
                cost = qty * price
                if cost <= cash and qty > 0:
                    cash -= cost
                    open_qty += qty
            elif action.startswith('SELL'):
                qty = float(tr['qty'])
                if qty > 0 and open_qty >= qty:
                    proceeds = qty * price
                    cash += proceeds
                    open_qty -= qty
            elif action == 'STOP_TRADING':
                # no position change here; strategy loop already halted.
                pass
            t_idx += 1

        equity = cash + open_qty * price
        df.at[i, 'cash'] = cash
        df.at[i, 'btc'] = open_qty
        df.at[i, 'equity'] = equity

    return df[['timestamp','equity','cash','btc']]

def merge_equity_curves(curves: dict):
    """
    curves: dict like {'DCA': df1, 'ATR': df2, 'Swing': df3}
    Returns a single DataFrame with timestamp + one equity column per strategy.
    """
    out = None
    for name, cdf in curves.items():
        c = cdf[['timestamp','equity']].rename(columns={'equity': name})
        out = c if out is None else out.merge(c, on='timestamp', how='outer')
    return out.sort_values('timestamp').reset_index(drop=True)
