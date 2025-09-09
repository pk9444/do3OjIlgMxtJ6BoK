from data.data_fetcher import get_binance_ohlcv, get_ohlcv
from indicators.technicals import compute_atr

def run_atr_daytrading_strategy(
    budget=10000, trade_size=1000, atr_period=14, atr_multiplier=1.5, max_drawdown=0.25
):
    """
    ATR-based Stop-Loss Day Trading Strategy with Portfolio Safeguards
    """

    #df = get_binance_ohlcv(symbol="BTCUSDT", interval="30m", limit=1000)
    df = get_ohlcv(symbol_binance="BTCUSDT", interval="30m", limit=1000, symbol_yf="BTC-USD")
    df = compute_atr(df, period=atr_period)

    trades = []
    initial_budget = budget
    portfolio_usd = budget
    portfolio_btc = 0.0
    open_trade = None
    safeguard_triggered = False

    for i in range(atr_period, len(df)):
        row = df.iloc[i]
        price = row["close"]
        atr = row["ATR"]

        # --- Global safeguard check ---
        portfolio_value = portfolio_usd + portfolio_btc * price
        if portfolio_value <= initial_budget * (1 - max_drawdown):
            safeguard_triggered = True
            # trades.append({
            #     "date": row["timestamp"],
            #     "action": "STOP_TRADING",
            #     "price": price,
            #     "portfolio_value": portfolio_value
            # })
            trades.append({
                "date": row["timestamp"],
                "action": "STOP_TRADING",
                "price": round(float(price), 2),
                "portfolio_value": round(float(portfolio_value), 2)
            })
            break  # stop further trading

        # --- Entry condition ---
        if open_trade is None and portfolio_usd >= trade_size:
            prev_close = df.iloc[i - 1]["close"]

            if price > prev_close:  # simple breakout entry
                qty = trade_size / price
                if qty > 0:  # safeguard: don’t buy zero BTC
                    portfolio_btc += qty
                    portfolio_usd -= trade_size
                    stop_loss = price - atr_multiplier * atr
                    target_price = price * 1.02  # +2% profit target

                    open_trade = {
                        "entry_price": price,
                        "qty": qty,
                        "stop_loss": stop_loss,
                        "target": target_price,
                        "entry_time": row["timestamp"]
                    }
                    # trades.append({
                    #     "date": row["timestamp"],
                    #     "action": "BUY",
                    #     "price": price,
                    #     "qty": qty,
                    #     "stop_loss": stop_loss,
                    #     "target": target_price,
                    #     "budget_left": portfolio_usd
                    # })
                    trades.append({
                    "date": row["timestamp"],
                    "action": "BUY",
                    "price": round(float(price), 2),
                    "qty": round(float(qty), 6),
                    "stop_loss": round(float(stop_loss), 2),
                    "target": round(float(target_price), 2),
                    "budget_left": round(float(portfolio_usd), 2)
                })

        # --- Exit condition ---
        elif open_trade is not None:
            # Stop-loss hit
            if price <= open_trade["stop_loss"] and open_trade["qty"] > 0:
                portfolio_btc -= open_trade["qty"]
                portfolio_usd += open_trade["qty"] * price
                # trades.append({
                #     "date": row["timestamp"],
                #     "action": "SELL_STOPLOSS",
                #     "price": price,
                #     "qty": open_trade["qty"],
                #     "budget_left": portfolio_usd
                # })

                trades.append({
                    "date": row["timestamp"],
                    "action": "SELL_STOPLOSS",
                    "price": round(float(price), 2),
                    "qty": round(float(open_trade["qty"]), 6),
                    "budget_left": round(float(portfolio_usd), 2)
                })
                open_trade = None

            # Profit target hit
            elif price >= open_trade["target"] and open_trade["qty"] > 0:
                portfolio_btc -= open_trade["qty"]
                portfolio_usd += open_trade["qty"] * price
                # trades.append({
                #     "date": row["timestamp"],
                #     "action": "SELL_TARGET",
                #     "price": price,
                #     "qty": open_trade["qty"],
                #     "budget_left": portfolio_usd
                # })

                trades.append({
                    "date": row["timestamp"],
                    "action": "SELL_TARGET",
                    "price": round(float(price), 2),
                    "qty": round(float(open_trade["qty"]), 6),
                    "budget_left": round(float(portfolio_usd), 2)
                })
                open_trade = None

    # Final valuation
    final_price = df.iloc[-1]["close"]
    final_value = portfolio_usd + portfolio_btc * final_price

    return {
        "trades": trades,
        "final_value_usd": final_value,
        "portfolio_usd": portfolio_usd,
        "portfolio_btc": portfolio_btc,
        "safeguard_triggered": safeguard_triggered
    }

# from data.data_fetcher import get_binance_ohlcv
# from indicators.technicals import compute_atr

# def run_atr_daytrading_strategy(budget=10000, trade_size=1000, atr_period=14, atr_multiplier=1.5):
#     """
#     ATR-based Stop-Loss Day Trading Strategy:
#     - Fetch 30m candles from Binance.
#     - Enter long trades when price rises above previous close (simple breakout rule for now).
#     - Stop-loss = entry_price - (atr_multiplier * ATR).
#     - Exit when stop-loss is hit OR when price rises 2% above entry.
#     - Track PnL and portfolio.

#     :param budget: Total trading budget
#     :param trade_size: Fixed USD amount per trade
#     :param atr_period: ATR calculation window
#     :param atr_multiplier: Stop-loss multiplier
#     :return: Trade history + portfolio summary
#     """
#     df = get_binance_ohlcv(symbol="BTCUSDT", interval="30m", limit=1000)
#     df = compute_atr(df, period=atr_period)

#     trades = []
#     portfolio_usd = budget
#     portfolio_btc = 0.0
#     open_trade = None

#     for i in range(atr_period, len(df)):
#         row = df.iloc[i]
#         price = row["close"]
#         atr = row["ATR"]

#         # If no open trade, check for entry
#         if open_trade is None and portfolio_usd >= trade_size:
#             prev_close = df.iloc[i - 1]["close"]

#             # Simple breakout rule: price > prev close
#             if price > prev_close:
#                 qty = trade_size / price
#                 portfolio_btc += qty
#                 portfolio_usd -= trade_size
#                 stop_loss = price - atr_multiplier * atr
#                 target_price = price * 1.02  # +2% profit target

#                 open_trade = {
#                     "entry_price": price,
#                     "qty": qty,
#                     "stop_loss": stop_loss,
#                     "target": target_price,
#                     "entry_time": row["timestamp"]
#                 }
#                 trades.append({
#                     "date": row["timestamp"],
#                     "action": "BUY",
#                     "price": price,
#                     "qty": qty,
#                     "stop_loss": stop_loss,
#                     "target": target_price,
#                     "budget_left": portfolio_usd
#                 })

#         # If trade is open, check for exit conditions
#         elif open_trade is not None:
#             # Stop-loss hit
#             if price <= open_trade["stop_loss"]:
#                 portfolio_btc -= open_trade["qty"]
#                 portfolio_usd += open_trade["qty"] * price
#                 trades.append({
#                     "date": row["timestamp"],
#                     "action": "SELL_STOPLOSS",
#                     "price": price,
#                     "qty": open_trade["qty"],
#                     "budget_left": portfolio_usd
#                 })
#                 open_trade = None

#             # Profit target hit
#             elif price >= open_trade["target"]:
#                 portfolio_btc -= open_trade["qty"]
#                 portfolio_usd += open_trade["qty"] * price
#                 trades.append({
#                     "date": row["timestamp"],
#                     "action": "SELL_TARGET",
#                     "price": price,
#                     "qty": open_trade["qty"],
#                     "budget_left": portfolio_usd
#                 })
#                 open_trade = None

#     # Final portfolio value
#     final_price = df.iloc[-1]["close"]
#     portfolio_value = portfolio_usd + portfolio_btc * final_price

#     return {
#         "trades": trades,
#         "final_value_usd": portfolio_value,
#         "portfolio_usd": portfolio_usd,
#         "portfolio_btc": portfolio_btc
#     }
