import random
from data.data_fetcher import get_binance_ohlcv, get_ohlcv
from indicators.technicals import compute_atr, compute_rsi, compute_sma, compute_ema

def run_swing_lstm_strategy(
    budget=10000, trade_size=2000, atr_period=14, atr_multiplier=2.0, profit_target=0.05
):
    """
    Swing trading strategy with placeholder LSTM predictions.
    - Uses 4h candles (slower timeframe).
    - For now: random 'long/flat' decision as mock prediction.
    - Entry: If model predicts UP -> BUY.
    - Exit: ATR stop-loss OR profit target (default +5%).
    """

    #df = get_binance_ohlcv(symbol="BTCUSDT", interval="4h", limit=1000)
    df = get_ohlcv(symbol_binance="BTCUSDT", interval="4h", limit=1000, symbol_yf="BTC-USD")

    # Compute indicators (features LSTM would use)
    df = compute_atr(df, period=atr_period)
    df = compute_rsi(df, period=14)
    df = compute_sma(df, period=50)
    df = compute_ema(df, period=50)

    trades = []
    portfolio_usd = budget
    portfolio_btc = 0.0
    open_trade = None

    for i in range(max(atr_period, 50), len(df)):
        row = df.iloc[i]
        price = row["close"]
        atr = row["ATR"]

        # --- Mock LSTM Prediction ---
        # Later: replace with actual LSTM.predict(features)
        prediction = random.choice(["UP", "FLAT"])  # placeholder

        # --- Entry Condition ---
        if open_trade is None and portfolio_usd >= trade_size:
            if prediction == "UP":
                qty = trade_size / price
                portfolio_btc += qty
                portfolio_usd -= trade_size
                stop_loss = price - atr_multiplier * atr
                target_price = price * (1 + profit_target)

                open_trade = {
                    "entry_price": price,
                    "qty": qty,
                    "stop_loss": stop_loss,
                    "target": target_price,
                    "entry_time": row["timestamp"]
                }
                trades.append({
                    "date": row["timestamp"],
                    "action": "BUY",
                    "price": round(float(price), 2),
                    "qty": round(float(qty), 6),
                    "stop_loss": round(float(stop_loss), 2),
                    "target": round(float(target_price), 2),
                    "budget_left": round(float(portfolio_usd), 2)
                })

        # --- Exit Condition ---
        elif open_trade is not None:
            if price <= open_trade["stop_loss"]:  # stop-loss hit
                portfolio_btc -= open_trade["qty"]
                portfolio_usd += open_trade["qty"] * price
                trades.append({
                    "date": row["timestamp"],
                    "action": "SELL_STOPLOSS",
                    "price": round(float(price), 2),
                    "qty": round(float(open_trade["qty"]), 6),
                    "budget_left": round(float(portfolio_usd), 2)
                })
                open_trade = None

            elif price >= open_trade["target"]:  # profit target hit
                portfolio_btc -= open_trade["qty"]
                portfolio_usd += open_trade["qty"] * price
                trades.append({
                    "date": row["timestamp"],
                    "action": "SELL_TARGET",
                    "price": round(float(price), 2),
                    "qty": round(float(open_trade["qty"]), 6),
                    "budget_left": round(float(portfolio_usd), 2)
                })
                open_trade = None

    # Final portfolio valuation
    final_price = df.iloc[-1]["close"]
    portfolio_value = portfolio_usd + portfolio_btc * final_price

    return {
        "trades": trades,
        "final_value_usd": round(float(portfolio_value), 2),
        "portfolio_usd": round(float(portfolio_usd), 2),
        "portfolio_btc": round(float(portfolio_btc), 6)
    }
