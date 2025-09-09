#from data.data_fetcher import get_binance_ohlcv

from data.data_fetcher import get_ohlcv

def run_dca_strategy(budget=10000, buy_amount=500, drop_trigger=0.03):
    """
    Simple DCA strategy:
    - Fetch daily candles from Binance (1d interval).
    - Buy a fixed amount when price drops >= drop_trigger (e.g., 3%) from last buy.
    - Stop when budget runs out.
    
    :param budget: Total budget allocated for DCA
    :param buy_amount: USD amount per buy
    :param drop_trigger: Fractional drop since last buy (e.g., 0.03 = 3%)
    :return: List of trades + final portfolio
    """
    #df = get_binance_ohlcv(symbol="BTCUSDT", interval="1d", limit=500)
    df = get_ohlcv(symbol_binance="BTCUSDT", interval="1d", limit=500, symbol_yf="BTC-USD")
    
    trades = []
    portfolio_btc = 0.0
    remaining_budget = budget
    last_buy_price = None

    for _, row in df.iterrows():
        close_price = row["close"]

        # If it's the first buy, just buy immediately
        if last_buy_price is None:
            qty = buy_amount / close_price
            portfolio_btc += qty
            remaining_budget -= buy_amount
            last_buy_price = close_price
            trades.append({
                "date": row["timestamp"],
                "price": close_price,
                "qty": qty,
                "action": "BUY",
                "budget_left": remaining_budget
            })
            continue

        # Check if drop_trigger condition met
        if (close_price <= last_buy_price * (1 - drop_trigger)) and remaining_budget >= buy_amount:
            qty = buy_amount / close_price
            portfolio_btc += qty
            remaining_budget -= buy_amount
            last_buy_price = close_price
            trades.append({
                "date": row["timestamp"],
                "price": close_price,
                "qty": qty,
                "action": "BUY",
                "budget_left": remaining_budget
            })

    # Final portfolio value
    final_price = df.iloc[-1]["close"]
    portfolio_value = portfolio_btc * final_price

    return {
        "trades": trades,
        "final_btc": portfolio_btc,
        "final_value_usd": portfolio_value,
        "remaining_budget": remaining_budget
    }
