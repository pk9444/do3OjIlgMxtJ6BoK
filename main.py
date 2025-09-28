from data.data_fetcher import get_binance_ohlcv
import yfinance as yf
import pandas as pd
from indicators.technicals import compute_atr, compute_rsi, compute_sma, compute_ema
from data.data_fetcher import get_ohlcv
from strategies.dca import run_dca_strategy
from strategies.atr_daytrading import run_atr_daytrading_strategy
from strategies.swing_lstm_reg import run_swing_lstm_reg_strategy

def main():
    df = get_ohlcv("BTCUSDT", "4h", 500, "BTC-USD")

    df = compute_atr(df, period=14)
    df = compute_rsi(df, period=14)
    df = compute_sma(df, period=50)
    df = compute_ema(df, period=50)

    print(df.tail())
    
if __name__ == "__main__":
    main()

    result = run_dca_strategy(budget=10000, buy_amount=500, drop_trigger=0.03)

    print("DCA Strategy Results:")
    print(f"Final BTC: {result['final_btc']:.6f}")
    print(f"Final Portfolio Value (USD): {result['final_value_usd']:.2f}")
    print(f"Remaining Budget: {result['remaining_budget']:.2f}")
    print(f"Total Trades: {len(result['trades'])}")

    print("\nTrade History (last 5):")
    for t in result["trades"][-5:]:
        print(t)
    

    result = run_atr_daytrading_strategy(budget=10000, trade_size=1000, atr_period=14, atr_multiplier=1.5)

    print("ATR Stop-Loss Day Trading Results:")
    print(f"Final Portfolio Value (USD): {result['final_value_usd']:.2f}")
    print(f"Remaining Cash: {result['portfolio_usd']:.2f}")
    print(f"Open BTC: {result['portfolio_btc']:.6f}")
    print(f"Total Trades: {len(result['trades'])}")

    print("\nTrade History (last 5):")
    for t in result["trades"][-5:]:
        print(t)
    

    result = run_swing_lstm_reg_strategy(budget=10000, trade_size=2000)

    print("Swing Trading (LSTM placeholder) Results:")
    print(f"Final Portfolio Value (USD): {result['final_value_usd']}")
    print(f"Remaining Cash: {result['portfolio_usd']}")
    print(f"Open BTC: {result['portfolio_btc']}")
    print(f"Total Trades: {len(result['trades'])}")

    print("\nTrade History (last 5):")
    for t in result["trades"][-5:]:
        print(t)




