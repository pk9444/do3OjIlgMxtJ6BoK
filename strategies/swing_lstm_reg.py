import torch
import torch.nn as nn
import joblib
import os
import pandas as pd
from data.data_fetcher import get_ohlcv
from indicators.technicals import compute_rsi, compute_ema, compute_macd, compute_atr, compute_sma

SEQ_LEN = 30
MODEL_PATH = "models/lstm_regressor.pt"
SCALER_X_PATH = "models/lstm_X_scaler.pkl"
SCALER_Y_PATH = "models/lstm_y_scaler.pkl"

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def run_swing_lstm_reg_strategy(
    budget=10000,
    trade_size=2000,
    prob_threshold=0.01  # ~1% move trigger
):
    # Graceful skip if artifacts missing
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH)):
        return {"trades": [], "final_value_usd": float(budget),
                "portfolio_usd": float(budget), "portfolio_btc": 0.0,
                "note": "LSTM regression model/scalers not found"}

    # Load model + scalers
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    input_dim = len(["close","rsi","ema_20","ema_50","macd","macd_signal","atr"])
    model = LSTMRegressor(input_dim=input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Get OHLCV
    df = get_ohlcv("BTCUSDT", "4h", 600)

    # df["rsi"] = compute_rsi(df["close"], 14)
    # df["ema_20"] = compute_ema(df["close"], 20)
    # df["ema_50"] = compute_ema(df["close"], 50)
    # df["macd"], df["macd_signal"] = compute_macd(df["close"])
    # df["atr"] = compute_atr(df)

    df = compute_atr(df, period=14)
    df = compute_rsi(df, period=14)
    df = compute_sma(df, period=50)
    df = compute_ema(df, period=50)
    df = compute_macd(df)
    df = df.dropna().reset_index(drop=True)

    #features = ["close","rsi","ema_20","ema_50","macd","macd_signal","atr"]
    features = ["close", "RSI", "SMA_50", "EMA_50", "MACD", "Signal", "ATR"]
    trades = []
    portfolio_btc, portfolio_usd = 0.0, budget

    # Roll through time
    for i in range(SEQ_LEN, len(df)-1):
        seq = df[features].iloc[i-SEQ_LEN:i].values
        seq_scaled = scaler_X.transform(seq)
        X_t = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = model(X_t).item()
        pred_price = scaler_y.inverse_transform([[pred_scaled]])[0][0]

        current_price = df["close"].iloc[i]
        date = df["timestamp"].iloc[i]

        # Simple trading logic
        if pred_price > current_price * (1 + prob_threshold) and portfolio_usd >= trade_size:
            qty = trade_size / current_price
            portfolio_btc += qty
            portfolio_usd -= trade_size
            trades.append({"date": date, "action": "BUY",
                           "price": current_price, "qty": qty,
                           "budget_left": portfolio_usd})
        elif pred_price < current_price * (1 - prob_threshold) and portfolio_btc > 0:
            portfolio_usd += portfolio_btc * current_price
            trades.append({"date": date, "action": "SELL",
                           "price": current_price, "qty": portfolio_btc,
                           "budget_left": portfolio_usd})
            portfolio_btc = 0.0

    final_value = portfolio_usd + portfolio_btc * df["close"].iloc[-1]
    return {"trades": trades, "final_value_usd": final_value,
            "portfolio_usd": portfolio_usd, "portfolio_btc": portfolio_btc}
