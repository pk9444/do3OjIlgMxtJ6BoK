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

"""
    run_swing_lstm_reg_strategy() : function to run the LSTM Regression model on the BTC Cancdles 
    @params : Budget - the total amount available to trade, init to 10K which can be changed
            trade - how much BTC can be traded in a transcation 
            prob_threshold - the differential b/w the BUY and SELL - set to 1% - can be changed 
    start_date: datetime of last known actual
"""
def run_swing_lstm_reg_strategy(
    budget=10000,
    trade_size=2000,
    prob_threshold=0.01,  # ~1% move trigger
    n_steps=7,
    freq="1D"
):
    # Graceful skip if artifacts missing
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH)):
        return {"trades": [], "final_value_usd": float(budget),
                "portfolio_usd": float(budget), "portfolio_btc": 0.0,
                "note": "LSTM regression model/scalers not found"}

    # Load model + scalers
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    input_dim = len(["prev_close","rsi","ema_20","ema_50","macd","macd_signal","atr"])
    model = LSTMRegressor(input_dim=input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Get OHLCV
    df = get_ohlcv("BTCUSDT", "4h", 600)

    df = compute_atr(df, period=14)
    df = compute_rsi(df, period=14)
    df = compute_sma(df, period=50)
    df = compute_ema(df, period=50)
    df = compute_macd(df)
    df = df.dropna().reset_index(drop=True)

    #features = ["close","rsi","ema_20","ema_50","macd","macd_signal","atr"]
    features = ["prev_close", "RSI", "SMA_50", "EMA_50", "MACD", "Signal", "ATR"]
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

    # Build last sequence for forecasting
    last_seq = df[features].iloc[-SEQ_LEN:].values
    last_seq_scaled = scaler_X.transform(last_seq)
    last_seq_tensor = torch.tensor(last_seq_scaled, dtype=torch.float32)

    forecast_df = forecast_future(
        model,
        last_seq_tensor,
        scaler_X,
        scaler_y,
        n_steps=n_steps,
        start_date=df["timestamp"].iloc[-1],
        freq=freq  # match your data interval
    )

    
    return {
            "trades": trades, 
            "final_value_usd": final_value,
            "portfolio_usd": portfolio_usd, 
            "portfolio_btc": portfolio_btc,
            "forecast": forecast_df
        }


def forecast_future(model, last_seq, scaler_X, scaler_y, n_steps=7, start_date=None, freq="1D"):
    """
    Generate n_steps BTC forecasts using trained LSTM regression model.
    
    last_seq : torch.Tensor, shape (seq_len, n_features)
    scaler   : fitted StandardScaler
    start_date: datetime of last known actual
    """
    preds = []
    seq = last_seq.clone().detach()

    for _ in range(n_steps):
        with torch.no_grad():
            pred_scaled = model(seq.unsqueeze(0)).item()
        pred_price = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        preds.append(pred_price)

        # roll window forward: drop oldest, append new pred as prev_close
        # NOTE: this assumes first column in features is prev_close
        new_row = seq[-1].clone()
        new_row[0] = scaler_X.transform([[pred_price] + [0]*(seq.shape[1]-1)])[0][0]  
        seq = torch.cat([seq[1:], new_row.unsqueeze(0)], dim=0)

    # Dates
    if start_date is not None:
        dates = pd.date_range(start=start_date, periods=n_steps+1, freq=freq)[1:]
    else:
        dates = list(range(n_steps))

    return pd.DataFrame({"date": dates, "predicted": preds})