# strategies/swing_lstm_actual.py (excerpt)

import os
import json
import torch
import torch.nn as nn
import joblib

SEQ_LEN = 30

class SwingLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=96, num_layers=2, dropout=0.3, output_dim=1):
        super(SwingLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # last timestep
        out = self.fc(out)
        return self.sigmoid(out)

# def run_swing_lstm_real_strategy(
#     budget=10000,
#     trade_size=2000,
#     atr_multiplier=2.0,
#     profit_target=0.05,
#     prob_threshold=0.55,
#     model_path="models/swing_lstm.pt",
#     scaler_path="models/swing_scaler.pkl",
#     config_path="models/swing_lstm_config.json"  # optional
# ):
#     # graceful skip if artifacts missing
#     if not os.path.exists(model_path) or not os.path.exists(scaler_path):
#         return {
#             "trades": [],
#             "final_value_usd": float(budget),
#             "portfolio_usd": float(budget),
#             "portfolio_btc": 0.0,
#             "note": "PyTorch model/scaler not found"
#         }

#     # load scaler
#     scaler = joblib.load(scaler_path)

#     # infer or load model config so it matches training
#     input_dim = 8  # features: ["close","RSI","SMA_50","EMA_50","ATR","MACD","Signal","volume"]
#     hidden_dim, num_layers, dropout = 96, 2, 0.3  # defaults to match your checkpoint
#     if os.path.exists(config_path):
#         with open(config_path, "r") as f:
#             cfg = json.load(f)
#         hidden_dim = int(cfg.get("hidden_dim", hidden_dim))
#         num_layers = int(cfg.get("num_layers", num_layers))
#         dropout    = float(cfg.get("dropout", dropout))

#     # build model with the SAME dims as training

   
#     model = SwingLSTM(input_dim=input_dim, hidden_dim=96, num_layers=2, dropout=0.3)
#     model.load_state_dict(torch.load(model_path, map_location="cpu"))
#     model.eval()
#     # model = SwingLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
#     # state = torch.load(model_path, map_location="cpu")
#     # model.load_state_dict(state)  # will succeed if dims match
#     # model.eval()

#     # ... (rest of your strategy logic unchanged)

def run_swing_lstm_real_strategy(
    budget=10000,
    trade_size=2000,
    atr_multiplier=2.0,
    profit_target=0.05,
    prob_threshold=0.55,
    model_path="models/swing_lstm.pt",
    scaler_path="models/swing_scaler.pkl",
    config_path="models/swing_lstm_config.json"
):
    try:
        # graceful skip if artifacts missing
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return {
                "trades": [],
                "final_value_usd": float(budget),
                "portfolio_usd": float(budget),
                "portfolio_btc": 0.0,
                "note": "PyTorch model/scaler not found"
            }

        # load scaler
        scaler = joblib.load(scaler_path)

        # infer or load config
        input_dim = 8
        hidden_dim, num_layers, dropout = 96, 2, 0.3
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                cfg = json.load(f)
            hidden_dim = int(cfg.get("hidden_dim", hidden_dim))
            num_layers = int(cfg.get("num_layers", num_layers))
            dropout    = float(cfg.get("dropout", dropout))

        # build + load model
        model = SwingLSTM(input_dim=input_dim,
                          hidden_dim=hidden_dim,
                          num_layers=num_layers,
                          dropout=dropout)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        # TODO: add actual trading logic here (placeholder for now)
        return {
            "trades": [],
            "final_value_usd": float(budget),
            "portfolio_usd": float(budget),
            "portfolio_btc": 0.0,
            "note": "Model loaded but strategy logic not yet implemented"
        }

    except Exception as e:
        # 🔒 Always return a dict
        return {
            "trades": [],
            "final_value_usd": float(budget),
            "portfolio_usd": float(budget),
            "portfolio_btc": 0.0,
            "note": f"Error loading model: {str(e)}"
        }