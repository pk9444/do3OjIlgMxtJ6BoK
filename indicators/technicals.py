import pandas as pd

def compute_atr(df, period=14):
    """
    Compute Average True Range (ATR).
    ATR = SMA(True Range, period)
    True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    """
    df = df.copy()
    df["prev_close"] = df["close"].shift(1)

    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = (df["high"] - df["prev_close"]).abs()
    df["tr3"] = (df["low"] - df["prev_close"]).abs()

    df["true_range"] = df[["tr1","tr2","tr3"]].max(axis=1)
    df["ATR"] = df["true_range"].rolling(window=period).mean()

    return df

def compute_rsi(df, period=14):
    """Relative Strength Index (RSI)"""
    df = df.copy()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def compute_sma(df, period=20):
    """Simple Moving Average"""
    df = df.copy()
    df[f"SMA_{period}"] = df["close"].rolling(window=period).mean()
    return df

def compute_ema(df, period=20):
    """Exponential Moving Average"""
    df = df.copy()
    df[f"EMA_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df

def compute_macd(df, fast=12, slow=26, signal=9):
    """
    Add MACD and Signal line to df.
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal)
    """
    df["EMA_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["EMA_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df