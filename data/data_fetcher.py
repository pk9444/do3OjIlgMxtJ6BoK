import requests
import pandas as pd
import yfinance as yf


BINANCE_URL = "https://api.binance.com/api/v3/klines"

def get_binance_ohlcv(symbol="BTCUSDT", interval="1d", limit=500):
    """
    Fetch OHLCV historical data from Binance.
    
    :param symbol: Trading pair (default BTC/USDT)
    :param interval: Candle interval (1m, 5m, 1h, 1d, etc.)
    :param limit: Number of candles (max 1000)
    :return: DataFrame with columns: [time, open, high, low, close, volume]
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(BINANCE_URL, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # Clean up
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)

    return df[["timestamp","open","high","low","close","volume"]]




def get_yf_ohlcv(symbol="BTC-USD", interval="4h", limit=1000):
    """
    Fetch OHLCV data from Yahoo Finance and make it consistent with Binance format.
    Returns:
        DataFrame with ['timestamp','open','high','low','close','volume']
    """
    # Fetch 1h data (since yfinance doesn't support 4h directly)
    df = yf.download(symbol, interval="1h", period="60d", progress=False)

    if df.empty:
        raise ValueError("No data fetched from Yahoo Finance")

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)  # keep only 'Open','High',...
    
    # Resample to 4h bars
    df = df.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    # Standardize column names like Binance
    df = df.dropna().reset_index()
    df.rename(columns={
        "Datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)

    return df


# --- Unified fetcher (NEW) ---
def get_ohlcv(symbol_binance="BTCUSDT", interval="4h", limit=1000,
              symbol_yf="BTC-USD"):
    """
    Try Binance first. If it fails, fallback to Yahoo Finance.
    Returns DataFrame with ['timestamp','open','high','low','close','volume']
    """
    try:
        print(f"Fetching {symbol_binance} from Binance...")
        return get_binance_ohlcv(symbol_binance, interval, limit)
    except Exception as e:
        print(" Binance fetch failed:", e)
        print("Fetching from Yahoo Finance instead...")
        return get_yf_ohlcv(symbol_yf, interval, limit)