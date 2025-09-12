# BITCOIN TRADING AGENT

## CONTEXT

The client is a fintech company focused on crypto trading and wants to build a smart bitcoin trading system designed to operate with minimal human supervision and continuously adapt to changing market conditions. 

The agent must dynamically manage budget allocation, shift between strategies, and make autonomous trading decisions while running 24/7. The BTC Agent takes into account different trading strategies and make decisions accordingly. Following strategies are to be tested:

- **Dollar Cost Averaging (DCA)** - For the *Value Investing* Strategy - Invest fix amounts at regular intervals 
- **Average True Range (ATR) Stop-Loss** - For the *Day Trading* Strategy - Trade at high frequency - Stop Loss when BTC drops below a certain threshold
- **LSTM Neural Network** - For the *Swing Trading* - Trade over time to maximize the profit at a BUY-SELL action

## METHODOLOGY FOR THE LSTM SWING TRADING 

<img width="1666" height="582" alt="Apziva_P5_Methodology" src="https://github.com/user-attachments/assets/d92bbf09-98ac-4b08-aa6b-eed54ea3a32b" />

## DATA DESCRIPTION 

The BTC data is fetched live in OHLCV (Open-High-Low-Close-Volume) from two public sources:
- Binance
- Yahoo Finance

Binance is the primary data source and if Binance does fetch live data due to any restriction, the agent fetches from Yahoo Finance. 

Each row in a BTC data is called a "Candle". So, we say the agent fetches live BTC Candles. 

### COLUMNS

The BTC Candles has five standard features:
- **Open** - opening price of the candle at the start of the given day
- **High** - highest price of the candle throughout the given day
- **Low** - lowest price of the candle throughout the given day
- **Close** - closing price of the candle at the end of the given day 
- **Volume** - total volumne or amount of BTC that was traded for the given day

Furthermore, five techincal indicators are added as calculated features:
- **SMA** - *Simple Moving Average* - Moving average based on the closing price over a rolling window of 14 days 
- **ATR** - *Average True Range* - average price range of a candle over 14 days - *True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))*
- **RSI** - *Relative Strength Index* - measure market momentum by comparing the speed and magnitude of recent price changes over 14 days
- **EMA** - *Exponential Moving Average* - a technical indicator that assigns greater weight to more recent price and help identify entry-exit points in a trade
- **MACD** - *Moving average convergence/divergence (MACD)* - momentum indicator that shows the relationship between two moving averages of a candle's price


## PROJECT GOALS

- Build a BTC Trading Agent that fetches live BTC data ands runs the three strategies on the candles.
- Analyze the Equity Trends, determine BUY-SELL markers and compares the performance of the three strategies based on standardized metrics.
- Ideally, the devised strategy, either the ATR Stop-Loss or the Swing Trading with LSTM should perform better than the DCA which is our baseline strategy.
- Deploy a LLM, that takes these strategies as context and generates trading recommendations to prompts accordingly.
- Compile a daily BTC log of 3-5 days and send it as an alert (additional goal).


## FEATURE ENGINEERING RESULTS

<img width="1488" height="1108" alt="Apziva_P5_Correlation_Map" src="https://github.com/user-attachments/assets/6dfa0434-33ba-4eaa-bf25-0dfa0ac43a3d" />

<img width="1276" height="702" alt="Apziva_P5_MI" src="https://github.com/user-attachments/assets/74cece44-82e9-4d54-a81b-e2e6b8d2da6e" />

<img width="1374" height="674" alt="Apziva_P5_L1_Reg" src="https://github.com/user-attachments/assets/77ce268f-22d5-4e44-a54b-8393440ea821" />


