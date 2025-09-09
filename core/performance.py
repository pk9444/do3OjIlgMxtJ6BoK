import numpy as np
import pandas as pd

"""
compute_performance_metrics() : function to compute and compile the performing metrics of each strategy
@params : trades - array containing the trades made by the strategy 
          initial budget - set by the user like 10K, 100K etc 
          final_value - final value of the BTC after a trade is made (Up or Down)
 @return : an output dictionary containing essential performance metrics  
"""
def compute_performance_metrics(trades, initial_budget, final_value):

    # Convert trades into PnL series
    pnl_list = []
    win_trades, loss_trades = [], []
    
    for t in trades:
        if "SELL" in t["action"]:
            # Profit/Loss for that trade
            # Approximation: PnL = (sell_price - buy_price) * qty
            # Need matching BUY for exact calc, here we just use budget changes
            pnl = t["budget_left"] - initial_budget  # running balance
            pnl_list.append(pnl)

            if t["action"] == "SELL_TARGET":
                win_trades.append(pnl)
            elif t["action"] == "SELL_STOPLOSS":
                loss_trades.append(pnl)

    # --- Basic Metrics ---
    total_return = (final_value - initial_budget) / initial_budget
    total_trades = len([t for t in trades if "BUY" in t["action"]])
    
    # Max Drawdown
    equity_curve = [initial_budget]
    running_value = initial_budget
    for t in trades:
        if "budget_left" in t:
            running_value = t["budget_left"]
            equity_curve.append(running_value)
    equity_curve = pd.Series(equity_curve)
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio (assuming daily-like returns, no risk-free rate)
    if len(pnl_list) > 1:
        returns = np.diff(pnl_list) / initial_budget
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe = 0
    
    # Win Rate
    total_sells = len(win_trades) + len(loss_trades)
    win_rate = (len(win_trades) / total_sells) * 100 if total_sells > 0 else 0
    
    # Profit Factor
    gross_profit = sum([max(0, x) for x in pnl_list])
    gross_loss = abs(sum([min(0, x) for x in pnl_list]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    return {
        "Final Value": round(final_value, 2),
        "PnL %": round(total_return * 100, 2),
        "Trades": total_trades,
        "Max Drawdown %": round(max_drawdown * 100, 2),
        "Sharpe": round(sharpe, 2),
        "Win Rate %": round(win_rate, 2),
        "Profit Factor": round(profit_factor, 2) if profit_factor != np.inf else "∞"
    }
