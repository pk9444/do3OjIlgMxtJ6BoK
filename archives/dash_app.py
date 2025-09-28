import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import os
from openai import OpenAI
from dotenv import load_dotenv
import datetime
# --- import your strategy + utils ---
from strategies.dca import run_dca_strategy
from strategies.atr_daytrading import run_atr_daytrading_strategy
from strategies.swing_lstm_test import run_swing_lstm_strategy        # placeholder LSTM
from strategies.swing_lstm_reg import run_swing_lstm_reg_strategy     # regression LSTM
from data.data_fetcher import get_ohlcv
from core.equity import equity_curve_from_trades, merge_equity_curves
from core.performance import compute_performance_metrics


#  Set your API key (better: load from env var)
load_dotenv()

x = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = x
client = OpenAI(api_key=x)


# ---------- Services ----------
def run_all_strategies(budget: float, n_steps=7, freq="1D"):
    """Run all 4 strategies and prepare equity/metrics/trades."""

    dca        = run_dca_strategy(budget=budget, buy_amount=500, drop_trigger=0.03)
    atr        = run_atr_daytrading_strategy(budget=budget, trade_size=budget * 0.10)
    swing_ph   = run_swing_lstm_strategy(budget=budget, trade_size=budget * 0.20)
    swing_reg  = run_swing_lstm_reg_strategy(budget=budget, trade_size=budget * 0.20, n_steps=n_steps, freq=freq)

    px_dca   = get_ohlcv("BTCUSDT", "1d", 500)
    px_atr   = get_ohlcv("BTCUSDT", "30m", 1000)
    px_swing = get_ohlcv("BTCUSDT", "4h", 1000)

    ec_dca   = equity_curve_from_trades(dca["trades"],        px_dca,   budget)
    ec_atr   = equity_curve_from_trades(atr["trades"],        px_atr,   budget)
    ec_sph   = equity_curve_from_trades(swing_ph["trades"],   px_swing, budget)
    ec_sreg  = equity_curve_from_trades(swing_reg["trades"],  px_swing, budget)

    merged = merge_equity_curves({
        "DCA":                       ec_dca,
        "ATR Stop-Loss":             ec_atr,
        "Swing (LSTM placeholder)":  ec_sph,
        "Swing (LSTM regression)":   ec_sreg
    })

    dca_m   = compute_performance_metrics(dca["trades"],        budget, dca["final_value_usd"])
    atr_m   = compute_performance_metrics(atr["trades"],        budget, atr["final_value_usd"])
    sph_m   = compute_performance_metrics(swing_ph["trades"],   budget, swing_ph["final_value_usd"])
    sreg_m  = compute_performance_metrics(swing_reg["trades"],  budget, swing_reg["final_value_usd"])

    metrics_df = pd.DataFrame(
        [dca_m, atr_m, sph_m, sreg_m],
        index=["DCA", "ATR Stop-Loss", "Swing (LSTM placeholder)", "Swing (LSTM regression)"]
    ).reset_index().rename(columns={"index": "Strategy"})

    trades_rows = []
    for strat_name, res in [
        ("DCA", dca),
        ("ATR", atr),
        ("Swing (placeholder)", swing_ph),
        ("Swing (regression)", swing_reg),
    ]:
        for t in res["trades"][-20:]:
            trades_rows.append({
                "Strategy": strat_name,
                "Date": str(t.get("date")),
                "Action": t.get("action"),
                "Price": (round(float(t["price"]), 2) if t.get("price") is not None else None),
                "Qty": (round(float(t["qty"]), 6) if t.get("qty") is not None else None),
                "Budget Left": (round(float(t["budget_left"]), 2) if t.get("budget_left") is not None else None),
                "Note": t.get("note") if "note" in t else None
            })
    trades_df = pd.DataFrame(trades_rows)

    return merged, trades_df, metrics_df, px_dca, dca, atr, swing_ph, swing_reg


# ---------- App ----------
external_stylesheets = [{
    "href": "https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap",
    "rel": "stylesheet"
}]

app = dash.Dash(__name__, 
            external_stylesheets=external_stylesheets,
            assets_folder="assets",
            suppress_callback_exceptions=True,
            )
app.title = "BTC Trading Agent"
app._favicon = "logo.png" 

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                margin: 0;
                padding: 0;
                background-color: #121212;
                font-family: 'Roboto', sans-serif;
            }

            /* Slim, styled scrollbar for chat history */
                #chat-history::-webkit-scrollbar {
                    width: 6px;   /* slimmer */
                }
                #chat-history::-webkit-scrollbar-track {
                    background: #1e1e1e;  /* matches panel */
                }
                #chat-history::-webkit-scrollbar-thumb {
                    background-color: #555; 
                    border-radius: 3px;
                }
                #chat-history::-webkit-scrollbar-thumb:hover {
                    background-color: #888;
                }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


# ---------- Layout ----------
app.layout = html.Div([

    
    html.H1("BTC Trading Agent", style={
        "textAlign": "center",
        "marginBottom": "20px"
    }),

    # Controls
    html.Div([
        html.Label("Budget (USD): ", style={"marginRight": "10px"}),
        dcc.Input(
            id="budget-input",
            type="number",
            value=10000,
            step=1000,
            style={"marginRight": "10px", "width": "150px", "padding": "5px"}
        ),

            html.Label("Forecast Steps:", style={"marginRight": "10px"}),
                dcc.Input(
                    id="steps-input", type="number", value=7, step=1,
                    style={"marginRight": "20px", "width":"80px"}
                ),

                html.Label("Frequency:", style={"marginRight": "10px"}),
                dcc.Input(
                    id="freq-input", type="text", value="1D",
                    style={"marginRight": "20px", "width":"80px"}
                ),
                
        html.Button(
            "Run Strategies",
            id="run-btn",
            n_clicks=0,
            style={
                "background": "#2e7d32",
                "color": "white",
                "border": "none",
                "padding": "8px 15px",
                "borderRadius": "8px",
                "cursor": "pointer"
            }
        )
    ], style={"margin": "20px", "textAlign": "center"}),

    dcc.Loading(
        id="loading-output",
        type="default",
        children=[

            html.Div([dcc.Graph(id="price-graph")], style={"margin": "20px"}),

            html.Div([dcc.Graph(id="equity-graph")], style={"margin": "20px"}),

            html.Div([
                html.H3("Recent Trades", style={"marginBottom": "10px"}),
                dash_table.DataTable(
                    id="trades-table",
                    columns=[{"name": c, "id": c} for c in
                             ["Strategy", "Date", "Action", "Price", "Qty", "Budget Left", "Note"]],
                    page_size=10,
                    style_table={"overflowX": "auto", "padding": "10px", "borderRadius": "10px"},
                    style_header={"backgroundColor": "#212121", "color": "white", "fontWeight": "bold",
                                  "border": "1px solid #333"},
                    style_cell={"backgroundColor": "#2c2c2c", "color": "white", "padding": "6px",
                                "fontSize": 13, "border": "1px solid #333"},
                    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#1e1e1e"}]
                )
            ], style={"margin": "60px"}),

            html.Div([
                html.H3("Performance Metrics", style={"marginBottom": "10px"}),
                dash_table.DataTable(
                    id="metrics-table",
                    style_table={"overflowX": "auto", "padding": "10px", "borderRadius": "10px"},
                    style_header={"backgroundColor": "#212121", "color": "white", "fontWeight": "bold",
                                  "border": "1px solid #333"},
                    style_cell={"backgroundColor": "#2c2c2c", "color": "white", "padding": "6px",
                                "fontSize": 13, "border": "1px solid #333"},
                    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#1e1e1e"}]
                )
            ], style={"margin": "60px"})
        ]
    ),

    # ---------- Chat Agent (collapsible, styled like ChatGPT) ----------
    html.Div([
        html.Button(
            "« 💬 Chat With Robin",
            id="chat-toggle",
            n_clicks=0,
            style={
                "position": "absolute", "top": "20px", "left": "-160px",
                "width": "150px", "height": "50px",
                "borderRadius": "8px", "border": "none",
                "background": "#2e7d32", "color": "white", "fontWeight": "bold",
                "cursor": "pointer", "boxShadow": "0 0 5px rgba(0,0,0,0.4)"
            }
        ),
        html.Div([
            html.H4(["Hi I am Robin! Your real-time BTC trading agent!", html.Br(), "How can I help you today?"], 
                    style={"marginTop": "0"}),

            # Scrollable chat history
            html.Div(id="chat-history", style={
                "whiteSpace": "pre-line", 
                "flex": "1",
                "overflowY": "auto",
                "padding": "10px",
                "backgroundColor": "#2c2c2c",
                "borderRadius": "8px",
                "marginBottom": "10px",
                "maxHeight": "80vh"
            }),

            # Input row
            html.Div([
                dcc.Input(
                    id="chat-input",
                    type="text",
                     n_submit=0,
                    placeholder="Ask me something...",
                    style={"flex": "1", "padding": "8px", "borderRadius": "5px", "border": "none"}
                ),
                html.Button(
                    "Send", id="chat-send", n_clicks=0,
                    style={
                        "marginLeft":"5px","background":"#2e7d32","color":"white",
                        "border":"none","padding":"8px 15px","borderRadius":"5px",
                        "cursor":"pointer"
                    }
                )
            ], style={"display":"flex", "marginBottom":"10px",  "marginTop": "10px"})
        ], id="chat-box", style={
            "height": "100%", "width": "500px",
            "backgroundColor": "#1e1e1e", "padding": "10px",
            "borderRadius": "0 0 0 10px", "color":"white",
            "boxShadow":"-2px 0 10px rgba(0,0,0,0.5)",
            "display": "none",
            "display": "flex", "flexDirection": "column"
        })
    ], style={
        "position": "fixed", "top": "0", "right": "0",
        "height": "100%", "zIndex": "1000",
        "display": "flex", "alignItems": "flex-start"
    })
], style={
    "backgroundColor": "#121212",
    "color": "white",
    "fontFamily": "Roboto, sans-serif",
    "minHeight": "100vh"
})


# ---------- Callbacks ----------
@app.callback(
    [
        Output("price-graph", "figure"),
        Output("equity-graph", "figure"),
        Output("trades-table", "data"),
        Output("metrics-table", "columns"),
        Output("metrics-table", "data")
    ],
    [Input("run-btn", "n_clicks")],
    [
        State("budget-input", "value"),
        State("steps-input", "value"),
        State("freq-input", "value")
     
     ],
    
)
def update_dashboard(n_clicks, budget, n_steps, freq):
    if n_clicks == 0:
        return go.Figure(), go.Figure(), [], [], []

    merged, trades_df, metrics_df, px_dca, dca, atr, swing_ph, swing_reg = run_all_strategies(budget, n_steps=n_steps, freq=freq)

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=px_dca["timestamp"], y=px_dca["close"],
        mode="lines", name="BTC Price", line=dict(color="#00cc96")
    ))

    # === Add forecast (if available) ===
    if swing_reg.get("forecast") is not None:
        forecast_df = swing_reg["forecast"]
        fig_price.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["predicted"],
            mode="lines+markers",
            name="LSTM Forecast",
            line=dict(dash="dot", color="yellow")
        ))

    for strat, res, color in [
        ("DCA", dca, "#1f77b4"),
        ("ATR", atr, "#ff7f0e"),
        ("Swing (placeholder)", swing_ph, "#d62728"),
        ("Swing (regression)", swing_reg, "#9467bd"),
    ]:
        buys = [t for t in res["trades"] if t.get("action") and "BUY" in t["action"]]
        sells = [t for t in res["trades"] if t.get("action") and "SELL" in t["action"]]
        fig_price.add_trace(go.Scatter(
            x=[t.get("date") for t in buys],
            y=[t.get("price") for t in buys],
            mode="markers",
            name=f"{strat} BUY",
            marker=dict(symbol="triangle-up", color=color, size=9)
        ))
        fig_price.add_trace(go.Scatter(
            x=[t.get("date") for t in sells],
            y=[t.get("price") for t in sells],
            mode="markers",
            name=f"{strat} SELL",
            marker=dict(symbol="triangle-down", color=color, size=9)
        ))

    fig_price.update_layout(
        title="BTC Price with Trade Markers",
        xaxis_title="Time", yaxis_title="Price (USD)",
        template="plotly_dark"
    )

    fig_equity = go.Figure()
    for col in ["DCA", "ATR Stop-Loss", "Swing (LSTM placeholder)", "Swing (LSTM regression)"]:
        if col in merged.columns:
            fig_equity.add_trace(go.Scatter(
                x=merged["timestamp"], y=merged[col],
                mode="lines", name=col
            ))
    fig_equity.update_layout(
        title="Equity Curves",
        xaxis_title="Time", yaxis_title="Portfolio Value (USD)",
        template="plotly_dark"
    )

    trades_data = trades_df.to_dict("records")
    metrics_columns = [{"name": c, "id": c} for c in metrics_df.columns]
    metrics_data = metrics_df.to_dict("records")

    return fig_price, fig_equity, trades_data, metrics_columns, metrics_data


@app.callback(
    Output("chat-history", "children"),
    [Input("chat-send", "n_clicks")],
    [
        State("chat-input", "value"),
        State("chat-history", "children"),
        State("metrics-table", "data"),
        State("trades-table", "data"),
    ],
)
def update_chat(n_clicks, user_msg, history, metrics_data, trades_data):
    if not n_clicks or not user_msg:
        return history

    try:
        # --- Build DataFrames from inputs ---
        mdf = pd.DataFrame(metrics_data) if metrics_data else pd.DataFrame()
        tdf = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()

        # Ensure numeric types for key metrics
        for col in [
            "Final Value", "PnL %", "Sharpe", 
            "Max Drawdown %", "Trades", 
            "Win Rate %", "Profit Factor"
        ]:
            if col in mdf.columns:
                mdf[col] = pd.to_numeric(mdf[col], errors="coerce")

        # --- Extract insights ---
        insights = []
        if not mdf.empty:
            if "PnL %" in mdf.columns and not mdf["PnL %"].isna().all():
                best_pnl_row = mdf.loc[mdf["PnL %"].idxmax()]
                insights.append(
                    f"Best PnL: {best_pnl_row.get('Strategy')} ({best_pnl_row.get('PnL %'):.2f}%)"
                )
            if "Sharpe" in mdf.columns and not mdf["Sharpe"].isna().all():
                best_sharpe_row = mdf.loc[mdf["Sharpe"].idxmax()]
                insights.append(
                    f"Best Sharpe: {best_sharpe_row.get('Strategy')} ({best_sharpe_row.get('Sharpe'):.2f})"
                )
            if "Max Drawdown %" in mdf.columns and not mdf["Max Drawdown %"].isna().all():
                worst_dd_row = mdf.loc[mdf["Max Drawdown %"].idxmin()]
                insights.append(
                    f"Worst drawdown: {worst_dd_row.get('Strategy')} ({worst_dd_row.get('Max Drawdown %'):.2f}%)"
                )
        insights_text = "\n".join(insights) if insights else "No metrics available yet."

        # --- Dates and last trade ---
        today = datetime.date.today().strftime("%Y-%m-%d")
        last_trade = tdf.tail(1).to_string(index=False) if not tdf.empty else "No trades yet."
        recent_trades_text = (
            tdf.tail(8).to_string(index=False) if not tdf.empty else "No trades yet."
        )
        metrics_table_text = mdf.to_string(index=False) if not mdf.empty else "No metrics yet."

        # --- Strategy rules (explicit, so LLM grounds advice) ---
        rules = (
            "• DCA: Buy $500 when price drops ≥3% from the last buy; long-term hold, no fixed sell.\n"
            "• ATR Day Trading: Position ≈10% of budget; stop-loss = entry − 1.5×ATR(14); "
            "target = entry + 2×ATR(14).\n"
            "• Swing LSTM (regression): BUY if model predicts ≥+1% vs current; SELL if ≤−1%; "
            "otherwise hold.\n"
            "• Portfolio safeguard: pause trading if equity drawdown ≥25%.\n"
        )

        # --- Full context for the LLM ---
        context = f"""
Today's date: {today}

Current Strategy Performance (summary):
{insights_text}

Strategy rules:
{rules}

Full performance metrics:
{metrics_table_text}

Most recent trade:
{last_trade}

Recent trades (last 8):
{recent_trades_text}

Instructions:
- Always ground BUY/SELL suggestions in the latest trade and ATR/thresholds.
- Do NOT invent future dates. If the next BUY/SELL date is unknown, give price/ATR conditions instead.
- Interpret "after today" relative to {today}.
"""

        # --- Call OpenAI ---
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a crypto trading assistant. Use the provided rules, metrics, and trades "
                               "to give concrete trading advice. When giving BUY/SELL recommendations, quote "
                               "thresholds (price, ATR, stop-loss/target). Never fabricate dates."
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_msg}"},
            ],
        )
        reply = resp.choices[0].message.content

    except Exception as e:
        reply = f"(Error: {e})"

    # --- Format chat bubbles ---
    user_bubble = html.Div(
        user_msg,
        style={
            "backgroundColor": "#2e7d32",
            "color": "white",
            "padding": "8px 12px",
            "borderRadius": "12px",
            "margin": "5px 0",
            "maxWidth": "80%",
            "marginLeft": "auto",
            "alignSelf": "flex-end",
        },
    )
    bot_bubble = html.Div(
        reply,
        style={
            "backgroundColor": "#333333",
            "color": "white",
            "padding": "8px 12px",
            "borderRadius": "12px",
            "margin": "5px 0",
            "maxWidth": "80%",
            "marginRight": "auto",
            "alignSelf": "flex-start",
        },
    )

    return (history or []) + [user_bubble, bot_bubble]


# ---------- Collapsible Chat Callback ----------
@app.callback(
    Output("chat-box", "style"),
    Input("chat-toggle", "n_clicks"),
    State("chat-box", "style")
)
def toggle_chat(n_clicks, style):
    if n_clicks % 2 == 1:  # collapsed
        return {**style, "display": "block"}
    else:  # open
        return {**style, "display": "none"}


# ---------- Runner ----------
if __name__ == "__main__":
    app.run(debug=True)
