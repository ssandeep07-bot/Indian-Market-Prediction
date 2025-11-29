import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from utils.data_manager import get_historical_data, get_nifty50_tickers
from utils.algorithm_engine import run_short_term_algo, run_long_term_algo, add_indicators
import numpy as np

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Personal Stock Recommender")

# --- Filters and Settings ---
st.sidebar.header("Filters & Settings")
holding_period = st.sidebar.radio(
    "Select Holding Period:",
    ('Short-Term (5-Day Forecast)', 'Long-Term (QVAL Proxy)'), # Mid-Term removed for clarity
    index=0
)
st.sidebar.markdown(f"**Next Refresh:** {datetime.now().strftime('%d %b, 4:30 PM IST')} (Daily EOD)")

# --- Data Fetching (runs once due to caching) ---
TICKER_LIST = get_nifty50_tickers()
@st.cache_data(show_spinner="Loading and backtesting 5+ years of data from Finnhub...")
def get_backtest_data():
    return get_historical_data(TICKER_LIST, period='5y')

full_data = get_backtest_data()

# --- Main Dashboard Logic ---
st.title("🧠 Personal Quant Recommendation System")
st.markdown(f"**Recommendations as of:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
st.markdown("---")

if full_data.empty:
    st.error("Cannot load market data. Please check Finnhub Key and API limits.")
    st.stop()

# --- 1. Algorithm Run and Recommendation Table ---
recommendations = []
unique_tickers = full_data['Ticker'].unique()

progress_text = f"Running {holding_period} Algorithm on {len(unique_tickers)} stocks..."
my_bar = st.progress(0, text=progress_text)

for i, ticker in enumerate(unique_tickers):
    my_bar.progress((i + 1) / len(unique_tickers), text=f"Processing {ticker}...")
    
    df_ticker = full_data[full_data['Ticker'] == ticker].copy()
    
    # Run the model based on sidebar selection
    if holding_period == 'Short-Term (5-Day Forecast)':
        signal, prescription, rationale = run_short_term_algo(df_ticker)
    elif holding_period == 'Long-Term (QVAL Proxy)':
        signal, prescription, rationale = run_long_term_algo(df_ticker)
    else:
        signal, prescription, rationale = "N/A", "Model not implemented.", "N/A"
        
    recommendations.append({
        "Symbol": ticker, 
        "Signal Strength": signal, 
        "Prescription (T/SL)": prescription,
        "Rationale/Why": rationale,
        "Refresh": "Daily EOD"
    })
my_bar.empty()

recommendations_df = pd.DataFrame(recommendations)
st.subheader(f"🎯 Today's {holding_period} Watchlist")

# Apply color coding
def color_signals(val):
    if "BUY" in val: return 'background-color: #d4edda; color: #155724'
    elif "SELL" in val: return 'background-color: #f8d7da; color: #721c24'
    else: return 'background-color: #ffeeba; color: #856404'

st.dataframe(
    recommendations_df.style.applymap(color_signals, subset=['Signal Strength']),
    hide_index=True,
    use_container_width=True
)

st.markdown("---")

# --- 2. Detailed Analysis & Backtesting Audit ---
st.subheader("🧪 Backtesting Audit & Detailed Analysis")
selected_ticker = st.selectbox(
    "Select a stock to view model performance and chart:",
    options=unique_tickers
)

if selected_ticker:
    df_ticker_analysis = full_data[full_data['Ticker'] == selected_ticker].copy()
    
    # Determine which function to use for backtesting based on the selected period
    if holding_period == 'Short-Term (5-Day Forecast)':
        kpis = run_short_term_algo(df_ticker_analysis, backtest=True)
    elif holding_period == 'Long-Term (QVAL Proxy)':
        kpis = run_long_term_algo(df_ticker_analysis, backtest=True)
    else:
        kpis = {"Sharpe Ratio": "N/A", "Max Drawdown": "N/A", "CAGR": "N/A", "Win Rate": "N/A", "Total Trades": "N/A"}


    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Sharpe Ratio", kpis.get("Sharpe Ratio", "N/A"), "Goal: > 1.0")
    col2.metric("Max Drawdown", kpis.get("Max Drawdown", "N/A"), "Goal: < 20%")
    col3.metric("CAGR", kpis.get("CAGR", "N/A"), "Goal: > Benchmark")
    col4.metric("Win Rate", kpis.get("Win Rate", "N/A"), "Total Trades: " + str(kpis.get("Total Trades", 0)))
    
    # --- Candlestick Chart with Historical Signals ---
    st.markdown(f"### Historical Signals for {selected_ticker}")
    
    # Use the robust add_indicators function to prepare the chart data
    df_chart = add_indicators(df_ticker_analysis.copy())
    
    # Rerun the signal logic to get the 'Final_Signal' column for charting (required since main algo returns tuple)
    df_chart['Final_Signal'] = 0 
    if not df_chart.empty:
        # Re-run indicator-based signal logic
        buy_condition = (df_chart['Close'] > df_chart['EMA50']) & (df_chart['MACD_Line'] > df_chart['MACD_Signal']) & (df_chart['RSI'] < 70)
        sell_condition = (df_chart['Close'] < df_chart['EMA50']) & (df_chart['MACD_Line'] < df_chart['MACD_Signal'])
        df_chart.loc[buy_condition, 'Final_Signal'] = 1
        df_chart.loc[sell_condition, 'Final_Signal'] = -1
    
    # Only use the last 252 days (1 year) for chart clarity
    df_chart = df_chart.iloc[-252:].reset_index()
    
    fig = go.Figure(data=[go.Candlestick(x=df_chart['Date'],
                                         open=df_chart['Open'], # Use Open/Close from Finnhub data
                                         high=df_chart['High'],
                                         low=df_chart['Low'],
                                         close=df_chart['Close'],
                                         name='Price')])

    buy_points = df_chart[df_chart['Final_Signal'] == 1.0]
    sell_points = df_chart[df_chart['Final_Signal'] == -1.0]

    fig.add_trace(go.Scatter(x=buy_points['Date'], y=buy_points['Close'] * 0.99,
                             mode='markers', marker_symbol='triangle-up',
                             marker=dict(size=10, color='green'), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_points['Date'], y=sell_points['Close'] * 1.01,
                             mode='markers', marker_symbol='triangle-down',
                             marker=dict(size=10, color='red'), name='Sell Signal'))

    fig.update_layout(xaxis_rangeslider_visible=False, height=500, title=f"{selected_ticker} Signals (Last 1 Year)")
    st.plotly_chart(fig, use_container_width=True)
    

