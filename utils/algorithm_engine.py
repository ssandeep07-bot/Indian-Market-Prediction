import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import timedelta
import streamlit as st # Keep st import for debugging/logging

# --- PREDICTION AND PRESCRIPTION CORE ---

def run_short_term_algo(df_ticker, backtest=False):
    """
    Short-Term (5-Day) Strategy: Triple-Confirmation Model.
    
    Args:
        df_ticker (pd.DataFrame): Historical data for one stock.
        backtest (bool): If True, returns KPIs. If False, returns latest signal.
    """
    # ----------------------------------------------------
    # FIX: Robustness Check - Ensure 'Close' column exists and DataFrame is large enough
    # ----------------------------------------------------
    if 'Close' not in df_ticker.columns:
        st.error(f"Algorithm Error: Input DataFrame is missing the 'Close' column.")
        return calculate_kpis(df_ticker) if backtest else ("HOLD", "Data Error: Missing Close Price.", "N/A")
    
    # Requires 200 days for 50-day EMA and clean calculations
    if df_ticker.empty or len(df_ticker) < 200:
        return calculate_kpis(df_ticker) if backtest else ("HOLD", "Insufficient data (<200 days).", "N/A")

    df = df_ticker.copy().set_index('Date').sort_index()

    # --- 1. Calculate Technical Indicators (using pandas-ta) ---
    
    # EMA50 (Trend Filter)
    df.ta.ema(close='Close', length=50, append=True)
    df['EMA50'] = df['EMA_50']
    
    # MACD (fast=12, slow=26, signal=9)
    # The FIX relies on the standard naming convention: MACD_12_26_9 and MACDS_12_26_9
    df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    df['MACD_Line'] = df['MACD_12_26_9']      
    df['MACD_Signal'] = df['MACDS_12_26_9'] 
    
    # RSI (Condition)
    # The error occurred here, likely because 'Close' was missing earlier
    df.ta.rsi(close='Close', length=14, append=True) 
    df['RSI'] = df['RSI_14']

    # --- 2. Generate Signals based on Triple-Confirmation (Logic remains the same) ---
    df['Trend_Filter'] = df['Close'] > df['EMA50']
    
    df['Momentum_Signal'] = np.where(df['MACD_Line'] > df['MACD_Signal'], 1.0, 0.0)
    df['Momentum_Signal'] = np.where(df['MACD_Line'] < df['MACD_Signal'], -1.0, df['Momentum_Signal'])

    buy_condition = (df['Trend_Filter'] == True) & (df['Momentum_Signal'] == 1.0) & (df['RSI'] < 70)
    sell_condition = (df['Trend_Filter'] == False) & (df['Momentum_Signal'] == -1.0)
    
    df['Final_Signal'] = 0
    df.loc[buy_condition, 'Final_Signal'] = 1
    df.loc[sell_condition, 'Final_Signal'] = -1
    
    # --- Output (Prediction/Prescription) ---
    if backtest:
        return calculate_kpis(df)
    
    # Latest Prediction (Prescription) - Use the last valid row
    last_row = df.iloc[-1]
    latest_signal = last_row['Final_Signal']
    latest_price = last_row['Close']
    
    if latest_signal == 1:
        target = latest_price * 1.03
        stop_loss = latest_price * 0.98
        rationale = "Trend UP (C > EMA50), MACD Buy Crossover, RSI OK (<70)."
        return "STRONG BUY", f"Entry: {latest_price:.2f} | T1: {target:.2f} | SL: {stop_loss:.2f}", rationale
    elif latest_signal == -1:
        target = latest_price * 0.97
        stop_loss = latest_price * 1.02
        rationale = "Trend DOWN (C < EMA50), MACD Sell Crossover."
        return "STRONG SELL", f"Exit: {latest_price:.2f} | T1: {target:.2f} | SL: {stop_loss:.2f}", rationale
    else:
        rationale = "Trend/Momentum Conflict (Neutral)."
        return "HOLD", "Monitor next refresh.", rationale

# (The calculate_kpis function remains the same)
def calculate_kpis(df_signals):
    """
    Calculates Backtesting Key Performance Metrics (KPMs)
    Simulates a 5-day holding period.
    """
    if 'Final_Signal' not in df_signals.columns or df_signals.empty:
        return {"Sharpe Ratio": "0.00", "Max Drawdown": "0.00%", "CAGR": "0.00%", "Win Rate": "0.0%", "Total Trades": 0}
        
    df_signals['Daily_Return'] = df_signals['Close'].pct_change()
    df_signals['Hold_Return'] = df_signals['Daily_Return'].shift(-5).rolling(5).sum()
    df_signals['Trade_Return'] = df_signals['Hold_Return'] * df_signals['Final_Signal'].shift(1)
    
    completed_trades = df_signals[df_signals['Final_Signal'] != 0].dropna(subset=['Trade_Return'])

    if completed_trades.empty:
        return {"Sharpe Ratio": "0.00", "Max Drawdown": "0.00%", "CAGR": "0.00%", "Win Rate": "0.0%", "Total Trades": 0}

    total_return = (completed_trades['Trade_Return'].fillna(0) + 1).prod() - 1
    cumulative_returns = (completed_trades['Trade_Return'].fillna(0) + 1).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    cagr = ((1 + total_return)**(252 / len(completed_trades.index))) - 1
    annual_volatility = completed_trades['Trade_Return'].std() * np.sqrt(252)
    risk_free_rate = 0.06
    sharpe_ratio = (cagr - risk_free_rate) / annual_volatility
    
    win_rate = completed_trades[completed_trades['Trade_Return'] > 0].shape[0] / completed_trades.shape[0]
    
    kpis = {
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown * 100:.2f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Win Rate": f"{win_rate * 100:.1f}%",
        "Total Trades": completed_trades.shape[0]
    }
    return kpis
