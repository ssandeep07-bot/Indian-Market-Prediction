import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st

# --- Helper Function to Process Indicators Robustly ---
def add_indicators(df):
    """Adds necessary indicators to the DataFrame using pandas-ta."""
    if df.empty or 'Close' not in df.columns:
        return df

    # Set index for reliable ta calculations
    df = df.set_index('Date').sort_index().copy()

    try:
        # EMA50 (Trend Filter)
        df.ta.ema(close='Close', length=50, append=True)
        df['EMA50'] = df['EMA_50']
        
        # MACD (Momentum)
        df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
        # FIX: Explicit check and fallback for MACD column names
        if 'MACD_12_26_9' in df.columns and 'MACDS_12_26_9' in df.columns:
            df['MACD_Line'] = df['MACD_12_26_9']      
            df['MACD_Signal'] = df['MACDS_12_26_9']
        else:
            st.warning("MACD columns not found. Skipping MACD signals for this stock.")
            df['MACD_Line'] = np.nan
            df['MACD_Signal'] = np.nan
        
        # RSI (Condition)
        df.ta.rsi(close='Close', length=14, append=True)
        df['RSI'] = df['RSI_14']
        
        # Drop initial NaN values created by indicators (e.g., first 50 days of EMA50)
        df.dropna(inplace=True)
        
    except Exception as e:
        st.error(f"Error during indicator calculation: {e}")
        return df.reset_index() # Return partial DF
        
    return df.reset_index()


# --- PREDICTION AND PRESCRIPTION CORE ---

def run_short_term_algo(df_ticker, backtest=False):
    """
    Short-Term (5-Day) Strategy: Triple-Confirmation Model.
    """
    if df_ticker.empty or len(df_ticker) < 200:
        return calculate_kpis(df_ticker) if backtest else ("HOLD", "Insufficient data (<200 days).", "N/A")

    df = add_indicators(df_ticker)
    
    # Must re-check length after dropping NaNs
    if df.empty or len(df) < 50:
        return calculate_kpis(df) if backtest else ("HOLD", "Data too short after cleaning.", "N/A")

    # --- 2. Generate Signals based on Triple-Confirmation ---
    df['Trend_Filter'] = df['Close'] > df['EMA50']
    
    # Check for MACD signal availability before using it
    if 'MACD_Line' not in df.columns or df['MACD_Line'].isnull().all():
        return calculate_kpis(df) if backtest else ("HOLD", "MACD calculation failed.", "N/A")

    # Momentum Signal: MACD Line > MACD Signal line
    df['Momentum_Signal'] = np.where(df['MACD_Line'] > df['MACD_Signal'], 1.0, 0.0)
    df['Momentum_Signal'] = np.where(df['MACD_Line'] < df['MACD_Signal'], -1.0, df['Momentum_Signal'])

    # Final BUY condition: Trend UP AND Momentum BUY AND RSI not Overbought (<70)
    buy_condition = (df['Trend_Filter'] == True) & (df['Momentum_Signal'] == 1.0) & (df['RSI'] < 70)
    
    # Final SELL condition: Trend DOWN AND Momentum SELL
    sell_condition = (df['Trend_Filter'] == False) & (df['Momentum_Signal'] == -1.0)
    
    df['Final_Signal'] = 0
    df.loc[buy_condition, 'Final_Signal'] = 1
    df.loc[sell_condition, 'Final_Signal'] = -1
    
    # --- Output ---
    if backtest:
        return calculate_kpis(df)
    
    # Latest Prediction (Prescription) - Use the last valid row
    last_row = df.iloc[-1]
    latest_signal = last_row['Final_Signal']
    latest_price = last_row['Close']
    
    # --- PRESCRIPTION ---
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

def calculate_kpis(df_signals):
    """
    Calculates Backtesting Key Performance Metrics (KPMs)
    Simulates a 5-day holding period.
    """
    # ... (KPI calculation logic remains the same, but relies on robust data) ...
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
