import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st # <-- ADD THIS LINE

# Rest of the code follows...
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
        
        # MACD (fast=12, slow=26, signal=9)
        df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
        # FIX: Access columns using the standard pandas-ta names
        df['MACD_Line'] = df['MACD_12_26_9']      
        df['MACD_Signal'] = df['MACDS_12_26_9']
        
        # RSI (Condition)
        df.ta.rsi(close='Close', length=14, append=True) 
        df['RSI'] = df['RSI_14']
        
        # Drop initial NaN values created by indicators (e.g., first 50 days of EMA50)
        df.dropna(inplace=True)
        
    except Exception as e:
        # If any indicator calculation fails due to missing data/key errors, log it
        st.error(f"Error during indicator calculation: {e}. Returning incomplete data.")
        return df.reset_index() 
        
    return df.reset_index()


# utils/algorithm_engine.py (Updated run_short_term_algo)

def run_short_term_algo(df_ticker, backtest=False):
    """
    Short-Term (5-Day) Strategy: Triple-Confirmation Model.
    """
    df = add_indicators(df_ticker)
    
    # --- FIX 1: IMMEDIATE CRITICAL COLUMN CHECK ---
    # If the required columns are not present after indicator calculation, stop processing.
    required_cols = ['Close', 'EMA50', 'MACD_Line', 'MACD_Signal', 'RSI']
    if not all(col in df.columns for col in required_cols):
        # This means add_indicators failed to create the essential columns (MACD, EMA, etc.)
        return calculate_kpis(df_ticker) if backtest else ("HOLD", "Critical indicators missing after calculation.", "N/A")
    
    # Must re-check length after dropping NaNs
    if df.empty or len(df) < 50:
        return calculate_kpis(df) if backtest else ("HOLD", "Data too short after cleaning.", "N/A")

    # --- 2. Generate Signals ---
    df['Trend_Filter'] = df['Close'] > df['EMA50']
    
    # FIX 2: We no longer need the explicit 'if MACD_Line not in df.columns' check here
    # because FIX 1 already validated it. The code below should now run smoothly.

    # Momentum Signal: MACD Line > MACD Signal line
    df['Momentum_Signal'] = np.where(df['MACD_Line'] > df['MACD_Signal'], 1.0, 0.0)
    df['Momentum_Signal'] = np.where(df['MACD_Line'] < df['MACD_Signal'], -1.0, df['Momentum_Signal'])
    # ... (Rest of the function remains the same, down to the final return) ...
# --- Long-Term Strategy (QVAL Proxy) ---
def run_long_term_algo(df_ticker, backtest=False):
    """
    Long-Term (1-Year) Strategy: Quantitative Value (QVAL) Proxy Model.
    """
    if df_ticker.empty or len(df_ticker) < 756: # 3 years * 252 trading days
        return calculate_kpis(df_ticker) if backtest else ("HOLD", "Insufficient data (<3 years) for Long-Term analysis.", "N/A")

    df = df_ticker.copy().set_index('Date').sort_index()
    close_series = df['Close']
    
    # 1. Quality Proxy: 3-Year CAGR
    period = 252 * 3
    if len(close_series) >= period:
        start_price = close_series.iloc[-period]
        end_price = close_series.iloc[-1]
        cagr = ((end_price / start_price)**(1/3)) - 1
    else:
        cagr = 0

    # 2. Valuation Proxy: Current Price relative to 1-Year Max (Lower is better/cheaper)
    max_price_1y = close_series.iloc[-252:].max()
    valuation_ratio = close_series.iloc[-1] / max_price_1y
    
    # 3. Stability Proxy: 1-Year Volatility
    stability_score = close_series.iloc[-252:].pct_change().std()
    
    # --- COMBINE SCORES ---
    # Proprietary score formula
    weighted_score = (cagr * 500) - (valuation_ratio * 2) - (stability_score * 50)
    
    # Initializing Final_Signal column for backtesting/output
    df['Final_Signal'] = 0 
    
    latest_price = close_series.iloc[-1]
    
    # --- Final Signal and Prescription ---
    if weighted_score > 0.5: 
        target = latest_price * 1.30 
        df.iloc[-1, df.columns.get_loc('Final_Signal')] = 1 
        rationale = f"Strong Quality (CAGR {cagr:.1%}) and Favorable Valuation ({valuation_ratio:.2f})."
        return "STRONG BUY (Long)", f"Entry: {latest_price:.2f} | Target: {target:.2f} | Score: {weighted_score:.2f}", rationale
        
    elif weighted_score < -0.5:
        target = latest_price * 0.70
        df.iloc[-1, df.columns.get_loc('Final_Signal')] = -1 
        rationale = f"Weak Quality (CAGR {cagr:.1%}) and High Volatility."
        return "STRONG SELL (Long)", f"Exit: {latest_price:.2f} | Target: {target:.2f} | Score: {weighted_score:.2f}", rationale
        
    else:
        rationale = f"Neutral Score ({weighted_score:.2f}). Waiting for better fundamentals."
        return "HOLD", "Monitor next month.", rationale


# --- KPI CALCULATION (Remains the same) ---
def calculate_kpis(df_signals):
    # This logic assumes the DataFrame passed to it has a 'Final_Signal' column.
    if 'Final_Signal' not in df_signals.columns or df_signals.empty or df_signals['Close'].isnull().all():
        return {"Sharpe Ratio": "0.00", "Max Drawdown": "0.00%", "CAGR": "0.00%", "Win Rate": "0.0%", "Total Trades": 0}
        
    df_signals['Daily_Return'] = df_signals['Close'].pct_change()
    df_signals['Hold_Return'] = df_signals['Daily_Return'].shift(-5).rolling(5).sum()
    df_signals['Trade_Return'] = df_signals['Hold_Return'] * df_signals['Final_Signal'].shift(1)
    
    completed_trades = df_signals[df_signals['Final_Signal'] != 0].dropna(subset=['Trade_Return'])

    if completed_trades.empty:
        return {"Sharpe Ratio": "0.00", "Max Drawdown": "0.00%", "CAGR": "0.00%", "Win Rate": "0.0%", "Total Trades": 0}

    # Aggregate Strategy Performance
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
