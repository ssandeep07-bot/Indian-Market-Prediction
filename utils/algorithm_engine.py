import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st

# --- Helper Function to Process Indicators Robustly ---
# utils/algorithm_engine.py (Updated add_indicators function)

def add_indicators(df):
    """Adds necessary indicators to the DataFrame using pandas-ta."""
    if df.empty or 'Close' not in df.columns:
        return df

    df = df.set_index('Date').sort_index().copy()

    try:
        # EMA50 (Trend Filter)
        df.ta.ema(close='Close', length=50, append=True)
        df['EMA50'] = df['EMA_50']
        
        # MACD (fast=12, slow=26, signal=9)
        df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
        
        # --- FIX: ROBUST MACD COLUMN IDENTIFICATION ---
        macd_cols = [col for col in df.columns if col.startswith('MACD_') and 'S_' in col]
        
        if len(macd_cols) == 1:
            # Found the correct Signal line name (e.g., MACD_S_12_26_9)
            df['MACD_Line'] = df['MACD_12_26_9'] # The MACD line name is usually reliable
            df['MACD_Signal'] = df[macd_cols[0]] # Use the dynamically found signal name
        else:
            # Fallback if names are completely missing or ambiguous
            st.warning("MACD columns not found after calculation. Skipping MACD signals.")
            df['MACD_Line'] = np.nan
            df['MACD_Signal'] = np.nan
            # Drop the other MACD columns to prevent confusion
            df.drop(columns=[c for c in df.columns if c.startswith('MACD_') and len(c) > 6], errors='ignore', inplace=True)

        # RSI (Condition)
        df.ta.rsi(close='Close', length=14, append=True) 
        df['RSI'] = df['RSI_14']
        
        df.dropna(inplace=True)
        
    except Exception as e:
        st.warning(f"Error during indicator calculation: {e}. Returning incomplete data.")
        # Ensure that incomplete columns are cleaned before returning
        for col in ['EMA50', 'MACD_Line', 'MACD_Signal', 'RSI']:
            if col in df.columns:
                df.drop(columns=[col], errors='ignore', inplace=True)
        return df.reset_index() 
        
    return df.reset_index()

# --- Short-Term Strategy (Triple-Confirmation) ---
def run_short_term_algo(df_ticker, backtest=False):
    """Short-Term (5-Day) Strategy: Triple-Confirmation Model."""
    df = add_indicators(df_ticker)
    
    # --- FIX 1: IMMEDIATE CRITICAL COLUMN CHECK ---
    required_cols = ['Close', 'EMA50', 'MACD_Line', 'MACD_Signal', 'RSI']
    if not all(col in df.columns for col in required_cols):
        # Gracefully exit if indicators are missing
        return calculate_kpis(df_ticker) if backtest else ("HOLD", "Critical indicators missing after calculation.", "N/A")
    
    if df.empty or len(df) < 50:
        return calculate_kpis(df) if backtest else ("HOLD", "Data too short after cleaning.", "N/A")

    # --- 2. Generate Signals ---
    df['Trend_Filter'] = df['Close'] > df['EMA50']
    
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


# --- Long-Term Strategy (QVAL Proxy) ---
def run_long_term_algo(df_ticker, backtest=False):
    """Long-Term (1-Year) Strategy: Quantitative Value (QVAL) Proxy Model."""
    if df_ticker.empty or len(df_ticker) < 756 or 'Close' not in df_ticker.columns:
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

    # 2. Valuation Proxy: Current Price relative to 1-Year Max
    max_price_1y = close_series.iloc[-252:].max()
    valuation_ratio = close_series.iloc[-1] / max_price_1y
    
    # 3. Stability Proxy: 1-Year Volatility
    stability_score = close_series.iloc[-252:].pct_change().std()
    
    # --- COMBINE SCORES ---
    weighted_score = (cagr * 500) - (valuation_ratio * 2) - (stability_score * 50)
    
    df['Final_Signal'] = 0 
    latest_price = close_series.iloc[-1]
    
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


# --- KPI CALCULATION ---
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
