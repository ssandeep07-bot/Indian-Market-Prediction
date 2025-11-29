import pandas as pd
import numpy as np
import talib
from datetime import timedelta

# --- PREDICTION AND PRESCRIPTION CORE ---

def run_short_term_algo(df_ticker, backtest=False):
    """
    Short-Term (5-Day) Strategy: Triple-Confirmation Model.
    
    Args:
        df_ticker (pd.DataFrame): Historical data for one stock.
        backtest (bool): If True, returns KPIs. If False, returns latest signal.
    """
    if df_ticker.empty or len(df_ticker) < 200:
        return calculate_kpis(df_ticker) if backtest else ("HOLD", "Insufficient data (<200 days).")

    # Ensure index is datetime for TA-Lib
    df = df_ticker.copy().set_index('Date').sort_index()

    # --- 1. Calculate Technical Indicators ---
    df['EMA50'] = talib.EMA(df['Close'], timeperiod=50)
    macd, macd_signal, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

    # --- 2. Generate Signals based on Triple-Confirmation ---
    
    # Condition 1: Long-term trend is up (Price > EMA50)
    df['Trend_Filter'] = df['Close'] > df['EMA50']
    
    # Condition 2: Momentum crossover (MACD line crosses above MACD Signal line)
    # 1.0 = Buy, -1.0 = Sell, 0.0 = Hold
    df['Momentum_Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1.0, 0.0)
    df['Momentum_Signal'] = np.where(df['MACD'] < df['MACD_Signal'], -1.0, df['Momentum_Signal'])

    # Condition 3: Avoid overbought conditions for BUY signal
    df['RSI_Filter'] = df['RSI'] < 70
    
    # --- 3. Final Prediction (The T+5 Signal) ---
    
    # Final BUY condition: Trend is UP AND Momentum is BUY AND RSI is not Overbought
    buy_condition = (df['Trend_Filter'] == True) & (df['Momentum_Signal'] == 1.0) & (df['RSI_Filter'] == True)
    
    # Final SELL condition (Simple reversal): Trend is DOWN AND Momentum is SELL
    sell_condition = (df['Trend_Filter'] == False) & (df['Momentum_Signal'] == -1.0)
    
    df['Final_Signal'] = 0
    df.loc[buy_condition, 'Final_Signal'] = 1
    df.loc[sell_condition, 'Final_Signal'] = -1
    
    # --- Output ---
    if backtest:
        return calculate_kpis(df)
    
    # Latest Prediction (Prescription)
    latest_signal = df['Final_Signal'].iloc[-1]
    latest_price = df['Close'].iloc[-1]
    
    if latest_signal == 1:
        # Prediction: Price will rise by 3% in next 5 days (Hypothetical Target)
        target = latest_price * 1.03
        stop_loss = latest_price * 0.98
        rationale = "MACD Buy, Price > EMA50, RSI OK."
        return "STRONG BUY", f"Entry: {latest_price:.2f} | T1: {target:.2f} | SL: {stop_loss:.2f}", rationale
    elif latest_signal == -1:
        target = latest_price * 0.97
        stop_loss = latest_price * 1.02
        rationale = "MACD Sell, Price < EMA50 (Weak Trend)."
        return "STRONG SELL", f"Exit: {latest_price:.2f} | T1: {target:.2f} | SL: {stop_loss:.2f}", rationale
    else:
        rationale = "Trend/Momentum Conflict or Neutral Signal."
        return "HOLD", "Monitor next refresh.", rationale

def calculate_kpis(df_signals):
    """
    Calculates Backtesting Key Performance Metrics (KPMs)
    Uses a 5-day holding period simulation for accuracy.
    """
    df_signals['Daily_Return'] = df_signals['Close'].pct_change()
    
    # Calculate returns based on a 5-day holding period after a buy signal
    # This simulates buying on Signal=1 and selling 5 days later.
    df_signals['Hold_Return'] = df_signals['Daily_Return'].shift(-5).rolling(5).sum()
    df_signals['Trade_Return'] = df_signals['Hold_Return'] * df_signals['Final_Signal'].shift(1)
    
    # Filter for completed trades where signal was not 0
    completed_trades = df_signals[df_signals['Final_Signal'] != 0].dropna(subset=['Trade_Return'])

    if completed_trades.empty:
        return {"Sharpe Ratio": "0.00", "Max Drawdown": "0.00%", "CAGR": "0.00%", "Win Rate": "0.0%", "Total Trades": 0}

    # Aggregate Strategy Performance
    total_return = (completed_trades['Trade_Return'].fillna(0) + 1).prod() - 1
    
    # Calculate Drawdown
    cumulative_returns = (completed_trades['Trade_Return'].fillna(0) + 1).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    # Annualize returns (assuming 252 trading days)
    cagr = ((1 + total_return)**(252 / len(completed_trades.index))) - 1
    annual_volatility = completed_trades['Trade_Return'].std() * np.sqrt(252)
    risk_free_rate = 0.06 # Indian risk-free rate
    sharpe_ratio = (cagr - risk_free_rate) / annual_volatility
    
    # Trade Statistics
    win_rate = completed_trades[completed_trades['Trade_Return'] > 0].shape[0] / completed_trades.shape[0]
    
    kpis = {
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown * 100:.2f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Win Rate": f"{win_rate * 100:.1f}%",
        "Total Trades": completed_trades.shape[0]
    }
    return kpis
