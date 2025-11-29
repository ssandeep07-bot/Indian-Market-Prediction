import pandas as pd
import yfinance as yf
from datetime import timedelta
import streamlit as st

@st.cache_data(ttl=timedelta(days=1)) # Cache historical data for 1 day (EOD refresh)
def get_historical_data(ticker_list, period='5y'):
    """
    Fetches historical data for a list of tickers.
    
    IMPORTANT: REPLACE yfinance with a reliable paid NSE/BSE data API for 
    production use, as yfinance is unreliable for specific Indian stock data.
    """
    if not ticker_list:
        st.warning("Ticker list is empty.")
        return pd.DataFrame()

    st.info(f"Fetching {period} data for {len(ticker_list)} stocks...")
    
    # Use .NS suffix for NSE stocks with yfinance
    yf_tickers = [t + ".NS" for t in ticker_list]
    
    try:
        # Download data using yfinance
        data = yf.download(yf_tickers, period=period, progress=False)
        
        # Check if data is multi-index (multiple tickers) or single index (one ticker)
        if isinstance(data.columns, pd.MultiIndex):
            # For multiple tickers, stack the dataframe
            data = data['Close'].stack().rename_axis(['Date', 'Ticker']).reset_index(name='Close')
            # Fetch Volume separately as stacking all columns can be slow/complex
            volume_data = yf.download(yf_tickers, period=period, progress=False)['Volume'].stack().rename_axis(['Date', 'Ticker']).reset_index(name='Volume')
            
            # Merge Close and Volume
            data = pd.merge(data, volume_data, on=['Date', 'Ticker'])
        else:
            # For a single ticker (if only one was requested), fix the structure
            data = data.reset_index().rename(columns={'Close': 'Close', 'Volume': 'Volume'})
            data['Ticker'] = ticker_list[0] # Manually assign ticker name
            
        # Clean ticker names (remove the .NS suffix)
        data['Ticker'] = data['Ticker'].str.replace('.NS', '', regex=False)
        
        return data[['Date', 'Ticker', 'Close', 'Volume']].sort_values(['Ticker', 'Date'])
        
    except Exception as e:
        st.error(f"Error fetching data using yfinance: {e}")
        return pd.DataFrame()

def get_nifty50_tickers():
    """
    Hypothetical function to get a list of highly liquid Nifty 50 stocks for the universe.
    """
    # This list represents your initial "Watchlist Universe" (Section 2 from previous answer)
    return [
        'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 
        'KOTAKBANK', 'HINDUNILVR', 'AXISBANK', 'LT', 'SBIN'
    ]
