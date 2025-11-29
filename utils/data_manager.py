import pandas as pd
import yfinance as yf
from datetime import timedelta
import streamlit as st

@st.cache_data(ttl=timedelta(days=1)) # Cache historical data for 1 day (EOD refresh)
def get_historical_data(ticker_list, period='5y'):
    """
    Fetches historical data for a list of tickers.
    
    NOTE: REPLACE yfinance with a reliable paid NSE/BSE data API for 
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
            # Stack the dataframe to get long format
            data = data.stack().rename_axis(['Date', 'Ticker']).reset_index()
            data.rename(columns={'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
            data = data[data['Ticker'].isin(yf_tickers)] # Filter out potential garbage
        else:
            # Handle single ticker case
            data = data.reset_index().rename(columns={'Close': 'Close', 'Volume': 'Volume'})
            data['Ticker'] = ticker_list[0] + '.NS' # Assign full ticker name
            
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
    return [
        'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 
        'KOTAKBANK', 'HINDUNILVR', 'AXISBANK', 'LT', 'SBIN'
    ]
