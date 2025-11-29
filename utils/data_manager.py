import pandas as pd
import finnhub
from datetime import timedelta, datetime
import streamlit as st
import time
import numpy as np

# --- 1. Finnhub API Client Setup ---
@st.cache_resource
def get_finnhub_client():
    """Initializes and returns the Finnhub client using Streamlit Secrets."""
    API_KEY = st.secrets.get("FINNHUB_KEY")
    if not API_KEY:
        st.error("Finnhub API Key not found. Please add FINNHUB_KEY to Streamlit Secrets.")
        return None
    return finnhub.Client(api_key=API_KEY)

# --- 2. Core Data Fetching Function ---
@st.cache_data(ttl=timedelta(days=1)) # Cache EOD data for 1 day
def get_historical_data(ticker_list, period='5y'):
    """
    Fetches historical data for a list of tickers from Finnhub (EOD).
    Finnhub uses the format: NSE:TICKER (e.g., NSE:RELIANCE).
    """
    client = get_finnhub_client()
    if client is None: return pd.DataFrame()
    
    st.info(f"Fetching 5 years (EOD) data for {len(ticker_list)} stocks from Finnhub...")
    
    all_data = []
    end_time = int(time.time())
    start_time = int((datetime.now() - timedelta(days=365 * 5)).timestamp()) # 5 years ago
    
    for i, ticker in enumerate(ticker_list):
        # Finnhub requires the exchange prefix for Indian stocks (NSE/BSE)
        symbol = f"NSE:{ticker}" 
        
        try:
            # Finnhub API call for candles (resolution 'D' for Daily)
            result = client.stock_candles(symbol, 'D', start_time, end_time)
            
            if 's' in result and result['s'] == 'ok':
                df = pd.DataFrame(result)
                # Finnhub candle data columns: c (close), h (high), l (low), o (open), v (volume), t (time)
                df.rename(columns={'t': 'Date', 'c': 'Close', 'v': 'Volume', 'h': 'High', 'l': 'Low', 'o': 'Open'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'], unit='s')
                df['Ticker'] = ticker
                
                # Filter to only keep required columns
                df = df[['Date', 'Ticker', 'Close', 'Volume', 'Open', 'High', 'Low']]
                all_data.append(df)
            else:
                # Log API-specific errors (e.g., Symbol not found, API limit reached)
                st.warning(f"Finnhub warning for {symbol}: {result.get('error', 'Data not OK')}")
            
            # Rate limit handling: Finnhub free tier pause (safe practice)
            time.sleep(0.5) 
            
        except Exception as e:
            st.error(f"Finnhub connection failed for {symbol}: {e}")
            
    if all_data:
        # Concatenate all individual stock DataFrames
        return pd.concat(all_data).sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    return pd.DataFrame()

# --- 3. Ticker List (No Change) ---
def get_nifty50_tickers():
    """
    Hypothetical function to get a list of highly liquid Nifty 50 stocks for the universe.
    """
    return [
        'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 
        'KOTAKBANK', 'HINDUNILVR', 'AXISBANK', 'LT', 'SBIN'
    ]
