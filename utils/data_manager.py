import pandas as pd
import streamlit as st
from datetime import timedelta
import psycopg2
import numpy as np

# --- 1. Supabase Connection Setup ---
@st.cache_resource
def get_db_connection():
    """Initializes and returns the database connection using Streamlit Secrets."""
    try:
        conn = psycopg2.connect(
            host=st.secrets["DB_HOST"],
            database=st.secrets["DB_DATABASE"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            port=st.secrets["DB_PORT"]
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# --- 2. Core Data Fetching Function (From Supabase) ---
@st.cache_data(ttl=timedelta(hours=4)) # Cache data locally for a few hours
def get_historical_data(ticker_list, period='5y'):
    """
    Fetches historical data for a list of tickers from the Supabase database.
    """
    conn = get_db_connection()
    if conn is None: return pd.DataFrame()

    try:
        # Construct the WHERE clause to fetch data for all required tickers
        ticker_list_str = "('" + "', '".join(ticker_list) + "')"
        
        query = f"""
        SELECT date, ticker, close, volume, "open", high, low
        FROM historical_data
        WHERE ticker IN {ticker_list_str}
        ORDER BY ticker, date;
        """
        
        # Read data directly into a pandas DataFrame
        df = pd.read_sql(query, conn)
        
        df.rename(columns={'ticker': 'Ticker', 'date': 'Date', 'open': 'Open'}, inplace=True)
        
        # Convert necessary columns to appropriate types
        df['Date'] = pd.to_datetime(df['Date'])
        
        return df
        
    except Exception as e:
        st.error(f"Error executing database query: {e}")
        return pd.DataFrame()
        
    finally:
        # Crucial: Close the connection after use
        if conn:
            conn.close()

# --- 3. Ticker List (No Change) ---
def get_nifty50_tickers():
    return [
        'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 
        'KOTAKBANK', 'HINDUNILVR', 'AXISBANK', 'LT', 'SBIN'
    ]
