import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- CONFIG ----------
API_BASE_URL = "http://nse-api-khaki.vercel.app:5000"

# Use plain NSE symbols (API defaults to NSE when no suffix is given) [web:8]
DEFAULT_SYMBOLS = {
    "Reliance Industries (NSE)": "RELIANCE",
    "Infosys (NSE)": "INFY",
    "HDFC Bank (NSE)": "HDFCBANK",
    "TCS (NSE)": "TCS",
    "ICICI Bank (NSE)": "ICICIBANK",
}

FORECAST_DAYS = 5
SYNTHETIC_HISTORY_DAYS = 60  # for charting, we simulate last 60 trading days


# ---------- DATA FUNCTIONS ----------
@st.cache_data(show_spinner=False)
def fetch_live_price(symbol: str) -> dict:
    """
    Fetch current price and basic info for a single symbol from Indian Stock Market API.
    Uses numeric response format res=num for easy parsing. [web:8]
    """
    url = f"{API_BASE_URL}/stock"
    params = {"symbol": symbol, "res": "num"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    # Expected shape (simplified): {"status": "success", "stock": {...}} [web:8]
    if data.get("status") != "success":
        return {}

    stock = data.get("stock") or {}
    return stock


@st.cache_data(show_spinner=False)
def build_synthetic_history(symbol: str) -> pd.DataFrame:
    """
    Build a synthetic recent price history from the current price.
    Because this API gives us real-time last_price but not full OHLC history,
    we create a simple random-walk history for UI purposes. [web:8][web:139]
    """
    stock = fetch_live_price(symbol)
    last_price = stock.get("last_price")

    if last_price is None:
        return pd.DataFrame()

    try:
        last_price = float(last_price)
    except Exception:
        return pd.DataFrame()

    # Generate BACKWARD synthetic path for the last N trading days
    rng = np.random.default_rng(seed=123)
    # daily pct changes ~ normal(0, 1%) – mild volatility
    daily_changes_pct = rng.normal(loc=0.0, scale=0.01, size=SYNTHETIC_HISTORY_DAYS)
    prices = [last_price]

    # Walk backwards (so we get plausible past prices leading to last_price)
    for change in daily_changes_pct[::-1]:
        prev_price = prices[0] / (1 + change)
        prices.insert(0, prev_price)

    # Generate past trading dates (skip weekends)
    dates = []
    current = dt.date.today()
    while len(dates) < len(prices):
        if current.weekday() < 5:  # Mon–Fri
            dates.append(current)
        current -= dt.timedelta(days=1)
    dates = sorted(dates)[-len(prices):]

    df = pd.DataFrame({"date": dates, "close": prices})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def get_next_trading_days(start_date: dt.date, n: int = 5):
    """
    Simple trading-day logic: skip weekends.
    Later we can replace with full NSE/BSE calendars. [web:8]
    """
    days = []
    current = start_date
    while len(days) < n:
        current = current + dt.timedelta(days=1)
        if current.weekday() < 5:
            days.append(current)
    return days


# ---------- DUMMY FORECAST MODEL (TO BE REPLACED LATER) ----------
def simple_dummy_forecast(last_close: float, last_date: dt.date, n_days: int = FORECAST_DAYS):
    """
    Placeholder forecast:
    - Starts from the last close price.
    - Adds small random drift.
    - Returns 5-day price path and a simple Buy/Sell/Hold suggestion.
    """
    if last_close is None or last_close <= 0:
        return pd.DataFrame(), 0.0, "HOLD"

    rng = np.random.default_rng(seed=42)
    daily_changes_pct = rng.normal(loc=0.001, scale=0.01, size=n_days)  # mean +0.1%, std 1%
    prices = []
    current_price = float(last_close)

    for change in daily_changes_pct:
        current_price = current_price * (1 + change)
        prices.append(current_price)

    dates = get_next_trading_days(last_date, n_days)

    forecast_df = pd.DataFrame(
        {
            "date": dates,
            "predicted_close": prices,
        }
    )

    expected_return_pct = (forecast_df["predicted_close"].iloc[-1] / float(last_close) - 1) * 100

    if expected_return_pct > 3:
        action = "BUY"
    elif expected_return_pct < -3:
        action = "SELL"
    else:
        action = "HOLD"

    return forecast_df, expected_return_pct, action


# ---------- STREAMLIT UI ----------
def main():
    st.set_page_config(
        page_title="Indian Market 5-Day Predictor",
        page_icon="📈",
        layout="wide",
    )

    st.title("Indian Market 5-Day Predictor")
    st.caption(
        "Personal research tool for NSE/BSE stocks (prototype UI, live price via Indian Stock Market API, "
        "forecast is a placeholder model)."
    )

    # Sidebar: stock selection and info
    st.sidebar.header("Configuration")
    symbol_label = st.sidebar.selectbox(
        "Select stock",
        options=list(DEFAULT_SYMBOLS.keys()),
        index=0,
    )
    symbol = DEFAULT_SYMBOLS[symbol_label]

    st.sidebar.markdown("**Forecast horizon:** 5 trading days")
    st.sidebar.markdown("Model: simple dummy drift model (to be upgraded to LSTM + sentiment).")

    # Fetch live data
    with st.spinner("Fetching live price..."):
        stock_info = fetch_live_price(symbol)

    if not stock_info:
        st.error("Unable to fetch live price data for this symbol. Please try another one or reload.")
        return

    last_price = stock_info.get("last_price")
    exchange = stock_info.get("exchange", "NSE")
    ticker = stock_info.get("ticker", symbol)
    change_pct = stock_info.get("percent_change")

    try:
        last_price_float = float(last_price)
    except Exception:
        st.error("Live price is not numeric, cannot proceed. Try reloading.")
        return

    st.subheader(f"Current overview: {symbol_label}")
    col_price, col_change = st.columns(2)
    with col_price:
        st.metric("Last traded price", f"{last_price_float:,.2f} INR")
    with col_change:
        try:
            if change_pct is not None:
                change_pct_float = float(change_pct)
                st.metric("Today's change", f"{change_pct_float:,.2f}%")
            else:
                st.metric("Today's change", "N/A")
        except Exception:
            st.metric("Today's change", "N/A")

    st.caption(f"Exchange: {exchange} | Ticker: {ticker}")

    # Synthetic historical chart (for now)
    st.markdown("### Price trend (synthetic last 60 trading days)")
    with st.spinner("Building synthetic history for charting..."):
        hist_df = build_synthetic_history(symbol)

    if hist_df is None or hist_df.empty:
        st.warning("Could not build a price history series. Only showing live price.")
    else:
        st.line_chart(hist_df["close"])

    # Forecast section
    st.markdown("### 5-day forecast & recommendation")
    last_date_for_forecast = dt.date.today()
    forecast_df, expected_return_pct, action = simple_dummy_forecast(
        last_close=last_price_float,
        last_date=last_date_for_forecast,
        n_days=FORECAST_DAYS,
    )

    if forecast_df is None or forecast_df.empty:
        st.warning("Forecast could not be generated. Please reload.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Expected 5-day return", f"{expected_return_pct:,.2f}%")
    with col2:
        st.metric("Suggested action", action)

    # Combine last actual with forecast for plotting
    combined = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": [last_date_for_forecast],
                    "price": [last_price_float],
                    "type": ["Actual last price"],
                }
            ),
            pd.DataFrame(
                {
                    "date": forecast_df["date"],
                    "price": forecast_df["predicted_close"].astype(float),
                    "type": ["Forecast"] * len(forecast_df),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = combined.set_index("date")

    st.markdown("#### Price path (live last price + next 5 trading days)")
    st.line_chart(combined["price"])

    st.markdown("#### Forecast table")
    st.dataframe(forecast_df.style.format({"predicted_close": "{:,.2f}"}))


if __name__ == "__main__":
    main()
