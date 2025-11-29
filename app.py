import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ---------- CONFIG ----------
DEFAULT_SYMBOLS = {
    "Reliance Industries (NSE)": "RELIANCE.NS",
    "Infosys (NSE)": "INFY.NS",
    "HDFC Bank (NSE)": "HDFCBANK.NS",
    "TCS (NSE)": "TCS.NS",
    "ICICI Bank (NSE)": "ICICIBANK.NS",
}

FORECAST_DAYS = 5


# ---------- DATA FUNCTIONS ----------
@st.cache_data(show_spinner=False)
def load_price_data(symbol: str, years: int = 2) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years)
    df = yf.download(symbol, start=start, end=end)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.lower)
    cols = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in cols if c in df.columns]].dropna()
    return df


def get_next_trading_days(last_date: pd.Timestamp, n: int = 5):
    # Simple trading-day logic: skip weekends.
    # Later we will plug in full NSE/BSE calendars.
    days = []
    current = last_date.date()
    while len(days) < n:
        current = current + dt.timedelta(days=1)
        if current.weekday() < 5:  # 0=Mon ... 4=Fri
            days.append(current)
    return days


# ---------- DUMMY FORECAST MODEL (TO BE REPLACED LATER) ----------
def simple_dummy_forecast(df: pd.DataFrame, n_days: int = FORECAST_DAYS):
    """
    Placeholder forecast:
    - Uses last close price.
    - Adds a small random drift.
    - Returns 5-day price path and a simple Buy/Sell/Hold suggestion.
    """
    if df is None or df.empty or "close" not in df.columns:
        return pd.DataFrame(), 0.0, "HOLD"

    last_close = df["close"].dropna().iloc[-1]
    dates = get_next_trading_days(df.index[-1], n_days)

    # Create a simple random walk around the last close
    rng = np.random.default_rng(seed=42)
    daily_changes_pct = rng.normal(loc=0.001, scale=0.01, size=n_days)  # mean +0.1%, std 1%
    prices = []
    current_price = float(last_close)

    for change in daily_changes_pct:
        current_price = current_price * (1 + change)
        prices.append(current_price)

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
    st.caption("Personal research tool for NSE/BSE stocks (prototype UI with placeholder model).")

    # Sidebar: stock selection and info
    st.sidebar.header("Configuration")
    symbol_label = st.sidebar.selectbox(
        "Select stock",
        options=list(DEFAULT_SYMBOLS.keys()),
        index=0,
    )
    symbol = DEFAULT_SYMBOLS[symbol_label]

    st.sidebar.markdown("**Forecast horizon:** 5 trading days")
    st.sidebar.markdown("Model: simple dummy drift model (will be upgraded to LSTM + sentiment).")

    # Load data
    with st.spinner("Loading market data..."):
        df = load_price_data(symbol)

    if df is None or df.empty or "close" not in df.columns:
        st.error("No price data available for this symbol right now. Please try another one or reload.")
        return

    # Clean data
    df = df.dropna(subset=["close"])
    if df.empty:
        st.error("No valid closing prices found for this symbol. Please try another one.")
        return

    last_close = df["close"].iloc[-1]

    st.subheader(f"Current overview: {symbol_label}")
    try:
        st.metric("Last close", f"{float(last_close):,.2f} INR")
    except Exception:
        st.metric("Last close", str(last_close))

    # Historical chart
    st.markdown("### Historical prices (last ~1 year)")
    try:
        st.line_chart(df["close"].tail(252))
    except Exception:
        st.line_chart(df["close"])

    # Forecast section
    st.markdown("### 5-day forecast & recommendation")
    forecast_df, expected_return_pct, action = simple_dummy_forecast(df, FORECAST_DAYS)

    if forecast_df is None or forecast_df.empty:
        st.warning("Forecast could not be generated. Please reload or try a different symbol.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Expected 5-day return",
            f"{expected_return_pct:,.2f}%",
        )
    with col2:
        st.metric("Suggested action", action)

    # Combine last actual with forecast for plotting
    combined = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": [df.index[-1].date()],
                    "price": [float(last_close)],
                    "type": ["Actual last close"],
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

    st.markdown("#### Price path (last close + next 5 trading days)")
    st.line_chart(combined["price"])

    st.markdown("#### Forecast table")
    st.dataframe(forecast_df.style.format({"predicted_close": "{:,.2f}"}))


if __name__ == "__main__":
    main()
