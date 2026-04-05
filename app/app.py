import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objects as go

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="Equity Price Prediction and Risk analysis Dashboard",
    layout="centered"
)

st.title("📈 Stock Price Forecasting App")
st.write("End-to-end time series forecasting using Facebook Prophet")

# -------------------------------
# User Inputs
# -------------------------------
stocks = ("AAPL", "GOOG", "MSFT", "AMZN")
selected_stock = st.selectbox("Select a stock", stocks)

n_years = st.slider("Years to forecast", 1, 4)
period = n_years * 365

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY, auto_adjust=False)
    data.reset_index(inplace=True)
    return data

st.text("Loading data...")
data = load_data(selected_stock)
st.text("Loading data... done!")

# -------------------------------
# Show Raw Data
# -------------------------------
st.subheader("Raw Market Data")
st.write(data.tail())

# -------------------------------
# Prepare Data for Prophet
# -------------------------------

if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
df = data[['Date', price_col]].copy()
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds']).dt.normalize().dt.tz_localize(None)
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Log-transform price so Prophet never predicts negative values
# and uncertainty intervals stay proportional (multiplicative noise)
df['y'] = np.log(df['y'])

# ---------------- TRAIN / TEST SPLIT (for evaluation only) ----------------
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df  = df.iloc[train_size:]

# ---------------- MODEL 1: trained on train_df → evaluation & strategy ----------------
eval_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.15
)
eval_model.add_country_holidays(country_name='US')
eval_model.fit(train_df)

forecast_test = eval_model.predict(test_df[['ds']])

# For components plot (covers full train window)
future_full = eval_model.make_future_dataframe(periods=len(test_df))
forecast_components = eval_model.predict(future_full)

# ---------------- MODEL 2: trained on ALL data → future forecast ----------------
full_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.15
)
full_model.add_country_holidays(country_name='US')
full_model.fit(df)   # <-- full dataset, so future extends from today

future_df = full_model.make_future_dataframe(periods=period)
future_df['ds'] = pd.to_datetime(future_df['ds']).dt.normalize().dt.tz_localize(None)
forecast_future = full_model.predict(future_df)
forecast_future['ds'] = pd.to_datetime(forecast_future['ds']).dt.normalize().dt.tz_localize(None)

last_date  = df['ds'].max()
future_only = forecast_future[forecast_future['ds'] > last_date].copy()
future_only.reset_index(drop=True, inplace=True)

if future_only.empty:
    st.error(
        f"⚠️ future_only is still empty.\n\n"
        f"last_date: {last_date}\n"
        f"forecast_future range: {forecast_future['ds'].min()} → {forecast_future['ds'].max()}"
    )
    st.stop()

# ---------------- EVALUATION ----------------
# Exp-transform back to price space for evaluation
y_true = np.exp(test_df['y'].values)
y_pred = np.exp(forecast_test['yhat'].values)

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

st.subheader("Model Performance")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")

# ---------------- STRATEGY ----------------
strategy_df = test_df.copy()
# Exp-transform back to real prices for return calculation
strategy_df['y']      = np.exp(strategy_df['y'])
strategy_df['y_pred'] = np.exp(forecast_test['yhat'].values)
strategy_df['returns'] = strategy_df['y'].pct_change()

# Signal: buy when predicted % change tomorrow is positive
strategy_df['pred_return'] = strategy_df['y_pred'].pct_change()
strategy_df['signal'] = (strategy_df['pred_return'].shift(1) > 0).astype(int)
strategy_df['strategy_returns'] = strategy_df['signal'] * strategy_df['returns']
strategy_df.dropna(inplace=True)

# ---------------- METRICS ----------------
strategy_df['cum_market']   = (1 + strategy_df['returns']).cumprod()
strategy_df['cum_strategy'] = (1 + strategy_df['strategy_returns']).cumprod()

n_days = len(strategy_df)
cagr   = (strategy_df['cum_strategy'].iloc[-1]) ** (252 / n_days) - 1
st.write(f"CAGR: {cagr:.2%}")

sharpe = (
    (strategy_df['strategy_returns'].mean() / strategy_df['strategy_returns'].std()) * np.sqrt(252)
    if strategy_df['strategy_returns'].std() != 0 else 0
)

cum         = strategy_df['cum_strategy']
rolling_max = cum.cummax()
max_drawdown = ((cum - rolling_max) / rolling_max).min()

st.subheader("📊 Strategy Performance")
st.write(f"Sharpe Ratio: {sharpe:.2f}")
st.write(f"Max Drawdown: {max_drawdown:.2%}")

fig_perf = go.Figure()
fig_perf.add_trace(go.Scatter(x=strategy_df['ds'], y=strategy_df['cum_market'],   name="Market Return"))
fig_perf.add_trace(go.Scatter(x=strategy_df['ds'], y=strategy_df['cum_strategy'], name="Strategy Return"))
fig_perf.update_layout(title="Strategy vs Market Performance", xaxis_title="Date", yaxis_title="Cumulative Return")
st.plotly_chart(fig_perf)

# ---------------- ACTUAL vs PREDICTED ----------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_df['ds'], y=np.exp(test_df['y'].values), mode='lines', name='Actual Price'))
fig.add_trace(go.Scatter(x=test_df['ds'], y=np.exp(forecast_test['yhat'].values), mode='lines', name='Predicted Price'))
fig.update_layout(title="Actual vs Predicted Prices", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig)

# ---------------- FORECAST COMPONENTS ----------------
st.subheader("Forecast Components")
st.write("Trend, Weekly & Yearly Seasonality")
st.pyplot(eval_model.plot_components(forecast_components))

# ---------------- FUTURE FORECAST PLOT ----------------
st.subheader(f"📈 Future Forecast (Next {n_years} Year{'s' if n_years > 1 else ''})")

fig_future = go.Figure()
fig_future.add_trace(go.Scatter(
    x=df['ds'], y=np.exp(df['y']),
    name="Historical", line=dict(color='blue')
))
# Upper bound first (no fill)
fig_future.add_trace(go.Scatter(
    x=future_only['ds'], y=np.exp(future_only['yhat_upper']),
    name="Upper Bound", line=dict(width=0), showlegend=False
))
# Lower bound fills UP to upper bound
fig_future.add_trace(go.Scatter(
    x=future_only['ds'], y=np.exp(future_only['yhat_lower']),
    name="Lower Bound", fill='tonexty',
    fillcolor='rgba(255,165,0,0.2)',
    line=dict(width=0), showlegend=False
))
# Forecast line on top
fig_future.add_trace(go.Scatter(
    x=future_only['ds'], y=np.exp(future_only['yhat']),
    name="Future Forecast", line=dict(color='orange')
))
fig_future.update_layout(
    title=f"Future Price Forecast — Next {n_years} Year{'s' if n_years > 1 else ''}",
    xaxis_title="Date", yaxis_title="Price"
)
st.plotly_chart(fig_future)

# ---------------- BASIC INSIGHT ----------------
latest_price = np.exp(df['y'].iloc[-1])
future_price = np.exp(future_only['yhat'].iloc[-1])

st.subheader("📊 Basic Insight")
if future_price > latest_price:
    st.success(f"📈 Upward trend  |  Current: ${latest_price:.2f}  →  Forecast end: ${future_price:.2f}")
else:
    st.error(f"📉 Downward trend  |  Current: ${latest_price:.2f}  →  Forecast end: ${future_price:.2f}")