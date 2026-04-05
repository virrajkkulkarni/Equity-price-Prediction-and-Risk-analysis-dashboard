import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# Page config to hide function objects
st.set_page_config(page_title="Stock Forecast", layout="wide")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Title
st.title('📈 Stock Forecast App')

# Sidebar for inputs
st.sidebar.header('User Input Parameters')

selected_stock = st.sidebar.selectbox(
    'Select stock for prediction',
    ('GOOG', 'AAPL', 'MSFT', 'GME', 'TSLA', 'AMZN')
)

n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Data loading function (won't display if cached properly)
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('📊 Raw Data')
st.write(f"Showing data from {START} to {TODAY}")
st.dataframe(data.tail())

# Plot function - defined but only called
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['Open'], 
        name="Stock Open",
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['Close'], 
        name="Stock Close",
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title='Stock Price History',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        xaxis_rangeslider_visible=True
    )
    return fig

# Actually call the function and display the plot
st.subheader('📈 Stock Price History')
fig = plot_raw_data()  # Call the function to get the figure
st.plotly_chart(fig, use_container_width=True)

# Forecasting section
st.subheader('🔮 Future Price Forecast')

# Prepare data for Prophet
df_train = data[['Date', 'Close']].copy()
df_train.columns = ['ds', 'y']

# Clean data
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train = df_train.dropna()

if len(df_train) > 10:
    with st.spinner('Training Prophet model...'):
        m = Prophet()
        m.fit(df_train)
        
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)
    
    st.success('Model trained successfully!')
    
    # Show forecast data
    st.write(f'Forecast for next {n_years} year(s)')
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    # Plot forecast
    st.write('### Forecast Plot')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Plot components
    st.write('### Forecast Components')
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)
    
else:
    st.error('Not enough data for prediction!')

# Add some info
st.sidebar.info(
    """
    **How to use:**
    1. Select a stock from the dropdown
    2. Choose prediction years
    3. View forecast results
    """
)