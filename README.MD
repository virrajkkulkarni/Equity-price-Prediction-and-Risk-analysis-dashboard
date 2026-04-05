📌 Overview

This project is an end-to-end financial time series pipeline that combines:

Stock price forecasting using Facebook Prophet
Model validation using MAE & RMSE
Strategy simulation using model predictions
Performance evaluation using Sharpe Ratio, CAGR, and Max Drawdown

It demonstrates how machine learning models can be extended into practical financial decision-making frameworks.

🚀 Features
📊 Interactive Streamlit web app
📉 Time-series forecasting using Prophet
📏 Model evaluation (MAE, RMSE)
📈 Trading strategy simulation
💰 Financial performance metrics:
Sharpe Ratio
CAGR
Maximum Drawdown
📅 Future price forecasting with confidence intervals
📌 Visual comparison: Actual vs Predicted
🛠️ Tech Stack
Python
Streamlit
Prophet
Pandas
NumPy
Plotly
yFinance
📂 Project Structure
stock-forecasting-prophet/
│
├── app.py                # Main Streamlit application
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/stock-forecasting-prophet.git
cd stock-forecasting-prophet
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the app
streamlit run app.py
📊 How It Works
1. Data Collection
Historical stock data is fetched using yfinance
2. Data Preparation
Cleaned and formatted for Prophet (ds, y)
Time-based train-test split applied
3. Forecasting
Prophet models:
Trend
Weekly seasonality
Yearly seasonality
4. Evaluation
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
5. Strategy Simulation
Buy signal generated based on predicted vs actual prices
Lookahead bias avoided using .shift(1)
6. Performance Metrics
Sharpe Ratio → risk-adjusted returns
CAGR → annual growth rate
Max Drawdown → downside risk
📈 Key Visualizations
Actual vs Predicted Prices
Strategy vs Market Performance
Forecast with Confidence Intervals
Trend & Seasonality Components
⚠️ Important Notes
This project is for educational purposes only
Prophet is not designed for high-frequency trading
The strategy is simplified and not production-ready
Financial markets are influenced by external factors not captured here
🧠 Key Learnings
Time-series forecasting concepts
Importance of proper validation in financial data
Avoiding lookahead bias
Translating ML predictions into trading strategies
Evaluating performance using financial metrics
📌 Future Improvements
Portfolio-level optimization
Advanced trading strategies
Feature engineering (technical indicators)
Hyperparameter tuning
Risk-adjusted allocation
👨‍💻 Author

Viraj Kulkarni

LinkedIn: https://www.linkedin.com/in/virrajkkulkarni/
GitHub: https://github.com/virrajkkulkarni