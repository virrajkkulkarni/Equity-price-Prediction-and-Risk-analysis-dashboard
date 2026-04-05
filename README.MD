# 📊 Equity Price Prediction & Risk Analysis Dashboard

## 📌 Overview
This project is an end-to-end financial time series pipeline that integrates **machine learning with financial decision-making**.

It combines:
- 📉 Stock price forecasting using Prophet  
- 📏 Model validation using MAE & RMSE  
- 📈 Strategy simulation based on predictions  
- 💰 Performance evaluation using Sharpe Ratio, CAGR, and Maximum Drawdown  

👉 The project demonstrates how predictive models can be extended into **actionable trading strategies and performance analysis frameworks**.

---

## 🚀 Key Features
- 📊 Interactive web application built with Streamlit  
- 📉 Time-series forecasting using Prophet  
- 📏 Model evaluation (MAE, RMSE)  
- 📈 Trading strategy simulation  
- 💰 Financial performance metrics:
  - Sharpe Ratio  
  - CAGR (Compound Annual Growth Rate)  
  - Maximum Drawdown  
- 📅 Future price forecasting with confidence intervals  
- 📊 Visual comparison: Actual vs Predicted prices  

---

## 🛠️ Tech Stack
- Python  
- Streamlit  
- Prophet  
- Pandas  
- NumPy  
- Plotly  
- yFinance  

---

## 📂 Project Structure

stock-forecasting-prophet/
│
├── app.py # Main Streamlit application
├── requirements.txt # Dependencies
├── data/ # Data & outputs
├── notebooks/ # Analysis scripts
└── README.md # Documentation


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/virrajkkulkarni/Equity-price-Prediction-and-Risk-analysis-dashboard
cd Equity-price-Prediction-and-Risk-analysis-dashboard
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the application
streamlit run app/app.py
📊 How It Works
1. Data Collection
Historical stock data fetched using yFinance
2. Data Preparation
Cleaned and structured into Prophet format (ds, y)
Time-based train-test split applied
3. Forecasting

Prophet models:

Trend
Weekly seasonality
Yearly seasonality
4. Model Evaluation
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
5. Strategy Simulation
Trading signals generated using predicted vs actual prices
Lookahead bias avoided using .shift(1)
6. Performance Evaluation
Sharpe Ratio → risk-adjusted returns
CAGR → annual growth rate
Maximum Drawdown → downside risk
📈 Key Visualizations
Actual vs Predicted Prices
Strategy vs Market Performance
Forecast with Confidence Intervals
Trend & Seasonality Components
⚠️ Disclaimer
This project is for educational purposes only
Prophet is not suitable for high-frequency trading
The trading strategy is simplified and not production-ready
Financial markets are influenced by external macro factors not included in this model
🧠 Key Learnings
Time-series forecasting using Prophet
Importance of proper validation in financial data
Avoiding lookahead bias in strategy design
Translating ML predictions into trading decisions
Evaluating strategies using financial metrics
🔮 Future Improvements
Portfolio-level optimization
Advanced trading strategies
Feature engineering (technical indicators)
Hyperparameter tuning
Risk-adjusted capital allocation
👨‍💻 Author

Viraj Kulkarni

🔗 LinkedIn: https://www.linkedin.com/in/virrajkkulkarni/
💻 GitHub: https://github.com/virrajkkulkarni