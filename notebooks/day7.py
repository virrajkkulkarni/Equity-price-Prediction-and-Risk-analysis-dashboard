# ============================================
# 📦 1. IMPORT LIBRARIES
# ============================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# 📈 2. LOAD & INSPECT DATA - FIXED VERSION
# ============================================

# Download Apple stock data
ticker = "AAPL"
stock = yf.download(ticker, start="2015-01-01", end="2024-12-31", progress=False)

# Debug: Print column names to see what yfinance returns
print("="*60)
print("📊 DATA INSPECTION")
print("="*60)
print(f"Stock: {ticker}")
print(f"Available Columns: {list(stock.columns)}")
print(f"Column Structure: {type(stock.columns)}")
print(f"Time Range: {stock.index[0].date()} to {stock.index[-1].date()}")
print(f"Total Days: {len(stock)}")
print("\n" + "="*60)

# Check if stock is MultiIndex (common issue with yfinance)
if isinstance(stock.columns, pd.MultiIndex):
    print("\n⚠️  MultiIndex detected! Flattening columns...")
    # Flatten the MultiIndex columns
    stock.columns = ['_'.join(col).strip() for col in stock.columns.values]
    print(f"New Columns: {list(stock.columns)}")
    
    # Map common column names
    column_mapping = {
        'Open': [col for col in stock.columns if 'Open' in col][0],
        'High': [col for col in stock.columns if 'High' in col][0],
        'Low': [col for col in stock.columns if 'Low' in col][0],
        'Close': [col for col in stock.columns if 'Close' in col][0],
        'Adj Close': [col for col in stock.columns if 'Adj' in col and 'Close' in col][0] if any('Adj' in col for col in stock.columns) else None,
        'Volume': [col for col in stock.columns if 'Volume' in col][0]
    }
    
    # Create a clean dataframe with standard column names
    clean_data = {}
    for standard_name, actual_name in column_mapping.items():
        if actual_name:
            clean_data[standard_name] = stock[actual_name]
    
    stock = pd.DataFrame(clean_data, index=stock.index)
    print(f"Standardized Columns: {list(stock.columns)}")

# Check for missing values
print("\n🔍 MISSING VALUES CHECK:")
print(stock.isnull().sum())
print(f"Missing Percentage: {(stock.isnull().sum().sum()/stock.size)*100:.2f}%")

# Fill any missing values (forward fill)
stock = stock.ffill()

# Basic statistics
print("\n📊 BASIC STATISTICS:")
print(stock.describe())

# ============================================
# 🎨 3. VISUALIZE PRICE BEHAVIOR - FIXED
# ============================================

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle(f'{ticker} - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Plot 1: Closing Price Over Time
axes[0, 0].plot(stock.index, stock['Close'], label='Close Price', linewidth=2, alpha=0.8, color='blue')
axes[0, 0].set_title('Closing Price Over Time', fontweight='bold')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Adjusted Close vs Close (only if Adj Close exists)
if 'Adj Close' in stock.columns:
    axes[0, 1].plot(stock.index, stock['Close'], label='Close', alpha=0.7, linewidth=1.5, color='blue')
    axes[0, 1].plot(stock.index, stock['Adj Close'], label='Adj Close', alpha=0.7, linewidth=1.5, color='red')
    axes[0, 1].set_title('Close vs Adjusted Close', fontweight='bold')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].legend()
else:
    axes[0, 1].plot(stock.index, stock['Close'], label='Close Price', alpha=0.8, linewidth=2, color='blue')
    axes[0, 1].set_title('Closing Price (Adj Close not available)', fontweight='bold')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Trading Volume Over Time
axes[1, 0].bar(stock.index, stock['Volume'], alpha=0.6, color='steelblue')
axes[1, 0].set_title('Trading Volume Over Time', fontweight='bold')
axes[1, 0].set_ylabel('Volume')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: OHLC Prices
axes[1, 1].plot(stock.index, stock['Open'], label='Open', alpha=0.5, linewidth=1, color='green')
axes[1, 1].plot(stock.index, stock['High'], label='High', alpha=0.5, linewidth=1, color='red')
axes[1, 1].plot(stock.index, stock['Low'], label='Low', alpha=0.5, linewidth=1, color='orange')
axes[1, 1].plot(stock.index, stock['Close'], label='Close', alpha=0.8, linewidth=2, color='blue')
axes[1, 1].set_title('OHLC Prices', fontweight='bold')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].legend(loc='upper left')
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Daily Price Range (High - Low)
daily_range = stock['High'] - stock['Low']
axes[2, 0].plot(stock.index, daily_range, color='coral', alpha=0.7, linewidth=1)
axes[2, 0].fill_between(stock.index, 0, daily_range, alpha=0.3, color='coral')
axes[2, 0].set_title('Daily Price Range (High - Low)', fontweight='bold')
axes[2, 0].set_ylabel('Range ($)')
axes[2, 0].grid(True, alpha=0.3)

# Plot 6: 50-day and 200-day Moving Averages
stock['MA_50'] = stock['Close'].rolling(window=50).mean()
stock['MA_200'] = stock['Close'].rolling(window=200).mean()
axes[2, 1].plot(stock.index, stock['Close'], label='Close', alpha=0.5, linewidth=1, color='gray')
axes[2, 1].plot(stock.index, stock['MA_50'], label='50-day MA', linewidth=2, color='blue')
axes[2, 1].plot(stock.index, stock['MA_200'], label='200-day MA', linewidth=2, color='red')
axes[2, 1].set_title('Moving Averages', fontweight='bold')
axes[2, 1].set_ylabel('Price ($)')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 📊 4. RETURNS ANALYSIS
# ============================================

# Calculate returns
stock['Returns'] = stock['Close'].pct_change() * 100  # Percentage returns
stock['Log_Returns'] = np.log(stock['Close'] / stock['Close'].shift(1)) * 100

# Remove NaN values
returns_data = stock['Returns'].dropna()
log_returns_data = stock['Log_Returns'].dropna()

print("\n" + "="*60)
print("📈 RETURNS ANALYSIS")
print("="*60)
print(f"Mean Daily Return: {returns_data.mean():.4f}%")
print(f"Std Dev of Returns: {returns_data.std():.4f}%")
print(f"Skewness: {returns_data.skew():.4f}")
print(f"Kurtosis: {returns_data.kurtosis():.4f}")
print(f"Minimum Return: {returns_data.min():.4f}%")
print(f"Maximum Return: {returns_data.max():.4f}%")
print(f"Median Return: {returns_data.median():.4f}%")

# Jarque-Bera test for normality
jb_test = stats.jarque_bera(returns_data)
print(f"\n📊 Normality Test (Jarque-Bera):")
print(f"  Statistic: {jb_test[0]:.2f}")
print(f"  p-value: {jb_test[1]:.10f}")
if jb_test[1] < 0.05:
    print(f"  ⚠️  Interpretation: NOT NORMAL (p < 0.05)")
else:
    print(f"  ✅ Interpretation: Normal (p ≥ 0.05)")

# ============================================
# 📊 5. RETURNS VISUALIZATION
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Returns Time Series
axes[0, 0].plot(returns_data.index, returns_data.values, alpha=0.7, linewidth=0.5, color='blue')
axes[0, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
axes[0, 0].axhline(y=returns_data.mean(), color='g', linestyle='--', alpha=0.5, label=f'Mean: {returns_data.mean():.3f}%')
axes[0, 0].set_title('Daily Returns Over Time', fontweight='bold')
axes[0, 0].set_ylabel('Returns (%)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Histogram of Returns
axes[0, 1].hist(returns_data, bins=100, alpha=0.7, edgecolor='black', density=True, color='skyblue')
# Add normal distribution curve
x = np.linspace(returns_data.min(), returns_data.max(), 1000)
pdf = stats.norm.pdf(x, returns_data.mean(), returns_data.std())
axes[0, 1].plot(x, pdf, 'r-', linewidth=2, label='Normal Distribution')
axes[0, 1].axvline(x=returns_data.mean(), color='g', linestyle='--', alpha=0.7, label=f'Mean: {returns_data.mean():.3f}%')
axes[0, 1].set_title('Distribution of Daily Returns', fontweight='bold')
axes[0, 1].set_xlabel('Returns (%)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Q-Q Plot
stats.probplot(returns_data, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (vs Normal Distribution)', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Cumulative Returns
cumulative_returns = (1 + returns_data/100).cumprod() - 1
axes[1, 1].plot(cumulative_returns.index, cumulative_returns.values * 100, linewidth=2, color='green')
axes[1, 1].set_title('Cumulative Returns', fontweight='bold')
axes[1, 1].set_ylabel('Cumulative Returns (%)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 🔄 6. ROLLING STATISTICS
# ============================================

# Calculate rolling statistics
window = 30
stock['Rolling_Mean'] = returns_data.rolling(window=window).mean()
stock['Rolling_Std'] = returns_data.rolling(window=window).std()

# Calculate rolling volatility (annualized)
stock['Rolling_Volatility'] = stock['Rolling_Std'] * np.sqrt(252)

fig, axes = plt.subplots(2, 1, figsize=(15, 8))

# Plot Rolling Mean
axes[0].plot(stock.index, stock['Rolling_Mean'], label=f'{window}-day Rolling Mean', linewidth=2, color='blue')
axes[0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
axes[0].set_title(f'{window}-Day Rolling Statistics', fontweight='bold')
axes[0].set_ylabel('Mean Returns (%)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot Rolling Volatility
axes[1].plot(stock.index, stock['Rolling_Volatility'], label=f'{window}-day Rolling Volatility', 
             linewidth=2, color='coral')
axes[1].axhline(y=returns_data.std() * np.sqrt(252), color='r', linestyle='--', alpha=0.5, 
                label=f'Overall: {returns_data.std() * np.sqrt(252):.2f}%')
axes[1].set_title(f'{window}-Day Rolling Volatility (Annualized)', fontweight='bold')
axes[1].set_ylabel('Volatility (%)')
axes[1].set_xlabel('Date')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 🔗 7. CORRELATION ANALYSIS (Optional - with SPY)
# ============================================

try:
    # Download S&P 500 ETF for comparison
    spy = yf.download("SPY", start="2015-01-01", end="2024-12-31", progress=False)
    
    # Handle MultiIndex columns for SPY
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = ['_'.join(col).strip() for col in spy.columns.values]
        close_col = [col for col in spy.columns if 'Close' in col][0]
        spy['Close'] = spy[close_col]
    
    spy['Returns'] = spy['Close'].pct_change() * 100
    spy_returns = spy['Returns'].dropna()
    
    # Align dates
    aligned_returns = pd.concat([returns_data, spy_returns], axis=1).dropna()
    aligned_returns.columns = [f'{ticker}_Returns', 'SPY_Returns']
    
    # Calculate correlation
    correlation = aligned_returns.corr().iloc[0, 1]
    rolling_corr = aligned_returns[f'{ticker}_Returns'].rolling(window=252).corr(aligned_returns['SPY_Returns'])
    
    print("\n" + "="*60)
    print("🔗 CORRELATION ANALYSIS")
    print("="*60)
    print(f"Correlation between {ticker} and SPY: {correlation:.4f}")
    print(f"Average Rolling (252-day) Correlation: {rolling_corr.mean():.4f}")
    
    # Plot correlation
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    axes[0].scatter(aligned_returns['SPY_Returns'], aligned_returns[f'{ticker}_Returns'], 
                    alpha=0.5, s=10, color='blue')
    axes[0].set_title(f'{ticker} vs SPY Returns', fontweight='bold')
    axes[0].set_xlabel('SPY Returns (%)')
    axes[0].set_ylabel(f'{ticker} Returns (%)')
    axes[0].grid(True, alpha=0.3)
    
    # Add regression line
    z = np.polyfit(aligned_returns['SPY_Returns'], aligned_returns[f'{ticker}_Returns'], 1)
    p = np.poly1d(z)
    axes[0].plot(aligned_returns['SPY_Returns'], p(aligned_returns['SPY_Returns']), 
                "r--", alpha=0.8, label=f'Beta = {z[0]:.3f}')
    axes[0].legend()
    
    # Rolling correlation
    axes[1].plot(rolling_corr.index, rolling_corr.values, linewidth=2, color='purple')
    axes[1].set_title('252-Day Rolling Correlation', fontweight='bold')
    axes[1].set_ylabel('Correlation')
    axes[1].set_xlabel('Date')
    axes[1].axhline(y=correlation, color='r', linestyle='--', alpha=0.5, label='Overall Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"\n⚠️  Correlation analysis skipped due to error: {e}")

# ============================================
# 📋 8. SUMMARY STATISTICS
# ============================================

# Calculate additional metrics
annual_return = ((1 + returns_data.mean()/100) ** 252 - 1) * 100
annual_volatility = returns_data.std() * np.sqrt(252)
sharpe_ratio = returns_data.mean() / returns_data.std() * np.sqrt(252) if returns_data.std() != 0 else 0

# Maximum drawdown
rolling_max = stock['Close'].expanding().max()
daily_drawdown = stock['Close'] / rolling_max - 1
max_drawdown = daily_drawdown.min() * 100

print("\n" + "="*60)
print("📋 SUMMARY OF FINDINGS")
print("="*60)
print(f"📈 STOCK: {ticker}")
print(f"📅 PERIOD: {len(stock)} trading days ({stock.index[0].date().year}-{stock.index[-1].date().year})")
print("\n📊 RETURNS ANALYSIS:")
print(f"   • Mean Daily Return: {returns_data.mean():.4f}%")
print(f"   • Annualized Return: {annual_return:.2f}%")
print(f"   • Daily Volatility: {returns_data.std():.4f}%")
print(f"   • Annual Volatility: {annual_volatility:.2f}%")
print(f"   • Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"   • Max Daily Return: {returns_data.max():.4f}%")
print(f"   • Min Daily Return: {returns_data.min():.4f}%")
print("\n📉 RISK METRICS:")
print(f"   • Maximum Drawdown: {max_drawdown:.2f}%")
print(f"   • Skewness: {returns_data.skew():.3f} ({'Negative' if returns_data.skew() < 0 else 'Positive'} skew)")
print(f"   • Kurtosis: {returns_data.kurtosis():.3f} ({'Fat tails' if returns_data.kurtosis() > 3 else 'Normal tails'})")
if 'correlation' in locals():
    print(f"   • Market Correlation: {correlation:.3f} with S&P 500")
print("\n✅ DISTRIBUTION:")
normality = "NOT NORMAL" if jb_test[1] < 0.05 else "NORMAL"
print(f"   • Distribution is {normality} (Jarque-Bera p={jb_test[1]:.6f})")

# ============================================
# 💾 9. SAVE RESULTS
# ============================================

# Save processed data
output_filename = f'{ticker}_processed_data.csv'
stock.to_csv(output_filename)
print(f"\n💾 Data saved to '{output_filename}'")

# Save summary statistics
summary_stats = {
    'Ticker': ticker,
    'Start_Date': stock.index[0].date(),
    'End_Date': stock.index[-1].date(),
    'Total_Days': len(stock),
    'Mean_Daily_Return': returns_data.mean(),
    'Std_Daily_Return': returns_data.std(),
    'Annualized_Return': annual_return,
    'Annualized_Volatility': annual_volatility,
    'Sharpe_Ratio': sharpe_ratio,
    'Skewness': returns_data.skew(),
    'Kurtosis': returns_data.kurtosis(),
    'Min_Return': returns_data.min(),
    'Max_Return': returns_data.max(),
    'Max_Drawdown': max_drawdown,
    'Normality_p_value': jb_test[1],
    'Correlation_with_SPY': correlation if 'correlation' in locals() else None
}

summary_df = pd.DataFrame([summary_stats])
summary_filename = f'{ticker}_summary_statistics.csv'
summary_df.to_csv(summary_filename, index=False)
print(f"📊 Summary statistics saved to '{summary_filename}'")

print("\n" + "="*60)
print("✅ EDA COMPLETE!")
print(f"✅ {len(stock)} days analyzed")
print(f"✅ {6} plots generated")
print(f"✅ {2} CSV files saved")
print("="*60)