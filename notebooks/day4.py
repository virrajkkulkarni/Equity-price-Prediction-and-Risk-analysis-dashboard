# day4_complete.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

print("="*70)
print("DAY 4: RETURNS & VOLATILITY - COMPLETE CODE")
print("="*70)

# ======================
# 1. LOAD DATA
# ======================
print("\n📊 LOADING DATA...")
ticker = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)  # 3 years for clarity

# Get adjusted data
stock = yf.Ticker(ticker)
data = stock.history(start=start_date, end=end_date, auto_adjust=True)
data.columns = [col for col in data.columns]  # Clean column names

print(f"Data loaded: {len(data)} trading days")
print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
print(f"Price range: ${data['Close'].min():.2f} to ${data['Close'].max():.2f}")

# ======================
# 2. CALCULATE RETURNS
# ======================
print("\n📈 CALCULATING RETURNS...")

# Method 1: Simple returns (for comparison)
data['Simple_Return'] = data['Close'].pct_change()

# Method 2: Log returns (preferred for modeling)
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

# Remove NaN values from returns
returns_data = data[['Log_Return', 'Simple_Return']].dropna()

print(f"Returns calculated: {len(returns_data)} days")
print(f"Average daily log return: {returns_data['Log_Return'].mean()*100:.4f}%")
print(f"Daily volatility (std): {returns_data['Log_Return'].std()*100:.4f}%")

# ======================
# 3. CALCULATE VOLATILITY
# ======================
print("\n📉 CALCULATING VOLATILITY...")

# Rolling volatility (standard deviation of returns)
window_sizes = [10, 30, 60]  # Days

for window in window_sizes:
    col_name = f'Vol_{window}D'
    data[col_name] = data['Log_Return'].rolling(window=window).std()
    
    # Annualize: daily_vol * sqrt(252 trading days/year)
    data[f'{col_name}_Annualized'] = data[col_name] * np.sqrt(252)
    
    # Get most recent value
    recent_vol = data[col_name].iloc[-1]
    if not pd.isna(recent_vol):
        print(f"  {window}-day volatility: {recent_vol*100:.2f}%")
        print(f"  Annualized: {recent_vol*np.sqrt(252)*100:.1f}%")

# ======================
# 4. CREATE COMPREHENSIVE PLOTS
# ======================
print("\n🎨 CREATING VISUALIZATIONS...")

# Create figure with subplots
fig = plt.figure(figsize=(15, 12))

# ===== PLOT 1: PRICE VS RETURNS =====
ax1 = plt.subplot(3, 2, 1)
ax1.plot(data.index, data['Close'], 'b-', linewidth=1.5, alpha=0.8)
ax1.set_title(f'{ticker} - Adjusted Close Price', fontsize=12, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# ===== PLOT 2: DAILY RETURNS =====
ax2 = plt.subplot(3, 2, 2)
colors = ['red' if x < 0 else 'green' for x in data['Log_Return']]
ax2.bar(data.index, data['Log_Return']*100, color=colors, alpha=0.6, width=1)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax2.set_title('Daily Log Returns (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Return (%)', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# ===== PLOT 3: RETURNS DISTRIBUTION =====
ax3 = plt.subplot(3, 2, 3)
returns_no_nan = data['Log_Return'].dropna() * 100  # Convert to %

# Histogram
n, bins, patches = ax3.hist(returns_no_nan, bins=50, alpha=0.7, 
                           color='steelblue', edgecolor='black', 
                           density=True)

# Add normal distribution curve
from scipy.stats import norm
mu, std = returns_no_nan.mean(), returns_no_nan.std()
x = np.linspace(returns_no_nan.min(), returns_no_nan.max(), 100)
p = norm.pdf(x, mu, std)
ax3.plot(x, p, 'r-', linewidth=2, label='Normal Distribution')

# Mark ±1σ, ±2σ
ax3.axvline(mu + std, color='orange', linestyle=':', alpha=0.6, label='±1σ')
ax3.axvline(mu - std, color='orange', linestyle=':', alpha=0.6)
ax3.axvline(mu + 2*std, color='red', linestyle=':', alpha=0.4, label='±2σ')
ax3.axvline(mu - 2*std, color='red', linestyle=':', alpha=0.4)

ax3.set_title('Returns Distribution vs Normal', fontsize=12, fontweight='bold')
ax3.set_xlabel('Daily Return (%)', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ===== PLOT 4: COMPARE RETURN TYPES =====
ax4 = plt.subplot(3, 2, 4)
sample_days = 100  # Show last 100 days for clarity
ax4.plot(data.index[-sample_days:], data['Simple_Return'][-sample_days:]*100, 
        'b-', linewidth=1, alpha=0.7, label='Simple Returns')
ax4.plot(data.index[-sample_days:], data['Log_Return'][-sample_days:]*100, 
        'r-', linewidth=1, alpha=0.7, label='Log Returns')
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax4.set_title('Simple vs Log Returns (Last 100 Days)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Date', fontsize=10)
ax4.set_ylabel('Return (%)', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# ===== PLOT 5: ROLLING VOLATILITY =====
ax5 = plt.subplot(3, 2, 5)
# Plot different window sizes
for window in window_sizes:
    vol_data = data[f'Vol_{window}D_Annualized'].dropna() * 100  # Convert to %
    ax5.plot(vol_data.index, vol_data, linewidth=1.5, alpha=0.7, 
            label=f'{window}-Day Window')

ax5.set_title('Rolling Volatility (Annualized)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Date', fontsize=10)
ax5.set_ylabel('Volatility (%)', fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.tick_params(axis='x', rotation=45)

# ===== PLOT 6: VOLATILITY CLUSTERING =====
ax6 = plt.subplot(3, 2, 6)
# Use 30-day volatility for demonstration
vol_30d = data['Vol_30D'].dropna() * 100

# Color by volatility regime
high_vol_threshold = vol_30d.quantile(0.75)
low_vol_threshold = vol_30d.quantile(0.25)

colors_vol = []
for vol in vol_30d:
    if vol > high_vol_threshold:
        colors_vol.append('red')    # High volatility
    elif vol < low_vol_threshold:
        colors_vol.append('green')  # Low volatility
    else:
        colors_vol.append('gray')   # Medium volatility

ax6.bar(vol_30d.index, vol_30d, color=colors_vol, alpha=0.6, width=1)
ax6.axhline(y=high_vol_threshold, color='red', linestyle='--', alpha=0.5, 
           label='High Vol Regime')
ax6.axhline(y=low_vol_threshold, color='green', linestyle='--', alpha=0.5,
           label='Low Vol Regime')

ax6.set_title('Volatility Clustering (30-Day)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Date', fontsize=10)
ax6.set_ylabel('Volatility (%)', fontsize=10)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

plt.suptitle(f'DAY 4: {ticker} Returns & Volatility Analysis', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('day4_complete_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualizations saved as 'day4_complete_analysis.png'")

# ======================
# 5. KEY STATISTICS
# ======================
print("\n📊 KEY STATISTICS:")
print("-" * 40)

# Returns statistics
print("RETURNS ANALYSIS:")
print(f"  Total period return: {((data['Close'].iloc[-1]/data['Close'].iloc[0])-1)*100:.1f}%")
print(f"  Average daily return: {data['Log_Return'].mean()*100:.4f}%")
print(f"  Daily volatility: {data['Log_Return'].std()*100:.4f}%")
print(f"  Annualized volatility: {data['Log_Return'].std()*np.sqrt(252)*100:.2f}%")
print(f"  Sharpe ratio (assuming 0% risk-free): {data['Log_Return'].mean()/data['Log_Return'].std():.3f}")

# Volatility statistics
print("\nVOLATILITY ANALYSIS:")
current_vol_30d = data['Vol_30D'].iloc[-1]
if not pd.isna(current_vol_30d):
    print(f"  Current 30-day volatility: {current_vol_30d*100:.2f}%")
    print(f"  Annualized: {current_vol_30d*np.sqrt(252)*100:.1f}%")

# Volatility range
vol_30d_clean = data['Vol_30D'].dropna()
print(f"  Max 30-day volatility: {vol_30d_clean.max()*100:.2f}%")
print(f"  Min 30-day volatility: {vol_30d_clean.min()*100:.2f}%")

# Extreme returns
print("\nEXTREME MOVES:")
print(f"  Best day: {data['Log_Return'].max()*100:.2f}%")
print(f"  Worst day: {data['Log_Return'].min()*100:.2f}%")
print(f"  Days > +5%: {(data['Log_Return'] > 0.05).sum()}")
print(f"  Days < -5%: {(data['Log_Return'] < -0.05).sum()}")

# ======================
# 6. REGIME ANALYSIS
# ======================
print("\n🔍 REGIME ANALYSIS:")
print("-" * 40)

# Identify high and low volatility periods
if len(vol_30d_clean) > 0:
    high_vol_periods = vol_30d_clean[vol_30d_clean > vol_30d_clean.quantile(0.75)]
    low_vol_periods = vol_30d_clean[vol_30d_clean < vol_30d_clean.quantile(0.25)]
    
    print(f"  High-volatility days: {len(high_vol_periods)}")
    print(f"  Low-volatility days: {len(low_vol_periods)}")
    
    # Check if we have COVID period in data
    if '2020-03' in data.index.strftime('%Y-%m').values:
        covid_vol = data.loc['2020-03', 'Vol_30D'].mean()
        if not pd.isna(covid_vol):
            print(f"  COVID period (Mar 2020) avg volatility: {covid_vol*100:.1f}%")

# ======================
# 7. ANALYST INSIGHTS
# ======================
print("\n" + "="*70)
print("ANALYST INSIGHTS - WRITE THIS IN YOUR PROJECT:")
print("="*70)

insights = """
1. RETURNS ANALYSIS:
   • AAPL shows slightly positive average daily returns with periods of high volatility
   • Returns distribution has fatter tails than normal distribution (more extreme moves)
   • Log and simple returns are similar for daily moves (<5%) but diverge for larger moves

2. VOLATILITY PATTERNS:
   • Volatility clusters in time - high-vol days follow high-vol days
   • Distinct volatility regimes visible (high, medium, low)
   • Rolling volatility shows how risk changes over time, not constant

3. KEY TAKEAWAYS FOR MODELING:
   • Use LOG RETURNS for time series modeling (better statistical properties)
   • Account for TIME-VARYING VOLATILITY in risk management
   • Consider VOLATILITY REGIMES when backtesting strategies
   • ROLLING WINDOWS more realistic than single train-test split

4. PRACTICAL IMPLICATIONS:
   • Position sizing should adapt to current volatility
   • Stop losses should be volatility-adjusted
   • Models need to handle changing volatility regimes
   • Risk forecasting as important as return forecasting
"""

print(insights)

# ======================
# 8. SAVE DATA FOR NEXT DAYS
# ======================
print("\n💾 SAVING PROCESSED DATA...")
# Save returns and volatility data for modeling
modeling_data = data[['Close', 'Log_Return', 'Vol_30D']].copy()
modeling_data.to_csv(f'{ticker}_returns_volatility.csv')
print(f"✅ Data saved as '{ticker}_returns_volatility.csv'")
print("   Columns: Close, Log_Return, Vol_30D")

print("\n" + "="*70)
print("DAY 4 COMPLETE! READY FOR TIME SERIES MODELING")
print("="*70)