# day3_eda_analyst.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

print("="*70)
print("DAY 3: EDA - THINK LIKE AN ANALYST")
print("="*70)

# ======================
# 1. SET UP THE DATA
# ======================
ticker = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)  # 5 years

print(f"\n📊 ANALYZING: {ticker} from {start_date.date()} to {end_date.date()}")
print("-" * 50)

# Get adjusted data
stock = yf.Ticker(ticker)
data = stock.history(start=start_date, end=end_date, auto_adjust=True)
data.columns = [col for col in data.columns]  # Ensure clean columns

print(f"Trading days: {len(data):,}")
print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")

# ======================
# 2. STEP 1: VISUAL INSPECTION - ANSWER KEY QUESTIONS
# ======================
print("\n" + "="*70)
print("STEP 1: VISUAL INSPECTION - ANSWER THESE QUESTIONS")
print("="*70)

# Calculate basic metrics for intuition
returns = data['Close'].pct_change()
volatility = returns.rolling(30).std()

print("\n📈 Quick Stats for Intuition:")
print(f"• Average daily return: {returns.mean()*100:.3f}%")
print(f"• Average daily volatility: {volatility.mean()*100:.3f}%")
print(f"• Maximum single-day gain: {returns.max()*100:.2f}%")
print(f"• Maximum single-day loss: {returns.min()*100:.2f}%")
print(f"• Days with >5% moves: {(returns.abs() > 0.05).sum()} days")

# ======================
# 3. STEP 2: PLOT PRICE - OBSERVE TREND CHANGES
# ======================
print("\n" + "="*70)
print("STEP 2: PRICE CHART - WHERE DOES TREND CHANGE?")
print("="*70)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Chart 1: Price with moving averages
ax1 = axes[0]
ax1.plot(data.index, data['Close'], linewidth=1.5, color='black', alpha=0.8, label='Adj Close')

# Add moving averages
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['MA_200'] = data['Close'].rolling(window=200).mean()
ax1.plot(data.index, data['MA_50'], 'orange', linewidth=1.5, alpha=0.7, label='50-day MA')
ax1.plot(data.index, data['MA_200'], 'red', linewidth=1.5, alpha=0.7, label='200-day MA')

# Mark major events (answer: "What looks abnormal?")
events = [
    ('2020-03-23', 'COVID Low', 'red', '↓'),
    ('2020-11-09', 'Vaccine News', 'green', '↑'),
    ('2022-01-03', 'Fed Hikes Start', 'orange', '↓'),
    ('2022-10-13', '2022 Bottom', 'purple', '↓'),
    ('2023-06-05', 'AI Rally', 'blue', '↑'),
]

for date_str, label, color, marker in events:
    try:
        date = pd.Timestamp(date_str)
        if date in data.index:
            price = data.loc[date, 'Close']
            ax1.scatter(date, price, color=color, s=200, zorder=5, marker=marker)
            ax1.annotate(label, xy=(date, price), xytext=(0, 15),
                        textcoords='offset points', fontsize=9, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2))
    except:
        continue

ax1.set_title(f'{ticker}: Spot the Trend Changes (Answer: They happen at events)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# ======================
# 4. STEP 3: ADD VOLUME - UNDERSTAND CONVICTION
# ======================
print("\n" + "="*70)
print("STEP 3: VOLUME - WHEN ARE MOVES CONVINCED?")
print("="*70)

ax2 = axes[1]

# Color volume bars by price direction
colors = ['red' if data['Close'].iloc[i] < data['Close'].iloc[i-1] else 'green' 
          for i in range(len(data))]
colors[0] = 'gray'  # First day

ax2.bar(data.index, data['Volume'] / 1e6, color=colors, alpha=0.6, width=1)

# Highlight high volume days
high_volume_threshold = data['Volume'].quantile(0.95)
high_volume_days = data[data['Volume'] > high_volume_threshold]
ax2.scatter(high_volume_days.index, high_volume_days['Volume'] / 1e6, 
           color='black', s=30, alpha=0.8, label='Top 5% Volume Days')

ax2.set_title('Trading Volume: Big Moves Need Big Volume', fontsize=14, fontweight='bold')
ax2.set_ylabel('Volume (Millions)', fontsize=12)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# ======================
# 5. STEP 4: MOVING AVERAGES - SEE THE LAG
# ======================
print("\n" + "="*70)
print("STEP 4: MOVING AVERAGES - SEE THE LAG PROBLEM")
print("="*70)

ax3 = axes[2]

# Calculate returns
daily_returns = data['Close'].pct_change() * 100

# Highlight MA crossovers
crossover_up = (data['MA_50'] > data['MA_200']) & (data['MA_50'].shift(1) <= data['MA_200'].shift(1))
crossover_down = (data['MA_50'] < data['MA_200']) & (data['MA_50'].shift(1) >= data['MA_200'].shift(1))

# Plot returns
ax3.bar(data.index, daily_returns, alpha=0.5, color='gray', label='Daily Returns')

# Mark crossover points
for date in data[crossover_up].index:
    ax3.axvline(x=date, color='green', alpha=0.3, linestyle='--', label='Golden Cross' if date == data[crossover_up].index[0] else "")
    
for date in data[crossover_down].index:
    ax3.axvline(x=date, color='red', alpha=0.3, linestyle='--', label='Death Cross' if date == data[crossover_down].index[0] else "")

# Add annotations for lag
if len(data[crossover_up]) > 0:
    first_golden = data[crossover_up].index[0]
    pre_cross_low = data['Close'][data.index < first_golden].min()
    post_cross_high = data['Close'][data.index > first_golden].max()
    move_before = (data.loc[first_golden, 'Close'] - pre_cross_low) / pre_cross_low * 100
    move_after = (post_cross_high - data.loc[first_golden, 'Close']) / data.loc[first_golden, 'Close'] * 100
    
    ax3.annotate(f'↑{move_before:.0f}% before cross\n↑{move_after:.0f}% after', 
                xy=(first_golden, 0), xytext=(0, 20),
                textcoords='offset points', ha='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))

ax3.set_title('Moving Average Crossovers: Good Confirmation, Bad Timing', fontsize=14, fontweight='bold')
ax3.set_ylabel('Daily Return (%)', fontsize=12)
ax3.set_xlabel('Date', fontsize=12)
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)

plt.tight_layout()
plt.savefig('day3_analyst_insights.png', dpi=300, bbox_inches='tight')
print("\n✅ Charts saved as 'day3_analyst_insights.png'")

# ======================
# 6. ANALYST WRITING EXERCISE
# ======================
print("\n" + "="*70)
print("ANALYST WRITING EXERCISE: ANSWER THESE QUESTIONS")
print("="*70)

print("\n📝 WRITE YOUR ANSWERS BELOW (10-12 lines total):")
print("-" * 50)

questions = [
    "1. What long-term trend do you see?",
    "2. Are there high-volatility periods? When?",
    "3. Does price behavior look stable over time?",
    "4. Would one global model fit all periods? Why/why not?",
    "5. Where would a moving average strategy fail?",
    "6. What does volume tell you about market conviction?"
]

for i, q in enumerate(questions, 1):
    print(f"\nQ{i}: {q}")
    print("   [Write your answer here]")
    print("   " + "_"*60)

# ======================
# 7. POWERFUL EXERCISE: COVID CRASH ANALYSIS
# ======================
print("\n" + "="*70)
print("POWERFUL EXERCISE: COVID CRASH DEEP DIVE")
print("="*70)

# Isolate COVID period
covid_start = '2020-02-15'
covid_end = '2020-08-15'
covid_data = data.loc[covid_start:covid_end]

if len(covid_data) > 0:
    covid_peak = covid_data['Close'].max()
    covid_trough = covid_data['Close'].min()
    covid_drop = (covid_trough - covid_peak) / covid_peak * 100
    recovery_date = covid_data[covid_data['Close'] >= covid_peak].index[0] if any(covid_data['Close'] >= covid_peak) else "Not recovered in period"
    recovery_days = (recovery_date - covid_data.index[0]).days if isinstance(recovery_date, pd.Timestamp) else "N/A"
    
    print(f"\n📉 COVID CRASH ANALYSIS ({covid_start} to {covid_end}):")
    print(f"   • Peak: ${covid_peak:.2f}")
    print(f"   • Trough: ${covid_trough:.2f}")
    print(f"   • Maximum drawdown: {covid_drop:.1f}%")
    print(f"   • Recovery date: {recovery_date}")
    print(f"   • Days to recover: {recovery_days}")
    
    print("\n🔍 ANSWER THESE (Write Below):")
    print("   1. Was the trend broken? [Yes/No + Why]")
    print("   2. How long did recovery take?")
    print("   3. Could a model trained pre-2020 predict this?")
    print("   4. What would have happened to a moving average strategy?")
else:
    print("\n⚠️ COVID period data not available in this date range")

# ======================
# 8. FINAL REFLECTION
# ======================
print("\n" + "="*70)
print("FINAL REFLECTION: CHECK YOUR UNDERSTANDING")
print("="*70)

print("\n✅ YOU STUDIED CORRECTLY IF YOU CAN SAY:")
print("   1. 'Markets shift regimes' - Found at least 3 regimes")
print("   2. 'Trends exist but break suddenly' - Saw COVID break")
print("   3. 'Volatility clusters' - High-vol days bunch together")
print("   4. 'Forecasting assumes stability — which markets violate'")
print("   5. 'Volume validates price moves' - Low volume rallies are suspect")

print("\n📊 YOUR TASK NOW:")
print("   • Stare at the saved chart for 5 minutes")
print("   • Write your analyst commentary")
print("   • Answer: 'Why would a single model struggle?'")
print("   • Save your written answers for project documentation")

print("\n" + "="*70)
print("DAY 3 COMPLETE: YOU'RE NOW THINKING LIKE AN ANALYST")
print("="*70)