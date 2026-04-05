# auto_adjust_comparison.py
import yfinance as yf
import pandas as pd

print("="*70)
print("VISUALIZING AUTO_ADJUST TRUE vs FALSE")
print("="*70)

# Get AAPL data for a period that includes the 2020 split
start_date = '2020-08-20'
end_date = '2020-09-10'

print(f"\nDate Range: {start_date} to {end_date}")
print("(Includes AAPL's 4-for-1 split on 2020-08-31)")

# ======================
# 1. WITHOUT AUTO_ADJUST (RAW DATA)
# ======================
print("\n" + "="*50)
print("1. WITHOUT auto_adjust=True (RAW DATA)")
print("="*50)

ticker = yf.Ticker('AAPL')
raw_data = ticker.history(start=start_date, end=end_date, auto_adjust=False)

print("\nColumns available:")
print(raw_data.columns.tolist())

print("\nData Head (RAW - unadjusted):")
print(raw_data[['Open', 'Close']].head(8))

# Show around split date
print("\n🔍 AROUND SPLIT DATE (RAW - Shows Fake Crash):")
split_period = raw_data.loc['2020-08-27':'2020-09-03']
for date, row in split_period.iterrows():
    print(f"{date.date():<15} Close: ${row['Close']:8.2f}")

# Calculate the "fake" return
pre_split = raw_data.loc['2020-08-28', 'Close']
post_split = raw_data.loc['2020-08-31', 'Close']
fake_return = (post_split - pre_split) / pre_split * 100
print(f"\n📉 FAKE RETURN from split: {fake_return:.1f}% (Looks like -75% crash!)")

# ======================
# 2. WITH AUTO_ADJUST (ADJUSTED DATA)
# ======================
print("\n" + "="*50)
print("2. WITH auto_adjust=True (ADJUSTED DATA)")
print("="*50)

adj_data = ticker.history(start=start_date, end=end_date, auto_adjust=True)

print("\nColumns available:")
print(adj_data.columns.tolist())

print("\nData Head (ADJUSTED - clean):")
print(adj_data[['Open', 'Close']].head(8))

# Show around split date
print("\n🔍 AROUND SPLIT DATE (ADJUSTED - Shows Real Move):")
split_period_adj = adj_data.loc['2020-08-27':'2020-09-03']
for date, row in split_period_adj.iterrows():
    print(f"{date.date():<15} Close: ${row['Close']:8.2f}")

# Calculate the real return
pre_split_adj = adj_data.loc['2020-08-28', 'Close']
post_split_adj = adj_data.loc['2020-08-31', 'Close']
real_return = (post_split_adj - pre_split_adj) / pre_split_adj * 100
print(f"\n📈 REAL RETURN from split: {real_return:.1f}% (Actual market move)")

# ======================
# 3. SIDE-BY-SIDE COMPARISON
# ======================
print("\n" + "="*70)
print("3. SIDE-BY-SIDE COMPARISON")
print("="*70)

comparison = pd.DataFrame({
    'Date': split_period.index.date,
    'Raw Close': split_period['Close'].values,
    'Adj Close': split_period_adj['Close'].values,
    'Difference': split_period['Close'].values - split_period_adj['Close'].values
})

print("\n📊 Comparison Table:")
print(comparison.to_string(index=False))

# ======================
# 4. VISUAL DIFFERENCES
# ======================
print("\n" + "="*50)
print("4. KEY DIFFERENCES VISUALIZED")
print("="*50)

# Show the dramatic difference
print("\n💰 PRICE COMPARISON:")
print(f"{'Date':<15} {'Raw Close':<12} {'Adj Close':<12} {'Explanation':<30}")
print("-" * 70)

# Pre-split dates
for date in ['2020-08-27', '2020-08-28']:
    raw = raw_data.loc[date, 'Close']
    adj = adj_data.loc[date, 'Close']
    print(f"{date:<15} ${raw:<11.2f} ${adj:<11.2f} Pre-split: Raw ÷ 4 = Adj")

# Split date
date = '2020-08-31'
raw = raw_data.loc[date, 'Close']
adj = adj_data.loc[date, 'Close']
print(f"{date:<15} ${raw:<11.2f} ${adj:<11.2f} Split day: Same!")

# Post-split dates
for date in ['2020-09-01', '2020-09-02']:
    raw = raw_data.loc[date, 'Close']
    adj = adj_data.loc[date, 'Close']
    print(f"{date:<15} ${raw:<11.2f} ${adj:<11.2f} Post-split: Same!")

# ======================
# 5. WHAT HAPPENS TO RETURNS?
# ======================
print("\n" + "="*50)
print("5. RETURNS COMPARISON (Critical for Models!)")
print("="*50)

# Calculate returns
raw_returns = raw_data['Close'].pct_change()
adj_returns = adj_data['Close'].pct_change()

print("\n📈 DAILY RETURNS Comparison:")
print(f"{'Date':<15} {'Raw Return':<15} {'Adj Return':<15}")
print("-" * 45)

# Show returns around split
for date in ['2020-08-28', '2020-08-31', '2020-09-01']:
    raw_ret = raw_returns.loc[date] * 100
    adj_ret = adj_returns.loc[date] * 100
    print(f"{date:<15} {raw_ret:>7.2f}%       {adj_ret:>7.2f}%")

print(f"\n⚠️  Split date return:")
print(f"   Raw:  {raw_returns.loc['2020-08-31']*100:.2f}% (Fake -75% crash!)")
print(f"   Adj:  {adj_returns.loc['2020-08-31']*100:.2f}% (Real 2% gain)")

# ======================
# 6. SUMMARY
# ======================
print("\n" + "="*70)
print("SUMMARY: Why auto_adjust=True Matters")
print("="*70)

print("""
✅ WITH auto_adjust=True:
   - Prices are ALREADY divided by 4 for pre-split dates
   - Smooth transition: $124 → $127 (real 2% gain)
   - Models learn REAL market patterns
   - Returns calculation is ACCURATE

❌ WITHOUT auto_adjust=True:
   - Pre-split: $499 (actual traded price)
   - Post-split: $127 (actual traded price)
   - Fake crash: $499 → $127 (looks like -75%!)
   - Models learn NONSENSE (predicting splits!)
   - Returns calculation is WRONG
""")

print("\n🎯 FOR YOUR PROJECT:")
print("ALWAYS use: yf.Ticker('AAPL').history(auto_adjust=True)")
print("Then use: data['Close'] (this is already adjusted!)")

# ======================
# 7. BONUS: Check for multiple columns
# ======================
print("\n" + "="*50)
print("7. COLUMN CHECK: What's different?")
print("="*50)

print("\n📋 WITHOUT auto_adjust (more columns):")
print(f"Columns: {raw_data.columns.tolist()}")
if 'Dividends' in raw_data.columns:
    print(f"Dividends column exists: Yes")
if 'Stock Splits' in raw_data.columns:
    print(f"Stock Splits column exists: Yes")

print("\n📋 WITH auto_adjust (cleaner columns):")
print(f"Columns: {adj_data.columns.tolist()}")
if 'Dividends' in adj_data.columns:
    print(f"Dividends column exists: Yes")
if 'Stock Splits' in adj_data.columns:
    print(f"Stock Splits column exists: Yes")

print("\n💡 Note: With auto_adjust=True, dividends are already")
print("subtracted from prices, so no separate column needed!")