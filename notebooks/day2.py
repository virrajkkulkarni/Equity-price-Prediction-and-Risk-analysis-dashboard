# day2_final.py - Works even without Adj Close column
import yfinance as yf
import pandas as pd
import numpy as np

print("="*60)
print("DAY 2: OHLCV ANALYSIS WITH MISSING ADJ CLOSE")
print("="*60)

# Method 1: Try with auto_adjust
print("\n1. Downloading with auto_adjust=True...")
try:
    data = yf.download('AAPL', start='2020-01-01', end='2025-01-01', 
                       auto_adjust=True, progress=False)
    print(f"Columns: {list(data.columns)}")
    
    if 'Close' in data.columns:
        print("✓ Using 'Close' column (already adjusted)")
        adj_close = data['Close']
    else:
        # Flatten columns if MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            if 'Close' in data.columns:
                adj_close = data['Close']
                print("✓ Using flattened 'Close' column")
        
except Exception as e:
    print(f"Error with auto_adjust: {e}")

# Method 2: Use Ticker.history()
print("\n2. Trying Ticker.history() method...")
try:
    ticker = yf.Ticker('AAPL')
    data2 = ticker.history(start='2020-01-01', end='2025-01-01', 
                          auto_adjust=True)
    print(f"Columns: {list(data2.columns)}")
    
    if 'Close' in data2.columns:
        print("✓ Success! Using this data.")
        adj_close = data2['Close']
    else:
        print("✗ Still no Close column")
        
except Exception as e:
    print(f"Error with Ticker.history(): {e}")

# Method 3: Get raw data and check for splits
print("\n3. Checking for corporate actions...")
try:
    ticker = yf.Ticker('AAPL')
    
    # Get splits
    splits = ticker.splits
    if len(splits) > 0:
        print(f"Found {len(splits[splits != 0])} stock split(s):")
        for date, ratio in splits[splits != 0].items():
            print(f"  {date.date()}: {ratio}:1 split")
    
    # Get dividends
    dividends = ticker.dividends
    if len(dividends) > 0:
        print(f"Found {len(dividends)} dividend payments")
        print(f"Recent dividend: ${dividends.iloc[-1]:.2f} on {dividends.index[-1].date()}")
    
    # Check actions in our date range
    splits_in_range = splits[(splits.index >= '2020-01-01') & 
                             (splits.index <= '2025-01-01')]
    if len(splits_in_range) > 0:
        print(f"\n⚠️ CRITICAL: {len(splits_in_range)} split(s) in analysis period!")
        print("You MUST use adjusted data for accurate analysis.")
        
except Exception as e:
    print(f"Error getting corporate actions: {e}")

# Show what we're working with
print("\n" + "="*60)
print("DATA SUMMARY")
print("="*60)

if 'adj_close' in locals():
    print(f"Data points: {len(adj_close)}")
    print(f"Date range: {adj_close.index[0].date()} to {adj_close.index[-1].date()}")
    print(f"Price range: ${adj_close.min():.2f} to ${adj_close.max():.2f}")
    
    # Calculate returns
    returns = adj_close.pct_change()
    print(f"Average daily return: {returns.mean()*100:.4f}%")
    print(f"Volatility (std dev): {returns.std()*100:.4f}%")
    
    print("\nFirst 5 adjusted prices:")
    print(adj_close.head())
    
    print("\nLast 5 adjusted prices:")
    print(adj_close.tail())
else:
    print("✗ Could not retrieve adjusted close prices")
    print("\nTROUBLESHOOTING:")
    print("1. Update yfinance: pip install yfinance --upgrade")
    print("2. Try different date range")
    print("3. Use 'auto_adjust=True' parameter")
    print("4. Check Yahoo Finance website directly")

# The key concept
print("\n" + "="*60)
print("KEY DAY 2 LEARNING: WHY ADJUSTED CLOSE MATTERS")
print("="*60)

print("""
Even without the column, remember:

1. **RAW CLOSE PROBLEMS:**
   - Shows artificial "crashes" on split dates
   - Doesn't include dividend returns
   - Creates discontinuous time series

2. **ADJUSTED CLOSE FIXES:**
   - Accounts for splits (e.g., 4:1 split ÷ prices by 4)
   - Accounts for dividends (subtracts from prices)
   - Creates smooth, continuous series for modeling

3. **FOR YOUR PROJECT:**
   - ALWAYS use adjusted data
   - If yfinance doesn't provide 'Adj Close', use 'auto_adjust=True'
   - Document your data source and adjustments
""")

# Quick check: AAPL had a 4-for-1 split on Aug 31, 2020
print("\nCHECK: AAPL 4-for-1 split (Aug 31, 2020)")
print("If using raw data, you'd see:")
print("  Aug 28, 2020: ~$500")
print("  Aug 31, 2020: ~$125  ← Looks like -75% crash!")
print("\nWith adjusted data:")
print("  Aug 28, 2020: ~$125")
print("  Aug 31, 2020: ~$125  ← Smooth transition!")