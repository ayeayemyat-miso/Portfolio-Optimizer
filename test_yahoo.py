# test_yahoo.py
import yfinance as yf
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"Testing AAPL from {start_date.date()} to {end_date.date()}")
print("-" * 50)

try:
    data = yf.Ticker("AAPL").history(start=start_date, end=end_date)
    if not data.empty:
        print(f"✅ Success! Got {len(data)} days of data")
        print("\nFirst 5 rows:")
        print(data.head())
    else:
        print("❌ No data returned")
except Exception as e:
    print(f"❌ Error: {e}")