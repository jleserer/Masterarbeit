import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# Define the stock tickers
stocks = {
    'SP500': '^GSPC',         # S&P 500
    'NASDAQ': '^IXIC',        # NASDAQ Composite
    'NIKKEI': '^N225',        # Nikkei 225
    'HANG_SENG': '^HSI',      # Hang Seng Index
    'DAX': '^GDAXI',          # DAX Performance Index
    '10-Year Bond': '^TNX',   # CBOE 10-Year Treasury Note Yield Index
    'USD/JPY': 'JPY=X',       # USD to JPY Exchange Rate
    'GBP/USD': 'GBPUSD=X'     # GBP to USD Exchange Rate
}

# Calculate the date range (50 years back from today)
end_date = datetime.now()
start_date = end_date - timedelta(days=50*365)

print(f"Downloading historical data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print("=" * 80)

# Download and save data for each stock
for name, ticker in stocks.items():
    print(f"\nDownloading {name} ({ticker})...")
    try:
        # Download historical data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if not data.empty:
            # Save to CSV file
            filename = f"{name}_historical_data.csv"
            data.to_csv(filename)
            print(f"✓ Successfully saved {len(data)} records to {filename}")
            print(f"  Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        else:
            print(f"✗ No data available for {name}")
    except Exception as e:
        print(f"✗ Error downloading {name}: {str(e)}")

print("\n" + "=" * 80)
print("Download complete!")
