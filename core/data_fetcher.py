# core/data_fetcher.py
"""
Fetch stock data from Yahoo Finance using batch download with retry logic
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import random

class PortfolioDataFetcher:
    """Fetch stock data from Yahoo Finance"""
    
    SECTORS = {
        'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'],
        'Financial': ['JPM', 'V', 'MA', 'BAC', 'GS'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV'],
        'Consumer': ['WMT', 'HD', 'COST', 'MCD', 'PG'],
        'Industrial/Energy': ['XOM', 'CVX', 'CAT', 'BA', 'GE']
    }
    
    @classmethod
    def get_all_tickers(cls):
        all_tickers = []
        for sector, tickers in cls.SECTORS.items():
            all_tickers.extend(tickers)
        return all_tickers
    
    @classmethod
    def get_sector_groups(cls):
        return cls.SECTORS
    
    @classmethod
    def fetch_data_with_retry(cls, tickers, start_date, end_date, max_retries=3):
        """Fetch data with retry logic for network issues"""
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = (2 ** attempt) + random.random() * 2
                    print(f"   Retry attempt {attempt + 1}/{max_retries} after {wait_time:.1f}s...")
                    time.sleep(wait_time)
                
                # Use yfinance's batch download
                data = yf.download(
                    tickers=tickers,
                    start=start_date,
                    end=end_date,
                    group_by='ticker',
                    auto_adjust=True,
                    progress=False,
                    threads=False  # Disable threading to avoid issues on Render
                )
                
                if not data.empty:
                    return data
                    
            except Exception as e:
                print(f"   Attempt {attempt + 1} failed: {str(e)[:100]}")
                continue
        
        return pd.DataFrame()
    
    @classmethod
    def create_sample_fallback(cls, tickers, start_date, end_date):
        """Create sample data as fallback when Yahoo Finance fails"""
        print("   ⚠️ Using sample data as fallback")
        
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(42)
        
        sample_data = {}
        for ticker in tickers[:10]:  # Limit to first 10 tickers for performance
            # Generate realistic price movements
            returns = np.random.randn(len(dates)) * 0.02
            price = 100 * np.exp(np.cumsum(returns))
            sample_data[ticker] = price
        
        df = pd.DataFrame(sample_data, index=dates)
        return df
    
    @classmethod
    def fetch_data(cls, tickers=None, start_date=None, end_date=None):
        """Fetch historical data using yfinance batch download with fallback"""
        
        if tickers is None:
            tickers = cls.get_all_tickers()
        
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=2*365)
        
        print(f"\n📊 Fetching {len(tickers)} stocks from Yahoo Finance")
        print(f"   From: {start_date.strftime('%Y-%m-%d')}")
        print(f"   To:   {end_date.strftime('%Y-%m-%d')}")
        print("-" * 60)
        
        try:
            # Try to fetch real data
            data = cls.fetch_data_with_retry(tickers, start_date, end_date)
            
            if data.empty:
                print("❌ No data retrieved from Yahoo Finance")
                print("   Using sample data as fallback...")
                prices = cls.create_sample_fallback(tickers, start_date, end_date)
                print(f"✅ Using sample data: {len(prices.columns)} stocks with {len(prices)} days")
                return prices
            
            # Extract close prices correctly
            if len(tickers) == 1:
                # Single ticker case
                if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
                    prices = data[['Close']].copy()
                    prices.columns = tickers
                else:
                    prices = data.copy()
            else:
                # Multiple tickers case
                try:
                    # Try to extract Close prices from multi-level columns
                    if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                        if 'Close' in data.columns.levels[1]:
                            prices = data.xs('Close', axis=1, level=1)
                        else:
                            # Fallback: use all data
                            prices = data
                    else:
                        prices = data
                except Exception as e:
                    print(f"   Error extracting prices: {e}")
                    prices = data
            
            # Ensure we have a DataFrame
            if not isinstance(prices, pd.DataFrame):
                prices = pd.DataFrame(prices)
            
            # Remove any columns that are empty
            prices = prices.dropna(axis=1, how='all')
            
            if prices.empty:
                print("❌ No valid price data extracted")
                print("   Using sample data as fallback...")
                prices = cls.create_sample_fallback(tickers, start_date, end_date)
                print(f"✅ Using sample data: {len(prices.columns)} stocks with {len(prices)} days")
                return prices
            
            print(f"✅ Loaded {len(prices.columns)} stocks with {len(prices)} days")
            return prices
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            print("   Using sample data as fallback...")
            import traceback
            traceback.print_exc()
            prices = cls.create_sample_fallback(tickers, start_date, end_date)
            print(f"✅ Using sample data: {len(prices.columns)} stocks with {len(prices)} days")
            return prices
    
    @classmethod
    def calculate_returns(cls, prices):
        """Calculate daily returns"""
        if prices.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        daily_returns = prices.pct_change().dropna()
        monthly_returns = prices.resample('M').last().pct_change().dropna()
        
        return daily_returns, monthly_returns
    
    @classmethod
    def get_risk_free_rate(cls):
        return 0.03