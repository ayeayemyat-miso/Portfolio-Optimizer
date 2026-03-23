# core/data_fetcher.py
"""
Fetch stock data from Yahoo Finance using batch download
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

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
    def fetch_data(cls, tickers=None, start_date=None, end_date=None):
        """Fetch historical data using yfinance batch download"""
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
            # Use yfinance's batch download
            data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                group_by='ticker',
                auto_adjust=True,
                progress=True,
                threads=True
            )
            
            if data.empty:
                print("❌ No data retrieved")
                return pd.DataFrame()
            
            # Extract close prices correctly
            if len(tickers) == 1:
                # Single ticker case
                prices = data[['Close']].copy()
                prices.columns = tickers
            else:
                # Multiple tickers case
                # Check if data has 'Close' at the top level (single ticker case already handled)
                # For multiple tickers, the structure is multi-level columns
                if 'Close' in data.columns.levels[1] if hasattr(data.columns, 'levels') else False:
                    # Multi-level columns
                    prices = data.xs('Close', axis=1, level=1)
                else:
                    # Fallback: try to get 'Close' from each ticker
                    prices = pd.DataFrame()
                    for ticker in tickers:
                        if ticker in data.columns.levels[0]:
                            prices[ticker] = data[ticker]['Close']
            
            # Remove any columns that are empty
            prices = prices.dropna(axis=1, how='all')
            
            print(f"✅ Loaded {len(prices.columns)} stocks with {len(prices)} days")
            return prices
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
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