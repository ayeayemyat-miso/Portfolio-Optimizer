# core/optimizer.py
"""
Modern portfolio optimization using Markowitz theory
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Modern Portfolio Optimization with various objectives"""
    
    def __init__(self, returns, risk_free_rate=0.03):
        self.returns = returns
        self.rf_daily = risk_free_rate / 252
        self.rf_annual = risk_free_rate
        self.n_assets = len(returns.columns)
        self.asset_names = returns.columns.tolist()
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
    
    def portfolio_performance(self, weights):
        """Calculate portfolio return, risk, and Sharpe ratio"""
        port_return_daily = np.sum(self.mean_returns * weights)
        port_return_annual = (1 + port_return_daily) ** 252 - 1
        port_risk_daily = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        port_risk_annual = port_risk_daily * np.sqrt(252)
        sharpe = (port_return_daily - self.rf_daily) / port_risk_daily if port_risk_daily > 0 else 0
        return port_return_annual, port_risk_annual, sharpe
    
    def optimize_max_sharpe(self):
        """Find weights that maximize Sharpe ratio"""
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        def negative_sharpe(weights):
            return -self.portfolio_performance(weights)[2]
        
        try:
            result = minimize(negative_sharpe, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            return result.x if result.success else initial_weights
        except:
            return initial_weights
    
    def optimize_min_volatility(self):
        """Find weights that minimize volatility"""
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        def portfolio_volatility(weights):
            return self.portfolio_performance(weights)[1]
        
        try:
            result = minimize(portfolio_volatility, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            return result.x if result.success else initial_weights
        except:
            return initial_weights
    
    def optimize_target_return(self, target_return_annual):
        """Find minimum risk portfolio for target annual return"""
        target_return_daily = (1 + target_return_annual) ** (1/252) - 1
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) - target_return_daily}
        ]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        try:
            result = minimize(lambda x: self.portfolio_performance(x)[1],
                            initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            return result.x if result.success else None
        except:
            return None
    
    def efficient_frontier(self, points=50):
        """Generate efficient frontier"""
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        min_return_annual = (1 + min_return) ** 252 - 1
        max_return_annual = (1 + max_return) ** 252 - 1
        target_returns = np.linspace(min_return_annual, max_return_annual, points)
        
        frontier = []
        for target in target_returns:
            weights = self.optimize_target_return(target)
            if weights is not None:
                ret, risk, sharpe = self.portfolio_performance(weights)
                frontier.append({'return': ret * 100, 'risk': risk, 'sharpe': sharpe, 'target': target})
        
        return pd.DataFrame(frontier)
    
    def get_weights_dataframe(self, weights):
        """Convert weights to readable DataFrame"""
        df = pd.DataFrame({
            'Asset': self.asset_names,
            'Weight (%)': np.round(weights * 100, 2)
        }).sort_values('Weight (%)', ascending=False)
        
        # Add sector mapping
        sector_map = {
            'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'GOOGL': 'Tech', 'META': 'Tech',
            'JPM': 'Financial', 'V': 'Financial', 'MA': 'Financial', 'BAC': 'Financial', 'GS': 'Financial',
            'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare', 'ABBV': 'Healthcare',
            'WMT': 'Consumer', 'HD': 'Consumer', 'COST': 'Consumer', 'MCD': 'Consumer', 'PG': 'Consumer',
            'XOM': 'Energy', 'CVX': 'Energy', 'CAT': 'Industrial', 'BA': 'Industrial', 'GE': 'Industrial'
        }
        df['Sector'] = df['Asset'].map(sector_map).fillna('Other')
        return df