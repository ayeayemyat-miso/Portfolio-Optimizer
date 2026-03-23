# core/treynor_black.py
"""
Treynor-Black active portfolio optimization
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

class TreynorBlackOptimizer:
    """Implements the Treynor-Black model"""
    
    def __init__(self, stock_returns, market_returns, risk_free_rate=0.03):
        self.stock_returns = stock_returns
        self.market_returns = market_returns
        self.rf = risk_free_rate / 252
    
    def compute_alpha_beta(self, stock):
        """Compute alpha, beta, residual variance, and p-value"""
        stock_ret = self.stock_returns[stock].dropna()
        market_ret = self.market_returns.loc[stock_ret.index].dropna()
        
        stock_excess = stock_ret - self.rf
        market_excess = market_ret - self.rf
        
        X = sm.add_constant(market_excess)
        model = sm.OLS(stock_excess, X).fit()
        
        return {
            'alpha': model.params[0],
            'beta': model.params[1],
            'p_value': model.pvalues[0],
            'resid_var': model.mse_resid,
            't_stat': model.tvalues[0]
        }
    
    def identify_mispriced_stocks(self, significance=0.10):
        """Identify stocks with statistically significant alpha"""
        results = []
        for stock in self.stock_returns.columns:
            try:
                stats = self.compute_alpha_beta(stock)
                results.append({
                    'ticker': stock,
                    'alpha_annual': stats['alpha'] * 252,
                    'beta': stats['beta'],
                    'p_value': stats['p_value'],
                    'resid_var_daily': stats['resid_var']
                })
            except:
                continue
        
        df = pd.DataFrame(results)
        if not df.empty:
            df['abs_alpha'] = df['alpha_annual'].abs()
            df = df.sort_values('abs_alpha', ascending=False)
        return df
    
    def compute_active_weights(self, mispriced_df):
        """Compute optimal active portfolio weights"""
        if mispriced_df.empty:
            return None
        
        # Raw active weights
        raw_weights = mispriced_df['alpha_annual'] / 252 / mispriced_df['resid_var_daily']
        raw_weights = raw_weights / raw_weights.sum()
        mispriced_df = mispriced_df.copy()
        mispriced_df['active_weight_raw'] = raw_weights
        
        # Active portfolio statistics
        active_alpha = (mispriced_df['alpha_annual'] / 252 * mispriced_df['active_weight_raw']).sum()
        active_beta = (mispriced_df['beta'] * mispriced_df['active_weight_raw']).sum()
        active_resid_var = ((mispriced_df['active_weight_raw']**2) * mispriced_df['resid_var_daily']).sum()
        
        # Market statistics
        market_ret_daily = self.market_returns.mean()
        market_var_daily = self.market_returns.var()
        market_excess_daily = market_ret_daily - self.rf
        
        # Optimal active weight
        w_active = (active_alpha / active_resid_var) / (market_excess_daily / market_var_daily)
        w_active = np.clip(w_active, -2, 2)
        w_market = 1 - w_active
        
        # Final weights
        market_weights = pd.Series(1/len(self.stock_returns.columns), index=self.stock_returns.columns)
        active_weights = mispriced_df.set_index('ticker')['active_weight_raw']
        final_weights = w_market * market_weights + w_active * active_weights.reindex(market_weights.index, fill_value=0)
        final_weights = final_weights / final_weights.sum()
        
        return {
            'mispriced_stocks': mispriced_df,
            'active_weights': active_weights,
            'w_active': w_active,
            'w_market': w_market,
            'active_alpha': active_alpha,
            'active_beta': active_beta,
            'active_resid_var': active_resid_var,
            'final_weights': final_weights
        }