# core/evaluator.py
"""
Portfolio performance evaluation metrics
"""

import numpy as np
import pandas as pd

class PerformanceEvaluator:
    """Calculate various portfolio performance metrics"""
    
    @staticmethod
    def annualized_return(returns):
        return (1 + returns.mean()) ** 252 - 1
    
    @staticmethod
    def annualized_volatility(returns):
        return returns.std() * np.sqrt(252)
    
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.03):
        daily_rf = risk_free_rate / 252
        excess_return = returns.mean() - daily_rf
        if returns.std() > 0:
            return (excess_return / returns.std()) * np.sqrt(252)
        return 0
    
    @staticmethod
    def sortino_ratio(returns, risk_free_rate=0.03, target=0):
        daily_rf = risk_free_rate / 252
        excess_return = returns.mean() - daily_rf
        downside_returns = returns[returns < target]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        if downside_std > 0:
            return (excess_return / downside_std) * np.sqrt(252)
        return 0
    
    @staticmethod
    def max_drawdown(cumulative_returns):
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def calmar_ratio(returns, cumulative_returns):
        ann_return = PerformanceEvaluator.annualized_return(returns)
        max_dd = PerformanceEvaluator.max_drawdown(cumulative_returns)
        if max_dd != 0:
            return ann_return / abs(max_dd)
        return 0
    
    @staticmethod
    def information_ratio(portfolio_returns, benchmark_returns):
        active_returns = portfolio_returns - benchmark_returns
        if active_returns.std() > 0:
            return (active_returns.mean() / active_returns.std()) * np.sqrt(252)
        return 0
    
    @staticmethod
    def get_all_metrics(returns, cumulative_returns, benchmark_returns=None, risk_free_rate=0.03):
        metrics = {
            'Annualized Return': f"{PerformanceEvaluator.annualized_return(returns)*100:.2f}%",
            'Annualized Volatility': f"{PerformanceEvaluator.annualized_volatility(returns)*100:.2f}%",
            'Sharpe Ratio': f"{PerformanceEvaluator.sharpe_ratio(returns, risk_free_rate):.2f}",
            'Sortino Ratio': f"{PerformanceEvaluator.sortino_ratio(returns, risk_free_rate):.2f}",
            'Maximum Drawdown': f"{PerformanceEvaluator.max_drawdown(cumulative_returns)*100:.2f}%",
            'Calmar Ratio': f"{PerformanceEvaluator.calmar_ratio(returns, cumulative_returns):.2f}"
        }
        if benchmark_returns is not None:
            metrics['Information Ratio'] = f"{PerformanceEvaluator.information_ratio(returns, benchmark_returns):.2f}"
        return metrics