"""
Smart Portfolio Optimizer - Complete Streamlit App
Includes: Markowitz, Treynor-Black, Value Screener, Monte Carlo, Portfolio Comparison, Risk Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta, datetime
import sys
import os
import yfinance as yf
from scipy import stats
import io

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data_fetcher import PortfolioDataFetcher as DataFetcher
from core.optimizer import PortfolioOptimizer
from core.evaluator import PerformanceEvaluator
from core.treynor_black import TreynorBlackOptimizer

# Page configuration
st.set_page_config(
    page_title="Smart Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'optimization_run' not in st.session_state:
    st.session_state.optimization_run = False
if 'optimal_weights' not in st.session_state:
    st.session_state.optimal_weights = None
if 'portfolio_stats' not in st.session_state:
    st.session_state.portfolio_stats = None
if 'returns_data' not in st.session_state:
    st.session_state.returns_data = None
if 'benchmark_returns' not in st.session_state:
    st.session_state.benchmark_returns = None
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []

BENCHMARKS = {
    'S&P 500': '^GSPC',
    'Nasdaq 100': '^NDX',
    'Dow Jones': '^DJI',
    'Russell 2000': '^RUT'
}

OPTIMIZATION_GOALS = {
    'Maximum Sharpe Ratio': 'sharpe',
    'Minimum Volatility': 'min_vol',
    'Target Return 6%': 0.06,
    'Target Return 8%': 0.08,
    'Target Return 10%': 0.10,
    'Target Return 12%': 0.12,
    'Target Return 15%': 0.15,
    'Target Return 20%': 0.20,
    'Equal Weight': 'equal'
}

# Helper functions
def get_sector(ticker):
    """Get sector for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('sector', 'Other')
    except:
        return 'Other'

def compute_var_cvar(returns, confidence=0.95):
    """Compute VaR and CVaR"""
    if len(returns) < 2:
        return np.nan, np.nan
    var = np.percentile(returns, (1-confidence)*100)
    cvar = returns[returns <= var].mean()
    return var, cvar

def compute_skewness_kurtosis(returns):
    """Compute skewness and kurtosis"""
    if len(returns) < 3:
        return np.nan, np.nan
    return stats.skew(returns), stats.kurtosis(returns)

def compute_rolling_sharpe(returns, window=63, rf=0.03/252):
    """Compute rolling Sharpe ratio"""
    rolling_returns = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    return (rolling_returns - rf) / rolling_vol

def get_valuation_metrics(ticker):
    """Get valuation metrics for a stock"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'pe': info.get('trailingPE'),
            'pb': info.get('priceToBook'),
            'dividend': info.get('dividendYield', 0),
            'roe': info.get('returnOnEquity'),
            'sector': info.get('sector', 'Unknown')
        }
    except:
        return {'pe': None, 'pb': None, 'dividend': 0, 'roe': None, 'sector': 'Unknown'}

def create_excel_report(weights_df, metrics, selected_tickers):
    """Create Excel report"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        weights_df.to_excel(writer, sheet_name='Portfolio Weights', index=False)
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ['Value']
        metrics_df.to_excel(writer, sheet_name='Performance Metrics')
        output.seek(0)
    return output

# ============================================================
# RISK ANALYSIS FUNCTIONS (NEW)
# ============================================================

def calculate_portfolio_risk_score(tickers, weights, betas, alphas):
    """Calculate portfolio risk score and position sizing recommendations"""
    
    # Calculate weighted average beta
    portfolio_beta = sum(weights[i] * betas[i] for i in range(len(tickers)))
    
    # Risk classification
    if portfolio_beta < 0.8:
        risk_level = "🟢 Conservative"
        risk_color = "green"
        risk_description = "Low volatility, defensive portfolio"
        suggested_leverage = "100% allocation"
    elif portfolio_beta < 1.2:
        risk_level = "🟡 Moderate"
        risk_color = "orange"
        risk_description = "Balanced, market-like risk"
        suggested_leverage = "100% allocation"
    elif portfolio_beta < 1.6:
        risk_level = "🟠 Aggressive"
        risk_color = "orange-red"
        risk_description = "High volatility, growth-focused"
        suggested_leverage = "80-100% allocation"
    else:
        risk_level = "🔴 Very Aggressive"
        risk_color = "red"
        risk_description = "Very high volatility, speculative"
        suggested_leverage = "60-80% allocation"
    
    # Calculate risk metrics
    total_risk = portfolio_beta * 0.15  # Assuming market vol 15%
    expected_annual_return = sum(weights[i] * alphas[i] for i in range(len(tickers))) + 0.10
    
    # Position sizing recommendations based on Beta
    position_sizing = []
    for i, ticker in enumerate(tickers):
        beta = betas[i]
        alpha = alphas[i]
        
        # Kelly-style position sizing based on Alpha/Beta ratio
        if alpha > 10 and beta < 1.5:
            suggested_weight = min(weights[i] * 1.2, 0.25)
            recommendation = "✅ Overweight"
            reason = f"High alpha ({alpha:.1f}%) with reasonable beta"
        elif alpha < -5:
            suggested_weight = max(weights[i] * 0.5, 0.02)
            recommendation = "❌ Underweight"
            reason = f"Negative alpha ({alpha:.1f}%) underperforming market"
        elif beta > 1.8:
            suggested_weight = min(weights[i] * 0.7, 0.15)
            recommendation = "⚠️ Reduce"
            reason = f"Very high beta ({beta:.2f}) increases portfolio risk"
        elif beta < 0.8:
            suggested_weight = min(weights[i] * 1.1, 0.20)
            recommendation = "✅ Hold/Add"
            reason = f"Low beta ({beta:.2f}) provides stability"
        else:
            suggested_weight = weights[i]
            recommendation = "Hold"
            reason = f"Well-balanced risk/return profile"
        
        position_sizing.append({
            'Ticker': ticker,
            'Current Weight': f"{weights[i]*100:.1f}%",
            'Beta': f"{beta:.2f}",
            'Alpha': f"{alpha:.1f}%",
            'Recommended Weight': f"{suggested_weight*100:.1f}%",
            'Action': recommendation,
            'Reason': reason
        })
    
    return {
        'portfolio_beta': portfolio_beta,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'risk_description': risk_description,
        'suggested_leverage': suggested_leverage,
        'total_risk': total_risk,
        'expected_return': expected_annual_return,
        'position_sizing': position_sizing
    }

def show_risk_analysis_tab(tickers, risk_free_rate):
    """Display portfolio risk analysis and position sizing recommendations"""
    st.header("🎯 Portfolio Risk Analysis & Position Sizing")
    
    # Get the current optimal weights from session state
    if st.session_state.optimal_weights is None:
        st.warning("Please run optimization first")
        return
    
    # Get returns and calculate betas/alphas
    rets = st.session_state.returns_data[tickers].dropna()
    common_idx = rets.index.intersection(st.session_state.benchmark_returns.index)
    
    if len(common_idx) < 60:
        st.warning("Insufficient data for risk analysis")
        return
    
    rets_aligned = rets.loc[common_idx]
    market_aligned = st.session_state.benchmark_returns.loc[common_idx]
    
    # Calculate betas and alphas for all tickers
    betas = []
    alphas = []
    for ticker in tickers:
        stock_ret = rets_aligned[ticker]
        market_ret = market_aligned
        mask = ~(stock_ret.isna() | market_ret.isna())
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            market_ret[mask], stock_ret[mask]
        )
        betas.append(slope)
        alphas.append(intercept * 252 * 100)
    
    # Get current weights
    weights = np.array([st.session_state.optimal_weights[t] for t in tickers])
    
    # Calculate risk score
    risk_analysis = calculate_portfolio_risk_score(tickers, weights, betas, alphas)
    
    # Display risk profile
    st.subheader("📊 Portfolio Risk Profile")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Portfolio Beta",
            f"{risk_analysis['portfolio_beta']:.2f}",
            delta=None
        )
    with col2:
        st.metric(
            "Risk Level",
            risk_analysis['risk_level'],
            delta=None
        )
    with col3:
        st.metric(
            "Expected Annual Volatility",
            f"{risk_analysis['total_risk']*100:.1f}%",
            delta=None
        )
    
    # Risk gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_analysis['portfolio_beta'],
        title = {'text': "Portfolio Beta Gauge"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 2.5], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.8], 'color': "lightgreen", 'name': 'Conservative'},
                {'range': [0.8, 1.2], 'color': "yellow", 'name': 'Moderate'},
                {'range': [1.2, 1.6], 'color': "orange", 'name': 'Aggressive'},
                {'range': [1.6, 2.5], 'color': "red", 'name': 'Very Aggressive'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_analysis['portfolio_beta']
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Investor profile questionnaire
    st.subheader("👤 Investor Profile Assessment")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Your Age", 20, 80, 35)
        investment_horizon = st.selectbox("Investment Horizon", 
                                          ["< 3 years", "3-5 years", "5-10 years", "> 10 years"])
    with col2:
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"],
            value="Moderate"
        )
        loss_tolerance = st.slider("Maximum acceptable loss (%)", 0, 50, 20)
    
    # Calculate recommended beta based on profile
    if risk_tolerance == "Very Conservative":
        target_beta = 0.6
    elif risk_tolerance == "Conservative":
        target_beta = 0.8
    elif risk_tolerance == "Moderate":
        target_beta = 1.0
    elif risk_tolerance == "Aggressive":
        target_beta = 1.3
    else:
        target_beta = 1.6
    
    # Age adjustment
    if age > 60:
        target_beta *= 0.7
    elif age < 30:
        target_beta *= 1.2
    
    # Horizon adjustment
    if investment_horizon == "< 3 years":
        target_beta *= 0.6
    elif investment_horizon == "3-5 years":
        target_beta *= 0.8
    elif investment_horizon == "> 10 years":
        target_beta *= 1.2
    
    target_beta = max(0.4, min(2.0, target_beta))
    
    # Compare current vs target
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Your Current Portfolio Beta", f"{risk_analysis['portfolio_beta']:.2f}")
    with col2:
        st.metric("Recommended Beta for You", f"{target_beta:.2f}", 
                 delta=f"{target_beta - risk_analysis['portfolio_beta']:.2f}")
    
    if risk_analysis['portfolio_beta'] > target_beta * 1.2:
        st.warning("⚠️ **Your portfolio is riskier than recommended.** Consider reducing high-beta positions.")
    elif risk_analysis['portfolio_beta'] < target_beta * 0.8:
        st.info("ℹ️ **Your portfolio is more conservative than recommended.** You could add growth stocks.")
    else:
        st.success("✅ **Your portfolio risk matches your profile!**")
    
    # Position sizing recommendations
    st.subheader("📋 Position Sizing Recommendations")
    
    sizing_df = pd.DataFrame(risk_analysis['position_sizing'])
    st.dataframe(sizing_df, use_container_width=True, hide_index=True)
    
    # Risk mitigation strategies
    st.subheader("🛡️ Risk Management Strategies")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Based on your portfolio:**")
        
        high_beta_stocks = [s for s in risk_analysis['position_sizing'] if float(s['Beta']) > 1.5]
        if high_beta_stocks:
            st.warning(f"Reduce exposure to high-beta stocks: {', '.join([s['Ticker'] for s in high_beta_stocks])}")
        
        negative_alpha = [s for s in risk_analysis['position_sizing'] if '%' in s['Alpha'] and float(s['Alpha'].replace('%', '')) < 0]
        if negative_alpha:
            st.warning(f"Consider selling underperformers: {', '.join([s['Ticker'] for s in negative_alpha])}")
    
    with col2:
        st.markdown("**📊 Suggested Actions:**")
        
        if risk_analysis['portfolio_beta'] > 1.3:
            st.info("1. Add bonds or defensive stocks to lower beta")
            st.info("2. Use stop-loss orders (5-10%)")
        elif risk_analysis['portfolio_beta'] < 0.7:
            st.info("1. Add growth stocks to increase returns")
            st.info("2. Consider small-cap exposure")
        else:
            st.success("1. Maintain current allocation")
            st.success("2. Rebalance quarterly")
    
    # Portfolio stress test
    st.subheader("📉 Portfolio Stress Test")
    
    market_shocks = {
        "Market Crash (-20%)": -0.20,
        "Market Correction (-10%)": -0.10,
        "Mild Downturn (-5%)": -0.05,
        "Market Rally (+10%)": 0.10,
        "Bull Market (+20%)": 0.20
    }
    
    shock_results = []
    for shock_name, shock_return in market_shocks.items():
        portfolio_impact = shock_return * risk_analysis['portfolio_beta']
        shock_results.append({
            'Scenario': shock_name,
            'Market Return': f"{shock_return*100:+.1f}%",
            'Expected Portfolio Return': f"{portfolio_impact*100:+.1f}%",
            'Impact': "🔴 Severe" if portfolio_impact < -0.15 else "🟡 Moderate" if portfolio_impact < -0.08 else "🟢 Mild"
        })
    
    shock_df = pd.DataFrame(shock_results)
    st.dataframe(shock_df, use_container_width=True, hide_index=True)
    
    # Final recommendation
    st.subheader("💡 Final Recommendation")
    
    if risk_analysis['portfolio_beta'] > target_beta:
        st.info(f"""
        **Action Plan to Reduce Risk:**
        1. Reduce position sizes in high-beta stocks
        2. Add defensive holdings (Consumer Staples, Utilities, Healthcare)
        3. Consider cash position (10-20%) for volatility buffer
        4. Set stop-loss at {loss_tolerance}% maximum loss
        
        **Expected Outcome:** Lower beta from {risk_analysis['portfolio_beta']:.2f} to {target_beta:.2f}
        """)
    else:
        st.success(f"""
        **Your portfolio is well-aligned with your risk profile!**
        
        - Current Beta: {risk_analysis['portfolio_beta']:.2f}
        - Target Beta: {target_beta:.2f}
        - Suggested max allocation per stock: 20-25%
        - Rebalance: Quarterly or when weights drift >5%
        
        Keep monitoring your portfolio's beta and rebalance as market conditions change.
        """)

# ============================================================
# MAIN FUNCTIONS
# ============================================================

def load_and_optimize(tickers, start_date, end_date, goal_value, risk_free_rate, benchmark_symbol):
    """Load data and run optimization"""
    try:
        # Fetch data for stocks using your existing fetcher
        fetcher = DataFetcher()
        prices = fetcher.fetch_data(tickers, start_date, end_date)
        
        if prices.empty:
            st.error("No data retrieved. Please check ticker symbols.")
            return
        
        daily_returns = prices.pct_change().dropna()
        
        # Remove timezone info for consistent indexing
        if daily_returns.index.tz is not None:
            daily_returns.index = daily_returns.index.tz_localize(None)
        
        st.session_state.returns_data = daily_returns
        
        # Fetch benchmark directly with yfinance
        with st.spinner(f"Loading {benchmark_symbol} benchmark..."):
            try:
                benchmark = yf.Ticker(benchmark_symbol)
                benchmark_hist = benchmark.history(start=start_date, end=end_date)
                
                if not benchmark_hist.empty:
                    benchmark_returns = benchmark_hist['Close'].pct_change().dropna()
                    
                    # Remove timezone
                    if benchmark_returns.index.tz is not None:
                        benchmark_returns.index = benchmark_returns.index.tz_localize(None)
                    
                    # Remove any NaN values
                    benchmark_returns = benchmark_returns.dropna()
                    
                    # Align dates with stock returns
                    common_dates = daily_returns.index.intersection(benchmark_returns.index)
                    
                    if len(common_dates) > 0:
                        st.session_state.benchmark_returns = benchmark_returns.loc[common_dates]
                        st.success(f"✅ Loaded {len(common_dates)} days of {benchmark_symbol} data")
                    else:
                        st.warning(f"No overlapping dates with benchmark")
                        st.session_state.benchmark_returns = None
                else:
                    st.warning(f"Could not fetch {benchmark_symbol} data")
                    st.session_state.benchmark_returns = None
            except Exception as e:
                st.warning(f"Benchmark error: {str(e)}")
                st.session_state.benchmark_returns = None
        
        # Optimize
        optimizer = PortfolioOptimizer(daily_returns, risk_free_rate)
        
        if goal_value == 'sharpe':
            weights = optimizer.optimize_max_sharpe()
            title = "Max Sharpe"
        elif goal_value == 'min_vol':
            weights = optimizer.optimize_min_volatility()
            title = "Min Volatility"
        elif goal_value == 'equal':
            weights = np.array([1/len(tickers)] * len(tickers))
            title = "Equal Weight"
        else:  # target return
            weights = optimizer.optimize_target_return(goal_value)
            if weights is None:
                weights = np.array([1/len(tickers)] * len(tickers))
                title = f"Target {goal_value*100:.0f}% (Fallback to Equal Weight)"
            else:
                title = f"Target {goal_value*100:.0f}%"
        
        if weights is None:
            weights = np.array([1/len(tickers)] * len(tickers))
        
        ret, risk, sharpe = optimizer.portfolio_performance(weights)
        
        weights_dict = {tickers[i]: weights[i] for i in range(len(tickers))}
        st.session_state.optimal_weights = weights_dict
        st.session_state.portfolio_stats = {
            'return': ret, 'volatility': risk, 'sharpe_ratio': sharpe,
            'strategy': title, 'n_assets': len(tickers)
        }
        st.session_state.optimization_run = True
        
        st.success(f"✅ Optimization complete! Using {title}")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.optimization_run = False

def show_markowitz_tab(tickers, goal_name, risk_free_rate, benchmark_name):
    """Display Markowitz analysis"""
    st.header("📊 Markowitz Portfolio Analysis")
    
    rets = st.session_state.returns_data[tickers].dropna()
    optimizer = PortfolioOptimizer(rets, risk_free_rate)
    
    weights = np.array([st.session_state.optimal_weights[t] for t in tickers])
    ret, risk, sharpe = optimizer.portfolio_performance(weights)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Expected Annual Return", f"{ret*100:.2f}%")
    with col2: st.metric("Annual Risk", f"{risk:.2f}%")
    with col3: st.metric("Sharpe Ratio", f"{sharpe:.3f}")
    with col4: st.metric("Strategy", goal_name)
    
    st.subheader("📈 Efficient Frontier")
    frontier = optimizer.efficient_frontier()
    
    fig = go.Figure()
    if not frontier.empty:
        fig.add_trace(go.Scatter(
            x=frontier['risk'], y=frontier['return'],
            mode='lines', name='Efficient Frontier',
            line=dict(color='#3498db', width=2)
        ))
    fig.add_trace(go.Scatter(
        x=[risk], y=[ret*100], mode='markers',
        name='Your Portfolio', marker=dict(size=20, color='red', symbol='star')
    ))
    fig.update_layout(xaxis_title='Risk (%)', yaxis_title='Return (%)', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Portfolio Allocation")
    weights_df = pd.DataFrame({'Asset': tickers, 'Weight (%)': weights*100})
    weights_df = weights_df.sort_values('Weight (%)', ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(weights_df, x='Asset', y='Weight (%)', color='Weight (%)',
                         color_continuous_scale='Viridis')
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        sector_weights = {}
        for ticker, weight in zip(tickers, weights):
            sector = get_sector(ticker)
            sector_weights[sector] = sector_weights.get(sector, 0) + weight*100
        
        sector_df = pd.DataFrame(list(sector_weights.items()), columns=['Sector', 'Weight (%)'])
        fig_pie = px.pie(sector_df, values='Weight (%)', names='Sector', hole=0.3)
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.subheader("📊 Rolling Sharpe Ratio (63-day)")
    port_returns = rets.dot(weights)
    rolling_sharpe = compute_rolling_sharpe(port_returns, window=63, rf=risk_free_rate/252)
    
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe,
                                   mode='lines', fill='tozeroy', name='Rolling Sharpe'))
    fig_roll.update_layout(height=400, xaxis_title='Date', yaxis_title='Sharpe Ratio')
    st.plotly_chart(fig_roll, use_container_width=True)
    
    st.subheader("📊 Risk Metrics")
    var95, cvar95 = compute_var_cvar(port_returns, 0.95)
    var99, cvar99 = compute_var_cvar(port_returns, 0.99)
    skew, kurt = compute_skewness_kurtosis(port_returns)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("VaR (95%)", f"{var95*100:.2f}%")
        st.metric("VaR (99%)", f"{var99*100:.2f}%")
    with col2:
        st.metric("CVaR (95%)", f"{cvar95*100:.2f}%")
        st.metric("CVaR (99%)", f"{cvar99*100:.2f}%")
    with col3:
        st.metric("Skewness", f"{skew:.2f}")
        st.metric("Kurtosis", f"{kurt:.2f}")

def show_treynor_black_tab(tickers, risk_free_rate):
    """Display Treynor-Black analysis"""
    st.header("📈 Treynor-Black Active Portfolio Model")
    
    if st.session_state.benchmark_returns is None:
        st.warning("⚠️ Benchmark data not available.")
        return
    
    rets = st.session_state.returns_data[tickers].copy()
    rets = rets.dropna()
    
    common_idx = rets.index.intersection(st.session_state.benchmark_returns.index)
    
    if len(common_idx) < 60:
        st.warning(f"Only {len(common_idx)} overlapping trading days. Need at least 60 days.")
        return
    
    rets_aligned = rets.loc[common_idx]
    market_aligned = st.session_state.benchmark_returns.loc[common_idx]
    
    yahoo_betas = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            yahoo_betas[ticker] = info.get('beta', None)
        except:
            yahoo_betas[ticker] = None
    
    results = []
    for ticker in tickers:
        try:
            stock_ret = rets_aligned[ticker]
            market_ret = market_aligned
            
            mask = ~(stock_ret.isna() | market_ret.isna())
            stock_ret_clean = stock_ret[mask]
            market_ret_clean = market_ret[mask]
            
            if len(stock_ret_clean) < 60:
                continue
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(market_ret_clean, stock_ret_clean)
            
            alpha_annual = intercept * 252 * 100
            beta_val = slope
            r_squared = r_value ** 2
            
            yahoo_beta = yahoo_betas.get(ticker)
            
            results.append({
                'Ticker': ticker,
                'Beta (Daily)': f"{beta_val:.2f}",
                'Yahoo Beta (5Y Monthly)': f"{yahoo_beta:.2f}" if yahoo_beta else 'N/A',
                'Difference': f"{abs(beta_val - yahoo_beta):.2f}" if yahoo_beta else 'N/A',
                'Alpha (Annual)': f"{alpha_annual:.2f}%",
                'R-Squared': f"{r_squared:.3f}",
                'p-value': f"{p_value:.4f}",
                'Significant': '✅ Yes' if p_value < 0.10 else '❌ No',
                'Days': len(stock_ret_clean)
            })
            
        except Exception as e:
            continue
    
    if not results:
        st.warning("Could not compute alphas.")
        return
    
    df = pd.DataFrame(results)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        positive_alpha = len([r for r in results if float(r['Alpha (Annual)'].replace('%', '')) > 0])
        st.metric("Stocks with Positive Alpha", f"{positive_alpha}/{len(results)}")
    
    with col2:
        sig_stocks = len([r for r in results if r['Significant'] == '✅ Yes'])
        st.metric("Statistically Significant", f"{sig_stocks}/{len(results)}")
    
    with col3:
        good_beta = len([r for r in results if r['Yahoo Beta (5Y Monthly)'] != 'N/A' and abs(float(r['Beta (Daily)']) - float(r['Yahoo Beta (5Y Monthly)'])) < 0.3])
        st.metric("Beta Within 0.3 of Yahoo", f"{good_beta}/{len(results)}")
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.info("""
    💡 **How to interpret:**
    - **Beta > 1.2**: More volatile than market
    - **Beta 0.8-1.2**: Similar volatility to market
    - **Beta < 0.8**: Less volatile than market
    - **Positive Alpha**: Stock outperformed benchmark
    - **p-value < 0.10**: Statistically significant relationship
    """)
    
    best_performers = [r for r in results if float(r['Alpha (Annual)'].replace('%', '')) > 5 and r['Significant'] == '✅ Yes']
    if best_performers:
        st.success(f"**Top Performers:** {', '.join([r['Ticker'] for r in best_performers[:3]])} show strong positive alpha.")

def show_value_screener_tab(tickers):
    """Display value screener"""
    st.header("💰 Value Stock Screener")
    
    stocks_data = []
    for ticker in tickers[:20]:
        metrics = get_valuation_metrics(ticker)
        rets = st.session_state.returns_data[ticker].dropna()
        annual_return = (1 + rets.mean()) ** 252 - 1
        annual_vol = rets.std() * np.sqrt(252)
        
        score = 3
        if metrics['pe'] and metrics['pe'] < 15: score += 2
        elif metrics['pe'] and metrics['pe'] < 20: score += 1
        if metrics['pb'] and metrics['pb'] < 2: score += 2
        elif metrics['pb'] and metrics['pb'] < 3: score += 1
        if annual_return > 0.15: score += 2
        elif annual_return > 0.08: score += 1
        if metrics['dividend'] and metrics['dividend'] > 0.02: score += 1
        
        stocks_data.append({
            'Ticker': ticker,
            'P/E': f"{metrics['pe']:.1f}" if metrics['pe'] else 'N/A',
            'P/B': f"{metrics['pb']:.1f}" if metrics['pb'] else 'N/A',
            'Div Yield': f"{metrics['dividend']*100:.1f}%" if metrics['dividend'] else 'N/A',
            'Ann Return': f"{annual_return*100:.1f}%",
            'Value Score': score,
            'Sector': metrics['sector']
        })
    
    stocks_data.sort(key=lambda x: x['Value Score'], reverse=True)
    df = pd.DataFrame(stocks_data)
    
    def color_score(val):
        if val >= 7:
            return 'background-color: #d4edda; color: #155724'
        elif val >= 5:
            return 'background-color: #fff3cd; color: #856404'
        return 'background-color: #f8d7da; color: #721c24'
    
    styled_df = df.style.map(color_score, subset=['Value Score'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ### Scoring System:
    - 🟢 **Strong Value (7+)** - Excellent fundamentals, consider buying
    - 🟡 **Good Value (5-6)** - Reasonable valuation  
    - 🟠 **Fair Value (3-4)** - Neutral, monitor
    - 🔴 **Poor Value (0-2)** - Overvalued or weak fundamentals
    """)

def show_monte_carlo_tab(tickers, goal_value, risk_free_rate):
    """Display Monte Carlo simulation"""
    st.header("🔄 Monte Carlo Simulation")
    
    rets = st.session_state.returns_data[tickers].dropna()
    optimizer = PortfolioOptimizer(rets, risk_free_rate)
    
    if goal_value == 'sharpe':
        weights = optimizer.optimize_max_sharpe()
    elif goal_value == 'min_vol':
        weights = optimizer.optimize_min_volatility()
    elif goal_value == 'equal':
        weights = np.array([1/len(tickers)] * len(tickers))
    else:
        weights = optimizer.optimize_target_return(goal_value)
    
    if weights is None:
        weights = np.array([1/len(tickers)] * len(tickers))
    
    n_sim = 1000
    n_days = 252
    mean = rets.mean()
    cov = rets.cov()
    
    np.random.seed(42)
    sim_returns = np.random.multivariate_normal(mean, cov, (n_days, n_sim))
    port_returns_sim = sim_returns @ weights
    cumulative_wealth = (1 + port_returns_sim).cumprod(axis=0)
    final_wealth = cumulative_wealth[-1, :]
    
    mean_final = np.mean(final_wealth)
    median_final = np.median(final_wealth)
    var95 = np.percentile(final_wealth, 5)
    var99 = np.percentile(final_wealth, 1)
    prob_loss = (final_wealth < 1).mean() * 100
    prob_gain_20 = (final_wealth > 1.2).mean() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Mean Final Value", f"${mean_final:.2f}")
    with col2: st.metric("Median Final Value", f"${median_final:.2f}")
    with col3: st.metric("VaR (95%)", f"${var95:.2f}")
    with col4: st.metric("Probability of Loss", f"{prob_loss:.1f}%")
    
    col1, col2 = st.columns(2)
    with col1: st.metric("Probability of 20%+ Gain", f"{prob_gain_20:.1f}%")
    with col2: st.metric("VaR (99%)", f"${var99:.2f}")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=final_wealth, nbinsx=50, name='Simulated Outcomes'))
    fig.add_vline(x=mean_final, line_dash="dash", line_color="red", annotation_text=f"Mean: ${mean_final:.2f}")
    fig.add_vline(x=1, line_dash="dot", line_color="gray", annotation_text="Initial")
    fig.update_layout(title='Distribution of Final Portfolio Value ($1 Initial Investment)',
                      xaxis_title='Final Value', yaxis_title='Frequency', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"Based on {n_sim:,} simulations over 1 year (252 trading days)")

def show_compare_tab(tickers, risk_free_rate):
    """Compare different portfolio strategies"""
    st.header("⚖️ Portfolio Strategy Comparison")
    
    rets = st.session_state.returns_data[tickers].dropna()
    optimizer = PortfolioOptimizer(rets, risk_free_rate)
    
    strategies = []
    
    w_sharpe = optimizer.optimize_max_sharpe()
    if w_sharpe is not None:
        r, risk, s = optimizer.portfolio_performance(w_sharpe)
        port_returns = rets.dot(w_sharpe)
        var95, _ = compute_var_cvar(port_returns, 0.95)
        strategies.append({'Strategy': 'Max Sharpe', 'Return': f"{r*100:.1f}%", 
                          'Risk': f"{risk:.1f}%", 'Sharpe': f"{s:.2f}", 'VaR 95%': f"{var95*100:.1f}%"})
    
    w_equal = np.array([1/len(tickers)] * len(tickers))
    r, risk, s = optimizer.portfolio_performance(w_equal)
    port_returns = rets.dot(w_equal)
    var95, _ = compute_var_cvar(port_returns, 0.95)
    strategies.append({'Strategy': 'Equal Weight', 'Return': f"{r*100:.1f}%",
                      'Risk': f"{risk:.1f}%", 'Sharpe': f"{s:.2f}", 'VaR 95%': f"{var95*100:.1f}%"})
    
    w_minvol = optimizer.optimize_min_volatility()
    if w_minvol is not None:
        r, risk, s = optimizer.portfolio_performance(w_minvol)
        port_returns = rets.dot(w_minvol)
        var95, _ = compute_var_cvar(port_returns, 0.95)
        strategies.append({'Strategy': 'Min Volatility', 'Return': f"{r*100:.1f}%",
                          'Risk': f"{risk:.1f}%", 'Sharpe': f"{s:.2f}", 'VaR 95%': f"{var95*100:.1f}%"})
    
    df = pd.DataFrame(strategies)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    best_sharpe = max(strategies, key=lambda x: float(x['Sharpe']))
    st.success(f"🏆 **Recommendation**: {best_sharpe['Strategy']} offers the best risk-adjusted returns with Sharpe Ratio of {best_sharpe['Sharpe']}")

# ============================================================
# MAIN APP
# ============================================================

def main():
    st.markdown('<h1 class="main-header">📊 Smart Portfolio Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("*AI-Powered Portfolio Optimization using Modern Portfolio Theory*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📅 Date Range")
        end_date = date.today()
        start_date = st.date_input("Start Date", end_date - timedelta(days=2*365))
        end_date = st.date_input("End Date", end_date)
        
        st.subheader("📈 Portfolio Assets")
        default_tickers = "AAPL,MSFT,GOOGL,AMZN,JPM"
        tickers_input = st.text_area(
            "Enter Stock Tickers (comma-separated)",
            default_tickers,
            help="Example: AAPL,MSFT,GOOGL,AMZN,JPM,TSLA"
        )
        selected_tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        st.session_state.selected_tickers = selected_tickers
        
        st.subheader("🎯 Optimization Goal")
        goal_name = st.selectbox("Goal", list(OPTIMIZATION_GOALS.keys()))
        goal_value = OPTIMIZATION_GOALS[goal_name]
        
        st.subheader("💰 Risk-Free Rate (%)")
        risk_free_rate = st.number_input("Rate", 0.0, 5.0, 3.0, 0.25) / 100
        
        st.subheader("📊 Benchmark")
        benchmark_name = st.selectbox("Benchmark", list(BENCHMARKS.keys()))
        benchmark_symbol = BENCHMARKS[benchmark_name]
        
        st.markdown("---")
        run_optimization = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        if st.session_state.optimization_run:
            weights_df = pd.DataFrame({
                'Asset': list(st.session_state.optimal_weights.keys()),
                'Weight (%)': [w*100 for w in st.session_state.optimal_weights.values()]
            })
            excel_data = create_excel_report(weights_df, st.session_state.portfolio_stats, selected_tickers)
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    if run_optimization and len(selected_tickers) >= 2:
        with st.spinner("Loading data and optimizing..."):
            load_and_optimize(selected_tickers, start_date, end_date, goal_value, risk_free_rate, benchmark_symbol)
    
    if st.session_state.optimization_run and len(st.session_state.selected_tickers) >= 2:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Markowitz", "📈 Treynor-Black", "💰 Value Screener", 
            "🔄 Monte Carlo", "⚖️ Compare", "🎯 Risk Analysis"
        ])
        
        with tab1:
            show_markowitz_tab(st.session_state.selected_tickers, goal_name, risk_free_rate, benchmark_name)
        
        with tab2:
            show_treynor_black_tab(st.session_state.selected_tickers, risk_free_rate)
        
        with tab3:
            show_value_screener_tab(st.session_state.selected_tickers)
        
        with tab4:
            show_monte_carlo_tab(st.session_state.selected_tickers, goal_value, risk_free_rate)
        
        with tab5:
            show_compare_tab(st.session_state.selected_tickers, risk_free_rate)
        
        with tab6:
            show_risk_analysis_tab(st.session_state.selected_tickers, risk_free_rate)

if __name__ == "__main__":
    main()
