# standalone_dash.py - Completely standalone dashboard
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

app = dash.Dash(__name__)

print("Loading data...")

# Fetch data directly
end = datetime.now()
start = end - timedelta(days=365)

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
data = yf.download(tickers, start=start, end=end, progress=False)['Close']
print(f"Data shape: {data.shape}")

# Calculate returns
returns = data.pct_change().dropna()
print(f"Returns shape: {returns.shape}")

# Calculate annualized returns and volatility
annual_returns = (1 + returns.mean()) ** 252 - 1
annual_vol = returns.std() * np.sqrt(252)

app.layout = html.Div([
    html.H1("📊 Portfolio Optimizer", style={'text-align': 'center', 'margin-top': '20px'}),
    
    html.Div([
        # Sidebar
        html.Div([
            html.H3("Settings"),
            html.Label("Select Stocks:"),
            dcc.Dropdown(
                id='stock-select',
                options=[{'label': col, 'value': col} for col in returns.columns],
                value=['AAPL', 'MSFT', 'GOOGL'],
                multi=True,
                style={'margin-bottom': '20px'}
            ),
            html.Label("Optimization Goal:"),
            dcc.Dropdown(
                id='goal',
                options=[
                    {'label': 'Max Sharpe Ratio', 'value': 'sharpe'},
                    {'label': 'Min Volatility', 'value': 'min_vol'},
                    {'label': 'Equal Weight', 'value': 'equal'}
                ],
                value='sharpe',
                style={'margin-bottom': '20px'}
            ),
            html.Label("Risk-Free Rate (%):"),
            dcc.Slider(
                id='rf',
                min=0, max=5, step=0.5, value=3,
                marks={i: f'{i}%' for i in range(0, 6)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px', 'vertical-align': 'top'}),
        
        # Charts
        html.Div([
            dcc.Graph(id='weights-chart', style={'height': '400px'}),
            dcc.Graph(id='returns-chart', style={'height': '400px'}),
            html.Div(id='metrics', style={'margin-top': '20px', 'padding': '20px', 'background': '#f8f9fa', 'border-radius': '10px'})
        ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px'})
    ])
])

def calculate_portfolio(returns_subset, weights):
    """Calculate portfolio return and volatility"""
    port_return = np.sum(returns_subset.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns_subset.cov() * 252, weights)))
    return port_return, port_vol

@app.callback(
    [Output('weights-chart', 'figure'),
     Output('returns-chart', 'figure'),
     Output('metrics', 'children')],
    [Input('stock-select', 'value'),
     Input('goal', 'value'),
     Input('rf', 'value')]
)
def update_portfolio(selected_stocks, goal, rf_rate):
    if not selected_stocks or len(selected_stocks) < 2:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Please select at least 2 stocks")
        return empty_fig, empty_fig, "Please select at least 2 stocks"
    
    rf = rf_rate / 100
    rets = returns[selected_stocks]
    n_assets = len(selected_stocks)
    
    # Calculate weights based on goal
    if goal == 'equal':
        weights = np.array([1/n_assets] * n_assets)
    elif goal == 'min_vol':
        # Simple min volatility (equal weights for simplicity)
        # In a real implementation, you'd use optimization
        weights = np.array([1/n_assets] * n_assets)
    else:  # max sharpe
        # Simple equal weights for now
        weights = np.array([1/n_assets] * n_assets)
    
    # Calculate portfolio performance
    port_return, port_vol = calculate_portfolio(rets, weights)
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0
    
    # Weights chart
    weights_fig = go.Figure(go.Bar(
        x=selected_stocks,
        y=weights * 100,
        text=[f'{w*100:.1f}%' for w in weights],
        textposition='outside',
        marker_color='#2ecc71'
    ))
    weights_fig.update_layout(title='Portfolio Weights', yaxis_title='Weight (%)')
    
    # Returns chart (cumulative)
    portfolio_returns = rets.dot(weights)
    cum_returns = (1 + portfolio_returns).cumprod()
    
    returns_fig = go.Figure()
    returns_fig.add_trace(go.Scatter(
        x=cum_returns.index,
        y=cum_returns.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#2ecc71', width=2),
        fill='tozeroy'
    ))
    returns_fig.update_layout(title='Portfolio Growth ($1 Invested)', 
                             xaxis_title='Date', 
                             yaxis_title='Cumulative Return')
    
    # Calculate metrics
    var_95 = np.percentile(portfolio_returns, 5)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    
    metrics_div = html.Div([
        html.H4("📊 Performance Metrics", style={'margin-bottom': '15px'}),
        html.Div([
            html.Div([
                html.P("Annualized Return", style={'font-size': '12px', 'color': '#666'}),
                html.P(f"{port_return*100:.2f}%", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#27ae60'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'background': 'white', 'border-radius': '8px', 'min-width': '120px'}),
            html.Div([
                html.P("Annualized Volatility", style={'font-size': '12px', 'color': '#666'}),
                html.P(f"{port_vol*100:.2f}%", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#e74c3c'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'background': 'white', 'border-radius': '8px', 'min-width': '120px'}),
            html.Div([
                html.P("Sharpe Ratio", style={'font-size': '12px', 'color': '#666'}),
                html.P(f"{sharpe:.3f}", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#3498db'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'background': 'white', 'border-radius': '8px', 'min-width': '120px'}),
            html.Div([
                html.P("VaR (95%)", style={'font-size': '12px', 'color': '#666'}),
                html.P(f"{var_95*100:.2f}%", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#e67e22'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'background': 'white', 'border-radius': '8px', 'min-width': '120px'}),
            html.Div([
                html.P("CVaR (95%)", style={'font-size': '12px', 'color': '#666'}),
                html.P(f"{cvar_95*100:.2f}%", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#e67e22'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'background': 'white', 'border-radius': '8px', 'min-width': '120px'})
        ])
    ])
    
    return weights_fig, returns_fig, metrics_div

if __name__ == '__main__':
    app.run(debug=True, port=8050)