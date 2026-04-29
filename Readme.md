# 📊 Smart Portfolio Optimizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://portfolio-optimizer-k62kg4m6pfmybapp75smp8b.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Live Demo

**Try it now:** [https://portfolio-optimizer-k62kg4m6pfmybapp75smp8b.streamlit.app/](https://portfolio-optimizer-k62kg4m6pfmybapp75smp8b.streamlit.app/)

> ⚡ **Note:** The first load may take 10-15 seconds as the app wakes up from free tier.

## 📖 About

Smart Portfolio Optimizer is a comprehensive investment analysis tool that helps you build optimal portfolios using **Modern Portfolio Theory** (Markowitz) and **Active Management** (Treynor-Black). Built with Streamlit and Yahoo Finance data.

### 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **📊 Markowitz Optimization** | Efficient frontier, optimal weights, Sharpe ratio maximization |
| **📈 Treynor-Black Model** | Alpha/Beta analysis with statistical significance testing |
| **💰 Value Screener** | Multi-factor stock scoring (P/E, P/B, momentum, dividend) |
| **🔄 Monte Carlo Simulation** | 1,000+ scenarios for risk analysis |
| **⚖️ Strategy Comparison** | Compare Sharpe, Equal Weight, and Min Volatility |
| **🎯 Risk Analysis** | Portfolio beta gauge, position sizing, stress tests |

## 📸 Dashboard Preview

### Markowitz Portfolio Analysis
![Markowitz Tab](<img width="664" height="401" alt="image" src="https://github.com/user-attachments/assets/9b3f5dc9-0cff-493b-ad00-3c1c5e1cba11" />
)

*Efficient frontier visualization with optimal portfolio selection*

### Treynor-Black Active Model
![Treynor-Black Tab](<img width="633" height="110" alt="image" src="https://github.com/user-attachments/assets/ded811b1-d1d5-4468-8502-47590cc05226" />
)

*Alpha and Beta calculation with Yahoo Finance comparison*

### Value Stock Screener
![Value Screener](<img width="662" height="318" alt="image" src="https://github.com/user-attachments/assets/36a4046a-a906-49c6-8522-5af9d9718bcc" />
)

*Multi-factor scoring system for stock valuation*

### Portfolio Risk Analysis
![Risk Analysis](<img width="704" height="347" alt="image" src="https://github.com/user-attachments/assets/4dcd1cca-ce78-406b-abf9-60afe6a3d639" />
)

*Portfolio beta gauge and position sizing recommendations*

## ✨ Features in Detail

### 1. 📊 Markowitz Portfolio Optimization
- **Efficient Frontier** - Visualize optimal risk-return combinations
- **Multiple Optimization Goals**
  - Maximum Sharpe Ratio (best risk-adjusted returns)
  - Minimum Volatility (safest portfolio)
  - Target Return (6%, 8%, 10%, 12%, 15%, 20%)
  - Equal Weight (diversification benchmark)
- **Performance Metrics**
  - Expected Annual Return & Volatility
  - Sharpe & Sortino Ratios
  - Maximum Drawdown & Calmar Ratio
  - Value at Risk (VaR 95% & 99%)
  - Conditional Value at Risk (CVaR)
  - Skewness & Kurtosis
- **Rolling Sharpe Ratio** (63-day window)
- **Sector Allocation** pie chart

### 2. 📈 Treynor-Black Active Model
- **Alpha Calculation** - Identify stocks outperforming benchmark
- **Beta Analysis** - Measure volatility relative to market
- **Statistical Significance** - p-value testing (90% confidence)
- **Comparison with Yahoo Finance** - Validate your calculations
- **Investment Insights** - Overweight/underweight recommendations

### 3. 💰 Value Stock Screener
- **Multi-Factor Scoring (1-10 scale)**
  - P/E Ratio (value factor)
  - P/B Ratio (value factor)
  - Historical Returns (momentum)
  - Volatility (risk factor)
  - Dividend Yield (income factor)
- **Color-Coded Recommendations**
  - 🟢 Strong Buy (7-10 points)
  - 🟡 Buy (5-6 points)
  - 🟠 Hold (3-4 points)
  - 🔴 Sell (0-2 points)

### 4. 🔄 Monte Carlo Simulation
- **1,000 simulations** over 1-year horizon
- **Distribution visualization** of final portfolio values
- **Risk Metrics**
  - Mean & Median final values
  - VaR at 95% and 99% confidence
  - Probability of loss
  - Probability of 20%+ gains

### 5. ⚖️ Strategy Comparison
Compare three portfolio strategies side-by-side:
- Your optimized portfolio (based on selected goal)
- Equal weight portfolio
- Minimum volatility portfolio

### 6. 🎯 Risk Analysis
- **Portfolio Beta Gauge** - Visual risk meter
- **Investor Profile Questionnaire**
  - Age-based adjustments
  - Investment horizon
  - Risk tolerance
  - Loss tolerance
- **Position Sizing Recommendations**
  - Overweight/underweight suggestions
  - Reason-based recommendations
- **Stress Test** - Portfolio impact under market scenarios (-20% to +20%)

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web application framework |
| **Python 3.12+** | Programming language |
| **Yahoo Finance** | Stock price data |
| **Plotly** | Interactive visualizations |
| **Pandas/NumPy** | Data manipulation |
| **SciPy** | Statistical calculations |
| **OpenPyXL** | Excel report generation |

## 📦 Installation

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Local Setup

```bash
# Clone the repository
git clone https://github.com/ayeayemyat-miso/Portfolio-Optimizer.git
cd Portfolio-Optimizer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
