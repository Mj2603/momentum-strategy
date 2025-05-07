# Momentum Strategy Implementation

A professional implementation of the well-documented Jegadeesh-Titman (12-1) momentum strategy. This strategy, first published in 1993, has been extensively researched and shown to be effective across different markets and time periods.

## Strategy Overview

The Jegadeesh-Titman momentum strategy is based on the following principles:
1. Rank stocks based on their 12-month returns, skipping the most recent month
2. Take long positions in the top performers and short positions in the bottom performers
3. Hold positions for one month
4. Rebalance monthly

This implementation includes:
- Original strategy methodology
- Risk management features
- Transaction cost modeling
- Comprehensive performance analysis

## Key Features

- Jegadeesh-Titman (12-1) momentum calculation
- Long-short portfolio construction with risk management
- Comprehensive backtesting framework with transaction costs
- Detailed performance analysis and visualization
- Advanced risk metrics calculation
- Market regime analysis

## Project Structure

```
momentum-strategy/
├── data/                  # Market data storage
├── momentum_strategy.py   # Core strategy implementation
├── backtest.py           # Advanced backtesting framework
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mj2603/momentum-strategy.git
cd momentum-strategy
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Strategy Implementation

```python
from momentum_strategy import MomentumStrategy

# Initialize strategy with a diversified set of stocks
symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech
    'JPM', 'BAC', 'GS', 'MS', 'WFC',          # Finance
    'JNJ', 'PFE', 'MRK', 'UNH', 'ABBV',       # Healthcare
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',        # Energy
    'PG', 'KO', 'PEP', 'WMT', 'COST'          # Consumer
]

strategy = MomentumStrategy(symbols)

# Run backtest
results = strategy.backtest('2018-01-01', '2023-12-31')

# Calculate and print metrics
metrics = strategy.calculate_metrics(results)
print("\nStrategy Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}")

# Plot results
strategy.plot_results(results)
```

### Advanced Analysis

```python
from backtest import BacktestAnalyzer

# Create analyzer
analyzer = BacktestAnalyzer(strategy)

# Perform detailed analysis
analysis = analyzer.analyze_returns(results)

# Generate comprehensive report
report = analyzer.generate_report(results, analysis)
print("\nDetailed Performance Report:")
for metric, value in report.items():
    print(f"{metric}: {value:.2f}")

# Plot analysis
analyzer.plot_analysis(results, analysis)
```

## Strategy Details

### Momentum Calculation
The strategy implements the original Jegadeesh-Titman methodology:
- Calculate 12-month returns (formation period)
- Skip the most recent month (skip period)
- Rank stocks based on the 12-1 momentum signal
- Rebalance monthly

### Portfolio Construction
- Systematic selection of top N stocks for long positions
- Strategic short positions in bottom N stocks
- Risk-weighted position sizing
- Monthly rebalancing with transaction cost consideration

### Performance Metrics
- Total Return and Annualized Return
- Risk-adjusted metrics (Sharpe Ratio, Sortino Ratio)
- Maximum Drawdown and Recovery Analysis
- Win Rate and Profit Factor
- Rolling Performance Metrics
- Regime-based Performance Analysis

## Academic Background

The Jegadeesh-Titman momentum strategy was first documented in the paper:
"Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency"
by Narasimhan Jegadeesh and Sheridan Titman (1993)

Key findings:
- Momentum strategies generate significant positive returns
- Returns persist for 3-12 months after formation
- Strategy works across different markets and time periods
- Returns are not explained by systematic risk factors

## Technical Requirements

- Python 3.8+
- pandas>=1.5.0
- numpy>=1.21.0
- yfinance>=0.2.0
- matplotlib>=3.5.0
- seaborn>=0.12.0

## Author

- **Mrityunjay Balkrishnan** - [GitHub Profile](https://github.com/Mj2603)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This project is for educational and research purposes only. The strategy implementation and backtesting results should not be used for actual trading without proper risk management and additional research. Past performance is not indicative of future results. 
