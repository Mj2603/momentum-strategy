import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class MomentumStrategy:
    """
    Implementation of the Jegadeesh-Titman (12-1) Momentum Strategy.
    This is a well-documented strategy that has shown consistent performance across markets.
    
    The strategy:
    1. Ranks stocks based on their 12-month returns, skipping the most recent month
    2. Takes long positions in the top performers and short positions in the bottom performers
    3. Holds positions for one month
    4. Rebalances monthly
    
    Author: Mrityunjay Balkrishnan
    GitHub: https://github.com/Mj2603
    """
    
    def __init__(self, 
                 symbols: List[str],
                 formation_period: int = 252,  # 12 months
                 skip_period: int = 21,        # 1 month
                 holding_period: int = 21,     # 1 month
                 top_n: int = 10,             # Number of stocks to hold long/short
                 transaction_cost: float = 0.001):  # 10 bps transaction cost
        """
        Initialize the Jegadeesh-Titman momentum strategy.
        
        Args:
            symbols: List of stock symbols to trade
            formation_period: Number of days to look back for momentum calculation (default: 252)
            skip_period: Number of days to skip at the end of formation period (default: 21)
            holding_period: Number of days to hold positions (default: 21)
            top_n: Number of stocks to hold in long/short positions
            transaction_cost: Transaction cost as a decimal
        """
        self.symbols = symbols
        self.formation_period = formation_period
        self.skip_period = skip_period
        self.holding_period = holding_period
        self.top_n = top_n
        self.transaction_cost = transaction_cost
        self.positions = {}
        self.portfolio_value = 100000  # Initial portfolio value
        self.returns_history = []
        self.transaction_history = []
        
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical price data for all symbols using yfinance.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with historical prices
        """
        data = {}
        for symbol in self.symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(start=start_date, end=end_date)
                data[symbol] = hist['Close']
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                
        return pd.DataFrame(data)
    
    def calculate_momentum(self, prices: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum scores using the Jegadeesh-Titman methodology.
        
        Args:
            prices: DataFrame with historical prices
            
        Returns:
            Series with momentum scores
        """
        # Calculate returns over the formation period, skipping the most recent month
        formation_returns = prices.pct_change(self.formation_period)
        skip_returns = prices.pct_change(self.skip_period)
        
        # Subtract the skip period returns to get the 12-1 momentum
        momentum_scores = formation_returns.iloc[-1] - skip_returns.iloc[-1]
        
        return momentum_scores
    
    def generate_signals(self, momentum_scores: pd.Series) -> Dict[str, float]:
        """
        Generate trading signals with risk management considerations.
        
        Args:
            momentum_scores: Series with momentum scores
            
        Returns:
            Dictionary with position sizes (-1 to 1)
        """
        # Sort stocks by momentum
        sorted_scores = momentum_scores.sort_values(ascending=False)
        
        # Select top and bottom stocks
        long_stocks = sorted_scores.head(self.top_n)
        short_stocks = sorted_scores.tail(self.top_n)
        
        # Generate position sizes with risk management
        positions = {}
        position_size = 1.0 / self.top_n  # Equal weight for each position
        
        # Apply position sizing with risk management
        for stock in long_stocks.index:
            # Add risk management: reduce position size for high volatility stocks
            vol = sorted_scores[stock] / sorted_scores.std()
            positions[stock] = position_size * (1 / (1 + vol))
            
        for stock in short_stocks.index:
            # Add risk management: reduce position size for high volatility stocks
            vol = sorted_scores[stock] / sorted_scores.std()
            positions[stock] = -position_size * (1 / (1 + vol))
            
        return positions
    
    def calculate_transaction_costs(self, old_positions: Dict[str, float], 
                                  new_positions: Dict[str, float]) -> float:
        """
        Calculate transaction costs for portfolio rebalancing.
        
        Args:
            old_positions: Dictionary with current positions
            new_positions: Dictionary with new positions
            
        Returns:
            Total transaction cost
        """
        total_cost = 0
        for stock in set(old_positions.keys()) | set(new_positions.keys()):
            old_pos = old_positions.get(stock, 0)
            new_pos = new_positions.get(stock, 0)
            turnover = abs(new_pos - old_pos)
            total_cost += turnover * self.transaction_cost
            
        return total_cost
    
    def backtest(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Run backtest of the Jegadeesh-Titman momentum strategy.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with backtest results
        """
        # Fetch historical data
        prices = self.fetch_data(start_date, end_date)
        
        # Initialize results
        portfolio_values = [self.portfolio_value]
        dates = [prices.index[0]]
        self.transaction_history = []
        
        # Run backtest
        for i in range(self.formation_period, len(prices), self.holding_period):
            # Calculate momentum scores
            momentum_scores = self.calculate_momentum(prices.iloc[:i])
            
            # Generate signals
            new_positions = self.generate_signals(momentum_scores)
            
            # Calculate transaction costs
            if self.positions:
                transaction_cost = self.calculate_transaction_costs(
                    self.positions, new_positions)
                self.transaction_history.append(transaction_cost)
                self.portfolio_value *= (1 - transaction_cost)
            
            # Update positions
            self.positions = new_positions
            
            # Calculate returns for holding period
            if i + self.holding_period <= len(prices):
                period_returns = prices.iloc[i:i+self.holding_period].pct_change()
                portfolio_return = sum(self.positions[stock] * period_returns[stock].sum() 
                                    for stock in self.positions.keys())
                
                # Update portfolio value
                self.portfolio_value *= (1 + portfolio_return)
                portfolio_values.append(self.portfolio_value)
                dates.append(prices.index[i])
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': dates,
            'Portfolio Value': portfolio_values
        })
        results.set_index('Date', inplace=True)
        
        return results
    
    def calculate_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for the strategy.
        
        Args:
            results: DataFrame with backtest results
            
        Returns:
            Dictionary with performance metrics
        """
        returns = results['Portfolio Value'].pct_change()
        
        # Calculate basic metrics
        total_return = (results['Portfolio Value'].iloc[-1] / results['Portfolio Value'].iloc[0] - 1) * 100
        annualized_return = returns.mean() * 252 * 100
        annualized_vol = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        
        # Calculate drawdown metrics
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / running_max - 1) * 100
        max_drawdown = drawdowns.min()
        
        # Calculate additional metrics
        positive_months = len(returns[returns > 0])
        negative_months = len(returns[returns < 0])
        win_rate = positive_months / (positive_months + negative_months) * 100
        
        # Calculate transaction costs
        total_transaction_cost = sum(self.transaction_history) * 100
        
        metrics = {
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'Annualized Volatility (%)': annualized_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Total Transaction Cost (%)': total_transaction_cost
        }
        
        return metrics
    
    def plot_results(self, results: pd.DataFrame):
        """
        Plot professional-grade backtest results.
        
        Args:
            results: DataFrame with backtest results
        """
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot portfolio value
        ax1.plot(results.index, results['Portfolio Value'], linewidth=2)
        ax1.set_title('Jegadeesh-Titman Momentum Strategy Performance', fontsize=14, pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdowns
        returns = results['Portfolio Value'].pct_change()
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / running_max - 1) * 100
        
        ax2.fill_between(results.index, drawdowns, 0, color='red', alpha=0.3)
        ax2.set_title('Portfolio Drawdowns', fontsize=14, pad=20)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage with a diversified set of stocks
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