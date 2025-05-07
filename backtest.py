import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from momentum_strategy import MomentumStrategy

class BacktestAnalyzer:
    """
    Advanced backtesting and analysis framework for the momentum strategy.
    Provides comprehensive performance analysis, risk metrics, and visualization tools.
    
    Author: Mrityunjay Balkrishnan
    GitHub: https://github.com/Mj2603
    """
    
    def __init__(self, strategy: MomentumStrategy):
        """
        Initialize the backtest analyzer with advanced analysis capabilities.
        
        Args:
            strategy: MomentumStrategy instance
        """
        self.strategy = strategy
        plt.style.use('seaborn')
        
    def analyze_returns(self, results: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Perform comprehensive return analysis with advanced metrics.
        
        Args:
            results: DataFrame with backtest results
            
        Returns:
            Dictionary with various return analyses
        """
        returns = results['Portfolio Value'].pct_change()
        
        # Calculate rolling metrics
        rolling_returns = returns.rolling(window=21).mean() * 252
        rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)
        rolling_sharpe = rolling_returns / rolling_vol
        
        # Calculate drawdowns
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / running_max - 1) * 100
        
        # Calculate regime detection
        volatility_regime = returns.rolling(window=63).std() * np.sqrt(252)
        trend_regime = returns.rolling(window=63).mean() * 252
        
        # Calculate additional metrics
        rolling_sortino = rolling_returns / (returns.rolling(window=21).apply(
            lambda x: np.sqrt(np.mean(x[x < 0]**2)) * np.sqrt(252)))
        
        return {
            'daily_returns': returns,
            'rolling_returns': rolling_returns,
            'rolling_volatility': rolling_vol,
            'rolling_sharpe': rolling_sharpe,
            'rolling_sortino': rolling_sortino,
            'drawdowns': drawdowns,
            'volatility_regime': volatility_regime,
            'trend_regime': trend_regime
        }
    
    def plot_analysis(self, results: pd.DataFrame, analysis: Dict[str, pd.DataFrame]):
        """
        Create professional-grade analysis visualizations.
        
        Args:
            results: DataFrame with backtest results
            analysis: Dictionary with analysis results
        """
        # Create subplots with custom styling
        fig = plt.figure(figsize=(15, 20))
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1])
        
        # Plot portfolio value
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(results.index, results['Portfolio Value'], linewidth=2)
        ax1.set_title('Portfolio Value Over Time', fontsize=14, pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Value ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot rolling metrics
        ax2 = fig.add_subplot(gs[1])
        analysis['rolling_returns'].plot(ax=ax2, label='Rolling Returns', linewidth=2)
        analysis['rolling_volatility'].plot(ax=ax2, label='Rolling Volatility', linewidth=2)
        ax2.set_title('Rolling Returns and Volatility', fontsize=14, pad=20)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Annualized %', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot drawdowns
        ax3 = fig.add_subplot(gs[2])
        analysis['drawdowns'].plot(ax=ax3, color='red', alpha=0.7)
        ax3.fill_between(analysis['drawdowns'].index, analysis['drawdowns'], 0, 
                        color='red', alpha=0.3)
        ax3.set_title('Portfolio Drawdowns', fontsize=14, pad=20)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Drawdown %', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Plot regime analysis
        ax4 = fig.add_subplot(gs[3])
        analysis['volatility_regime'].plot(ax=ax4, label='Volatility Regime', linewidth=2)
        analysis['trend_regime'].plot(ax=ax4, label='Trend Regime', linewidth=2)
        ax4.set_title('Market Regime Analysis', fontsize=14, pad=20)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.set_ylabel('Annualized %', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, results: pd.DataFrame, analysis: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Generate a comprehensive performance report with advanced metrics.
        
        Args:
            results: DataFrame with backtest results
            analysis: Dictionary with analysis results
            
        Returns:
            Dictionary with detailed performance metrics
        """
        returns = analysis['daily_returns']
        
        # Calculate basic metrics
        total_return = (results['Portfolio Value'].iloc[-1] / results['Portfolio Value'].iloc[0] - 1) * 100
        annualized_return = returns.mean() * 252 * 100
        annualized_vol = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        max_drawdown = analysis['drawdowns'].min()
        
        # Calculate additional metrics
        positive_months = len(returns[returns > 0])
        negative_months = len(returns[returns < 0])
        win_rate = positive_months / (positive_months + negative_months) * 100
        
        # Calculate rolling metrics
        rolling_sharpe = analysis['rolling_sharpe'].mean()
        rolling_vol = analysis['rolling_volatility'].mean()
        rolling_sortino = analysis['rolling_sortino'].mean()
        
        # Calculate regime metrics
        high_vol_regime = analysis['volatility_regime'].mean()
        trend_strength = analysis['trend_regime'].mean()
        
        # Calculate risk-adjusted metrics
        downside_vol = returns[returns < 0].std() * np.sqrt(252) * 100
        sortino_ratio = (returns.mean() * 252) / (downside_vol / 100)
        
        return {
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'Annualized Volatility (%)': annualized_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Average Rolling Sharpe': rolling_sharpe,
            'Average Rolling Sortino': rolling_sortino,
            'Average Rolling Volatility (%)': rolling_vol,
            'Average Volatility Regime (%)': high_vol_regime * 100,
            'Average Trend Strength (%)': trend_strength * 100
        }

if __name__ == "__main__":
    # Example usage
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
    strategy = MomentumStrategy(symbols)
    
    # Run backtest
    results = strategy.backtest('2020-01-01', '2023-12-31')
    
    # Create analyzer and perform analysis
    analyzer = BacktestAnalyzer(strategy)
    analysis = analyzer.analyze_returns(results)
    
    # Generate and print report
    report = analyzer.generate_report(results, analysis)
    print("\nDetailed Performance Report:")
    for metric, value in report.items():
        print(f"{metric}: {value:.2f}")
    
    # Plot analysis
    analyzer.plot_analysis(results, analysis) 