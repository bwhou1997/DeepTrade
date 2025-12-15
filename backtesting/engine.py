"""
Backtesting engine using backtrader
"""
import backtrader as bt
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
import os


class BacktestEngine:
    """
    Backtesting engine wrapper around backtrader.
    
    Args:
        data_path: Path to CSV file with OHLCV data
        initial_cash: Starting cash amount
        commission: Commission rate (e.g., 0.001 for 0.1%)
        fromdate: Start date (optional)
        todate: End date (optional)
    """
    
    def __init__(
        self,
        data_path: str,
        initial_cash: float = 10000.0,
        commission: float = 0.001,
        fromdate: Optional[str] = None,
        todate: Optional[str] = None,
    ):
        self.data_path = data_path
        self.initial_cash = initial_cash
        self.commission = commission
        self.fromdate = fromdate
        self.todate = todate
        
        self.cerebro = bt.Cerebro()
        self.strategy = None
        
        # Setup cerebro
        self._setup_cerebro()
        self._load_data()
    
    def _setup_cerebro(self):
        """Setup backtrader cerebro with broker settings."""
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    def _load_data(self):
        """Load OHLCV data from CSV."""
        # Read CSV
        df = pd.read_csv(self.data_path)
        
        # Ensure date column exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.rename(columns={'Date': 'date'}, inplace=True)
        else:
            raise ValueError("CSV must have a 'date' or 'Date' column")
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        col_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        }
        
        for old_col, new_col in col_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Convert to backtrader data feed
        data_feed = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Use index
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1,  # Not used
        )
        
        # Apply date filters if provided
        if self.fromdate:
            data_feed.fromdate = datetime.strptime(self.fromdate, '%Y-%m-%d')
        if self.todate:
            data_feed.todate = datetime.strptime(self.todate, '%Y-%m-%d')
        
        self.cerebro.adddata(data_feed)
    
    def add_strategy(self, strategy, **kwargs):
        """
        Add trading strategy.
        
        Args:
            strategy: Strategy class or instance
            **kwargs: Strategy parameters
        """
        if isinstance(strategy, bt.Strategy):
            # Strategy instance
            self.strategy = strategy
            self.cerebro.addstrategy(type(strategy), **kwargs)
        else:
            # Strategy class
            self.cerebro.addstrategy(strategy, **kwargs)
    
    def run(self) -> Dict[str, Any]:
        """
        Run backtest.
        
        Returns:
            Dictionary with backtest results
        """
        print(f'Starting Portfolio Value: {self.cerebro.broker.getvalue():.2f}')
        
        # Run backtest
        strategies = self.cerebro.run()
        
        # Get final value
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        print(f'Final Portfolio Value: {final_value:.2f}')
        print(f'Total Return: {total_return:.2%}')
        
        # Extract analyzer results
        results = {
            'initial_value': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
        }
        
        # Get analyzer results from first strategy
        if strategies:
            strat = strategies[0]
            
            # Sharpe ratio
            if hasattr(strat.analyzers, 'sharpe'):
                sharpe = strat.analyzers.sharpe.get_analysis()
                results['sharpe_ratio'] = sharpe.get('sharperatio', None)
            
            # Drawdown
            if hasattr(strat.analyzers, 'drawdown'):
                dd = strat.analyzers.drawdown.get_analysis()
                results['max_drawdown'] = dd.get('max', {}).get('drawdown', None)
                results['max_drawdown_period'] = dd.get('max', {}).get('len', None)
            
            # Returns
            if hasattr(strat.analyzers, 'returns'):
                ret = strat.analyzers.returns.get_analysis()
                results['annual_return'] = ret.get('rnorm100', None)
            
            # Trade analysis
            if hasattr(strat.analyzers, 'trades'):
                trades = strat.analyzers.trades.get_analysis()
                results['total_trades'] = trades.get('total', {}).get('total', 0)
                results['won'] = trades.get('won', {}).get('total', 0)
                results['lost'] = trades.get('lost', {}).get('total', 0)
                results['win_rate'] = (
                    results['won'] / results['total_trades'] 
                    if results['total_trades'] > 0 else 0
                )
        
        return results
    
    def plot(self, save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            save_path: Optional path to save plot
        """
        figs = self.cerebro.plot(style='candlestick', barup='green', bardown='red')
        
        if save_path and figs:
            fig = figs[0][0]
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    engine = BacktestEngine(
        data_path="data/AAPL.csv",
        initial_cash=10000,
        commission=0.001
    )
    
    from .strategy import SimpleMAStrategy
    engine.add_strategy(SimpleMAStrategy)
    
    results = engine.run()
    print(results)


