"""
Trading strategies for backtesting
"""
import backtrader as bt
import numpy as np
import torch
from typing import Optional


class BaseStrategy(bt.Strategy):
    """
    Base strategy class for backtrader.
    """
    params = dict(
        lookback=60,
        size=1,
    )

    def __init__(self):
        self.order = None

    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

    def next(self):
        """Called for each bar."""
        raise NotImplementedError("Subclasses must implement next()")


class MLStrategy(BaseStrategy):
    """
    Machine learning-based trading strategy.
    
    Uses a trained model to predict market direction and execute trades.
    
    Args:
        model: Trained PyTorch model
        lookback: Number of historical bars to use for prediction
        size: Position size
        threshold: Confidence threshold for trading (optional)
    """
    
    params = dict(
        lookback=60,
        size=1,
        threshold=None,  # If None, always trade on prediction
    )

    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.model.eval()  # Set to evaluation mode
        
        # Update params if provided
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)

    def _get_features(self):
        """
        Extract features from current bar data.
        
        Returns:
            features: numpy array of shape (lookback, num_features)
        """
        # Get OHLCV data
        lookback = self.p.lookback
        
        if len(self.data) < lookback:
            return None
        
        # Extract features: [open, high, low, close, volume]
        features = []
        for i in range(lookback):
            idx = -lookback + i
            bar_features = [
                self.data.open[idx],
                self.data.high[idx],
                self.data.low[idx],
                self.data.close[idx],
                self.data.volume[idx],
            ]
            features.append(bar_features)
        
        return np.array(features, dtype=np.float32)

    def _predict(self, features):
        """
        Get prediction from model.
        
        Args:
            features: numpy array of shape (lookback, num_features)
        
        Returns:
            prediction: 0 (DOWN), 1 (FLAT), or 2 (UP)
            confidence: prediction confidence/probability
        """
        # Convert to tensor
        features_tensor = torch.from_numpy(features).unsqueeze(0)  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred].item()
        
        return pred, confidence

    def next(self):
        """Execute strategy logic for each bar."""
        # Ensure we have enough history
        if len(self.data) < self.p.lookback:
            return

        # Avoid duplicate orders
        if self.order:
            return

        # Get features
        features = self._get_features()
        if features is None:
            return

        # Get prediction
        pred, confidence = self._predict(features)

        # Check threshold if set
        if self.p.threshold is not None and confidence < self.p.threshold:
            return

        # Execute trades
        if pred == 2 and not self.position:  # UP signal, no position
            self.order = self.buy(size=self.p.size)
            self.log(f'BUY CREATE: Close={self.data.close[0]:.2f}, Confidence={confidence:.3f}')

        elif pred == 0 and self.position:  # DOWN signal, has position
            self.order = self.sell(size=self.p.size)
            self.log(f'SELL CREATE: Close={self.data.close[0]:.2f}, Confidence={confidence:.3f}')

    def log(self, txt, dt=None):
        """Logging function."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')


class SimpleMAStrategy(BaseStrategy):
    """
    Simple moving average crossover strategy (example).
    """
    params = dict(
        fast_period=10,
        slow_period=30,
        size=1,
    )

    def __init__(self):
        super().__init__()
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)

    def next(self):
        if self.order:
            return

        # Buy signal: fast MA crosses above slow MA
        if not self.position:
            if self.fast_ma[0] > self.slow_ma[0] and self.fast_ma[-1] <= self.slow_ma[-1]:
                self.order = self.buy(size=self.p.size)
                self.log(f'BUY CREATE: Close={self.data.close[0]:.2f}')

        # Sell signal: fast MA crosses below slow MA
        else:
            if self.fast_ma[0] < self.slow_ma[0] and self.fast_ma[-1] >= self.slow_ma[-1]:
                self.order = self.sell(size=self.p.size)
                self.log(f'SELL CREATE: Close={self.data.close[0]:.2f}')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')


