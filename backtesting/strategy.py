"""
Trading strategies for backtesting
"""
import backtrader as bt
import numpy as np
import torch


# ======================================================================
# Base Strategy
# ======================================================================
class BaseStrategy(bt.Strategy):
    """
    Base strategy class for backtrader strategies.
    """

    params = dict(
        size=1,
    )

    def __init__(self):
        self.order = None  # IMPORTANT: order lifecycle tracking

    def notify_order(self, order):
        """
        Order status callback.
        """
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            self.log(
                f"{'BUY' if order.isbuy() else 'SELL'} EXECUTED @ "
                f"{order.executed.price:.2f}"
            )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled / Margin / Rejected")

        self.order = None  # reset after completion

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")


# ======================================================================
# ML Strategy
# ======================================================================
class MLStrategy(BaseStrategy):
    """
    Active ML strategy aligned with k-step classification.

    Label:
        0 = DOWN
        1 = FLAT
        2 = UP
    """

    params = dict(
        lookback=60,
        hold_period=5,      # == k
        size=1,
        threshold=0.0,      # entry threshold
        reverse_threshold=0.55,  # stronger condition for reverse
    )

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

        self.bar_entered = None

    # --------------------------------------------------
    # Feature extraction (same as training)
    # --------------------------------------------------
    def _get_features(self):
        if len(self.data) < self.p.lookback:
            return None

        feats = np.zeros((self.p.lookback, 5), dtype=np.float32)
        for i in range(self.p.lookback):
            idx = -self.p.lookback + i
            feats[i] = [
                self.data.open[idx],
                self.data.high[idx],
                self.data.low[idx],
                self.data.close[idx],
                self.data.volume[idx],
            ]
        return feats

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    def _predict(self, features):
        """
        features: np.ndarray, shape (T, d)
        """

        # --------------------------------------------------
        # 1. Rolling normalization (NO FUTURE INFO)
        # --------------------------------------------------
        eps = 1e-8
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + eps
        features_norm = (features - mean) / std

        # --------------------------------------------------
        # 2. To tensor
        # --------------------------------------------------
        x = torch.from_numpy(features_norm).float().unsqueeze(0)  # (1, T, d)

        # --------------------------------------------------
        # 3. Model inference
        # --------------------------------------------------
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            conf = probs[0, pred].item()

        return pred, conf


    # --------------------------------------------------
    # Order notification (CRITICAL)
    # --------------------------------------------------
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.bar_entered = len(self)
                self.log(f"BUY EXECUTED @ {order.executed.price:.2f}")
            else:
                self.log(f"SELL EXECUTED @ {order.executed.price:.2f}")
                self.bar_entered = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Failed")

        self.order = None

    # --------------------------------------------------
    # Trading logic
    # --------------------------------------------------
    def next(self):
        if self.order:
            return

        features = self._get_features()
        if features is None:
            return

        pred, conf = self._predict(features)


        # print(f"[DEBUG] date {self.datas[0].datetime.date(0).isoformat()} Pred: {pred}, Conf: {conf:.3f}")
        # ==========================
        # ENTRY
        # ==========================
        if not self.position:
            # allow re-entry immediately after exit
            if pred == 2 and conf >= self.p.threshold:
                self.order = self.buy(size=self.p.size)
                self.log(
                    f"BUY CREATE @ {self.data.close[0]:.2f}, conf={conf:.3f}"
                )
            return

        # ==========================
        # EXIT
        # ==========================
        bars_held = len(self) - self.bar_entered

        # 1) time-based exit (core k-step logic)
        if bars_held >= self.p.hold_period:
            self.order = self.sell(size=self.p.size)
            self.log(
                f"SELL (time exit) @ {self.data.close[0]:.2f}, held={bars_held}"
            )
            return

        # 2) strong reverse only
        if pred == 0 and conf >= self.p.reverse_threshold:
            self.order = self.sell(size=self.p.size)
            self.log(
                f"SELL (reverse) @ {self.data.close[0]:.2f}, conf={conf:.3f}"
            )


# ======================================================================
# Simple Moving Average Strategy
# ======================================================================
class SimpleMAStrategy(BaseStrategy):
    """
    Simple moving average crossover strategy.
    """

    params = dict(
        fast_period=10,
        slow_period=30,
        size=1,
    )

    def __init__(self):
        super().__init__()

        self.fast_ma = bt.ind.SMA(
            self.data.close, period=self.p.fast_period
        )
        self.slow_ma = bt.ind.SMA(
            self.data.close, period=self.p.slow_period
        )

        self.log(
            f"Using MA Strategy | fast={self.p.fast_period}, "
            f"slow={self.p.slow_period}"
        )

    def next(self):
        if self.order:
            return

        # Golden cross → BUY
        if not self.position:
            if (
                self.fast_ma[0] > self.slow_ma[0]
                and self.fast_ma[-1] <= self.slow_ma[-1]
            ):
                self.order = self.buy(size=self.p.size)
                self.log(f"BUY CREATE @ {self.data.close[0]:.2f}")

        # Death cross → SELL
        else:
            if (
                self.fast_ma[0] < self.slow_ma[0]
                and self.fast_ma[-1] >= self.slow_ma[-1]
            ):
                self.order = self.sell(size=self.p.size)
                self.log(f"SELL CREATE @ {self.data.close[0]:.2f}")
