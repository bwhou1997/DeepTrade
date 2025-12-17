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
                self.log(
                    f"BUY EXECUTED @ {order.executed.price:.2f}, "
                    f"size={order.executed.size}"
                )
            else:
                self.bar_entered = None
                self.log(
                    f"SELL EXECUTED @ {order.executed.price:.2f}, "
                    f"size={order.executed.size}"
                )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            status_map = {
                order.Canceled: "CANCELED",
                order.Margin: "MARGIN (Insufficient Cash)",
                order.Rejected: "REJECTED",
            }

            self.log(
                f"ORDER FAILED | "
                f"type={order.ordtypename()} | "
                f"status={status_map[order.status]} | "
                f"size={order.created.size} | "
                f"price={order.created.price} | "
                f"cash={self.broker.getcash():.2f}"
            )

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

        self.log(
            f"[DEBUG] Pred={pred}, Conf={conf:.3f}, "
            f"Pos={self.position.size}, Cash={self.broker.getcash():.2f}"
        )

        price = self.data.close[0]

        # ----------------------------------
        # 2. ENTRY: ALL-IN
        # ----------------------------------
        if not self.position:
            if pred == 2 and conf >= self.p.threshold:

                if self.p.size is not None:
                    size = self.p.size
                else:
                    cash = self.broker.getcash()
                    comminfo = self.broker.getcommissioninfo(self.data)
                    commission = comminfo.p.commission
                    size = int(cash / (price * (1.0 + commission)))

                if size > 0:
                    self.order = self.buy(size=size)
                    self.bar_entered = len(self)
            return
        # if not self.position:
        #     # allow re-entry immediately after exit
        #     if pred == 2 and conf >= self.p.threshold:
        #         self.order = self.buy(size=self.p.size)
        #         self.log(
        #             f"BUY CREATE @ {self.data.close[0]:.2f}, conf={conf:.3f}"
        #         )
        #     return

        # ==========================
        # EXIT
        # ==========================
        bars_held = len(self) - self.bar_entered

        if bars_held >= self.p.hold_period:
            self.order = self.close()
            self.log(
                f"SELL ALL (time exit) @ {price:.2f}, "
                f"held={bars_held}"
            )
            return

        if pred == 0 and conf >= self.p.reverse_threshold:
            self.order = self.close()
            self.log(
                f"SELL ALL (reverse) @ {price:.2f}, "
                f"conf={conf:.3f}"
            )
        # # 1) time-based exit (core k-step logic)
        # if bars_held >= self.p.hold_period:
        #     self.order = self.sell(size=self.p.size)
        #     self.log(
        #         f"SELL (time exit) @ {self.data.close[0]:.2f}, held={bars_held}"
        #     )
        #     return

        # # 2) strong reverse only
        # if pred == 0 and conf >= self.p.reverse_threshold:
        #     self.order = self.sell(size=self.p.size)
        #     self.log(
        #         f"SELL (reverse) @ {self.data.close[0]:.2f}, conf={conf:.3f}"
        #     )


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

class EMACrossStrategy(BaseStrategy):
    params = dict(
        fast_period=12,
        slow_period=26,
        size=1,
    )

    def __init__(self):
        super().__init__()
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.fast_period)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.slow_period)

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.ema_fast[0] > self.ema_slow[0] and self.ema_fast[-1] <= self.ema_slow[-1]:
                self.order = self.buy(size=self.p.size)
        else:
            if self.ema_fast[0] < self.ema_slow[0] and self.ema_fast[-1] >= self.ema_slow[-1]:
                self.order = self.sell(size=self.p.size)

class MACDStrategy(BaseStrategy):
    params = dict(size=1)

    def __init__(self):
        super().__init__()
        self.macd = bt.ind.MACD(self.data.close)

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.macd.macd[0] > self.macd.signal[0] and self.macd.macd[-1] <= self.macd.signal[-1]:
                self.order = self.buy(size=self.p.size)
        else:
            if self.macd.macd[0] < self.macd.signal[0]:
                self.order = self.sell(size=self.p.size)


class MAcrossover(bt.Strategy): 
	# Moving average parameters
	params = (('pfast',20),('pslow',50),)

	def log(self, txt, dt=None):
		dt = dt or self.datas[0].datetime.date(0)
		print(f'{dt.isoformat()} {txt}') # Comment this line when running optimization

	def __init__(self):
		self.dataclose = self.datas[0].close
		
		# Order variable will contain ongoing order details/status
		self.order = None

		# Instantiate moving averages
		self.fast_sma = bt.indicators.MovingAverageSimple(self.datas[0], period=self.params.pfast)
		self.slow_sma = bt.indicators.MovingAverageSimple(self.datas[0], period=self.params.pslow)
		
		''' Using the built-in crossover indicator
		self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)'''


	def notify_order(self, order):
		if order.status in [order.Submitted, order.Accepted]:
			# An active Buy/Sell order has been submitted/accepted - Nothing to do
			return

		# Check if an order has been completed
		# Attention: broker could reject order if not enough cash
		if order.status in [order.Completed]:
			if order.isbuy():
				self.log(f'BUY EXECUTED, {order.executed.price:.2f}')
			elif order.issell():
				self.log(f'SELL EXECUTED, {order.executed.price:.2f}')
			self.bar_executed = len(self)

		elif order.status in [order.Canceled, order.Margin, order.Rejected]:
			self.log('Order Canceled/Margin/Rejected')

		# Reset orders
		self.order = None

	def next(self):
		''' Logic for using the built-in crossover indicator
		
		if self.crossover > 0: # Fast ma crosses above slow ma
			pass # Signal for buy order
		elif self.crossover < 0: # Fast ma crosses below slow ma
			pass # Signal for sell order
		'''

		# Check for open orders
		if self.order:
			return

		# Check if we are in the market
		if not self.position:
			# We are not in the market, look for a signal to OPEN trades
				
			#If the 20 SMA is above the 50 SMA
			if self.fast_sma[0] > self.slow_sma[0] and self.fast_sma[-1] < self.slow_sma[-1]:
				self.log(f'BUY CREATE {self.dataclose[0]:2f}')
				# Keep track of the created order to avoid a 2nd order
				self.order = self.buy()
			#Otherwise if the 20 SMA is below the 50 SMA   
			elif self.fast_sma[0] < self.slow_sma[0] and self.fast_sma[-1] > self.slow_sma[-1]:
				self.log(f'SELL CREATE {self.dataclose[0]:2f}')
				# Keep track of the created order to avoid a 2nd order
				self.order = self.sell()
		else:
			# We are already in the market, look for a signal to CLOSE trades
			if len(self) >= (self.bar_executed + 5):
				self.log(f'CLOSE CREATE {self.dataclose[0]:2f}')
				self.order = self.close()

