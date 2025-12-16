"""
Backtesting engine using backtrader
"""
import backtrader as bt
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any


class BacktestEngine:
    """
    Backtesting engine wrapper around backtrader.
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

        self._setup_cerebro()
        self._load_data()

    # ------------------------------------------------------------------
    # Cerebro setup
    # ------------------------------------------------------------------
    def _setup_cerebro(self):
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=self.commission)

        # Analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_data(self):
        df = pd.read_csv(self.data_path)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        elif "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"], utc=True)
        else:
            raise ValueError("CSV must contain 'date' or 'Date' column")

        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        for old, new in col_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]

        df = df.sort_values("date").set_index("date")

        # ✅ CRITICAL FIX: filter in pandas
        if self.fromdate:
            df = df[df.index >= pd.Timestamp(self.fromdate, tz="UTC")]
        if self.todate:
            df = df[df.index <= pd.Timestamp(self.todate, tz="UTC")]

        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )

        self.cerebro.adddata(data)


    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------
    def add_strategy(self, strategy_cls, **kwargs):
        """
        Add a Strategy CLASS (not instance).
        """
        if not issubclass(strategy_cls, bt.Strategy):
            raise TypeError("strategy_cls must be a backtrader Strategy class")

        self.cerebro.addstrategy(strategy_cls, **kwargs)

    # ------------------------------------------------------------------
    # Run backtest
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        print(f"Starting Portfolio Value: {self.cerebro.broker.getvalue():.2f}")

        strategies = self.cerebro.run()
        strat = strategies[0]

        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash

        print(f"Final Portfolio Value: {final_value:.2f}")
        print(f"Total Return: {total_return:.2%}")

        results = {
            "initial_value": self.initial_cash,
            "final_value": final_value,
            "total_return": total_return,
        }

        # --- analyzers ---
        if hasattr(strat.analyzers, "sharpe"):
            res = strat.analyzers.sharpe.get_analysis()
            results["sharpe_ratio"] = res.get("sharperatio")

        if hasattr(strat.analyzers, "drawdown"):
            dd = strat.analyzers.drawdown.get_analysis()
            results["max_drawdown"] = dd.get("max", {}).get("drawdown")
            results["max_drawdown_len"] = dd.get("max", {}).get("len")

        if hasattr(strat.analyzers, "returns"):
            ret = strat.analyzers.returns.get_analysis()
            results["annual_return"] = ret.get("rnorm100")

        if hasattr(strat.analyzers, "trades"):
            t = strat.analyzers.trades.get_analysis()
            total = t.get("total", {}).get("total", 0)
            won = t.get("won", {}).get("total", 0)
            lost = t.get("lost", {}).get("total", 0)

            results.update(
                {
                    "total_trades": total,
                    "won": won,
                    "lost": lost,
                    "win_rate": won / total if total > 0 else 0.0,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def plot(self, save_path: Optional[str] = None):
        figs = self.cerebro.plot(style="candlestick")

        if save_path and figs:
            fig = figs[0][0]
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from strategy import SimpleMAStrategy

    engine = BacktestEngine(
        data_path="../data/data/msft/MSFT.csv",
        initial_cash=10000,
        commission=0.001,
    )

    engine.add_strategy(
        SimpleMAStrategy,
        fast_period=10,
        slow_period=30,
    )

    results = engine.run()
    print(results)
