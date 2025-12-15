# indicator_bundle.py
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


class IndicatorBundle:
    def __init__(self, config: dict):
        self.use = config.get("use", True)
        self.indicators = config.get("indicators", {})

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use or not self.indicators:
            return df

        df = df.copy()

        for name, params in self.indicators.items():
            if name == "sma":
                for w in params:
                    df[f"sma_{w}"] = SMAIndicator(
                        df["close"], window=w
                    ).sma_indicator()

            elif name == "ema":
                for w in params:
                    df[f"ema_{w}"] = EMAIndicator(
                        df["close"], window=w
                    ).ema_indicator()

            elif name == "rsi":
                for w in params:
                    df[f"rsi_{w}"] = RSIIndicator(
                        df["close"], window=w
                    ).rsi()

            elif name == "macd":
                for fast, slow, signal in params:
                    macd = MACD(
                        df["close"],
                        window_fast=fast,
                        window_slow=slow,
                        window_sign=signal,
                    )
                    df[f"macd_{fast}_{slow}"] = macd.macd()
                    df[f"macd_signal_{signal}"] = macd.macd_signal()

            elif name == "bbands":
                for w in params:
                    bb = BollingerBands(df["close"], window=w)
                    df[f"bb_low_{w}"] = bb.bollinger_lband()
                    df[f"bb_mid_{w}"] = bb.bollinger_mavg()
                    df[f"bb_high_{w}"] = bb.bollinger_hband()

            elif name == "atr":
                for w in params:
                    atr = AverageTrueRange(
                        df["high"], df["low"], df["close"], window=w
                    )
                    df[f"atr_{w}"] = atr.average_true_range()

            else:
                raise ValueError(f"Unknown indicator: {name}")

        return df
