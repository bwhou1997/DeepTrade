import os
import time
from typing import Optional, Dict, List, Union
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from .registry import register_universe, UNIVERSE_REGISTRY
import os
import pandas as pd
import requests
from io import StringIO

@register_universe("nasdaq100")
def get_nasdaq100_symbols() -> list[str]:
    """
    Get NASDAQ-100 symbols from Wikipedia (robust version).
    """

    print("[INFO] Fetching NASDAQ-100 symbols from Wikipedia...")

    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        tables = pd.read_html(StringIO(resp.text))

        df = None
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if "ticker" in cols or "symbol" in cols:
                df = t
                break

        if df is None:
            raise RuntimeError("Could not find NASDAQ-100 ticker table")

        # 找到 ticker 列
        ticker_col = None
        for c in df.columns:
            if str(c).lower() in ("ticker", "symbol", "trading symbol"):
                ticker_col = c
                break

        if ticker_col is None:
            raise RuntimeError("Ticker column not found in NASDAQ-100 table")

        symbols = (
            df[ticker_col]
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
            .tolist()
        )

        # 过滤明显不是 ticker 的垃圾行
        symbols = [
            s for s in symbols
            if s.isupper() and 1 <= len(s) <= 6
        ]

        print(f"[INFO] NASDAQ-100 symbols fetched: {len(symbols)}")

        return symbols

    except Exception as e:
        raise RuntimeError(
            "Failed to fetch NASDAQ-100 symbols"
        ) from e


@register_universe("sp500")
def get_sp500_symbols() -> list[str]:
    """
    Get S&P 500 symbols with local cache + Wikipedia fallback.
    This version is production-safe.
    """

    print("[INFO] Fetching S&P 500 symbols from Wikipedia...")

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        tables = pd.read_html(StringIO(resp.text))
        df = tables[0]

        symbols = df["Symbol"].str.replace(".", "-", regex=False).tolist()

        return symbols

    except Exception as e:
        raise RuntimeError(
            "Failed to fetch S&P 500 symbols. "
            "Check internet or provide local cache."
        ) from e



# ============================================================
# 2. Interval-aware start date adjustment
# ============================================================

def adjust_start_for_interval(start: str, interval: str) -> str:
    """
    Yahoo Finance limits intraday data history.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    now = datetime.now()

    if interval == "1h":
        max_days = 730
    elif interval in ("30m", "15m", "5m"):
        max_days = 60
    elif interval == "1m":
        max_days = 7
    else:
        return start  # 1d, 1wk, etc.

    earliest_allowed = now - timedelta(days=max_days)

    if start_dt < earliest_allowed:
        print(
            f"[WARN] {interval} data limited to last {max_days} days. "
            f"Adjusting start date from {start} to {earliest_allowed.date()}"
        )
        return earliest_allowed.strftime("%Y-%m-%d")

    return start


# ============================================================
# 3. Download single stock
# ============================================================

def download_stock_data(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.
    """
    try:
        start = adjust_start_for_interval(start, interval)

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError("No data returned")

        # Reset index and standardize columns
        df.reset_index(inplace=True)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Normalize datetime column name
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)
        elif "date" not in df.columns:
            raise ValueError("No date/datetime column found")

        cols = ["date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)

        return df

    except Exception as e:
        raise RuntimeError(f"{symbol}: {e}")


# ============================================================
# 4. Download multiple stocks (safe + throttled)
# ============================================================

def download_multiple_stocks(
    symbols: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    save_dir: Optional[str] = None,
    sleep_sec: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """
    Download data for multiple stocks with throttling.
    """
    data = {}
    failed = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, symbol in enumerate(symbols, 1):
        try:
            save_path = (
                os.path.join(save_dir, f"{symbol}.csv")
                if save_dir
                else None
            )

            df = download_stock_data(
                symbol=symbol,
                start=start,
                end=end,
                interval=interval,
                save_path=save_path,
            )

            data[symbol] = df
            print(f"[{i}/{len(symbols)}] OK  {symbol:<6} rows={len(df)}")

        except Exception as e:
            print(f"[{i}/{len(symbols)}] FAIL {symbol:<6} {e}")
            failed.append(symbol)

        time.sleep(sleep_sec)

    print("\n================ SUMMARY ================")
    print(f"Success: {len(data)}")
    print(f"Failed : {len(failed)}")
    if failed:
        print("Failed symbols:", failed)

    return data


def YahooFinanceDownloader(
    symbols: Union[str, List[str]],
    start: str,
    end: str,
    interval: str = "1d",
    save_dir: Optional[str] = None,
    sleep_sec: float = 1.0,
    max_stocks: Optional[int] = None,
):
    """
    symbols:
        - List[str]: explicit stock list
        - str: universe name (e.g. 'sp500')
    """

    # --------------------------------------------------
    # 1. Resolve symbols
    # --------------------------------------------------
    if isinstance(symbols, str):
        universe_name = symbols

        if universe_name not in UNIVERSE_REGISTRY:
            raise KeyError(
                f"Universe '{universe_name}' not registered. "
                f"Available: {list(UNIVERSE_REGISTRY.keys())}"
            )

        print(f"[INFO] Resolving universe '{universe_name}'")
        symbols_list = UNIVERSE_REGISTRY[universe_name]()
    elif isinstance(symbols, list):
        symbols_list = symbols

    else:
        raise TypeError(
            "symbols must be either List[str] or str (universe name)"
        )

    print(f"[INFO] Downloading {len(symbols_list)} symbols")

    # --------------------------------------------------
    # 2. Download
    # --------------------------------------------------
    return download_multiple_stocks(
        symbols=symbols_list[:max_stocks] if max_stocks is not None else symbols_list,
        start=start,
        end=end,
        interval=interval,
        save_dir=save_dir,
        sleep_sec=sleep_sec,
    )


# ============================================================
# 5. Main entry
# ============================================================

if __name__ == "__main__":

    INTERVAL = "1d"
    START_DATE = "2010-01-01"
    END_DATE = "2020-01-01"
    SAVE_DIR = "data/msft"
    MAX_STOCKS = None

    SYMBOLS: str | List[str] = ['MSFT'] 
    # SYMBOLS = "nasdaq100", "sp500"
    # SYMBOLS = ["AAPL", "MSFT",...]

    YahooFinanceDownloader(
        symbols=SYMBOLS,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        save_dir=SAVE_DIR,
        max_stocks=MAX_STOCKS,
    )
