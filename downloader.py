from data import YahooFinanceDownloader
from typing import List, Union, Optional



INTERVAL = "1d"
START_DATE = "2007-01-01"
END_DATE = "2025-12-01"
SAVE_DIR = "./data/data/msft"
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