"""
Stock data download utilities
"""
import pandas as pd
import yfinance as yf
from typing import Optional
from datetime import datetime
import os


def download_stock_data(
    symbol: str,
    start: str,
    end: str,
    interval: str = '1d',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1h', '1m', etc.)
        save_path: Optional path to save CSV file
    
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Rename columns to standard format
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if 'date' in df.columns:
            df.rename(columns={'date': 'date'}, inplace=True)
        elif 'datetime' in df.columns:
            df.rename(columns={'datetime': 'date'}, inplace=True)
        
        # Select only OHLCV columns
        cols_to_keep = ['date'] + [col for col in required_cols if col in df.columns]
        df = df[cols_to_keep]
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Data saved to {save_path}")
        
        return df
    
    except Exception as e:
        raise RuntimeError(f"Error downloading data for {symbol}: {str(e)}")


def download_multiple_stocks(
    symbols: list,
    start: str,
    end: str,
    interval: str = '1d',
    save_dir: Optional[str] = None
) -> dict:
    """
    Download data for multiple stocks.
    
    Args:
        symbols: List of stock ticker symbols
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        interval: Data interval
        save_dir: Optional directory to save CSV files
    
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    data_dict = {}
    
    for symbol in symbols:
        try:
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"{symbol}.csv")
            
            df = download_stock_data(symbol, start, end, interval, save_path)
            data_dict[symbol] = df
            print(f"Downloaded {symbol}: {len(df)} rows")
        
        except Exception as e:
            print(f"Failed to download {symbol}: {e}")
            continue
    
    return data_dict


if __name__ == "__main__":
    # Example usage
    df = download_stock_data('AAPL', '2020-01-01', '2023-12-31', save_path='data/AAPL.csv')
    print(df.head())
    print(f"\nDownloaded {len(df)} rows")


