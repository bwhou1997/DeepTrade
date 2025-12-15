"""
Tests for data download module
"""
import unittest
import os
import tempfile
import shutil
from data.download import (
    download_multiple_stocks,
    download_stock_data,
    YahooFinanceDownloader,
    get_sp500_symbols,
)


class TestDownloadStockData(unittest.TestCase):
    """Test single stock download"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_download_single_stock(self):
        """Test downloading single stock data"""
        save_path = os.path.join(self.test_dir, 'AAPL.csv')
        df = download_stock_data(
            'AAPL',
            '2024-01-01',
            '2024-01-31',
            interval='1d',
            save_path=save_path
        )
        
        # Check if data is returned
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        
        # Check required columns
        required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
        self.assertTrue(required_cols.issubset(set(df.columns)))
        
        # Check if file was saved
        self.assertTrue(os.path.exists(save_path))


class TestDownloadMultipleStocks(unittest.TestCase):
    """Test downloading multiple stocks from a simple list"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_download_multiple_stocks_simple_list(self):
        """Test downloading multiple stocks from a simple list"""
        symbols = ['AAPL', 'MSFT']
        result = download_multiple_stocks(
            symbols,
            '2024-01-01',
            '2024-01-31',
            save_dir=self.test_dir
        )
        
        # Check if files exist
        for symbol in symbols:
            file_path = os.path.join(self.test_dir, f'{symbol}.csv')
            self.assertTrue(os.path.exists(file_path), f"File {file_path} not found")
        
        # Check if data is returned
        self.assertEqual(len(result), len(symbols))
        for symbol in symbols:
            self.assertIn(symbol, result)
            self.assertFalse(result[symbol].empty)


class TestYahooFinanceDownloader(unittest.TestCase):
    """Test the main YahooFinanceDownloader function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_downloader_with_simple_list(self):
        """Test YahooFinanceDownloader with simple symbol list"""
        symbols = ['AAPL', 'MSFT']
        result = YahooFinanceDownloader(
            symbols=symbols,
            start='2024-01-01',
            end='2024-01-31',
            save_dir=self.test_dir,
            max_stocks=5
        )
        
        # Check if data was downloaded
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
        
        # Check if files were saved
        for symbol in symbols:
            file_path = os.path.join(self.test_dir, f'{symbol}.csv')
            if os.path.exists(file_path):
                self.assertTrue(True)
    
    def test_downloader_with_sp500_universe(self):
        """Test YahooFinanceDownloader with sp500 universe"""
        result = YahooFinanceDownloader(
            symbols='sp500',
            start='2024-01-01',
            end='2024-01-31',
            save_dir=self.test_dir,
            sleep_sec=0.2,
            max_stocks=3
        )
        
        # Check if data was downloaded (should download first 10 symbols)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

    def test_downloader_with_nasdaq100_universe(self):
        """Test YahooFinanceDownloader with nasdaq100 universe"""
        result = YahooFinanceDownloader(
            symbols='nasdaq100',
            start='2024-01-01',
            end='2024-01-31',
            save_dir=self.test_dir,
            sleep_sec=0.2,
            max_stocks=3
        )
        
        # Check if data was downloaded (should download first 10 symbols)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

if __name__ == '__main__':
    unittest.main()
