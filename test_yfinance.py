"""
Simple YFinance Test Script

This standalone script tests the basic functionality of yfinance 
to fetch data for AAPL without any custom rate limiting or complex logic.
"""

import yfinance as yf
import pandas as pd
import time

def test_company_info():
    """Test fetching company info for AAPL"""
    print("\n===== Testing Company Info =====")
    try:
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        # Print a subset of the info to keep the output manageable
        important_keys = ['symbol', 'shortName', 'sector', 'industry', 'marketCap', 'currentPrice']
        for key in important_keys:
            if key in info:
                print(f"{key}: {info[key]}")
            else:
                print(f"{key}: Not available")
                
        return True
    except Exception as e:
        print(f"Error fetching company info: {e}")
        return False

def test_price_history():
    """Test fetching price history for AAPL"""
    print("\n===== Testing Price History =====")
    try:
        ticker = yf.Ticker("AAPL")
        
        # Get last 5 days of data
        hist = ticker.history(period="5d")
        
        if hist.empty:
            print("No price history found")
            return False
        
        print(f"Got {len(hist)} days of price history")
        print(hist.head())
        return True
    except Exception as e:
        print(f"Error fetching price history: {e}")
        return False

def test_news():
    """Test fetching news for AAPL"""
    print("\n===== Testing News =====")
    try:
        ticker = yf.Ticker("AAPL")
        news = ticker.news
        
        if not news:
            print("No news found")
            return False
        
        print(f"Got {len(news)} news articles")
        for i, article in enumerate(news[:3]):  # Show top 3 articles
            print(f"Article {i+1}: {article.get('title', 'No title')}")
        return True
    except Exception as e:
        print(f"Error fetching news: {e}")
        return False

def test_quarterly_financials():
    """Test fetching quarterly financials for AAPL"""
    print("\n===== Testing Quarterly Financials =====")
    try:
        ticker = yf.Ticker("AAPL")
        financials = ticker.quarterly_financials
        
        if financials.empty:
            print("No quarterly financials found")
            return False
        
        print("Quarterly financials data shape:", financials.shape)
        print(financials.head(3))
        return True
    except Exception as e:
        print(f"Error fetching quarterly financials: {e}")
        return False

def run_all_tests():
    """Run all yfinance tests"""
    print(f"Testing yfinance version: {yf.__version__}")
    print("Starting tests for AAPL ticker...")
    
    tests = [
        ("Company Info", test_company_info),
        ("Price History", test_price_history),
        ("News", test_news),
        ("Quarterly Financials", test_quarterly_financials)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\nStarting test: {name}")
        start_time = time.time()
        success = test_func()
        elapsed = time.time() - start_time
        results.append((name, success, elapsed))
        print(f"Test {name} {'PASSED' if success else 'FAILED'} in {elapsed:.2f} seconds")
        
        # Add a delay between tests to avoid rate limiting
        if not name == tests[-1][0]:  # Don't wait after the last test
            wait_time = 2.0
            print(f"Waiting {wait_time} seconds before next test...")
            time.sleep(wait_time)
    
    # Print summary
    print("\n===== Test Summary =====")
    all_passed = all(success for _, success, _ in results)
    print(f"Overall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    for name, success, elapsed in results:
        status = "PASSED" if success else "FAILED"
        print(f"{name}: {status} ({elapsed:.2f}s)")

if __name__ == "__main__":
    run_all_tests()
