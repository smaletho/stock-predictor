"""
Yahoo Finance API Client

This module handles all interactions with the Yahoo Finance API through yfinance,
including data fetching and transformation.
"""

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from newsapi import NewsApiClient

# Load environment variables
load_dotenv()

# Configure YFinance session parameters globally
yf.set_tz_cache_location(os.path.join(os.path.dirname(__file__), 'data', 'yf_cache'))

# Create directory for cache if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'data', 'yf_cache'), exist_ok=True)

class YahooFinanceClient:
    """Client for interacting with Yahoo Finance API via yfinance"""
    
    def __init__(self, newsapi_key=None, max_retries=3, backoff_factor=2):
        """Initialize the client and optional NewsAPI client for news data
        
        Args:
            newsapi_key: Optional NewsAPI key for better news coverage
            max_retries: Maximum number of retries for requests
            backoff_factor: Backoff factor for retry delay (seconds)
        """
        # Force reload the environment variables
        load_dotenv(override=True)
        
        # Rate limiting settings
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Minimum seconds between requests
        
        # The new yfinance version (0.2.61+) handles sessions internally using curl_cffi
        # We should not create our own session as it would conflict with yfinance's internal handling
        # Let yfinance handle its own sessions and connections
        
        # NewsAPI for better news coverage (optional)
        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_KEY')
        self.use_newsapi = self.newsapi_key is not None
        
        if self.use_newsapi:
            print(f"Using NewsAPI for enhanced news coverage")
            self.newsapi = NewsApiClient(api_key=self.newsapi_key)
        else:
            print("NewsAPI key not found, will use Yahoo Finance for news (limited)")
        
        print("Yahoo Finance client initialized with custom session and proper headers")
        
        # We'll skip initial connection test to reduce API calls
        print("Yahoo Finance client ready - will make API calls only when needed")
        
        # Create cache directory for downloads
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
    
    def _test_connection(self):
        """Test the connection to Yahoo Finance API"""
        print("Testing connection to Yahoo Finance...")
        ticker = yf.Ticker("SPY")
        quote = ticker.history(period="1d")
        if not quote.empty:
            print("Connection test successful!")
        else:
            raise Exception("Failed to retrieve data in connection test")
    
    def _rate_limited_request(self, func, *args, **kwargs):
        """Execute a function with rate limiting and retry logic
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function call
        """
        # Add delay between requests to avoid rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            # Add random jitter to delay to avoid synchronized requests
            sleep_time = self.min_request_interval - time_since_last_request + random.uniform(0.1, 1.0)
            print(f"Rate limiting: Waiting {sleep_time:.2f} seconds before request")
            time.sleep(sleep_time)
        
        # Update last request time
        self.last_request_time = time.time()
        
        # Try the request with retries
        for retry in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if retry == self.max_retries:
                    # Last retry attempt failed, re-raise the exception
                    raise
                
                # Calculate backoff delay
                backoff_delay = self.backoff_factor * (2 ** retry) + random.uniform(0, 1)
                print(f"Request failed: {e}. Retrying in {backoff_delay:.2f} seconds (attempt {retry + 1}/{self.max_retries})")
                time.sleep(backoff_delay)
    
    def get_company_overview(self, ticker):
        """Get company overview data"""
        try:
            # Create a Ticker object and use rate limiting
            def _fetch_info():
                stock = yf.Ticker(ticker)
                return stock.info
            
            # Get company info with rate limiting
            info = self._rate_limited_request(_fetch_info)
            
            # Map Yahoo Finance fields to match the expected format
            # from Alpha Vantage for compatibility
            company_data = {
                "Symbol": ticker,
                "AssetType": info.get("quoteType", "Common Stock"),
                "Name": info.get("shortName", ""),
                "Description": info.get("longBusinessSummary", ""),
                "Exchange": info.get("exchange", ""),
                "Currency": info.get("currency", "USD"),
                "Country": info.get("country", ""),
                "Sector": info.get("sector", ""),
                "Industry": info.get("industry", ""),
                "Address": info.get("address1", ""),
                "FiscalYearEnd": info.get("lastFiscalYearEnd", ""),
                "LatestQuarter": info.get("lastDividendDate", ""),
                "MarketCapitalization": info.get("marketCap", ""),
                "EBITDA": info.get("ebitda", ""),
                "PERatio": info.get("trailingPE", ""),
                "PEGRatio": info.get("pegRatio", ""),
                "BookValue": info.get("bookValue", ""),
                "DividendPerShare": info.get("lastDividendValue", ""),
                "DividendYield": info.get("dividendYield", ""),
                "EPS": info.get("trailingEps", ""),
                "RevenuePerShareTTM": info.get("revenuePerShare", ""),
                "ProfitMargin": info.get("profitMargins", ""),
                "OperatingMarginTTM": info.get("operatingMargins", ""),
                "ReturnOnAssetsTTM": info.get("returnOnAssets", ""),
                "ReturnOnEquityTTM": info.get("returnOnEquity", ""),
                "RevenueTTM": info.get("totalRevenue", ""),
                "GrossProfitTTM": info.get("grossProfits", ""),
                "DilutedEPSTTM": info.get("trailingEps", ""),
                "QuarterlyEarningsGrowthYOY": info.get("earningsQuarterlyGrowth", ""),
                "QuarterlyRevenueGrowthYOY": info.get("revenueQuarterlyGrowth", ""),
                "AnalystTargetPrice": info.get("targetMeanPrice", ""),
                "TrailingPE": info.get("trailingPE", ""),
                "ForwardPE": info.get("forwardPE", ""),
                "PriceToSalesRatioTTM": info.get("priceToSalesTrailing12Months", ""),
                "PriceToBookRatio": info.get("priceToBook", ""),
                "EVToRevenue": info.get("enterpriseToRevenue", ""),
                "EVToEBITDA": info.get("enterpriseToEbitda", ""),
                "Beta": info.get("beta", ""),
                "52WeekHigh": info.get("fiftyTwoWeekHigh", ""),
                "52WeekLow": info.get("fiftyTwoWeekLow", ""),
                "50DayMovingAverage": info.get("fiftyDayAverage", ""),
                "200DayMovingAverage": info.get("twoHundredDayAverage", ""),
                "SharesOutstanding": info.get("sharesOutstanding", ""),
                "SharesFloat": info.get("floatShares", ""),
                "SharesShort": info.get("sharesShort", ""),
                "SharesShortPriorMonth": info.get("sharesShortPriorMonth", ""),
                "ShortRatio": info.get("shortRatio", ""),
                "ShortPercentOutstanding": info.get("shortPercentOfFloat", ""),
                "ShortPercentFloat": info.get("shortPercentOfFloat", ""),
                "PercentInsiders": info.get("heldPercentInsiders", ""),
                "PercentInstitutions": info.get("heldPercentInstitutions", ""),
                "ForwardAnnualDividendRate": info.get("dividendRate", ""),
                "ForwardAnnualDividendYield": info.get("dividendYield", ""),
                "PayoutRatio": info.get("payoutRatio", ""),
                "DividendDate": info.get("dividendDate", ""),
                "ExDividendDate": info.get("exDividendDate", ""),
                "LastSplitFactor": info.get("lastSplitFactor", ""),
                "LastSplitDate": info.get("lastSplitDate", "")
            }
            
            return company_data
        except Exception as e:
            print(f"Error fetching company data: {e}")
            return {"Symbol": ticker}
    
    def get_daily_adjusted(self, ticker, outputsize='compact'):
        """Get daily adjusted time series data
        
        Args:
            ticker: Stock symbol
            outputsize: 'compact' for last 100 data points, 'full' for maximum available
        """
        try:
            # Determine period based on outputsize
            period = "3mo" if outputsize == 'compact' else "max"
            
            # Define function to fetch history with rate limiting
            def _fetch_history():
                stock = yf.Ticker(ticker)
                return stock.history(period=period, interval="1d")
            
            # Get historical data with rate limiting
            hist = self._rate_limited_request(_fetch_history)
            
            # Try a different period if the first attempt returns empty data
            if hist.empty:
                print(f"No data found for {ticker} with period={period}, trying 1mo period")
                period = "1mo"
                
                def _fetch_history_fallback():
                    stock = yf.Ticker(ticker)
                    return stock.history(period=period, interval="1d")
                
                hist = self._rate_limited_request(_fetch_history_fallback)
            
            if hist.empty:
                return pd.DataFrame()
            
            # Rename columns to match Alpha Vantage format for compatibility
            hist.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Stock Splits': 'split_coefficient'
            }, inplace=True)
            
            # Add 'adjusted close' column if it doesn't exist
            if 'adjusted close' not in hist.columns and 'close' in hist.columns:
                hist['adjusted close'] = hist['close']
            
            return hist
        except Exception as e:
            print(f"Error fetching daily data: {e}")
            return pd.DataFrame()
    
    def get_weekly_adjusted(self, ticker):
        """Get weekly adjusted time series data"""
        try:
            # Define function to fetch history with rate limiting
            def _fetch_history():
                stock = yf.Ticker(ticker)
                return stock.history(period="5y", interval="1wk")
            
            # Get historical data with rate limiting
            hist = self._rate_limited_request(_fetch_history)
            
            # Try a different period if the first attempt returns empty data
            if hist.empty:
                print(f"No weekly data found for {ticker} with period=5y, trying 1y period")
                
                def _fetch_history_fallback():
                    stock = yf.Ticker(ticker)
                    return stock.history(period="1y", interval="1wk")
                
                hist = self._rate_limited_request(_fetch_history_fallback)
            
            if hist.empty:
                return pd.DataFrame()
            
            # Rename columns to match Alpha Vantage format for compatibility
            hist.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Stock Splits': 'split_coefficient'
            }, inplace=True)
            
            # Add 'adjusted close' column if it doesn't exist
            if 'adjusted close' not in hist.columns and 'close' in hist.columns:
                hist['adjusted close'] = hist['close']
            
            return hist
        except Exception as e:
            print(f"Error fetching weekly data: {e}")
            return pd.DataFrame()
    
    def get_monthly_adjusted(self, ticker):
        """Get monthly adjusted time series data"""
        try:
            # Define function to fetch history with rate limiting
            def _fetch_history():
                stock = yf.Ticker(ticker)
                return stock.history(period="10y", interval="1mo")
            
            # Get historical data with rate limiting
            hist = self._rate_limited_request(_fetch_history)
            
            # Try a different period if the first attempt returns empty data
            if hist.empty:
                print(f"No monthly data found for {ticker} with period=10y, trying 2y period")
                
                def _fetch_history_fallback():
                    stock = yf.Ticker(ticker)
                    return stock.history(period="2y", interval="1mo")
                
                hist = self._rate_limited_request(_fetch_history_fallback)
            
            if hist.empty:
                return pd.DataFrame()
            
            # Rename columns to match Alpha Vantage format for compatibility
            hist.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Stock Splits': 'split_coefficient'
            }, inplace=True)
            
            # Add 'adjusted close' column if it doesn't exist
            if 'adjusted close' not in hist.columns and 'close' in hist.columns:
                hist['adjusted close'] = hist['close']
            
            return hist
        except Exception as e:
            print(f"Error fetching monthly data: {e}")
            return pd.DataFrame()
    
    def get_intraday(self, ticker, interval='60m', outputsize='compact'):
        """Get intraday time series data
        
        Args:
            ticker: Stock symbol
            interval: Time interval between data points (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h)
            outputsize: 'compact' for last 100 data points, 'full' for maximum available
        """
        try:
            # Map interval to yfinance format
            yf_interval = interval.replace('min', 'm').replace('hour', 'h')
            
            # Determine period based on outputsize and interval
            if outputsize == 'compact':
                period = "1d" if 'm' in yf_interval else "5d"
            else:
                period = "7d" if 'm' in yf_interval else "60d"
            
            # Define function to fetch history with rate limiting
            def _fetch_history():
                stock = yf.Ticker(ticker)
                return stock.history(period=period, interval=yf_interval)
            
            # Get historical data with rate limiting
            hist = self._rate_limited_request(_fetch_history)
            
            # Try a different period if the first attempt returns empty data
            if hist.empty:
                print(f"No intraday data found for {ticker} with period={period}, trying fallback period")
                # Try a different period as fallback
                fallback_period = "7d"  # Longer period as fallback
                
                def _fetch_history_fallback():
                    stock = yf.Ticker(ticker)
                    return stock.history(period=fallback_period, interval=yf_interval)
                
                hist = self._rate_limited_request(_fetch_history_fallback)
            
            if hist.empty:
                return pd.DataFrame()
            
            # Rename columns to match Alpha Vantage format for compatibility
            hist.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Stock Splits': 'split_coefficient'
            }, inplace=True)
            
            # Add 'adjusted close' column if it doesn't exist
            if 'adjusted close' not in hist.columns and 'close' in hist.columns:
                hist['adjusted close'] = hist['close']
            
            return hist
        except Exception as e:
            print(f"Error fetching intraday data: {e}")
            return pd.DataFrame()
    
    def get_news(self, ticker):
        """Get news about the ticker
        
        Returns news articles related to the ticker.
        """
        try:
            if self.use_newsapi:
                # Use NewsAPI for better news coverage
                articles = self.newsapi.get_everything(
                    q=f"{ticker} stock",
                    language='en',
                    sort_by='relevancy',
                    page_size=20
                )
                
                # Format articles to match the expected format from Alpha Vantage
                news_articles = []
                for article in articles.get('articles', []):
                    news_articles.append({
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'time_published': article.get('publishedAt', ''),
                        'summary': article.get('description', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'category': 'news',
                        'relevance_score': '0.8',  # Default relevance score
                        'ticker_sentiment': [{
                            'ticker': ticker,
                            'relevance_score': '0.8',
                            'ticker_sentiment_score': '0.0',  # Neutral by default
                            'ticker_sentiment_label': 'Neutral'
                        }]
                    })
                
                return news_articles
            else:
                # Use Yahoo Finance news (limited)
                stock = yf.Ticker(ticker)
                news = stock.news
                
                # Format news to match the expected format from Alpha Vantage
                news_articles = []
                for article in news:
                    news_articles.append({
                        'title': article.get('title', ''),
                        'url': article.get('link', ''),
                        'time_published': datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat(),
                        'summary': article.get('summary', ''),
                        'source': article.get('publisher', ''),
                        'category': 'news',
                        'relevance_score': '0.8',  # Default relevance score
                        'ticker_sentiment': [{
                            'ticker': ticker,
                            'relevance_score': '0.8',
                            'ticker_sentiment_score': '0.0',  # Neutral by default
                            'ticker_sentiment_label': 'Neutral'
                        }]
                    })
                
                return news_articles
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def get_news_by_topics(self, topics_list):
        """Get news by topics instead of tickers
        
        Args:
            topics_list: List of topic keywords
            
        Returns news articles related to the topics.
        """
        try:
            if self.use_newsapi:
                # Join topics into a query string
                query = " OR ".join(topics_list)
                
                # Use NewsAPI for better news coverage
                articles = self.newsapi.get_everything(
                    q=query,
                    language='en',
                    sort_by='relevancy',
                    page_size=20
                )
                
                # Format articles to match the expected format from Alpha Vantage
                news_articles = []
                for article in articles.get('articles', []):
                    news_articles.append({
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'time_published': article.get('publishedAt', ''),
                        'summary': article.get('description', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'category': 'topic',
                        'source': 'topic_search',
                        'relevance_score': '0.8',  # Default relevance score
                        'ticker_sentiment': []
                    })
                
                return news_articles
            else:
                # Try to get finance-related news using Yahoo Finance
                # This is a fallback and won't be topic-specific
                print("Warning: NewsAPI key not available, returning general financial news")
                
                try:
                    # Get market news from Yahoo Finance
                    market_news = yf.Ticker("^GSPC").news  # S&P 500 news
                    
                    # Format news to match the expected format from Alpha Vantage
                    news_articles = []
                    for article in market_news:
                        news_articles.append({
                            'title': article.get('title', ''),
                            'url': article.get('link', ''),
                            'time_published': datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat(),
                            'summary': article.get('summary', ''),
                            'source': article.get('publisher', ''),
                            'category': 'market',
                            'source': 'topic_search',
                            'relevance_score': '0.5',  # Lower relevance as it's not topic-specific
                            'ticker_sentiment': []
                        })
                    
                    return news_articles
                except Exception as e:
                    print(f"Error fetching general news: {e}")
                    return []
        except Exception as e:
            print(f"Error in get_news_by_topics: {e}")
            return []
    
    def get_sector_performance(self):
        """Get sector performance data"""
        try:
            # Use predefined sector ETFs as proxies for sector performance
            sectors = {
                "Technology": "XLK",
                "Financial": "XLF",
                "Health Care": "XLV",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Energy": "XLE",
                "Utilities": "XLU",
                "Materials": "XLB",
                "Industrial": "XLI",
                "Real Estate": "XLRE",
                "Communication Services": "XLC"
            }
            
            sector_data = {
                "Rank A: Real-Time Performance": {},
                "Rank B: 1 Day Performance": {},
                "Rank C: 5 Day Performance": {},
                "Rank D: 1 Month Performance": {},
                "Rank E: 3 Month Performance": {},
                "Rank F: Year-to-Date (YTD) Performance": {},
                "Rank G: 1 Year Performance": {},
                "Rank H: 3 Year Performance": {},
                "Rank I: 5 Year Performance": {},
                "Rank J: 10 Year Performance": {}
            }
            
            # Get performance data for each sector
            for sector_name, ticker in sectors.items():
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1y")
                    
                    if not hist.empty:
                        latest_close = hist['Close'].iloc[-1]
                        prev_day_close = hist['Close'].iloc[-2] if len(hist) > 1 else latest_close
                        five_day_close = hist['Close'].iloc[-6] if len(hist) > 5 else latest_close
                        month_close = hist['Close'].iloc[-22] if len(hist) > 21 else latest_close
                        three_month_close = hist['Close'].iloc[-66] if len(hist) > 65 else latest_close
                        ytd_close = hist.loc[hist.index.year == hist.index[0].year][0] if hist.index[0].year == datetime.now().year else latest_close
                        year_close = hist['Close'].iloc[0] if len(hist) > 250 else latest_close
                        
                        # Calculate percentage changes
                        real_time_change = 0  # Real-time not available through yfinance
                        one_day_change = ((latest_close - prev_day_close) / prev_day_close) * 100
                        five_day_change = ((latest_close - five_day_close) / five_day_close) * 100
                        month_change = ((latest_close - month_close) / month_close) * 100
                        three_month_change = ((latest_close - three_month_close) / three_month_close) * 100
                        ytd_change = ((latest_close - ytd_close) / ytd_close) * 100
                        year_change = ((latest_close - year_close) / year_close) * 100
                        
                        # Format the changes as strings with + or - prefix
                        sector_data["Rank A: Real-Time Performance"][sector_name] = f"{real_time_change:+.2f}%"
                        sector_data["Rank B: 1 Day Performance"][sector_name] = f"{one_day_change:+.2f}%"
                        sector_data["Rank C: 5 Day Performance"][sector_name] = f"{five_day_change:+.2f}%"
                        sector_data["Rank D: 1 Month Performance"][sector_name] = f"{month_change:+.2f}%"
                        sector_data["Rank E: 3 Month Performance"][sector_name] = f"{three_month_change:+.2f}%"
                        sector_data["Rank F: Year-to-Date (YTD) Performance"][sector_name] = f"{ytd_change:+.2f}%"
                        sector_data["Rank G: 1 Year Performance"][sector_name] = f"{year_change:+.2f}%"
                        
                        # Multi-year performance is left blank as it would require more historical data
                except Exception as e:
                    print(f"Error fetching sector data for {sector_name}: {e}")
            
            return sector_data
        except Exception as e:
            print(f"Error fetching sector performance: {e}")
            return {}
    
    def get_quote(self, ticker):
        """Get current quote for a symbol"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get the latest quote information
            quote = stock.info
            
            # Format the data to match Alpha Vantage format
            formatted_quote = {
                'symbol': ticker,
                'open': quote.get('regularMarketOpen', 0),
                'high': quote.get('regularMarketDayHigh', 0),
                'low': quote.get('regularMarketDayLow', 0),
                'price': quote.get('regularMarketPrice', 0),
                'volume': quote.get('regularMarketVolume', 0),
                'latest trading day': datetime.now().strftime('%Y-%m-%d'),
                'previous close': quote.get('regularMarketPreviousClose', 0),
                'change': quote.get('regularMarketChange', 0),
                'change percent': f"{quote.get('regularMarketChangePercent', 0) * 100:.2f}%"
            }
            
            return formatted_quote
        except Exception as e:
            print(f"Error fetching quote: {e}")
            return {}
