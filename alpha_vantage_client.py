"""
Alpha Vantage API Client

This module handles all interactions with the Alpha Vantage API,
including data fetching and transformation.
"""

import os
import requests
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AlphaVantageClient:
    """Client for interacting with the Alpha Vantage API"""
    
    def __init__(self, api_key=None):
        """Initialize with API key from environment or passed directly"""
        # Force reload the environment variables to ensure we get the latest values
        load_dotenv(override=True)
        
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required. Set it in .env file or pass directly.")
        if self.api_key == "your_api_key_here":
            raise ValueError("Please replace the placeholder API key in .env file with your actual Alpha Vantage API key.")
            
        print(f"Using Alpha Vantage API key: {self.api_key[:4]}...{self.api_key[-4:]}")
        self.base_url = "https://www.alphavantage.co/query"
    
    def _make_request(self, params):
        """Make a request to the Alpha Vantage API with the given parameters"""
        # Make sure we're using the actual API key from the .env file
        params['apikey'] = self.api_key
        print(f"Making API request with parameters: {params}")
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        # Debug: Print the raw response text
        print(f"API response: {response.text[:500]}..." if len(response.text) > 500 else f"API response: {response.text}")
        
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            raise Exception(f"API error: {data['Error Message']}")
        if 'Information' in data and 'call frequency' in data['Information']:
            raise Exception(f"API limit reached: {data['Information']}")
        if 'Note' in data:
            print(f"API Note: {data['Note']}")
            
        return data
    
    def get_company_overview(self, ticker):
        """Get company overview data"""
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker
        }
        return self._make_request(params)
    
    def get_daily_adjusted(self, ticker, outputsize='compact'):
        """Get daily adjusted time series data
        
        Args:
            ticker: Stock symbol
            outputsize: 'compact' for last 100 data points, 'full' for up to 20 years
        """
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': ticker,
            'outputsize': outputsize
        }
        data = self._make_request(params)
        
        # Convert to DataFrame
        time_series = data.get('Time Series (Daily)', {})
        if not time_series:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        return df
    
    def get_weekly_adjusted(self, ticker):
        """Get weekly adjusted time series data"""
        params = {
            'function': 'TIME_SERIES_WEEKLY_ADJUSTED',
            'symbol': ticker
        }
        data = self._make_request(params)
        
        # Convert to DataFrame
        time_series = data.get('Weekly Adjusted Time Series', {})
        if not time_series:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        return df
    
    def get_monthly_adjusted(self, ticker):
        """Get monthly adjusted time series data"""
        params = {
            'function': 'TIME_SERIES_MONTHLY_ADJUSTED',
            'symbol': ticker
        }
        data = self._make_request(params)
        
        # Convert to DataFrame
        time_series = data.get('Monthly Adjusted Time Series', {})
        if not time_series:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        return df
    
    def get_intraday(self, ticker, interval='60min', outputsize='compact'):
        """Get intraday time series data
        
        Args:
            ticker: Stock symbol
            interval: Time interval between data points (1min, 5min, 15min, 30min, 60min)
            outputsize: 'compact' for last 100 data points, 'full' for up to 20 years
        """
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': ticker,
            'interval': interval,
            'outputsize': outputsize
        }
        data = self._make_request(params)
        
        # Convert to DataFrame
        time_series_key = f"Time Series ({interval})"
        time_series = data.get(time_series_key, {})
        if not time_series:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        return df
    
    def get_news(self, ticker):
        """Get news about the ticker
        
        Returns news articles related to the ticker.
        """
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'sort': 'RELEVANCE'
        }
        data = self._make_request(params)
        
        # Get news feed
        feed = data.get('feed', [])
        if not feed:
            return []
            
        return feed
        
    def get_news_by_topics(self, topics_list):
        """Get news by topics instead of tickers
        
        Args:
            topics_list: List of topic keywords
            
        Returns news articles related to the topics.
        """
        # Join the topics into a comma-separated string
        # According to Alpha Vantage docs, topics should be comma-separated
        topics = ','.join(topics_list)
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'topics': topics,  # Use topics parameter instead of tickers
            'sort': 'RELEVANCE',
            'limit': 50  # Limit results to 50 articles
        }
        
        try:
            data = self._make_request(params)
            
            # Get news feed
            feed = data.get('feed', [])
            if not feed:
                return []
                
            # Mark these articles with their source
            for article in feed:
                article['source'] = 'topic_search'
                
            return feed
        except Exception as e:
            print(f"Error in get_news_by_topics: {e}")
            return []
    
    def get_sector_performance(self):
        """Get sector performance data"""
        params = {
            'function': 'SECTOR'
        }
        return self._make_request(params)
    
    def get_quote(self, ticker):
        """Get current quote for a symbol"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': ticker
        }
        data = self._make_request(params)
        quote = data.get('Global Quote', {})
        return {k.split('. ')[1]: v for k, v in quote.items()} if quote else {}
