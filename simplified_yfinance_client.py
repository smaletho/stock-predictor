"""
Simplified Yahoo Finance Client

This module provides a simplified interface to the Yahoo Finance API via yfinance,
focusing only on the core functionality needed for the stock predictor application.
"""

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from newsapi import NewsApiClient

# Load environment variables
load_dotenv()

class SimplifiedYahooFinanceClient:
    """A simplified client for interacting with Yahoo Finance API via yfinance"""
    
    def __init__(self, newsapi_key=None):
        """Initialize the client with optional NewsAPI for better news coverage"""
        # Force reload the environment variables
        load_dotenv(override=True)
        
        # Set up NewsAPI client if key is available
        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_KEY')
        self.use_newsapi = self.newsapi_key is not None
        
        if self.use_newsapi:
            print(f"Using NewsAPI for enhanced news coverage")
            self.newsapi = NewsApiClient(api_key=self.newsapi_key)
        else:
            print("NewsAPI key not found, will use Yahoo Finance for news (limited)")
            
        print("Simplified Yahoo Finance client initialized")
    
    def get_company_overview(self, ticker):
        """Get company overview data for the given ticker"""
        print(f"Fetching company data for {ticker}...")
        try:
            # Create a Ticker object
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Map Yahoo Finance fields to match the expected format
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
                "MarketCapitalization": info.get("marketCap", ""),
                "PERatio": info.get("trailingPE", ""),
                "PEGRatio": info.get("pegRatio", ""),
                "BookValue": info.get("bookValue", ""),
                "DividendYield": info.get("dividendYield", ""),
                "EPS": info.get("trailingEps", ""),
                "ProfitMargin": info.get("profitMargins", ""),
                "AnalystTargetPrice": info.get("targetMeanPrice", ""),
                "52WeekHigh": info.get("fiftyTwoWeekHigh", ""),
                "52WeekLow": info.get("fiftyTwoWeekLow", ""),
                "50DayMovingAverage": info.get("fiftyDayAverage", ""),
                "200DayMovingAverage": info.get("twoHundredDayAverage", ""),
            }
            
            return company_data
        except Exception as e:
            print(f"Error fetching company data: {e}")
            return {"Symbol": ticker}
    
    def get_daily_adjusted(self, ticker, outputsize='compact'):
        """Get daily adjusted time series data"""
        print(f"Fetching daily data for {ticker}...")
        try:
            # Determine period based on outputsize
            period = "3mo" if outputsize == 'compact' else "max"
            
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval="1d")
            
            if hist.empty:
                print(f"No daily data found for {ticker}, trying shorter period...")
                hist = stock.history(period="1mo", interval="1d")
            
            if hist.empty:
                return pd.DataFrame()
            
            # Rename columns to match expected format
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
        print(f"Fetching weekly data for {ticker}...")
        try:
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y", interval="1wk")
            
            if hist.empty:
                return pd.DataFrame()
            
            # Rename columns to match expected format
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
        print(f"Fetching monthly data for {ticker}...")
        try:
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y", interval="1mo")
            
            if hist.empty:
                return pd.DataFrame()
            
            # Rename columns to match expected format
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
    
    def get_intraday(self, ticker, interval='60min', outputsize='compact'):
        """Get intraday time series data"""
        print(f"Fetching intraday data for {ticker}...")
        try:
            # Map interval to yfinance format
            yf_interval = interval.replace('min', 'm').replace('hour', 'h')
            
            # Determine period based on outputsize and interval
            period = "1d" if outputsize == 'compact' else "5d"
            
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=yf_interval)
            
            if hist.empty:
                print(f"No intraday data found, trying longer period...")
                hist = stock.history(period="5d", interval=yf_interval)
            
            if hist.empty:
                return pd.DataFrame()
            
            # Rename columns to match expected format
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
    
    def get_company_news(self, ticker):
        """Get news about the company (used by main.py)"""
        return self.get_news(ticker)
    
    def get_news(self, ticker):
        """Get news about the ticker"""
        print(f"Fetching news for {ticker}...")
        try:
            if self.use_newsapi:
                # Use NewsAPI for better news coverage
                # Get articles related to the ticker symbol
                articles = self.newsapi.get_everything(
                    q=ticker,
                    language='en',
                    sort_by='relevancy',
                    page_size=10
                )
                
                # Get articles related to the company name (may get more relevant results)
                stock = yf.Ticker(ticker)
                company_name = stock.info.get('shortName', '')
                
                # If we have a company name, search for that too
                if company_name and company_name.lower() != ticker.lower():
                    company_articles = self.newsapi.get_everything(
                        q=company_name,
                        language='en',
                        sort_by='relevancy',
                        page_size=10
                    )
                    
                    # Combine articles
                    all_articles = {
                        'articles': articles.get('articles', []) + company_articles.get('articles', [])
                    }
                else:
                    all_articles = articles
                
                # Format articles to match the expected format
                news_articles = []
                for article in all_articles.get('articles', []):
                    news_articles.append({
                        'title': article.get('title', 'No title'),
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
                
                # Format news to match the expected format
                news_articles = []
                for article in news:
                    news_articles.append({
                        'title': article.get('title', 'No title'),
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
        """Get news by topics instead of tickers"""
        print(f"Fetching news for topics: {topics_list}...")
        try:
            if self.use_newsapi:
                # Join topics into a query string
                query = " OR ".join(topics_list)
                
                # Use NewsAPI for better news coverage
                articles = self.newsapi.get_everything(
                    q=query,
                    language='en',
                    sort_by='relevancy',
                    page_size=10
                )
                
                # Format articles to match the expected format
                news_articles = []
                for article in articles.get('articles', []):
                    news_articles.append({
                        'title': article.get('title', 'No title'),
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
                # For Yahoo Finance, we'll just use general market news instead
                print("Warning: NewsAPI key not available, returning general financial news")
                stock = yf.Ticker("^GSPC")  # S&P 500 as a proxy for market news
                news = stock.news
                
                # Format news to match the expected format
                news_articles = []
                for article in news:
                    news_articles.append({
                        'title': article.get('title', 'No title'),
                        'url': article.get('link', ''),
                        'time_published': datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat(),
                        'summary': article.get('summary', ''),
                        'source': article.get('publisher', ''),
                        'category': 'market',
                        'source': 'topic_search',
                        'relevance_score': '0.5',
                        'ticker_sentiment': []
                    })
                
                return news_articles
        except Exception as e:
            print(f"Error in get_news_by_topics: {e}")
            return []
