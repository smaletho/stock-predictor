"""
Stock Predictor

A Python application for predicting and analyzing stocks using Alpha Vantage API
and LLM-based sentiment analysis.

Usage:
    python main.py --ticker AAPL
"""

import os
import sys
import json
import argparse
import asyncio
import pandas as pd
import httpx
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

from simplified_yfinance_client import SimplifiedYahooFinanceClient
from technical_analysis import TechnicalAnalysis
from enhanced_technical_analysis import EnhancedTechnicalAnalysis
from sentiment_analysis import SentimentAnalyzer
from report_generator import ReportGenerator
from data_enrichment import DataEnrichment

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Stock Predictor - Technical and Sentiment Analysis")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports")
    parser.add_argument("--api-mode", type=str, choices=["full", "cached"], default=None, 
                      help="API usage mode: 'full' uses all endpoints (more API calls), 'cached' uses only local data if available (no API calls)")
    parser.add_argument("--skip-news", action="store_true", help="Skip news fetching to save API calls")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for caching data")
    return parser.parse_args()

def ensure_directory(directory):
    """Ensure a directory exists"""
    os.makedirs(directory, exist_ok=True)
    return directory

def load_cached_data(file_path):
    """Load data from a cache file if it exists"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading cached data: {e}")
    return None

def save_cached_data(data, filepath):
    """Save data to a cached file handling pandas Timestamp objects"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Handle DataFrame conversion to JSON-serializable format
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to dict with string indices
            data_dict = {}
            for idx, row in data.iterrows():
                # Convert Timestamp index to string
                key = idx.strftime('%Y-%m-%d %H:%M:%S') if hasattr(idx, 'strftime') else str(idx)
                data_dict[key] = row.to_dict()
            data = data_dict
        
        # Handle dict with Timestamp keys
        elif isinstance(data, dict):
            # Convert any Timestamp keys to strings
            data_dict = {}
            for key, value in data.items():
                # Convert Timestamp key to string
                if hasattr(key, 'strftime'):
                    key = key.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    key = str(key)
                data_dict[key] = value
            data = data_dict
            
        with open(filepath, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving cached data: {e}")
        print("Continuing without caching...")
        # Continue execution without caching

async def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    ticker = args.ticker.upper()
    output_dir = args.output_dir
    api_mode = args.api_mode
    skip_news = args.skip_news
    data_dir = ensure_directory(args.data_dir)
    
    # Create cache directory for this ticker
    ticker_cache_dir = ensure_directory(os.path.join(data_dir, ticker))
    
    # Define standard mode (formerly minimal) as the default
    if api_mode is None:
        api_mode = "standard"
    
    print(f"Analyzing stock: {ticker} (API Mode: {api_mode})")
    
    # Create a progress bar for the overall analysis
    overall_steps = 8 if not skip_news else 6
    progress_bar = tqdm(total=overall_steps, desc="Analysis Progress", position=0)
    
    # Initialize API client
    try:
        yahoo_finance = SimplifiedYahooFinanceClient()
        progress_bar.update(1)  # Step 1: Initialize API client
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Initialize analysis modules
    technical_analyzer = TechnicalAnalysis()
    enhanced_technical_analyzer = EnhancedTechnicalAnalysis()
    sentiment_analyzer = SentimentAnalyzer()
    data_enricher = DataEnrichment(data_dir)
    report_generator = ReportGenerator(output_dir=output_dir)
    progress_bar.update(1)  # Step 2: Initialize analysis modules
    
    # Fetch company data - can be cached as it doesn't change often
    company_cache_file = os.path.join(ticker_cache_dir, "company_overview.json")
    company_data = None
    
    if api_mode == "cached":
        company_data = load_cached_data(company_cache_file)
        if company_data:
            print(f"Using cached company data for {ticker}")
    
    if not company_data:
        print(f"Fetching company data for {ticker}...")
        try:
            company_data = yahoo_finance.get_company_overview(ticker)
            # Cache the data for future use
            save_cached_data(company_data, company_cache_file)
        except Exception as e:
            print(f"Error fetching company data: {e}")
            # Try to use cached data as fallback
            company_data = load_cached_data(company_cache_file)
            if not company_data:
                company_data = {"Symbol": ticker}
    
    progress_bar.update(1)  # Step 3: Company data fetched
    
    # Fetch price data with optimizations based on API mode
    print("Fetching price data...")
    
    # Cache file paths
    daily_cache_file = os.path.join(ticker_cache_dir, "daily.json")
    weekly_cache_file = os.path.join(ticker_cache_dir, "weekly.json")
    monthly_cache_file = os.path.join(ticker_cache_dir, "monthly.json")
    
    # Initialize DataFrames
    df_daily = pd.DataFrame()
    df_weekly = pd.DataFrame()
    df_monthly = pd.DataFrame()
    
    try:
        # Try to load cached data first if in cached mode
        if api_mode == "cached":
            daily_data = load_cached_data(daily_cache_file)
            weekly_data = load_cached_data(weekly_cache_file)
            monthly_data = load_cached_data(monthly_cache_file)
            
            if daily_data:
                print("Using cached daily data")
                df_daily = pd.DataFrame.from_dict(daily_data, orient='index')
                df_daily.index = pd.to_datetime(df_daily.index)
                df_daily = df_daily.sort_index()
                # Convert columns to numeric
                for col in df_daily.columns:
                    df_daily[col] = pd.to_numeric(df_daily[col])
            
            if weekly_data:
                print("Using cached weekly data")
                df_weekly = pd.DataFrame.from_dict(weekly_data, orient='index')
                df_weekly.index = pd.to_datetime(df_weekly.index)
                df_weekly = df_weekly.sort_index()
                # Convert columns to numeric
                for col in df_weekly.columns:
                    df_weekly[col] = pd.to_numeric(df_weekly[col])
            
            if monthly_data:
                print("Using cached monthly data")
                df_monthly = pd.DataFrame.from_dict(monthly_data, orient='index')
                df_monthly.index = pd.to_datetime(df_monthly.index)
                df_monthly = df_monthly.sort_index()
                # Convert columns to numeric
                for col in df_monthly.columns:
                    df_monthly[col] = pd.to_numeric(df_monthly[col])
        
        # If we don't have cached data or not in cached mode, fetch from API
        if df_daily.empty:
            # Daily data (compact = last 100 data points, which is free)
            df_daily = yahoo_finance.get_daily_adjusted(ticker, outputsize="compact")
            if not df_daily.empty:
                # Cache the data
                save_cached_data(df_daily.to_dict(orient='index'), daily_cache_file)
            elif api_mode != "minimal":
                # Try intraday as fallback only in full mode
                print("Daily data not available, trying intraday data...")
                df_intraday = yahoo_finance.get_intraday(ticker, interval="60min", outputsize="compact")
                if not df_intraday.empty:
                    df_daily = df_intraday
                    # Cache the data
                    save_cached_data(df_daily.to_dict(orient='index'), daily_cache_file)
        
        if df_daily.empty:
            print(f"Error: No price data found for {ticker}")
            sys.exit(1)
        
        # In standard mode (or what was formerly "minimal" mode), derive weekly and monthly from daily data
        if api_mode != "full":
            print("Using derived weekly and monthly data from daily data to save API calls")
            # Convert daily to weekly
            if df_weekly.empty:
                df_weekly = df_daily.resample('W').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                if 'adjusted close' in df_daily.columns:
                    df_weekly['adjusted close'] = df_daily['adjusted close'].resample('W').last()
                # Cache derived weekly data
                save_cached_data(df_weekly.to_dict(orient='index'), weekly_cache_file)
            
            # Convert daily to monthly
            if df_monthly.empty:
                df_monthly = df_daily.resample('ME').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                if 'adjusted close' in df_daily.columns:
                    df_monthly['adjusted close'] = df_daily['adjusted close'].resample('ME').last()
                # Cache derived monthly data
                save_cached_data(df_monthly.to_dict(orient='index'), monthly_cache_file)
        else:
            # In full mode, get actual weekly and monthly data if we don't have it cached
            if df_weekly.empty:
                df_weekly = yahoo_finance.get_weekly_adjusted(ticker)
                if not df_weekly.empty:
                    save_cached_data(df_weekly.to_dict(orient='index'), weekly_cache_file)
            
            if df_monthly.empty:
                df_monthly = yahoo_finance.get_monthly_adjusted(ticker)
                if not df_monthly.empty:
                    save_cached_data(df_monthly.to_dict(orient='index'), monthly_cache_file)
    except Exception as e:
        print(f"Error fetching price data: {e}")
        sys.exit(1)
    
    # Perform technical analysis
    print("Performing technical analysis...")
    # Analyze daily data
    df_daily = TechnicalAnalysis.analyze_all(df_daily)
    
    # Analyze weekly data
    df_weekly = TechnicalAnalysis.analyze_all(df_weekly)
    
    # Analyze monthly data
    df_monthly = TechnicalAnalysis.analyze_all(df_monthly)
    
    # Get technical signals from the most recent day
    technical_signals = {}
    if not df_daily.empty:
        last_day = df_daily.iloc[-1]
        for col in last_day.index:
            if any(col.endswith(suffix) for suffix in ['_Signal', '_Trend', '_Crossover']):
                technical_signals[col] = last_day[col]
    
    # Calculate price targets
    price_targets = TechnicalAnalysis.calculate_price_targets(df_daily)
    
    # Run enhanced technical analysis
    print("Performing enhanced technical analysis...")
    enhanced_signals = enhanced_technical_analyzer.analyze(df_daily)
    
    # Merge enhanced analysis with regular analysis
    technical_signals.update(enhanced_signals)
    
    # Prepare Fibonacci levels for price targets
    if 'Fib_23.6' in enhanced_signals:
        price_targets['fibonacci_levels'] = {
            '23.6%': enhanced_signals.get('Fib_23.6'),
            '38.2%': enhanced_signals.get('Fib_38.2'),
            '50.0%': enhanced_signals.get('Fib_50.0'),
            '61.8%': enhanced_signals.get('Fib_61.8'),
            '78.6%': enhanced_signals.get('Fib_78.6')
        }
    
    # Add patterns to technical signals if any were detected
    if 'Patterns' in enhanced_signals and enhanced_signals['Patterns'] != ['No significant patterns detected']:
        technical_signals['Detected_Patterns'] = enhanced_signals['Patterns']
    
    progress_bar.update(1)  # Step 4: Technical analysis completed
    
    # Fetch news data only if not skipped
    news_cache_file = os.path.join(ticker_cache_dir, "news.json")
    news_articles = []
    
    if not skip_news:
        print("Fetching news data...")
        
        # Try to use cached news if in cached mode
        if api_mode == "cached":
            cached_news = load_cached_data(news_cache_file)
            if cached_news:
                print("Using cached news data")
                news_articles = cached_news
        
        if not news_articles:
            try:
                print(f"Fetching news for {ticker}...")
                news_articles = yahoo_finance.get_company_news(ticker)
                
                # Cache the news data
                save_cached_data(news_articles, news_cache_file)
            except Exception as e:
                print(f"Error fetching news: {e}")
                # Try to use cached data as fallback
                news_articles = load_cached_data(news_cache_file) or []
    
    progress_bar.update(1)  # Step 5: News data fetched
    
    # Fetch additional data enrichment features
    print("Fetching enrichment data...")
    
    # Get earnings data
    earnings_data = data_enricher.get_earnings_data(ticker)
    
    # Get sector and industry context
    sector_data = data_enricher.get_sector_data(ticker)
    
    # Get options data
    options_data = data_enricher.get_options_data(ticker)
    
    # Get economic indicators
    economic_data = data_enricher.get_economic_indicators()
    
    # Add the enrichment data to the company data
    company_data['earnings'] = earnings_data
    company_data['sector'] = sector_data
    company_data['options'] = options_data
    company_data['economic'] = economic_data
    
    progress_bar.update(1)  # Step 6: Enrichment data fetched
    
    # Handle skipped news case
    if skip_news:
        print("Skipping news fetching to save API calls")
        # Try to use cached news
        cached_news = load_cached_data(news_cache_file)
        if cached_news:
            print("Using cached news data instead")
            news_articles = cached_news
    
    # Perform sentiment analysis
    print("Analyzing news sentiment...")
    sentiment_results = {}
    news_summary = ""
    
    if news_articles:
        try:
            # Check if Ollama is running by making a simple request
            print("Checking Ollama availability...")
            ollama_available = False
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{sentiment_analyzer.ollama_host.rstrip('/')}/api/tags",
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        ollama_available = True
                        print("Ollama is available, proceeding with sentiment analysis")
                    else:
                        print(f"Ollama returned status code {response.status_code}")
            except Exception as e:
                print(f"Ollama not available: {e}")
                
            if ollama_available:
                sentiment_results = await sentiment_analyzer.analyze_news_articles(news_articles)
                news_summary = await sentiment_analyzer.summarize_news(news_articles)
            else:
                # Fall back to simple rule-based analysis
                print("Using rule-based sentiment analysis fallback")
                sentiment_results = {
                    "overall_sentiment": "neutral",
                    "confidence": 50,
                    "articles_analyzed": len(news_articles),
                    "bullish_count": 0,
                    "bearish_count": 0,
                    "neutral_count": len(news_articles),
                    "key_positive_points": ["Automated analysis due to Ollama unavailability"],
                    "key_negative_points": [],
                    "key_neutral_points": ["To enable LLM-based sentiment analysis, make sure Ollama is running with llama3 model"]
                }
                news_summary = "Sentiment analysis unavailable. Please ensure Ollama is running with the llama3 model."
        except Exception as e:
            print(f"Error in sentiment analysis process: {e}")
            sentiment_results = {
                "overall_sentiment": "neutral",
                "confidence": 50,
                "articles_analyzed": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "key_positive_points": [],
                "key_negative_points": [],
                "key_neutral_points": [f"Error analyzing sentiment: {e}"]
            }
            news_summary = "No news summary available due to an error."
    else:
        print("No news articles available for sentiment analysis")
        sentiment_results = {
            "overall_sentiment": "neutral",
            "confidence": 0,
            "articles_analyzed": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "key_positive_points": [],
            "key_negative_points": [],
            "key_neutral_points": ["No news articles were available for analysis"]
        }
        news_summary = "No news summary available due to lack of news articles."
    
    progress_bar.update(1)  # Step 7: Sentiment analysis completed
    
    # Generate report
    print("Generating analysis report...")
    report = report_generator.generate_report(
        ticker,
        company_data,
        df_daily,
        df_weekly,
        df_monthly,
        technical_signals,
        price_targets,
        sentiment_results,
        news_summary
    )
    
    # Save report
    report_path = report_generator.save_report(report, ticker)
    
    # Complete the progress bar
    progress_bar.update(1)  # Step 8: Report generated
    progress_bar.close()
    
    print(f"\nAnalysis complete!")
    print(f"Report saved to: {report_path}")
    
    # Print summary
    price_col = 'adjusted close' if 'adjusted close' in df_daily.columns else 'close'
    current_price = df_daily[price_col].iloc[-1]
    
    print("\nSummary:")
    print(f"Current Price: ${current_price:.2f}")
    
    # Print price targets
    print("\nPrice Targets:")
    print(f"Short-term Target: ${price_targets['short_term_target']:.2f}")
    print(f"Medium-term Target: ${price_targets['medium_term_target']:.2f}")
    print(f"Long-term Target: ${price_targets['long_term_target']:.2f}")
    
    # Print support and resistance levels
    print("\nSupport Levels:")
    for level in price_targets['supports']:
        print(f"  ${level:.2f}")
        
    print("\nResistance Levels:")
    for level in price_targets['resistances']:
        print(f"  ${level:.2f}")
    
    # Print trend outlook
    print("\nTrend Outlook:")
    short_term = technical_signals.get("Short_Term_Trend", "Neutral")
    medium_term = technical_signals.get("Medium_Term_Trend", "Neutral")
    long_term = technical_signals.get("Long_Term_Trend", "Neutral")
    
    print(f"Short-term (Days): {short_term}")
    print(f"Medium-term (Weeks): {medium_term}")
    print(f"Long-term (Months): {long_term}")
    
    # Print sentiment
    sentiment = sentiment_results.get("overall_sentiment", "neutral").capitalize()
    print(f"\nNews Sentiment: {sentiment} (Confidence: {sentiment_results.get('confidence', 0)}%)")
    
    # Print key factors
    if sentiment_results.get("key_positive_points"):
        print("\nKey Positive Factors:")
        for point in sentiment_results.get("key_positive_points", [])[:3]:
            print(f"  - {point}")
            
    if sentiment_results.get("key_negative_points"):
        print("\nKey Negative Factors:")
        for point in sentiment_results.get("key_negative_points", [])[:3]:
            print(f"  - {point}")
    
    # Final recommendation
    bullish_signals = sum(1 for signal in [short_term, medium_term, long_term, sentiment] 
                         if signal.lower() == "bullish")
    bearish_signals = sum(1 for signal in [short_term, medium_term, long_term, sentiment] 
                         if signal.lower() == "bearish")
    
    if bullish_signals > bearish_signals:
        recommendation = "Bullish"
        action = "Consider buying or holding"
    elif bearish_signals > bullish_signals:
        recommendation = "Bearish"
        action = "Consider selling or avoiding"
    else:
        recommendation = "Neutral"
        action = "Monitor for clearer signals"
    
    print(f"\nOverall Recommendation: {recommendation}")
    print(f"Suggested Action: {action}")

if __name__ == "__main__":
    asyncio.run(main())
