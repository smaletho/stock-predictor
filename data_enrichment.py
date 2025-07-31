"""
Data Enrichment Module

This module enhances stock data with additional information like earnings data,
sector performance, options metrics, and relevant economic indicators.
"""

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import finviz
import httpx
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataEnrichment:
    """Class for enriching stock data with additional information"""
    
    def __init__(self, cache_dir: str = "data"):
        """Initialize the data enrichment module"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Alpha Vantage API key for economic data
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        # Initialize clients if API keys are available
        if self.alpha_vantage_key:
            self.alpha_vantage = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        else:
            self.alpha_vantage = None
    
    def get_earnings_data(self, ticker: str) -> Dict:
        """
        Get earnings data for a stock
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with earnings data
        """
        print(f"Fetching earnings data for {ticker}...")
        
        # Create cache directory for this ticker if it doesn't exist
        ticker_cache_dir = os.path.join(self.cache_dir, ticker)
        os.makedirs(ticker_cache_dir, exist_ok=True)
        
        # Cache file path
        cache_file = os.path.join(ticker_cache_dir, "earnings_data.json")
        
        # Check if cache exists and is less than 24 hours old
        if os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age < timedelta(hours=24):
                # Use cached data if available and recent
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error loading cached earnings data: {e}")
        
        # Fetch earnings data using yfinance
        try:
            stock = yf.Ticker(ticker)
            
            # Get earnings dates - handle both DataFrame and dict returns from yfinance
            earnings_data = {}
            
            try:
                # Get calendar data safely
                calendar = stock.calendar
                
                # Handle the case where calendar might be a DataFrame
                if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                    # Extract next earnings date if available
                    try:
                        earnings_data["next_earnings_date"] = calendar.loc["Earnings Date"].iloc[0].strftime('%Y-%m-%d')
                    except (KeyError, AttributeError, IndexError):
                        earnings_data["next_earnings_date"] = "Unknown"
                # Handle the case where calendar might be a dictionary
                elif isinstance(calendar, dict) and calendar:
                    try:
                        # Try to extract the earnings date from the dictionary
                        if "Earnings Date" in calendar:
                            date_val = calendar["Earnings Date"]
                            if hasattr(date_val, 'strftime'):
                                earnings_data["next_earnings_date"] = date_val.strftime('%Y-%m-%d')
                            else:
                                earnings_data["next_earnings_date"] = str(date_val)
                        else:
                            earnings_data["next_earnings_date"] = "Unknown"
                    except Exception:
                        earnings_data["next_earnings_date"] = "Unknown"
                else:
                    earnings_data["next_earnings_date"] = "Unknown"
            except Exception as e:
                print(f"Error getting earnings calendar: {e}")
                earnings_data["next_earnings_date"] = "Unknown"
            
            # Get historical earnings
            try:
                earnings_history = stock.earnings_history
                
                # Check if earnings_history is a DataFrame and not empty
                if isinstance(earnings_history, pd.DataFrame) and not earnings_history.empty:
                    # Convert to list of dicts for JSON serialization
                    earnings_data["history"] = []
                    
                    for idx, row in earnings_history.iterrows():
                        try:
                            earnings_data["history"].append({
                                "date": idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                                "eps_estimate": float(row.get("EPS Estimate", 0)),
                                "eps_actual": float(row.get("Reported EPS", 0)),
                                "surprise": float(row.get("Surprise(%)", 0))
                            })
                        except (ValueError, AttributeError, KeyError) as e:
                            print(f"Error processing earnings history row: {e}")
                else:
                    # Handle case when earnings_history is not a DataFrame or is empty
                    earnings_data["history"] = []
                    print("No earnings history available")
            except Exception as e:
                print(f"Error retrieving earnings history: {e}")
                earnings_data["history"] = []
            
            # Calculate earnings surprise streak (consecutive beats or misses)
            if "history" in earnings_data and earnings_data["history"]:
                earnings_data["beat_streak"] = 0
                earnings_data["miss_streak"] = 0
                
                for entry in earnings_data["history"]:
                    if entry["surprise"] > 0:
                        earnings_data["beat_streak"] += 1
                        earnings_data["miss_streak"] = 0
                    elif entry["surprise"] < 0:
                        earnings_data["beat_streak"] = 0
                        earnings_data["miss_streak"] += 1
                    else:
                        earnings_data["beat_streak"] = 0
                        earnings_data["miss_streak"] = 0
            
            # Save to cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(earnings_data, f)
            except Exception as e:
                print(f"Error saving earnings data to cache: {e}")
            
            return earnings_data
        
        except Exception as e:
            print(f"Error fetching earnings data: {e}")
            return {"error": str(e)}
    
    def get_sector_data(self, ticker: str) -> Dict:
        """
        Get sector and industry performance data for context
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with sector and industry data
        """
        print(f"Fetching sector data for {ticker}...")
        
        # Create cache directory for this ticker if it doesn't exist
        ticker_cache_dir = os.path.join(self.cache_dir, ticker)
        os.makedirs(ticker_cache_dir, exist_ok=True)
        
        # Cache file path
        cache_file = os.path.join(ticker_cache_dir, "sector_data.json")
        
        # Check if cache exists and is less than 12 hours old
        if os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age < timedelta(hours=12):
                # Use cached data if available and recent
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error loading cached sector data: {e}")
        
        # Fetch sector data
        try:
            # Get stock information directly from Yahoo Finance
            stock = yf.Ticker(ticker)
            stock_info = stock.info
            
            sector_data = {}
            
            # Extract sector and industry information
            sector_data["sector"] = stock_info.get("sector", "Unknown")
            sector_data["industry"] = stock_info.get("industry", "Unknown")
            
            # Get relevant metrics directly from Yahoo Finance to avoid dependency issues
            sector_data["market_cap"] = stock_info.get("marketCap", "Unknown")
            sector_data["pe_ratio"] = stock_info.get("trailingPE", "Unknown")
            sector_data["forward_pe"] = stock_info.get("forwardPE", "Unknown")
            sector_data["peg_ratio"] = stock_info.get("pegRatio", "Unknown")
            sector_data["price_to_book"] = stock_info.get("priceToBook", "Unknown")
            sector_data["dividend_yield"] = stock_info.get("dividendYield", "Unknown")
            sector_data["eps_growth"] = stock_info.get("earningsQuarterlyGrowth", "Unknown")
            sector_data["profit_margin"] = stock_info.get("profitMargins", "Unknown")
            
            # Calculate sector performance metrics directly
            try:
                # Get the sector ETF performance instead of using FinViz
                sector = sector_data.get("sector", "")
                sector_etfs = {
                    "Technology": "XLK",
                    "Information Technology": "XLK",
                    "Financials": "XLF",
                    "Healthcare": "XLV",
                    "Consumer Discretionary": "XLY",
                    "Consumer Staples": "XLP",
                    "Energy": "XLE",
                    "Industrials": "XLI",
                    "Materials": "XLB",
                    "Utilities": "XLU",
                    "Real Estate": "XLRE",
                    "Communication Services": "XLC"
                }
                
                matching_etf = None
                for sector_name, etf in sector_etfs.items():
                    if sector and sector_name.lower() in sector.lower():
                        matching_etf = etf
                        break
                
                if matching_etf:
                    etf_data = yf.Ticker(matching_etf).history(period="1mo")
                    if not etf_data.empty:
                        sector_data["sector_etf"] = matching_etf
                        
                        # Calculate performance metrics
                        last_price = etf_data['Close'].iloc[-1]
                        day_ago = etf_data['Close'].iloc[-2] if len(etf_data) > 1 else last_price
                        week_ago = etf_data['Close'].iloc[-5] if len(etf_data) > 5 else last_price
                        month_ago = etf_data['Close'].iloc[-20] if len(etf_data) > 20 else last_price
                        
                        # Calculate percentage changes
                        sector_data["sector_perf_today"] = f"{((last_price / day_ago) - 1) * 100:.2f}%" if day_ago != 0 else "0%"
                        sector_data["sector_perf_week"] = f"{((last_price / week_ago) - 1) * 100:.2f}%" if week_ago != 0 else "0%"
                        sector_data["sector_perf_month"] = f"{((last_price / month_ago) - 1) * 100:.2f}%" if month_ago != 0 else "0%"
                else:
                    # If no matching ETF found, use default values
                    sector_data["sector_perf_today"] = "N/A"
                    sector_data["sector_perf_week"] = "N/A"
                    sector_data["sector_perf_month"] = "N/A"
            except Exception as e:
                print(f"Error calculating sector performance: {e}")
                # Set defaults if data isn't available
                sector_data["sector_perf_today"] = "N/A"
                sector_data["sector_perf_week"] = "N/A"
                sector_data["sector_perf_month"] = "N/A"
            
            # Get sector ETF performance
            sector_etfs = {
                "Technology": "XLK",
                "Financials": "XLF",
                "Healthcare": "XLV",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Energy": "XLE",
                "Industrials": "XLI",
                "Materials": "XLB",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
                "Communication Services": "XLC"
            }
            
            # Find the closest matching sector ETF
            sector = sector_data.get("sector", "")
            matching_etf = None
            
            for sector_name, etf_symbol in sector_etfs.items():
                if sector_name.lower() in sector.lower():
                    matching_etf = etf_symbol
                    break
            
            if matching_etf:
                # Get ETF performance
                try:
                    etf = yf.Ticker(matching_etf)
                    etf_history = etf.history(period="1y")
                    
                    if not etf_history.empty:
                        current_price = etf_history['Close'].iloc[-1]
                        week_ago_price = etf_history['Close'].iloc[-5] if len(etf_history) >= 5 else None
                        month_ago_price = etf_history['Close'].iloc[-22] if len(etf_history) >= 22 else None
                        year_ago_price = etf_history['Close'].iloc[-252] if len(etf_history) >= 252 else None
                        
                        sector_data["sector_etf"] = matching_etf
                        sector_data["sector_etf_current"] = float(current_price)
                        
                        if week_ago_price:
                            sector_data["sector_etf_week_perf"] = float(((current_price / week_ago_price) - 1) * 100)
                        
                        if month_ago_price:
                            sector_data["sector_etf_month_perf"] = float(((current_price / month_ago_price) - 1) * 100)
                        
                        if year_ago_price:
                            sector_data["sector_etf_year_perf"] = float(((current_price / year_ago_price) - 1) * 100)
                except Exception as e:
                    print(f"Error fetching sector ETF data: {e}")
            
            # Get peer comparison
            try:
                if "sector" in sector_data and "industry" in sector_data:
                    # Use finviz to get peers (stocks in the same industry)
                    # This is a basic implementation and could be enhanced
                    peers = []
                    
                    # Add basic peer comparison from yahoo finance
                    if "recommendationKey" in stock_info:
                        sector_data["recommendation"] = stock_info["recommendationKey"]
                    
                    # Add any provided peer symbols from Yahoo Finance
                    if hasattr(stock, 'recommendations') and stock.recommendations is not None:
                        if 'Similar' in stock.recommendations:
                            peers = stock.recommendations['Similar'].tolist()
                    
                    # Limit to 5 peers
                    sector_data["peers"] = peers[:5] if peers else []
            except Exception as e:
                print(f"Error fetching peer data: {e}")
            
            # Save to cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(sector_data, f)
            except Exception as e:
                print(f"Error saving sector data to cache: {e}")
            
            return sector_data
        
        except Exception as e:
            print(f"Error fetching sector data: {e}")
            return {"error": str(e)}
    
    def get_options_data(self, ticker: str) -> Dict:
        """
        Get options data for a stock
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with options data
        """
        print(f"Fetching options data for {ticker}...")
        
        # Create cache directory for this ticker if it doesn't exist
        ticker_cache_dir = os.path.join(self.cache_dir, ticker)
        os.makedirs(ticker_cache_dir, exist_ok=True)
        
        # Cache file path
        cache_file = os.path.join(ticker_cache_dir, "options_data.json")
        
        # Check if cache exists and is less than 6 hours old
        if os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age < timedelta(hours=6):
                # Use cached data if available and recent
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error loading cached options data: {e}")
        
        # Fetch options data
        try:
            # Get stock information
            stock = yf.Ticker(ticker)
            
            options_data = {}
            
            # Get options expiration dates
            expirations = stock.options
            
            if not expirations or len(expirations) == 0:
                return {"error": "No options data available"}
            
            # Get nearest expiration date
            nearest_expiration = expirations[0]
            
            # Get options chain for nearest expiration
            options_chain = stock.option_chain(nearest_expiration)
            
            # Process calls
            calls_df = options_chain.calls
            if not calls_df.empty:
                # Extract key metrics
                options_data["call_volume"] = int(calls_df['volume'].sum())
                options_data["call_open_interest"] = int(calls_df['openInterest'].sum())
                
                # Calculate put/call ratio
                options_data["expiration_date"] = nearest_expiration
            
            # Process puts
            puts_df = options_chain.puts
            if not puts_df.empty:
                # Extract key metrics
                options_data["put_volume"] = int(puts_df['volume'].sum())
                options_data["put_open_interest"] = int(puts_df['openInterest'].sum())
            
            # Calculate put/call ratios
            if "call_volume" in options_data and "put_volume" in options_data and options_data["call_volume"] > 0:
                options_data["put_call_ratio_volume"] = options_data["put_volume"] / options_data["call_volume"]
            
            if "call_open_interest" in options_data and "put_open_interest" in options_data and options_data["call_open_interest"] > 0:
                options_data["put_call_ratio_oi"] = options_data["put_open_interest"] / options_data["call_open_interest"]
            
            # Implied volatility
            if not calls_df.empty and 'impliedVolatility' in calls_df.columns:
                options_data["avg_implied_volatility_calls"] = float(calls_df['impliedVolatility'].mean())
            
            if not puts_df.empty and 'impliedVolatility' in puts_df.columns:
                options_data["avg_implied_volatility_puts"] = float(puts_df['impliedVolatility'].mean())
            
            # Calculate overall implied volatility
            if "avg_implied_volatility_calls" in options_data and "avg_implied_volatility_puts" in options_data:
                options_data["avg_implied_volatility"] = (options_data["avg_implied_volatility_calls"] + 
                                                         options_data["avg_implied_volatility_puts"]) / 2
            
            # Market sentiment based on put/call ratio
            if "put_call_ratio_volume" in options_data:
                if options_data["put_call_ratio_volume"] > 1.2:
                    options_data["options_sentiment"] = "Bearish"
                elif options_data["put_call_ratio_volume"] < 0.8:
                    options_data["options_sentiment"] = "Bullish"
                else:
                    options_data["options_sentiment"] = "Neutral"
            
            # Save to cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(options_data, f)
            except Exception as e:
                print(f"Error saving options data to cache: {e}")
            
            return options_data
        
        except Exception as e:
            print(f"Error fetching options data: {e}")
            return {"error": str(e)}
    
    def get_economic_indicators(self) -> Dict:
        """
        Get relevant economic indicators
        
        Returns:
            Dictionary with economic indicator data
        """
        print("Fetching economic indicators...")
        
        # Cache file path
        cache_file = os.path.join(self.cache_dir, "economic_indicators.json")
        
        # Check if cache exists and is less than 24 hours old
        if os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age < timedelta(hours=24):
                # Use cached data if available and recent
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error loading cached economic indicators: {e}")
        
        # Fetch economic indicators
        indicators = {}
        
        try:
            # If Alpha Vantage API key is available, fetch data
            if self.alpha_vantage:
                # Get real GDP
                try:
                    gdp_data, _ = self.alpha_vantage.get_real_gdp()
                    if not gdp_data.empty:
                        latest_gdp = gdp_data.iloc[-1]['value']
                        prev_gdp = gdp_data.iloc[-2]['value']
                        gdp_growth = (latest_gdp - prev_gdp) / prev_gdp * 100
                        
                        indicators["real_gdp"] = float(latest_gdp)
                        indicators["gdp_growth"] = float(gdp_growth)
                except Exception as e:
                    print(f"Error fetching GDP data: {e}")
                
                # Get inflation rate (CPI)
                try:
                    cpi_data, _ = self.alpha_vantage.get_cpi()
                    if not cpi_data.empty:
                        latest_cpi = cpi_data.iloc[-1]['value']
                        prev_cpi = cpi_data.iloc[-13]['value']  # 12 months ago
                        inflation_rate = (latest_cpi - prev_cpi) / prev_cpi * 100
                        
                        indicators["cpi"] = float(latest_cpi)
                        indicators["inflation_rate"] = float(inflation_rate)
                except Exception as e:
                    print(f"Error fetching CPI data: {e}")
                
                # Get unemployment rate
                try:
                    unemployment_data, _ = self.alpha_vantage.get_unemployment()
                    if not unemployment_data.empty:
                        latest_unemployment = unemployment_data.iloc[-1]['value']
                        indicators["unemployment_rate"] = float(latest_unemployment)
                except Exception as e:
                    print(f"Error fetching unemployment data: {e}")
            
            # Fetch Treasury yield data - use more reliable ticker symbols and add fallbacks
            try:
                # Try multiple ticker symbols for 10-Year Treasury Yield
                treasury_10y_tickers = ["^TNX", "TNX", "TYX", "US10Y=X", "^TYX"]
                treasury_10y_data = None
                
                # Try each ticker until we get data
                for ticker in treasury_10y_tickers:
                    try:
                        treasury_10y = yf.Ticker(ticker)
                        data = treasury_10y.history(period="1mo")
                        if not data.empty:
                            treasury_10y_data = data
                            print(f"Found 10-year Treasury data using {ticker}")
                            break
                    except:
                        continue
                
                # Set 10-year yield if we found data
                if treasury_10y_data is not None and not treasury_10y_data.empty:
                    indicators["treasury_10y_yield"] = float(treasury_10y_data['Close'].iloc[-1])
                else:
                    # Use a reasonable default if we can't get data
                    indicators["treasury_10y_yield"] = 4.2  # Set a reasonable default based on current rates
                    print("Using default value for 10-year Treasury yield")
                
                # Try multiple ticker symbols for 2-Year Treasury Yield
                treasury_2y_tickers = ["^UST2Y", "US2Y=X", "UST2Y", "^TUZ"]
                treasury_2y_data = None
                
                # Try each ticker until we get data
                for ticker in treasury_2y_tickers:
                    try:
                        treasury_2y = yf.Ticker(ticker)
                        data = treasury_2y.history(period="1mo")
                        if not data.empty:
                            treasury_2y_data = data
                            print(f"Found 2-year Treasury data using {ticker}")
                            break
                    except:
                        continue
                
                # Set 2-year yield if we found data
                if treasury_2y_data is not None and not treasury_2y_data.empty:
                    indicators["treasury_2y_yield"] = float(treasury_2y_data['Close'].iloc[-1])
                else:
                    # Use a reasonable default if we can't get data
                    indicators["treasury_2y_yield"] = 4.3  # Set a reasonable default based on current rates
                    print("Using default value for 2-year Treasury yield")
                    
                # Calculate yield curve (10Y - 2Y)
                indicators["yield_curve"] = indicators["treasury_10y_yield"] - indicators["treasury_2y_yield"]
                
                # Interpret yield curve
                if indicators["yield_curve"] < 0:
                    indicators["yield_curve_status"] = "Inverted (Recession Warning)"
                elif indicators["yield_curve"] < 0.5:
                    indicators["yield_curve_status"] = "Flat (Caution)"
                else:
                    indicators["yield_curve_status"] = "Normal (Healthy Economy)"
            except Exception as e:
                print(f"Error fetching Treasury yield data: {e}")
                # Set defaults in case of error
                indicators["treasury_10y_yield"] = 4.2
                indicators["treasury_2y_yield"] = 4.3
                indicators["yield_curve"] = -0.1
                indicators["yield_curve_status"] = "Default (Data Unavailable)"
            
            # Get market indices
            try:
                # S&P 500
                sp500 = yf.Ticker("^GSPC")
                sp500_data = sp500.history(period="1mo")
                
                if not sp500_data.empty:
                    current_sp500 = sp500_data['Close'].iloc[-1]
                    week_ago_sp500 = sp500_data['Close'].iloc[-5] if len(sp500_data) >= 5 else None
                    month_ago_sp500 = sp500_data['Close'].iloc[-20] if len(sp500_data) >= 20 else None
                    
                    indicators["sp500"] = float(current_sp500)
                    
                    if week_ago_sp500:
                        indicators["sp500_week_change"] = float(((current_sp500 / week_ago_sp500) - 1) * 100)
                    
                    if month_ago_sp500:
                        indicators["sp500_month_change"] = float(((current_sp500 / month_ago_sp500) - 1) * 100)
                
                # VIX (Volatility Index)
                vix = yf.Ticker("^VIX")
                vix_data = vix.history(period="1mo")
                
                if not vix_data.empty:
                    indicators["vix"] = float(vix_data['Close'].iloc[-1])
                    
                    # Interpret VIX
                    if indicators["vix"] < 15:
                        indicators["market_volatility"] = "Low"
                    elif indicators["vix"] < 25:
                        indicators["market_volatility"] = "Moderate"
                    else:
                        indicators["market_volatility"] = "High"
            except Exception as e:
                print(f"Error fetching market indices: {e}")
            
            # Save to cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(indicators, f)
            except Exception as e:
                print(f"Error saving economic indicators to cache: {e}")
            
            return indicators
        
        except Exception as e:
            print(f"Error fetching economic indicators: {e}")
            return {"error": str(e)}
