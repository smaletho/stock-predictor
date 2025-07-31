"""
Technical Analysis Module

This module provides functions for performing technical analysis on stock data,
including calculating moving averages, RSI, MACD, and generating trading signals.
"""

import numpy as np
import pandas as pd


class TechnicalAnalysis:
    """Technical analysis tools for stock data"""
    
    @staticmethod
    def add_moving_averages(df, periods=(5, 10, 20, 50, 100, 200)):
        """
        Add simple moving averages (SMA) to the dataframe
        
        Args:
            df: DataFrame with 'close' or 'adjusted close' column
            periods: Tuple of periods to calculate SMAs for
        
        Returns:
            DataFrame with added SMA columns
        """
        price_col = 'adjusted close' if 'adjusted close' in df.columns else 'close'
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate SMAs
        for period in periods:
            result_df[f'SMA_{period}'] = result_df[price_col].rolling(window=period).mean()
            
        return result_df
    
    @staticmethod
    def add_exponential_moving_averages(df, periods=(5, 10, 20, 50, 100, 200)):
        """
        Add exponential moving averages (EMA) to the dataframe
        
        Args:
            df: DataFrame with 'close' or 'adjusted close' column
            periods: Tuple of periods to calculate EMAs for
        
        Returns:
            DataFrame with added EMA columns
        """
        price_col = 'adjusted close' if 'adjusted close' in df.columns else 'close'
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate EMAs
        for period in periods:
            result_df[f'EMA_{period}'] = result_df[price_col].ewm(span=period, adjust=False).mean()
            
        return result_df
    
    @staticmethod
    def add_rsi(df, period=14):
        """
        Add Relative Strength Index (RSI) to the dataframe
        
        Args:
            df: DataFrame with 'close' or 'adjusted close' column
            period: Period for RSI calculation
        
        Returns:
            DataFrame with added RSI column
        """
        price_col = 'adjusted close' if 'adjusted close' in df.columns else 'close'
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate price changes
        delta = result_df[price_col].diff()
        
        # Calculate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        result_df['RSI'] = rsi
        
        return result_df
    
    @staticmethod
    def add_macd(df, fast_period=12, slow_period=26, signal_period=9):
        """
        Add Moving Average Convergence Divergence (MACD) to the dataframe
        
        Args:
            df: DataFrame with 'close' or 'adjusted close' column
            fast_period: Period for fast EMA
            slow_period: Period for slow EMA
            signal_period: Period for signal line
        
        Returns:
            DataFrame with added MACD columns
        """
        price_col = 'adjusted close' if 'adjusted close' in df.columns else 'close'
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate MACD components
        fast_ema = result_df[price_col].ewm(span=fast_period, adjust=False).mean()
        slow_ema = result_df[price_col].ewm(span=slow_period, adjust=False).mean()
        
        # MACD line
        result_df['MACD'] = fast_ema - slow_ema
        
        # Signal line
        result_df['MACD_Signal'] = result_df['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Histogram
        result_df['MACD_Histogram'] = result_df['MACD'] - result_df['MACD_Signal']
        
        return result_df
    
    @staticmethod
    def add_bollinger_bands(df, period=20, std_dev=2):
        """
        Add Bollinger Bands to the dataframe
        
        Args:
            df: DataFrame with 'close' or 'adjusted close' column
            period: Period for moving average
            std_dev: Number of standard deviations for bands
        
        Returns:
            DataFrame with added Bollinger Bands columns
        """
        price_col = 'adjusted close' if 'adjusted close' in df.columns else 'close'
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate middle band (SMA)
        result_df['BB_Middle'] = result_df[price_col].rolling(window=period).mean()
        
        # Calculate standard deviation
        result_df['BB_StdDev'] = result_df[price_col].rolling(window=period).std()
        
        # Calculate upper and lower bands
        result_df['BB_Upper'] = result_df['BB_Middle'] + (result_df['BB_StdDev'] * std_dev)
        result_df['BB_Lower'] = result_df['BB_Middle'] - (result_df['BB_StdDev'] * std_dev)
        
        # Calculate %B (where price is relative to the bands)
        result_df['BB_B'] = (result_df[price_col] - result_df['BB_Lower']) / (result_df['BB_Upper'] - result_df['BB_Lower'])
        
        return result_df
    
    @staticmethod
    def add_volume_indicators(df):
        """
        Add volume-based indicators to the dataframe
        
        Args:
            df: DataFrame with 'volume' column
        
        Returns:
            DataFrame with added volume indicators
        """
        if 'volume' not in df.columns:
            return df
            
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Volume moving averages
        result_df['Volume_SMA_5'] = result_df['volume'].rolling(window=5).mean()
        result_df['Volume_SMA_20'] = result_df['volume'].rolling(window=20).mean()
        
        # Volume relative to its moving average
        result_df['Volume_Ratio'] = result_df['volume'] / result_df['Volume_SMA_20']
        
        # On-Balance Volume (OBV)
        price_col = 'adjusted close' if 'adjusted close' in df.columns else 'close'
        price_change = result_df[price_col].diff()
        
        obv = [0]
        for i in range(1, len(result_df)):
            if price_change.iloc[i] > 0:
                obv.append(obv[-1] + result_df['volume'].iloc[i])
            elif price_change.iloc[i] < 0:
                obv.append(obv[-1] - result_df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
                
        result_df['OBV'] = obv
        
        return result_df
    
    @staticmethod
    def generate_signals(df):
        """
        Generate trading signals based on technical indicators
        
        Args:
            df: DataFrame with technical indicators
        
        Returns:
            DataFrame with added signal columns
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Moving Average Crossover Signals
        if 'SMA_20' in result_df.columns and 'SMA_50' in result_df.columns:
            # Bullish crossover (short-term MA crosses above long-term MA)
            result_df['MA_Crossover'] = np.where(
                (result_df['SMA_20'].shift(1) <= result_df['SMA_50'].shift(1)) & 
                (result_df['SMA_20'] > result_df['SMA_50']),
                'Bullish',
                np.where(
                    (result_df['SMA_20'].shift(1) >= result_df['SMA_50'].shift(1)) & 
                    (result_df['SMA_20'] < result_df['SMA_50']),
                    'Bearish',
                    'Neutral'
                )
            )
        
        # RSI Signals
        if 'RSI' in result_df.columns:
            result_df['RSI_Signal'] = np.where(
                result_df['RSI'] < 30, 'Oversold',
                np.where(
                    result_df['RSI'] > 70, 'Overbought',
                    'Neutral'
                )
            )
        
        # MACD Signals
        if 'MACD' in result_df.columns and 'MACD_Signal' in result_df.columns:
            result_df['MACD_Crossover'] = np.where(
                (result_df['MACD'].shift(1) <= result_df['MACD_Signal'].shift(1)) & 
                (result_df['MACD'] > result_df['MACD_Signal']),
                'Bullish',
                np.where(
                    (result_df['MACD'].shift(1) >= result_df['MACD_Signal'].shift(1)) & 
                    (result_df['MACD'] < result_df['MACD_Signal']),
                    'Bearish',
                    'Neutral'
                )
            )
        
        # Bollinger Band Signals
        if 'BB_B' in result_df.columns:
            result_df['BB_Signal'] = np.where(
                result_df['BB_B'] < 0, 'Strong Buy',
                np.where(
                    result_df['BB_B'] < 0.2, 'Buy',
                    np.where(
                        result_df['BB_B'] > 1, 'Strong Sell',
                        np.where(
                            result_df['BB_B'] > 0.8, 'Sell',
                            'Hold'
                        )
                    )
                )
            )
        
        # Volume Signals
        if 'Volume_Ratio' in result_df.columns:
            result_df['Volume_Signal'] = np.where(
                result_df['Volume_Ratio'] > 1.5, 'High Volume',
                np.where(
                    result_df['Volume_Ratio'] < 0.5, 'Low Volume',
                    'Normal Volume'
                )
            )
        
        # Overall Trend
        price_col = 'adjusted close' if 'adjusted close' in df.columns else 'close'
        if 'SMA_20' in result_df.columns and 'SMA_50' in result_df.columns and 'SMA_200' in result_df.columns:
            # Long-term trend (price above/below 200-day MA)
            long_term = np.where(
                result_df[price_col] > result_df['SMA_200'], 'Bullish',
                np.where(
                    result_df[price_col] < result_df['SMA_200'], 'Bearish',
                    'Neutral'
                )
            )
            
            # Medium-term trend (price above/below 50-day MA)
            medium_term = np.where(
                result_df[price_col] > result_df['SMA_50'], 'Bullish',
                np.where(
                    result_df[price_col] < result_df['SMA_50'], 'Bearish',
                    'Neutral'
                )
            )
            
            # Short-term trend (price above/below 20-day MA)
            short_term = np.where(
                result_df[price_col] > result_df['SMA_20'], 'Bullish',
                np.where(
                    result_df[price_col] < result_df['SMA_20'], 'Bearish',
                    'Neutral'
                )
            )
            
            result_df['Long_Term_Trend'] = long_term
            result_df['Medium_Term_Trend'] = medium_term
            result_df['Short_Term_Trend'] = short_term
        
        return result_df
    
    @staticmethod
    def calculate_price_targets(df, current_price=None):
        """
        Calculate potential price targets based on technical analysis
        
        Args:
            df: DataFrame with technical indicators
            current_price: Current price of the stock (if None, use the last price in df)
        
        Returns:
            Dictionary with price targets
        """
        price_col = 'adjusted close' if 'adjusted close' in df.columns else 'close'
        
        if current_price is None:
            current_price = df[price_col].iloc[-1]
        
        # Get recent price data
        recent_data = df.iloc[-100:] if len(df) > 100 else df
        
        # Calculate targets
        targets = {
            'current_price': current_price,
            'supports': [],
            'resistances': [],
            'short_term_target': None,
            'medium_term_target': None,
            'long_term_target': None
        }
        
        # Find support and resistance levels using recent lows and highs
        lows = recent_data[price_col].nsmallest(5)
        highs = recent_data[price_col].nlargest(5)
        
        # Add supports (below current price)
        for low in lows:
            if low < current_price:
                targets['supports'].append(round(low, 2))
        
        # Add resistances (above current price)
        for high in highs:
            if high > current_price:
                targets['resistances'].append(round(high, 2))
        
        # Sort and limit to 3 levels
        targets['supports'] = sorted(targets['supports'], reverse=True)[:3]
        targets['resistances'] = sorted(targets['resistances'])[:3]
        
        # Calculate Fibonacci retracement levels for targets
        if len(recent_data) > 0:
            highest_high = recent_data[price_col].max()
            lowest_low = recent_data[price_col].min()
            price_range = highest_high - lowest_low
            
            # Fibonacci levels for uptrend: 23.6%, 38.2%, 61.8%, 100%, 161.8%
            if recent_data[price_col].iloc[-1] > recent_data[price_col].iloc[0]:  # In uptrend
                targets['short_term_target'] = round(current_price + (price_range * 0.236), 2)
                targets['medium_term_target'] = round(current_price + (price_range * 0.618), 2)
                targets['long_term_target'] = round(current_price + (price_range * 1.618), 2)
            # Fibonacci levels for downtrend
            else:
                targets['short_term_target'] = round(current_price - (price_range * 0.236), 2)
                targets['medium_term_target'] = round(current_price - (price_range * 0.618), 2)
                targets['long_term_target'] = round(current_price - (price_range * 1.0), 2)
        
        return targets
    
    @staticmethod
    def analyze_all(df):
        """
        Perform all technical analysis on the given dataframe
        
        Args:
            df: DataFrame with price and volume data
        
        Returns:
            DataFrame with all technical indicators and signals
        """
        df = TechnicalAnalysis.add_moving_averages(df)
        df = TechnicalAnalysis.add_exponential_moving_averages(df)
        df = TechnicalAnalysis.add_rsi(df)
        df = TechnicalAnalysis.add_macd(df)
        df = TechnicalAnalysis.add_bollinger_bands(df)
        df = TechnicalAnalysis.add_volume_indicators(df)
        df = TechnicalAnalysis.generate_signals(df)
        
        return df
