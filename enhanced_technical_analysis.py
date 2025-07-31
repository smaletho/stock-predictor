"""
Enhanced Technical Analysis Module

This module provides advanced technical analysis indicators and pattern recognition
for stock price data.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional

class EnhancedTechnicalAnalysis:
    """Enhanced technical analysis for stock price data"""
    
    def __init__(self):
        """Initialize the enhanced technical analysis module"""
        pass
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform enhanced technical analysis on price data
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with analysis results
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure we have the right columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain {col} column")
        
        # Calculate various indicators
        results = {}
        
        # Basic indicators from the original module
        results.update(self._calculate_basic_indicators(df))
        
        # Additional indicators
        results.update(self._calculate_vwap(df))
        results.update(self._calculate_volatility_metrics(df))
        results.update(self._calculate_ichimoku(df))
        results.update(self._calculate_fibonacci_levels(df))
        results.update(self._identify_patterns(df))
        
        return results
    
    def _calculate_basic_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate basic technical indicators"""
        results = {}
        
        # RSI (Relative Strength Index)
        df['rsi'] = ta.rsi(df['close'], length=14)
        results['RSI'] = df['rsi'].iloc[-1]
        results['RSI_Signal'] = 'Oversold' if results['RSI'] < 30 else 'Overbought' if results['RSI'] > 70 else 'Neutral'
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.macd(df['close'])
        df = pd.concat([df, macd], axis=1)
        results['MACD'] = df['MACD_12_26_9'].iloc[-1]
        results['MACD_Signal'] = df['MACDs_12_26_9'].iloc[-1]
        results['MACD_Histogram'] = df['MACDh_12_26_9'].iloc[-1]
        results['MACD_Trend'] = 'Bullish' if results['MACD'] > results['MACD_Signal'] else 'Bearish'
        
        # Moving Averages - handle case when not enough data available
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        
        # Define current price first
        current_price = df['close'].iloc[-1]
        
        # Get SMA values if available, otherwise use defaults to avoid None comparison issues
        results['SMA_20'] = df['sma_20'].iloc[-1] if not pd.isna(df['sma_20'].iloc[-1]) else current_price
        results['SMA_50'] = df['sma_50'].iloc[-1] if not pd.isna(df['sma_50'].iloc[-1]) else current_price
        results['SMA_200'] = df['sma_200'].iloc[-1] if not pd.isna(df['sma_200'].iloc[-1]) else current_price
        
        # Store availability flags for the indicators
        results['SMA_20_Available'] = not pd.isna(df['sma_20'].iloc[-1])
        results['SMA_50_Available'] = not pd.isna(df['sma_50'].iloc[-1])
        results['SMA_200_Available'] = not pd.isna(df['sma_200'].iloc[-1])
        
        # Determine trends based on MA crossovers - but only if the data is available
        
        # Price vs SMA comparisons
        if results['SMA_20_Available']:
            results['Price_vs_SMA20'] = 'Above' if current_price > results['SMA_20'] else 'Below'
        else:
            results['Price_vs_SMA20'] = 'Unknown (insufficient data)'
            
        if results['SMA_50_Available']:
            results['Price_vs_SMA50'] = 'Above' if current_price > results['SMA_50'] else 'Below'
        else:
            results['Price_vs_SMA50'] = 'Unknown (insufficient data)'
            
        if results['SMA_200_Available']:
            results['Price_vs_SMA200'] = 'Above' if current_price > results['SMA_200'] else 'Below'
        else:
            results['Price_vs_SMA200'] = 'Unknown (insufficient data)'
        
        # Golden/Death Cross - but only if we have both SMAs available
        if results['SMA_50_Available'] and results['SMA_200_Available']:
            # Need at least 2 data points for each SMA to check for crossover
            if len(df) > 2 and not pd.isna(df['sma_50'].iloc[-2]) and not pd.isna(df['sma_200'].iloc[-2]):
                results['Golden_Cross'] = results['SMA_50'] > results['SMA_200'] and df['sma_50'].iloc[-2] <= df['sma_200'].iloc[-2]
                results['Death_Cross'] = results['SMA_50'] < results['SMA_200'] and df['sma_50'].iloc[-2] >= df['sma_200'].iloc[-2]
            else:
                results['Golden_Cross'] = False
                results['Death_Cross'] = False
        else:
            results['Golden_Cross'] = False
            results['Death_Cross'] = False
            
        # Short/Medium/Long Term Trends
        if results['SMA_20_Available']:
            results['Short_Term_Trend'] = 'Bullish' if current_price > results['SMA_20'] else 'Bearish'
        else:
            results['Short_Term_Trend'] = 'Neutral (insufficient data)'
            
        if results['SMA_50_Available']:
            results['Medium_Term_Trend'] = 'Bullish' if current_price > results['SMA_50'] else 'Bearish'
        else:
            results['Medium_Term_Trend'] = 'Neutral (insufficient data)'
            
        if results['SMA_200_Available']:
            results['Long_Term_Trend'] = 'Bullish' if current_price > results['SMA_200'] else 'Bearish'
        else:
            results['Long_Term_Trend'] = 'Neutral (insufficient data)'
        
        return results
    
    def _calculate_vwap(self, df: pd.DataFrame) -> Dict:
        """Calculate VWAP (Volume Weighted Average Price)"""
        results = {}
        
        try:
            # Calculate VWAP manually to avoid timezone warning
            # VWAP = ∑(Price * Volume) / ∑(Volume)
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['price_volume'] = df['typical_price'] * df['volume']
            
            # Calculate cumulative values
            df['cumulative_price_volume'] = df['price_volume'].cumsum()
            df['cumulative_volume'] = df['volume'].cumsum()
            
            # Calculate VWAP
            df['vwap'] = df['cumulative_price_volume'] / df['cumulative_volume']
            
            # Get the most recent VWAP value
            if not pd.isna(df['vwap'].iloc[-1]):
                results['VWAP'] = df['vwap'].iloc[-1]
            else:
                results['VWAP'] = df['close'].iloc[-1]  # Default to current price if NA
            
            # VWAP signal
            current_price = df['close'].iloc[-1]
            results['VWAP_Signal'] = 'Bullish' if current_price > results['VWAP'] else 'Bearish'
        except Exception as e:
            # If anything goes wrong, use defaults
            current_price = df['close'].iloc[-1]
            results['VWAP'] = current_price
            results['VWAP_Signal'] = 'Neutral'
            results['VWAP_Error'] = str(e)
        
        return results
    
    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility metrics like ATR and Bollinger Bands"""
        results = {}
        
        # ATR (Average True Range)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        results['ATR'] = df['atr'].iloc[-1]
        results['ATR_Percent'] = (results['ATR'] / df['close'].iloc[-1]) * 100
        
        # Bollinger Bands
        bollinger = ta.bbands(df['close'], length=20, std=2)
        df = pd.concat([df, bollinger], axis=1)
        
        results['BB_Upper'] = df['BBU_20_2.0'].iloc[-1]
        results['BB_Middle'] = df['BBM_20_2.0'].iloc[-1]
        results['BB_Lower'] = df['BBL_20_2.0'].iloc[-1]
        
        # Bollinger Band Width
        results['BB_Width'] = (results['BB_Upper'] - results['BB_Lower']) / results['BB_Middle']
        
        # BB signals
        current_price = df['close'].iloc[-1]
        if current_price > results['BB_Upper']:
            results['BB_Signal'] = 'Overbought'
        elif current_price < results['BB_Lower']:
            results['BB_Signal'] = 'Oversold'
        else:
            results['BB_Signal'] = 'Neutral'
        
        # Volatility state based on BB width
        bb_width_history = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
        mean_bb_width = bb_width_history.mean()
        
        if results['BB_Width'] < mean_bb_width * 0.8:
            results['Volatility_State'] = 'Low (Potential Breakout Coming)'
        elif results['BB_Width'] > mean_bb_width * 1.2:
            results['Volatility_State'] = 'High (Potential Reversal Coming)'
        else:
            results['Volatility_State'] = 'Normal'
        
        return results
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """Calculate Ichimoku Cloud indicators"""
        results = {}
        
        try:
            # Calculate Ichimoku Cloud - we'll calculate this manually
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            period9_high = df['high'].rolling(window=9).max()
            period9_low = df['low'].rolling(window=9).min()
            tenkan_sen = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            period26_high = df['high'].rolling(window=26).max()
            period26_low = df['low'].rolling(window=26).min()
            kijun_sen = (period26_high + period26_low) / 2
            
            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            senkou_span_a = (tenkan_sen + kijun_sen) / 2
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            period52_high = df['high'].rolling(window=52).max()
            period52_low = df['low'].rolling(window=52).min()
            senkou_span_b = (period52_high + period52_low) / 2
            
            # Chikou Span (Lagging Span): Close plotted 26 periods in the past
            chikou_span = df['close'].shift(-26)
            
            # Add the calculated values to the dataframe
            df['tenkan_sen'] = tenkan_sen
            df['kijun_sen'] = kijun_sen
            df['senkou_span_a'] = senkou_span_a
            df['senkou_span_b'] = senkou_span_b
            df['chikou_span'] = chikou_span
        
            # Extract relevant Ichimoku components for the most recent data point
            if not pd.isna(df['tenkan_sen'].iloc[-1]):
                results['Tenkan_Sen'] = df['tenkan_sen'].iloc[-1]  # Conversion Line
            else:
                results['Tenkan_Sen'] = df['close'].iloc[-1]  # Default to price if NA
                
            if not pd.isna(df['kijun_sen'].iloc[-1]):
                results['Kijun_Sen'] = df['kijun_sen'].iloc[-1]  # Base Line
            else:
                results['Kijun_Sen'] = df['close'].iloc[-1]  # Default to price if NA
                
            if not pd.isna(df['senkou_span_a'].iloc[-1]):
                results['Senkou_Span_A'] = df['senkou_span_a'].iloc[-1]  # Leading Span A
            else:
                results['Senkou_Span_A'] = df['close'].iloc[-1]  # Default to price if NA
                
            if not pd.isna(df['senkou_span_b'].iloc[-1]):
                results['Senkou_Span_B'] = df['senkou_span_b'].iloc[-1]  # Leading Span B
            else:
                results['Senkou_Span_B'] = df['close'].iloc[-1]  # Default to price if NA
                
            if not pd.isna(df['chikou_span'].iloc[-1]):
                results['Chikou_Span'] = df['chikou_span'].iloc[-1]  # Lagging Span
            else:
                results['Chikou_Span'] = df['close'].iloc[-1]  # Default to price if NA
        except Exception as e:
            # If anything goes wrong, use defaults that won't break the analysis
            current_price = df['close'].iloc[-1]
            results['Tenkan_Sen'] = current_price
            results['Kijun_Sen'] = current_price
            results['Senkou_Span_A'] = current_price
            results['Senkou_Span_B'] = current_price
            results['Chikou_Span'] = current_price
            results['Ichimoku_Error'] = str(e)
        
        # Determine cloud color (bullish or bearish)
        results['Cloud_Color'] = 'Bullish' if results['Senkou_Span_A'] > results['Senkou_Span_B'] else 'Bearish'
        
        # Price position relative to the cloud
        current_price = df['close'].iloc[-1]
        if current_price > max(results['Senkou_Span_A'], results['Senkou_Span_B']):
            results['Price_vs_Cloud'] = 'Above (Bullish)'
        elif current_price < min(results['Senkou_Span_A'], results['Senkou_Span_B']):
            results['Price_vs_Cloud'] = 'Below (Bearish)'
        else:
            results['Price_vs_Cloud'] = 'Inside (Neutral)'
        
        # TK Cross (Tenkan/Kijun Cross)
        try:
            # Check if we have at least 2 data points to detect a crossover
            if len(df) > 2:
                # Check if required data is available
                if not pd.isna(df['tenkan_sen'].iloc[-2]) and not pd.isna(df['kijun_sen'].iloc[-2]):
                    # Bullish TK Cross: Tenkan crosses above Kijun
                    if (results['Tenkan_Sen'] > results['Kijun_Sen'] and 
                        df['tenkan_sen'].iloc[-2] <= df['kijun_sen'].iloc[-2]):
                        results['TK_Cross'] = 'Bullish'
                    # Bearish TK Cross: Tenkan crosses below Kijun
                    elif (results['Tenkan_Sen'] < results['Kijun_Sen'] and 
                          df['tenkan_sen'].iloc[-2] >= df['kijun_sen'].iloc[-2]):
                        results['TK_Cross'] = 'Bearish'
                    else:
                        results['TK_Cross'] = 'None'
                else:
                    results['TK_Cross'] = 'Unknown (insufficient data)'
            else:
                results['TK_Cross'] = 'Unknown (insufficient data)'
        except Exception as e:
            results['TK_Cross'] = 'None'
            results['TK_Cross_Error'] = str(e)
        
        # Overall Ichimoku signal
        if (results['Price_vs_Cloud'] == 'Above (Bullish)' and 
            results['Cloud_Color'] == 'Bullish' and 
            results['TK_Cross'] == 'Bullish'):
            results['Ichimoku_Signal'] = 'Strong Buy'
        elif (results['Price_vs_Cloud'] == 'Below (Bearish)' and 
              results['Cloud_Color'] == 'Bearish' and 
              results['TK_Cross'] == 'Bearish'):
            results['Ichimoku_Signal'] = 'Strong Sell'
        elif results['Price_vs_Cloud'] == 'Above (Bullish)':
            results['Ichimoku_Signal'] = 'Buy'
        elif results['Price_vs_Cloud'] == 'Below (Bearish)':
            results['Ichimoku_Signal'] = 'Sell'
        else:
            results['Ichimoku_Signal'] = 'Neutral'
        
        return results
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate Fibonacci retracement and extension levels"""
        results = {}
        
        # Find significant high and low points in the last 90 days (or all data if less)
        period = min(90, len(df))
        recent_df = df.iloc[-period:]
        
        high_point = recent_df['high'].max()
        high_idx = recent_df['high'].idxmax()
        
        low_point = recent_df['low'].min()
        low_idx = recent_df['low'].idxmin()
        
        # Determine if the trend is up or down based on which came first
        if high_idx > low_idx:
            # Uptrend: use low to high for retracement
            trend = 'Uptrend'
            fib_diff = high_point - low_point
            
            # Retracement levels (down from high)
            results['Fib_0'] = high_point
            results['Fib_23.6'] = high_point - 0.236 * fib_diff
            results['Fib_38.2'] = high_point - 0.382 * fib_diff
            results['Fib_50.0'] = high_point - 0.5 * fib_diff
            results['Fib_61.8'] = high_point - 0.618 * fib_diff
            results['Fib_78.6'] = high_point - 0.786 * fib_diff
            results['Fib_100'] = low_point
            
            # Extension levels (above high)
            results['Fib_Extension_127'] = high_point + 0.127 * fib_diff
            results['Fib_Extension_162'] = high_point + 0.618 * fib_diff
            results['Fib_Extension_200'] = high_point + 1.0 * fib_diff
        else:
            # Downtrend: use high to low for retracement
            trend = 'Downtrend'
            fib_diff = high_point - low_point
            
            # Retracement levels (up from low)
            results['Fib_0'] = low_point
            results['Fib_23.6'] = low_point + 0.236 * fib_diff
            results['Fib_38.2'] = low_point + 0.382 * fib_diff
            results['Fib_50.0'] = low_point + 0.5 * fib_diff
            results['Fib_61.8'] = low_point + 0.618 * fib_diff
            results['Fib_78.6'] = low_point + 0.786 * fib_diff
            results['Fib_100'] = high_point
            
            # Extension levels (below low)
            results['Fib_Extension_127'] = low_point - 0.127 * fib_diff
            results['Fib_Extension_162'] = low_point - 0.618 * fib_diff
            results['Fib_Extension_200'] = low_point - 1.0 * fib_diff
        
        results['Fibonacci_Trend'] = trend
        
        # Find nearest Fibonacci level to current price
        current_price = df['close'].iloc[-1]
        
        fib_levels = {k: v for k, v in results.items() if k.startswith('Fib_')}
        nearest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price))
        
        results['Nearest_Fib_Level'] = nearest_fib[0]
        results['Nearest_Fib_Value'] = nearest_fib[1]
        
        return results
    
    def _identify_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify common chart patterns"""
        results = {}
        patterns = []
        
        # Need at least 30 days of data for pattern recognition
        if len(df) < 30:
            results['Patterns'] = 'Insufficient data for pattern recognition'
            return results
        
        # Double Top pattern
        double_top = self._check_double_top(df)
        if double_top:
            patterns.append('Double Top (Bearish)')
        
        # Double Bottom pattern
        double_bottom = self._check_double_bottom(df)
        if double_bottom:
            patterns.append('Double Bottom (Bullish)')
        
        # Head and Shoulders pattern
        head_shoulders = self._check_head_and_shoulders(df)
        if head_shoulders:
            patterns.append('Head and Shoulders (Bearish)')
        
        # Inverse Head and Shoulders pattern
        inv_head_shoulders = self._check_inverse_head_and_shoulders(df)
        if inv_head_shoulders:
            patterns.append('Inverse Head and Shoulders (Bullish)')
        
        # Check for bullish/bearish engulfing patterns
        engulfing = self._check_engulfing_patterns(df)
        if engulfing:
            patterns.append(engulfing)
        
        # Check for trend channels
        channel = self._check_trend_channel(df)
        if channel:
            patterns.append(channel)
        
        results['Patterns'] = patterns if patterns else ['No significant patterns detected']
        results['Pattern_Signal'] = self._get_pattern_signal(patterns)
        
        return results
    
    def _check_double_top(self, df: pd.DataFrame) -> bool:
        """Check for a double top pattern"""
        try:
            # Get local maxima
            # Order=5 means it compares with 5 points on each side
            max_idx = argrelextrema(df['high'].values, np.greater, order=5)[0]
            
            if len(max_idx) < 2:
                return False
            
            # Check the last two peaks
            peak1_idx, peak2_idx = max_idx[-2], max_idx[-1]
            peak1, peak2 = df['high'].iloc[peak1_idx], df['high'].iloc[peak2_idx]
            
            # Peaks should be within 3% of each other
            if abs(peak1 - peak2) / peak1 > 0.03:
                return False
            
            # Should be at least 10 days apart
            if peak2_idx - peak1_idx < 10:
                return False
            
            # There should be a trough in between
            # Use integer indexing to find the minimum in the slice
            trough_slice = df['low'].iloc[peak1_idx:peak2_idx]
            trough_loc = trough_slice.values.argmin()  # Position within the slice
            trough = trough_slice.iloc[trough_loc]     # Value at that position
            
            # Trough should be at least 3% below peaks
            if (peak1 - trough) / peak1 < 0.03 or (peak2 - trough) / peak2 < 0.03:
                return False
            
            # The second peak should be recent (within last 20 days)
            if len(df) - 1 - peak2_idx > 20:
                return False
            
            return True
        except Exception as e:
            # If any error occurs, log it and return False
            print(f"Error in double top detection: {e}")
            return False
    
    def _check_double_bottom(self, df: pd.DataFrame) -> bool:
        """Check for a double bottom pattern"""
        try:
            # Get local minima
            min_idx = argrelextrema(df['low'].values, np.less, order=5)[0]
            
            if len(min_idx) < 2:
                return False
            
            # Check the last two troughs
            trough1_idx, trough2_idx = min_idx[-2], min_idx[-1]
            trough1, trough2 = df['low'].iloc[trough1_idx], df['low'].iloc[trough2_idx]
            
            # Troughs should be within 3% of each other
            if abs(trough1 - trough2) / trough1 > 0.03:
                return False
            
            # Should be at least 10 days apart
            if trough2_idx - trough1_idx < 10:
                return False
            
            # There should be a peak in between
            # Use integer indexing to find the maximum in the slice
            peak_slice = df['high'].iloc[trough1_idx:trough2_idx]
            peak_loc = peak_slice.values.argmax()  # Position within the slice
            peak = peak_slice.iloc[peak_loc]       # Value at that position
            
            # Peak should be at least 3% above troughs
            if (peak - trough1) / trough1 < 0.03 or (peak - trough2) / trough2 < 0.03:
                return False
            
            # The second trough should be recent (within last 20 days)
            if len(df) - 1 - trough2_idx > 20:
                return False
            
            return True
        except Exception as e:
            # If any error occurs, log it and return False
            print(f"Error in double bottom detection: {e}")
            return False
    
    def _check_head_and_shoulders(self, df: pd.DataFrame) -> bool:
        """Check for a head and shoulders pattern"""
        try:
            # Get local maxima
            max_idx = argrelextrema(df['high'].values, np.greater, order=5)[0]
            
            if len(max_idx) < 3:
                return False
            
            # Need the last three peaks
            shoulder1_idx, head_idx, shoulder2_idx = max_idx[-3], max_idx[-2], max_idx[-1]
            shoulder1 = df['high'].iloc[shoulder1_idx]
            head = df['high'].iloc[head_idx]
            shoulder2 = df['high'].iloc[shoulder2_idx]
            
            # Head should be higher than shoulders
            if head <= shoulder1 or head <= shoulder2:
                return False
            
            # Shoulders should be within 5% of each other's height
            if abs(shoulder1 - shoulder2) / shoulder1 > 0.05:
                return False
            
            # There should be a trough between each peak
            # Use integer indexing to find the minimum in the slices
            trough1_slice = df['low'].iloc[shoulder1_idx:head_idx]
            trough1_loc = trough1_slice.values.argmin()
            trough1_val = trough1_slice.iloc[trough1_loc]
            
            trough2_slice = df['low'].iloc[head_idx:shoulder2_idx]
            trough2_loc = trough2_slice.values.argmin()
            trough2_val = trough2_slice.iloc[trough2_loc]
            
            # Actual indices in the original DataFrame
            trough1_idx = shoulder1_idx + trough1_loc
            trough2_idx = head_idx + trough2_loc
            
            # Check neckline (connect the troughs)
            if trough2_idx > trough1_idx:  # Ensure proper order
                neckline_slope = (trough2_val - trough1_val) / (trough2_idx - trough1_idx)
                # Additional check - neckline should be relatively flat
                if abs(neckline_slope) > 0.01:  # Allow slight slope
                    return False
            
            # The second shoulder should be recent (within last 20 days)
            if len(df) - 1 - shoulder2_idx > 20:
                return False
            
            return True
        except Exception as e:
            # If any error occurs, log it and return False
            print(f"Error in head and shoulders detection: {e}")
            return False
    
    def _check_inverse_head_and_shoulders(self, df: pd.DataFrame) -> bool:
        """Check for an inverse head and shoulders pattern"""
        try:
            # Get local minima
            min_idx = argrelextrema(df['low'].values, np.less, order=5)[0]
            
            if len(min_idx) < 3:
                return False
            
            # Need the last three troughs
            shoulder1_idx, head_idx, shoulder2_idx = min_idx[-3], min_idx[-2], min_idx[-1]
            shoulder1 = df['low'].iloc[shoulder1_idx]
            head = df['low'].iloc[head_idx]
            shoulder2 = df['low'].iloc[shoulder2_idx]
            
            # Head should be lower than shoulders
            if head >= shoulder1 or head >= shoulder2:
                return False
            
            # Shoulders should be within 5% of each other's height
            if abs(shoulder1 - shoulder2) / shoulder1 > 0.05:
                return False
            
            # There should be a peak between each trough
            # Use integer indexing to find the maximum in the slices
            peak1_slice = df['high'].iloc[shoulder1_idx:head_idx]
            peak1_loc = peak1_slice.values.argmax()
            peak1_val = peak1_slice.iloc[peak1_loc]
            
            peak2_slice = df['high'].iloc[head_idx:shoulder2_idx]
            peak2_loc = peak2_slice.values.argmax()
            peak2_val = peak2_slice.iloc[peak2_loc]
            
            # Actual indices in the original DataFrame
            peak1_idx = shoulder1_idx + peak1_loc
            peak2_idx = head_idx + peak2_loc
            
            # Check neckline (connect the peaks)
            if peak2_idx > peak1_idx:  # Ensure proper order
                neckline_slope = (peak2_val - peak1_val) / (peak2_idx - peak1_idx)
                # Additional check - neckline should be relatively flat
                if abs(neckline_slope) > 0.01:  # Allow slight slope
                    return False
            
            # The second shoulder should be recent (within last 20 days)
            if len(df) - 1 - shoulder2_idx > 20:
                return False
            
            return True
        except Exception as e:
            # If any error occurs, log it and return False
            print(f"Error in inverse head and shoulders detection: {e}")
            return False
    
    def _check_engulfing_patterns(self, df: pd.DataFrame) -> Optional[str]:
        """Check for bullish or bearish engulfing patterns"""
        # Need at least 2 days of data
        if len(df) < 2:
            return None
        
        # Check the last two days
        current_day = df.iloc[-1]
        previous_day = df.iloc[-2]
        
        # Bullish engulfing
        if (previous_day['close'] < previous_day['open'] and  # Previous day was bearish
            current_day['open'] < previous_day['close'] and  # Open below previous close
            current_day['close'] > previous_day['open']):  # Close above previous open
            return 'Bullish Engulfing (Bullish)'
        
        # Bearish engulfing
        if (previous_day['close'] > previous_day['open'] and  # Previous day was bullish
            current_day['open'] > previous_day['close'] and  # Open above previous close
            current_day['close'] < previous_day['open']):  # Close below previous open
            return 'Bearish Engulfing (Bearish)'
        
        return None
    
    def _check_trend_channel(self, df: pd.DataFrame) -> Optional[str]:
        """Check for trend channels"""
        # Need at least 20 days of data for trend channels
        if len(df) < 20:
            return None
        
        # Use only the last 60 days
        period = min(60, len(df))
        recent_df = df.iloc[-period:]
        
        # Calculate linear regression
        y = recent_df['close'].values
        x = np.arange(len(y))
        
        slope, intercept = np.polyfit(x, y, 1)
        
        # Determine trend direction
        if slope > 0:
            return 'Upward Channel (Bullish)'
        elif slope < 0:
            return 'Downward Channel (Bearish)'
        else:
            return 'Sideways Channel (Neutral)'
    
    def _get_pattern_signal(self, patterns: List[str]) -> str:
        """Determine the overall signal from detected patterns"""
        if not patterns or patterns == ['No significant patterns detected']:
            return 'Neutral'
        
        bullish_count = sum(1 for pattern in patterns if 'Bullish' in pattern)
        bearish_count = sum(1 for pattern in patterns if 'Bearish' in pattern)
        
        if bullish_count > bearish_count:
            return 'Bullish'
        elif bearish_count > bullish_count:
            return 'Bearish'
        else:
            return 'Neutral'
