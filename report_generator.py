"""
Report Generator Module

This module handles generating formatted reports from stock analysis data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from datetime import datetime
import os

class ReportGenerator:
    """Class for generating analysis reports"""
    
    def __init__(self, output_dir="reports"):
        """Initialize with output directory"""
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up styling for plots
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
    
    def _format_ticker_data(self, company_data, df_daily, df_weekly, df_monthly):
        """Format basic ticker data for the report"""
        report = []
        
        # Company Overview
        report.append("## Company Overview\n")
        
        overview_data = [
            ["Symbol", company_data.get("Symbol", "N/A")],
            ["Name", company_data.get("Name", "N/A")],
            ["Exchange", company_data.get("Exchange", "N/A")],
            ["Sector", company_data.get("Sector", "N/A")],
            ["Industry", company_data.get("Industry", "N/A")],
            ["Market Cap", company_data.get("MarketCapitalization", "N/A")],
            ["PE Ratio", company_data.get("PERatio", "N/A")],
            ["EPS", company_data.get("EPS", "N/A")],
            ["Dividend Yield", company_data.get("DividendYield", "N/A")],
            ["52-Week High", company_data.get("52WeekHigh", "N/A")],
            ["52-Week Low", company_data.get("52WeekLow", "N/A")]
        ]
        
        report.append(tabulate(overview_data, tablefmt="pipe") + "\n\n")
        
        # Current Price Data
        if not df_daily.empty:
            report.append("## Current Price Data\n")
            
            last_day = df_daily.iloc[-1]
            prev_day = df_daily.iloc[-2] if len(df_daily) > 1 else None
            
            price_col = 'adjusted close' if 'adjusted close' in df_daily.columns else 'close'
            current_price = last_day[price_col]
            
            price_data = [
                ["Current Price", f"${current_price:.2f}"],
                ["Date", last_day.name.strftime('%Y-%m-%d')]
            ]
            
            if prev_day is not None:
                change = current_price - prev_day[price_col]
                pct_change = (change / prev_day[price_col]) * 100
                price_data.append(["Change", f"${change:.2f} ({pct_change:.2f}%)"])
            
            price_data.extend([
                ["Volume", f"{last_day['volume']:,}"],
                ["Open", f"${last_day['open']:.2f}"],
                ["High", f"${last_day['high']:.2f}"],
                ["Low", f"${last_day['low']:.2f}"]
            ])
            
            report.append(tabulate(price_data, tablefmt="pipe") + "\n\n")
        
        return "\n".join(report)
    
    def _format_technical_analysis(self, df_daily, df_weekly, df_monthly, price_targets):
        """Format technical analysis data for the report"""
        report = []
        
        # Price Targets
        report.append("## Price Targets\n")
        
        target_data = [
            ["Current Price", f"${price_targets['current_price']:.2f}"],
            ["Short-term Target", f"${price_targets['short_term_target']:.2f}"],
            ["Medium-term Target", f"${price_targets['medium_term_target']:.2f}"],
            ["Long-term Target", f"${price_targets['long_term_target']:.2f}"]
        ]
        
        report.append(tabulate(target_data, tablefmt="pipe") + "\n\n")
        
        # Support and Resistance Levels
        report.append("### Support and Resistance Levels\n")
        
        support_data = [["Support Levels"]]
        for level in price_targets['supports']:
            support_data.append([f"${level:.2f}"])
        
        resistance_data = [["Resistance Levels"]]
        for level in price_targets['resistances']:
            resistance_data.append([f"${level:.2f}"])
        
        report.append(tabulate(support_data, tablefmt="pipe") + "\n\n")
        report.append(tabulate(resistance_data, tablefmt="pipe") + "\n\n")
        
        # Technical Indicators
        report.append("## Technical Indicators\n")
        
        if not df_daily.empty:
            last_day = df_daily.iloc[-1]
            
            indicators = []
            
            # Moving Averages
            ma_columns = [col for col in last_day.index if col.startswith('SMA_') or col.startswith('EMA_')]
            for col in ma_columns:
                indicators.append([col, f"${last_day[col]:.2f}"])
            
            # RSI
            if 'RSI' in last_day.index:
                indicators.append(["RSI", f"{last_day['RSI']:.2f}"])
            
            # MACD
            if 'MACD' in last_day.index:
                indicators.append(["MACD", f"{last_day['MACD']:.4f}"])
            if 'MACD_Signal' in last_day.index:
                indicators.append(["MACD Signal", f"{last_day['MACD_Signal']:.4f}"])
            if 'MACD_Histogram' in last_day.index:
                indicators.append(["MACD Histogram", f"{last_day['MACD_Histogram']:.4f}"])
            
            # Bollinger Bands
            if 'BB_Upper' in last_day.index:
                indicators.append(["Bollinger Upper", f"${last_day['BB_Upper']:.2f}"])
            if 'BB_Middle' in last_day.index:
                indicators.append(["Bollinger Middle", f"${last_day['BB_Middle']:.2f}"])
            if 'BB_Lower' in last_day.index:
                indicators.append(["Bollinger Lower", f"${last_day['BB_Lower']:.2f}"])
            
            report.append(tabulate(indicators, tablefmt="pipe") + "\n\n")
        
        # Trading Signals
        report.append("## Trading Signals\n")
        
        if not df_daily.empty:
            last_day = df_daily.iloc[-1]
            
            signals = []
            
            # Trend signals
            if 'Short_Term_Trend' in last_day.index:
                signals.append(["Short-term Trend (Daily)", last_day['Short_Term_Trend']])
            
            if not df_weekly.empty and 'Short_Term_Trend' in df_weekly.iloc[-1].index:
                signals.append(["Medium-term Trend (Weekly)", df_weekly.iloc[-1]['Short_Term_Trend']])
            
            if not df_monthly.empty and 'Short_Term_Trend' in df_monthly.iloc[-1].index:
                signals.append(["Long-term Trend (Monthly)", df_monthly.iloc[-1]['Short_Term_Trend']])
            
            # Other signals
            signal_columns = [
                ('MA_Crossover', 'Moving Average Crossover'),
                ('RSI_Signal', 'RSI'),
                ('MACD_Crossover', 'MACD Crossover'),
                ('BB_Signal', 'Bollinger Bands'),
                ('Volume_Signal', 'Volume')
            ]
            
            for col, label in signal_columns:
                if col in last_day.index:
                    signals.append([label, last_day[col]])
            
            report.append(tabulate(signals, tablefmt="pipe") + "\n\n")
        
        return "\n".join(report)
    
    def _format_sentiment_analysis(self, sentiment_results, news_summary):
        """Format sentiment analysis data for the report"""
        report = []
        
        # Overall Sentiment
        report.append("## News Sentiment Analysis\n")
        
        sentiment = sentiment_results.get("overall_sentiment", "neutral").capitalize()
        confidence = sentiment_results.get("confidence", 0)
        articles_analyzed = sentiment_results.get("articles_analyzed", 0)
        
        report.append(f"**Overall Sentiment:** {sentiment} (Confidence: {confidence}%)\n")
        report.append(f"**Articles Analyzed:** {articles_analyzed}\n\n")
        
        # Sentiment breakdown
        bullish = sentiment_results.get("bullish_count", 0)
        bearish = sentiment_results.get("bearish_count", 0)
        neutral = sentiment_results.get("neutral_count", 0)
        
        sentiment_data = [
            ["Sentiment", "Count", "Percentage"],
            ["Bullish", bullish, f"{(bullish/articles_analyzed)*100:.1f}%" if articles_analyzed else "0%"],
            ["Bearish", bearish, f"{(bearish/articles_analyzed)*100:.1f}%" if articles_analyzed else "0%"],
            ["Neutral", neutral, f"{(neutral/articles_analyzed)*100:.1f}%" if articles_analyzed else "0%"]
        ]
        
        report.append(tabulate(sentiment_data, headers="firstrow", tablefmt="pipe") + "\n\n")
        
        # Key Points
        report.append("### Key Points from News Analysis\n")
        
        # Positive points
        positive_points = sentiment_results.get("key_positive_points", [])
        if positive_points:
            report.append("**Positive Factors:**\n")
            for point in positive_points:
                report.append(f"- {point}\n")
            report.append("\n")
        
        # Negative points
        negative_points = sentiment_results.get("key_negative_points", [])
        if negative_points:
            report.append("**Negative Factors:**\n")
            for point in negative_points:
                report.append(f"- {point}\n")
            report.append("\n")
        
        # Neutral points
        neutral_points = sentiment_results.get("key_neutral_points", [])
        if neutral_points:
            report.append("**Neutral Factors:**\n")
            for point in neutral_points:
                report.append(f"- {point}\n")
            report.append("\n")
        
        # News Summary
        report.append("### News Summary\n")
        report.append(news_summary + "\n\n")
        
        # Recent Articles
        article_results = sentiment_results.get("article_results", [])
        if article_results:
            report.append("### Recent Articles\n")
            
            for i, article in enumerate(article_results[:5], 1):
                title = article.get("title", "No title")
                url = article.get("url", "#")
                sentiment = article.get("sentiment", {}).get("sentiment", "neutral").capitalize()
                
                report.append(f"{i}. [{title}]({url}) - **{sentiment}**\n")
            
            report.append("\n")
        
        return "\n".join(report)
    
    def _format_final_analysis(self, ticker, technical_signals, sentiment_results):
        """Format the final analysis and recommendation"""
        report = []
        
        report.append("## Overall Analysis and Recommendation\n")
        
        # Extract key signals for different timeframes
        short_term = "Neutral"
        medium_term = "Neutral"
        long_term = "Neutral"
        
        if "Short_Term_Trend" in technical_signals:
            short_term = technical_signals["Short_Term_Trend"]
        
        if "Medium_Term_Trend" in technical_signals:
            medium_term = technical_signals["Medium_Term_Trend"]
        
        if "Long_Term_Trend" in technical_signals:
            long_term = technical_signals["Long_Term_Trend"]
        
        # Get sentiment
        sentiment = sentiment_results.get("overall_sentiment", "neutral").capitalize()
        
        # Summary table
        summary_data = [
            ["Timeframe", "Outlook"],
            ["Short-term (Days)", short_term],
            ["Medium-term (Weeks)", medium_term],
            ["Long-term (Months)", long_term],
            ["News Sentiment", sentiment]
        ]
        
        report.append(tabulate(summary_data, headers="firstrow", tablefmt="pipe") + "\n\n")
        
        # Generate final recommendation
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
        
        report.append(f"**Final Recommendation for {ticker}: {recommendation}**\n")
        report.append(f"**Suggested Action: {action}**\n\n")
        
        # Additional analysis
        report.append("### Analysis Summary\n")
        
        # Combine technical and sentiment factors
        factors = []
        
        # Add technical factors
        if "MA_Crossover" in technical_signals:
            factors.append(f"Moving Average Crossover: {technical_signals['MA_Crossover']}")
        
        if "RSI_Signal" in technical_signals:
            factors.append(f"RSI: {technical_signals['RSI_Signal']}")
        
        if "MACD_Crossover" in technical_signals:
            factors.append(f"MACD: {technical_signals['MACD_Crossover']}")
        
        if "BB_Signal" in technical_signals:
            factors.append(f"Bollinger Bands: {technical_signals['BB_Signal']}")
        
        if "Volume_Signal" in technical_signals:
            factors.append(f"Volume: {technical_signals['Volume_Signal']}")
        
        # Add sentiment factors
        positive_points = sentiment_results.get("key_positive_points", [])
        negative_points = sentiment_results.get("key_negative_points", [])
        
        # Create bullet points
        report.append("Key factors influencing this analysis:\n")
        
        for factor in factors:
            report.append(f"- {factor}\n")
        
        for point in positive_points[:2]:
            report.append(f"- Positive: {point}\n")
        
        for point in negative_points[:2]:
            report.append(f"- Negative: {point}\n")
        
        report.append("\n")
        
        # Disclaimer
        report.append("*Disclaimer: This analysis is generated automatically and should not be considered as financial advice. Always conduct your own research before making investment decisions.*\n")
        
        return "\n".join(report)
    
    def generate_price_chart(self, ticker, df_daily, output_file=None):
        """
        Generate a price chart with moving averages
        
        Args:
            ticker: Stock ticker symbol
            df_daily: DataFrame with daily price data
            output_file: Output file path (if None, will be generated)
        
        Returns:
            Path to the generated chart file
        """
        if df_daily.empty:
            return None
        
        if output_file is None:
            os.makedirs(os.path.join(self.output_dir, "charts"), exist_ok=True)
            output_file = os.path.join(self.output_dir, "charts", f"{ticker}_price_chart.png")
        
        # Create a new figure
        plt.figure(figsize=(12, 8))
        
        # Get price column
        price_col = 'adjusted close' if 'adjusted close' in df_daily.columns else 'close'
        
        # Get recent data (last 180 days)
        recent_data = df_daily.iloc[-180:] if len(df_daily) > 180 else df_daily
        
        # Plot price
        plt.plot(recent_data.index, recent_data[price_col], label='Price', linewidth=2)
        
        # Plot moving averages
        ma_columns = [col for col in recent_data.columns if col.startswith('SMA_')]
        for ma in ['SMA_20', 'SMA_50', 'SMA_200']:
            if ma in recent_data.columns:
                plt.plot(recent_data.index, recent_data[ma], label=ma, linewidth=1.5)
        
        # Set title and labels
        plt.title(f"{ticker} Price Chart with Moving Averages", fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        # Save the chart
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def generate_report(self, ticker, company_data, df_daily, df_weekly, df_monthly, 
                       technical_signals, price_targets, sentiment_results, news_summary):
        """
        Generate a complete analysis report
        
        Args:
            ticker: Stock ticker symbol
            company_data: Company overview data
            df_daily: DataFrame with daily price data
            df_weekly: DataFrame with weekly price data
            df_monthly: DataFrame with monthly price data
            technical_signals: Dictionary with technical signals
            price_targets: Dictionary with price targets
            sentiment_results: Dictionary with sentiment analysis results
            news_summary: String with news summary
        
        Returns:
            String with the complete report
        """
        report_parts = []
        
        # Report header
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_parts.append(f"# Stock Analysis Report: {ticker}\n")
        report_parts.append(f"*Generated on: {timestamp}*\n\n")
        
        # Basic ticker data
        report_parts.append(self._format_ticker_data(company_data, df_daily, df_weekly, df_monthly))
        
        # Technical analysis
        report_parts.append(self._format_technical_analysis(df_daily, df_weekly, df_monthly, price_targets))
        
        # Sentiment analysis
        report_parts.append(self._format_sentiment_analysis(sentiment_results, news_summary))
        
        # Final analysis and recommendation
        report_parts.append(self._format_final_analysis(ticker, technical_signals, sentiment_results))
        
        # Generate price chart
        chart_path = self.generate_price_chart(ticker, df_daily)
        if chart_path:
            report_parts.append(f"\n\n## Price Chart\n\n![{ticker} Price Chart]({chart_path})\n")
        
        return "\n".join(report_parts)
    
    def save_report(self, report, ticker):
        """
        Save the report to a file
        
        Args:
            report: Report string
            ticker: Stock ticker symbol
        
        Returns:
            Path to the saved report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_report_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        return filepath
