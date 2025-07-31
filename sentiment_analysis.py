"""
Sentiment Analysis Module

This module handles sentiment analysis of news articles using Ollama's llama3 model.
"""

import os
import json
import asyncio
import httpx
import ollama
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SentimentAnalyzer:
    """Class for analyzing sentiment of news articles using Ollama or fallback methods"""
    
    def __init__(self, ollama_host=None, model="llama3"):
        """Initialize with Ollama host and model name"""
        print("Initializing SentimentAnalyzer...")
        # Set Ollama host if provided, otherwise use environment variable or default
        self.ollama_host = ollama_host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.model = model
        self.ollama_available = False
        
        # Check if Ollama is available during initialization
        try:
            # Configure Ollama client with the host
            if self.ollama_host != 'http://localhost:11434':
                ollama.host = self.ollama_host
            
            # Simple test to see if Ollama is running
            # We'll use a direct call instead of checking for specific models
            # This avoids potential issues with httpx in the ollama client
            try:
                # Just check if we can connect - this is a very light operation
                models = ollama.list()
                # If we get here, Ollama is available
                self.ollama_available = True
                
                # Check for model availability
                available_models = [model['name'] for model in models.get('models', [])]
                if self.model in available_models:
                    print(f"Ollama is available at {self.ollama_host} with model {self.model}")
                elif available_models:
                    # If the model isn't found but others are available, use the first one
                    self.model = available_models[0]
                    print(f"Model {self.model} not found. Using available model: {self.model} instead")
                else:
                    # No models available
                    print("No models available in Ollama")
                    self.ollama_available = False
            except Exception as inner_e:
                print(f"Unable to list Ollama models: {inner_e}")
                self.ollama_available = False
        except Exception as e:
            print(f"Ollama not available: {e}")
            self.ollama_available = False
            print("Will use rule-based sentiment analysis as fallback")
    
    async def analyze_text(self, text, max_length=4000):
        """
        Analyze the sentiment of the given text
        
        Args:
            text: Text to analyze
            max_length: Maximum length of text to analyze
        
        Returns:
            Dictionary with sentiment analysis results
        """
        # Truncate text if it's too long
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Prepare prompt for the model
        prompt = f"""
        Analyze the sentiment of the following text from a financial/stock market perspective.
        
        TEXT:
        {text}
        
        Please provide a detailed sentiment analysis with the following structure:
        1. Overall sentiment (bullish, bearish, or neutral)
        2. Confidence score (0-100)
        3. Key positive points
        4. Key negative points
        5. Key neutral points
        
        Format your response as JSON with the following keys:
        - sentiment: "bullish", "bearish", or "neutral"
        - confidence: integer between 0 and 100
        - positive_points: array of strings
        - negative_points: array of strings
        - neutral_points: array of strings
        """
        
        # Use Ollama client library
        if not self.ollama_available:
            return self._get_default_sentiment_result("Ollama not available")
        
        try:
            print(f"Analyzing text with {self.model} model...")
            
            # Create a coroutine to run the synchronous Ollama generate function
            start_time = asyncio.get_event_loop().time()
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,  # Default executor
                lambda: ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": 0.1,  # Lower temperature for more consistent responses
                        "num_predict": 1024  # Limit response length
                    }
                )
            )
            
            elapsed = asyncio.get_event_loop().time() - start_time
            print(f"Analysis completed in {elapsed:.2f} seconds")
            
            # Extract the response text
            response_text = response.get('response', '{}')
            
            # Try to extract JSON from the response
            try:
                # First, try to parse as is
                sentiment_data = json.loads(response_text)
            except json.JSONDecodeError:
                # If that fails, try to find JSON within the text
                try:
                    # Look for JSON between triple backticks
                    import re
                    json_match = re.search(r'```json\s*([\s\S]+?)\s*```', response_text)
                    if json_match:
                        sentiment_data = json.loads(json_match.group(1))
                    else:
                        # Try to find text that looks like JSON (surrounded by curly braces)
                        json_match = re.search(r'\{[\s\S]+\}', response_text)
                        if json_match:
                            sentiment_data = json.loads(json_match.group(0))
                        else:
                            raise json.JSONDecodeError("Could not find JSON in response", response_text, 0)
                except json.JSONDecodeError:
                    print("Could not parse JSON from Ollama response")
                    return self._get_default_sentiment_result("Error: Could not parse JSON from Ollama response")
            
            # Validate and format sentiment data
            sentiment = sentiment_data.get("sentiment", "neutral").lower()
            confidence = int(sentiment_data.get("confidence", 50))
            
            # Ensure sentiment is one of the expected values
            if sentiment not in ["bullish", "bearish", "neutral"]:
                sentiment = "neutral"
            
            # Ensure confidence is within expected range
            confidence = max(0, min(100, confidence))
            
            # Prepare the result
            result = {
                "sentiment": sentiment,
                "confidence": confidence,
                "positive_points": sentiment_data.get("positive_points", []),
                "negative_points": sentiment_data.get("negative_points", []),
                "neutral_points": sentiment_data.get("neutral_points", []),
                "error": False,
                "error_message": ""
            }
            
            return result
            
        except Exception as e:
            print(f"Error using Ollama client: {e}")
            return self._get_default_sentiment_result(f"Error: {str(e)}")
    
    def _get_default_sentiment_result(self, error_message):
        """Return a default sentiment result with the given error message"""
        # For now, we'll use a simple rule-based analysis as fallback
        return {
            "sentiment": "neutral",
            "confidence": 50,
            "positive_points": ["Automated fallback analysis activated"],
            "negative_points": [],
            "neutral_points": [error_message]
        }
    
    async def analyze_news_articles(self, news_articles):
        """
        Analyze sentiment of a list of news articles using Ollama or a rule-based fallback
        
        Args:
            news_articles: List of news article dictionaries
        
        Returns:
            Dictionary with sentiment analysis results
        """
        if not news_articles:
            return {
                "overall_sentiment": "neutral",
                "confidence": 50,
                "articles_analyzed": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "key_positive_points": [],
                "key_negative_points": [],
                "key_neutral_points": ["No news articles to analyze"]
            }
        
        print("Analyzing news sentiment...")
        
        # Check if Ollama is available
        if not self.ollama_available:
            print("Ollama not available, using rule-based sentiment analysis")
            return self._rule_based_sentiment_analysis(news_articles)
        
        print(f"Ollama is available with model {self.model}, using it for sentiment analysis")
        
        try:
            # Prepare text for analysis
            # Limit to 5 articles to avoid overwhelming the model
            articles_to_analyze = news_articles[:5]  
            
            # Format the news articles for the prompt
            combined_text = ""
            for i, article in enumerate(articles_to_analyze, 1):
                title = article.get("title", "No title")
                source = article.get("source", "Unknown source")
                date = article.get("time_published", "Unknown date")[:10]  # Just the date part
                summary = article.get("summary", "")[:200]  # Limit summary length
                
                combined_text += f"Article {i}:\n"
                combined_text += f"Title: {title}\n"
                combined_text += f"Source: {source}\n"
                combined_text += f"Date: {date}\n"
                combined_text += f"Summary: {summary}\n\n"
            
            # Create a clear, concise prompt
            prompt = f"""Analyze the sentiment of these financial news articles about a stock or company:

{combined_text}

Provide a detailed financial sentiment analysis with the following structure:
1. Overall sentiment (bullish, bearish, or neutral)
2. Confidence score (0-100)
3. Key positive points (2-3 bullet points)
4. Key negative points (2-3 bullet points)
5. Key neutral points (1-2 bullet points)

Format your response as JSON with the following keys:
- sentiment: "bullish", "bearish", or "neutral"
- confidence: integer between 0 and 100
- positive_points: array of strings
- negative_points: array of strings
- neutral_points: array of strings

Response format (JSON only):"""
            
            # Call Ollama with a 5-minute timeout
            print(f"Starting sentiment analysis of news articles...")
            analysis_start_time = asyncio.get_event_loop().time()
            try:
                sentiment_result = await asyncio.wait_for(
                    self.analyze_text(prompt),
                    timeout=300.0  # 5 minutes timeout
                )
                
                analysis_elapsed = asyncio.get_event_loop().time() - analysis_start_time
                print(f"Sentiment analysis completed in {analysis_elapsed:.2f} seconds")
                
                if not sentiment_result.get("error", False):
                    sentiment = sentiment_result.get('sentiment', 'unknown')
                    confidence = sentiment_result.get('confidence', 0)
                    print(f"News sentiment: {sentiment.capitalize()} (Confidence: {confidence}%)")
                    
                    # Format the result to match expected structure
                    return {
                        "overall_sentiment": sentiment_result.get("sentiment", "neutral"),
                        "confidence": sentiment_result.get("confidence", 50),
                        "articles_analyzed": len(articles_to_analyze),
                        "bullish_count": 1 if sentiment_result.get("sentiment") == "bullish" else 0,
                        "bearish_count": 1 if sentiment_result.get("sentiment") == "bearish" else 0,
                        "neutral_count": 1 if sentiment_result.get("sentiment") == "neutral" else 0,
                        "key_positive_points": sentiment_result.get("positive_points", []),
                        "key_negative_points": sentiment_result.get("negative_points", []),
                        "key_neutral_points": sentiment_result.get("neutral_points", [])
                    }
            except asyncio.TimeoutError:
                analysis_elapsed = asyncio.get_event_loop().time() - analysis_start_time
                print(f"Ollama analysis timed out after {analysis_elapsed:.2f} seconds")
            except Exception as e:
                print(f"Error during sentiment analysis: {str(e)}")
        
        except Exception as e:
            print(f"Error preparing news for analysis: {type(e).__name__}: {str(e)}")
        
        # If we reached here, Ollama failed - use rule-based fallback
        print("Falling back to rule-based sentiment analysis")
        return self._rule_based_sentiment_analysis(news_articles)
    
    def _rule_based_sentiment_analysis(self, news_articles):
        """Simple rule-based sentiment analysis for news articles"""
        # Limit the number of articles to analyze
        articles_to_analyze = news_articles[:10]
        
        # Keywords for sentiment analysis
        bullish_keywords = ['bullish', 'surge', 'soar', 'gain', 'growth', 'positive', 'jump', 
                          'outperform', 'beat', 'exceed', 'upgrade', 'strong', 'upside', 'buy', 
                          'higher', 'rally', 'record', 'profit', 'rise']
        bearish_keywords = ['bearish', 'fall', 'drop', 'decline', 'loss', 'negative', 'tumble', 
                          'underperform', 'miss', 'downgrade', 'weak', 'downside', 'sell', 
                          'lower', 'risk', 'disappointing', 'concern', 'worry', 'lose']
        
        # Analyze each article
        article_results = []
        for article in articles_to_analyze:
            title = article.get("title", "")
            summary = article.get("summary", "")
            
            # Combine for analysis
            text = f"{title} {summary}".lower()
            
            # Count sentiment keywords
            bullish_count = sum(1 for word in bullish_keywords if word in text)
            bearish_count = sum(1 for word in bearish_keywords if word in text)
            
            # Determine sentiment
            if bullish_count > bearish_count:
                sentiment = "bullish"
                confidence = min(100, int((bullish_count / (bullish_count + bearish_count + 1)) * 100))
            elif bearish_count > bullish_count:
                sentiment = "bearish"
                confidence = min(100, int((bearish_count / (bullish_count + bearish_count + 1)) * 100))
            else:
                sentiment = "neutral"
                confidence = 50
            
            # Create points based on keywords found
            positive_points = [f"Keyword '{word}' found in article" for word in bullish_keywords if word in text][:3]
            negative_points = [f"Keyword '{word}' found in article" for word in bearish_keywords if word in text][:3]
            
            # Create sentiment result
            sentiment_result = {
                "sentiment": sentiment,
                "confidence": confidence,
                "positive_points": positive_points if positive_points else ["No strongly positive elements identified"],
                "negative_points": negative_points if negative_points else ["No strongly negative elements identified"],
                "neutral_points": ["Analysis performed using keyword matching (Ollama unavailable)"]
            }
            
            article_results.append({
                "title": title,
                "url": article.get("url", ""),
                "published_at": article.get("time_published", ""),
                "sentiment": sentiment_result,
                "analyzed_by": "rule-based"
            })
        
        # Count overall sentiment
        bullish_count = sum(1 for article in article_results if article["sentiment"]["sentiment"] == "bullish")
        bearish_count = sum(1 for article in article_results if article["sentiment"]["sentiment"] == "bearish")
        neutral_count = sum(1 for article in article_results if article["sentiment"]["sentiment"] == "neutral")
        
        # Determine overall sentiment
        if bullish_count > bearish_count:
            overall_sentiment = "bullish"
            confidence = min(100, int((bullish_count / len(article_results)) * 70))  # 70% max confidence for rule-based
        elif bearish_count > bullish_count:
            overall_sentiment = "bearish"
            confidence = min(100, int((bearish_count / len(article_results)) * 70))  # 70% max confidence for rule-based
        else:
            overall_sentiment = "neutral"
            confidence = 50
        
        # Extract key points
        key_positive_points = ["Automated keyword-based analysis detected positive sentiment"]
        key_negative_points = ["Automated keyword-based analysis detected negative sentiment"]
        for article in article_results:
            sentiment = article["sentiment"]
            if sentiment["sentiment"] == "bullish" and sentiment["positive_points"]:
                key_positive_points.append(f"Article '{article['title'][:40]}...' shows positive signals")
            if sentiment["sentiment"] == "bearish" and sentiment["negative_points"]:
                key_negative_points.append(f"Article '{article['title'][:40]}...' shows negative signals")
        
        return {
            "overall_sentiment": overall_sentiment,
            "confidence": confidence,
            "articles_analyzed": len(article_results),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "key_positive_points": key_positive_points[:5],
            "key_negative_points": key_negative_points[:5],
            "key_neutral_points": ["Sentiment analysis performed using rule-based analysis as Ollama was unavailable"],
            "article_results": article_results
        }
        
    def _simple_rule_based_sentiment(self, title, summary):
        """A simple rule-based sentiment analyzer for a single article"""
        # Keywords for sentiment analysis
        bullish_keywords = ['bullish', 'surge', 'soar', 'gain', 'growth', 'positive', 'jump', 
                          'outperform', 'beat', 'exceed', 'upgrade', 'strong', 'upside', 'buy', 
                          'higher', 'rally', 'record', 'profit', 'rise']
        bearish_keywords = ['bearish', 'fall', 'drop', 'decline', 'loss', 'negative', 'tumble', 
                          'underperform', 'miss', 'downgrade', 'weak', 'downside', 'sell', 
                          'lower', 'risk', 'disappointing', 'concern', 'worry', 'lose']
        
        # Combine for analysis
        text = f"{title} {summary}".lower()
        
        # Count sentiment keywords
        bullish_count = sum(1 for word in bullish_keywords if word in text)
        bearish_count = sum(1 for word in bearish_keywords if word in text)
        
        # Determine sentiment
        if bullish_count > bearish_count:
            sentiment = "bullish"
            confidence = min(100, int((bullish_count / (bullish_count + bearish_count + 1)) * 100))
        elif bearish_count > bullish_count:
            sentiment = "bearish"
            confidence = min(100, int((bearish_count / (bullish_count + bearish_count + 1)) * 100))
        else:
            sentiment = "neutral"
            confidence = 50
        
        # Create points based on keywords found
        positive_points = [f"Keyword '{word}' found in article" for word in bullish_keywords if word in text][:3]
        negative_points = [f"Keyword '{word}' found in article" for word in bearish_keywords if word in text][:3]
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_points": positive_points if positive_points else ["No strongly positive elements identified"],
            "negative_points": negative_points if negative_points else ["No strongly negative elements identified"],
            "neutral_points": ["Analysis performed using keyword matching due to Ollama unavailability"]
        }
    
    async def summarize_news(self, news_articles):
        """
        Generate a summary of news articles
        
        Args:
            news_articles: List of news article dictionaries
        
        Returns:
            String with news summary
        """
        if not news_articles:
            return "No news articles available."
            
        # Check if Ollama is available
        if not self.ollama_available:
            return "News summary not available - Ollama service is not accessible."
            
        print(f"[DEBUG] Generating news summary with {self.model}...")
        
        # Limit the number of articles to summarize
        articles_to_summarize = news_articles[:5]
        
        # Prepare text for summarization
        text = "Recent news articles:\n\n"
        
        for i, article in enumerate(articles_to_summarize, 1):
            title = article.get("title", "No title")
            summary = article.get("summary", "No summary")
            date = article.get("time_published", "Unknown date")[:10]  # Just get the date part
            
            text += f"{i}. {title}\n"
            text += f"   Date: {date}\n"
            text += f"   Summary: {summary[:300]}\n\n"
        
        # Prepare prompt for the model
        prompt = f"""Summarize the following financial news articles, focusing on key events, market impacts, and overall sentiment:

{text}

Provide a concise 3-5 sentence summary that captures the most important information and the overall market sentiment from these articles.

Summary:"""
        
        try:
            # Use Ollama client library
            print("Generating news summary...")
            summary_start_time = asyncio.get_event_loop().time()
            
            # Create a coroutine to run the synchronous Ollama generate function
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,  # Default executor
                    lambda: ollama.generate(
                        model=self.model,
                        prompt=prompt,
                        options={
                            "temperature": 0.2,  # Lower temperature for factual summaries
                            "num_predict": 512  # Limit response length for summary
                        }
                    )
                ),
                timeout=120.0  # 2 minute timeout
            )
            
            summary_elapsed = asyncio.get_event_loop().time() - summary_start_time
            print(f"News summary completed in {summary_elapsed:.2f} seconds")
            
            # Extract and return the summary
            return response.get('response', "No summary available.")
            
        except asyncio.TimeoutError:
            print("News summary timed out")
            return "News summary unavailable - processing timed out."
        except Exception as e:
            print(f"Error generating news summary: {str(e)}")
            return "Error generating news summary."
