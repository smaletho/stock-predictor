# Stock Predictor

A Python application for predicting and analyzing stocks using Yahoo Finance data and LLM-based sentiment analysis.

## Features

- Stock data retrieval using Yahoo Finance API via yfinance
- Technical analysis with multiple moving averages and trade signals
- News sentiment analysis using Ollama's LLM models
- Multiple timeframe analysis (daily, weekly, monthly, 3-month)
- Optional enhanced news coverage using NewsAPI
- Comprehensive output with price targets and detailed analysis

## Setup

1. Ensure you have Python installed on your system
2. Clone this repository
3. Create and activate the virtual environment:
   ```
   # Create virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Configure your environment in the `.env` file (optional):
   ```
   # Optional: Add NewsAPI key for enhanced news coverage
   NEWSAPI_KEY=your_newsapi_key_here
   
   # Ollama host URL (default: http://localhost:11434)
   OLLAMA_HOST=http://localhost:11434
   ```
6. Install and start Ollama for sentiment analysis:
   - Download from [Ollama.ai](https://ollama.ai/)
   - Install the llama3 model: `ollama pull llama3`
   - Ensure Ollama is running before starting the application

## Usage

Run the main script with a stock ticker:

```
python main.py --ticker AAPL
```

Additional options:

```
python main.py --ticker MSFT --output-dir reports --api-mode cached --skip-news
```

Command-line arguments:
- `--ticker`: Stock ticker symbol (required)
- `--output-dir`: Directory for saving reports (default: "reports")
- `--api-mode`: API usage mode: 'full' or 'cached' (default: "standard")
- `--skip-news`: Skip news fetching to save API calls
- `--data-dir`: Directory for caching data (default: "data")

## Output

The application will generate:
- Technical analysis summary with short, medium, and long-term indicators
- Sentiment analysis of related news using Ollama LLM
- Price targets and predictions with support and resistance levels
- Trading signals based on multiple moving averages
- Overall market sentiment with confidence score
- Comprehensive analysis report saved in Markdown format

## Dependencies

- yfinance: Yahoo Finance data access
- pandas: Data analysis and manipulation
- matplotlib & seaborn: Data visualization
- ollama: Python client for Ollama LLM
- NewsAPI (optional): Enhanced news retrieval

## Notes

- The Ollama sentiment analysis requires a running Ollama instance with the llama3 model
- If Ollama is not available, the application will fall back to rule-based sentiment analysis
- Cached data is used when available to minimize API calls
