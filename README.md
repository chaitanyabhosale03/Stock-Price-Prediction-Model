# Stock Price Prediction Dashboard

This project is a web-based Stock Price Prediction Dashboard built with Streamlit. It allows users to analyze, visualize, and predict stock prices using various machine learning models and technical indicators. The app also provides sentiment analysis and generates PDF reports for selected stocks.

> **IMPORTANT NOTE:**
> If the app is not working due to data fetching issues, try using [polygon.io](https://polygon.io/) as a data source. If you don't know how to do this, message me for help.

## Features
- **User Login**: Simple authentication for dashboard access.
- **Stock Data Fetching**: Download historical stock data using Yahoo Finance.
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and more.
- **Machine Learning Models**: Linear Regression, Random Forest, XGBoost, and LSTM for price prediction.
- **Interactive Charts**: Visualize price, volume, and technical indicators.
- **Sentiment Analysis**: Analyze news headlines for sentiment.
- **PDF Report Generation**: Download a summary report for any stock.
- **Portfolio Tracker**: Sidebar watchlist with live price updates.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.

## How to Run Locally
1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd Stock_Model
   ```
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Add stock tickers**
   - Place your `stock_tickers.xlsx` file in the project directory, or use the default tickers.
4. **Run the app**
   ```sh
   streamlit run appv2.1.3.py
   ```
5. **Login**
   - Username: `admin`
   - Password: `admin`

## File Structure
- `appv2.1.3.py` — Main Streamlit app
- `requirements.txt` — Python dependencies
- `stock_tickers.xlsx` — List of stock symbols (optional)
- `stock_prediction_model.h5` — Pre-trained LSTM model (optional)

## License
This project is for educational purposes.
