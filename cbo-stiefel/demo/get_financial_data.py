# filename: get_financial_data.py

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import sys

# --- Configuration ---

# 1. Define your asset universe (N assets)
#    (Using a sample of 55 S&P 500 stocks as an example)
#TICKERS = [
#    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'JPM',
#    'JNJ', 'V', 'XOM', 'WMT', 'PG', 'UNH', 'MA', 'HD', 'CVX', 'MRK',
#    'BAC', 'KO', 'PEP', 'LLY', 'AVGO', 'COST', 'TMO', 'ABBV', 'MCD',
#    'CSCO', 'CRM', 'PFE', 'ADBE', 'LIN', 'ABT', 'DIS', 'NFLX', 'CMCSA',
#    'VZ', 'PM', 'INTC', 'AMD', 'NEE', 'NKE', 'HON', 'UPS', 'CAT', 'IBM',
#    'SBUX', 'GS', 'RTX', 'LOW', 'BLK', 'DE', 'BA', 'C'
#]  # n = 55 assets
TICKERS = [
    # US Equity Sectors & Sizes
    'SPY', 'QQQ', 'IWM', 'MDY',
    'XLF', 'XLK', 'XLV', 'XLE',
    'XLI', 'XLY', 'XLP', 'XLU',
    'XLB', 'VNQ',

    # International Equities
    'EFA', 'EEM', 'EWJ', 'EWG',
    'EWU', 'EWQ', 'MCHI', 'INDA',
    'EWA', 'EWC', 'EWZ',

    # Fixed Income / Bonds
    'AGG', 'TLT', 'IEF', 'SHY',
    'TIP', 'LQD', 'HYG', 'BNDX',

    # Commodities
    'GLD', 'SLV', 'USO', 'UNG',
    'DBC', 'DBA', 'COPX',

    # Currencies & Volatility
    'UUP', 'FXE', 'FXY', 'VXX'
]

# 2. Define your historical time window (T observations)
#    (Using 200 weekly periods, as in the research paper)
END_DATE = datetime.now()
START_DATE = END_DATE - pd.DateOffset(weeks=200)

# 3. Define the data interval
#    ("1wk" for weekly, "1d" for daily)
INTERVAL = "1wk"

# 4. Define the output file name
OUTPUT_FILE = "asset_returns.csv"


# --- End Configuration ---


def fetch_price_data(tickers, start, end, interval):
    """
    Downloads historical 'Adj Close' prices from Yahoo Finance.
    """
    print(f"Downloading {len(tickers)} assets from {start.date()} to {end.date()}...")
    try:
        data = yf.download(tickers, start=START_DATE, end=END_DATE, interval=interval, auto_adjust=False)

        if data.empty:
            print("Error: No data downloaded. Check tickers and date range.")
            return None

        # Select 'Adj Close' prices, which account for dividends and splits
        prices = data['Adj Close']

        # Drop any assets that have no data for the whole period
        prices = prices.dropna(axis=1, how='all')

        if prices.empty:
            print("Error: All tickers were invalid or had no data.")
            return None

        print(f"Successfully downloaded data for {prices.shape[1]} assets.")
        return prices

    except Exception as e:
        print(f"An error occurred during download: {e}")
        return None


def calculate_log_returns(prices_df):
    """
    Calculates the log-differenced returns from a DataFrame of prices.
    Log returns (log_rtn = ln(P_t / P_{t-1})) are standard in finance.
    """
    print("Calculating log-differenced returns...")
    # ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
    log_returns = np.log(prices_df / prices_df.shift(1))

    # The first row will be NaN after shifting
    log_returns = log_returns.dropna(axis=0, how='all')

    print(f"Calculated returns matrix of shape: {log_returns.shape} (T x N)")
    return log_returns


def save_returns_to_csv(returns_df, filename):
    """
    Saves the final returns DataFrame to a CSV file.
    """
    try:
        returns_df.to_csv(filename)
        print(f"\nSuccessfully saved returns to '{filename}'")
        print("You can now run 'full_droppca_suite.py' again.")
    except Exception as e:
        print(f"Error: Could not save file. {e}")
        sys.exit(1)


def main():
    """
    Main script execution logic.
    """
    if len(TICKERS) == 0:
        print("Error: TICKERS list is empty. Please add stock tickers.")
        sys.exit(1)

    # 1. Fetch Prices
    prices = fetch_price_data(TICKERS, START_DATE, END_DATE, INTERVAL)

    if prices is None:
        print("Exiting due to data download failure.")
        sys.exit(1)

    # 2. Calculate Returns
    returns = calculate_log_returns(prices)

    # 3. Save to File
    save_returns_to_csv(returns, OUTPUT_FILE)


if __name__ == "__main__":
    main()