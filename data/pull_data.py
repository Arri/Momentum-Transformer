import os
from typing import List
import pandas as pd
import yfinance as yf
from settings.default import STOCK_TICKERS, YFINANCE_OUTPUT_FOLDER, YFINANCE_START_DATE, YFINANCE_END_DATE


def pull_yfinance_data(ticker: str, start_date=YFINANCE_START_DATE, end_date=YFINANCE_END_DATE) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance for a specific ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        start_date: Start date for historical data (YYYY-MM-DD).
        end_date: End date for historical data (YYYY-MM-DD).

    Returns:
        DataFrame with historical stock data.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    print(f"Found for the ticker {ticker}: {data}")
    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    data = data.rename(
        columns={
            "Date": "date",
            "Adj Close": "adjclose",
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
        }
    )
    # data = data.reset_index(drop=False)
    # print("======>>>> ", data.columns)
    
    return data[["close", "open", "high", "low", "volume"]]


def save_yfinance_data(tickers: List[str], start_date=YFINANCE_START_DATE, end_date=YFINANCE_END_DATE):
    """
    Download and save yfinance data for multiple tickers.

    Args:
        tickers: List of stock ticker symbols.
        start_date: Start date for historical data (YYYY-MM-DD).
        end_date: End date for historical data (YYYY-MM-DD).
    """
    for ticker in tickers:
        try:
            data = pull_yfinance_data(ticker, start_date, end_date)
            output_path = os.path.join(YFINANCE_OUTPUT_FOLDER(), f"{ticker}.csv")
            data.to_csv(output_path)
            print(f"Saved data for {ticker} to {output_path}")
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")


if __name__ == "__main__":
    save_yfinance_data(STOCK_TICKERS)
