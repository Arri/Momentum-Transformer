#########################################################################
# This is STEP 1
# Run this 
#    * On the PC in wsl2,
#    * in the anaconda environment 'trade',
#    * from the folder ./trading_momentum_transformer
#    * with e.g.:
# > python -m data.download_yfinance_data --symbols LDOS RAIFY NFLX AMZN GOOGL BIIB AMD TSLA TMUS AAPL GE BAC C --start_date 2010-01-01 --end_date 2025-01-01 --output_dir data/yfinance
# or if you want to use the default 10 tickers (see settings/default.py):
# > python -m data.download_yfinance_data --start_date 2010-01-01 --end_date 2025-01-01
#########################################################################

import yfinance as yf
import os
import argparse
import logging
import pandas as pd

def download_stock_data(symbols, start_date, end_date, output_dir):
    """
    Download historical stock data for a list of symbols using yfinance.

    Args:
        symbols (list): List of stock symbols to download.
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.
        output_dir (str): Directory to save the CSV files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        filename="download_yfinance_data.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting yfinance data download.")

    for symbol in symbols:
        try:
            print(f"Downloading data for {symbol}...")
            logging.info(f"Downloading data for {symbol}.")

            # Fetch data
            data = yf.download(symbol, start=start_date, end=end_date)

            # Check if data is not empty
            if not data.empty:
                # Convert the first part of the MultiIndex column names to lowercase
                new_columns = [(level1.lower(), level2) for level1, level2 in data.columns]
                data.columns = pd.MultiIndex.from_tuples(new_columns, names=data.columns.names)
                
                # Save the data to a CSV file
                file_path = os.path.join(output_dir, f"{symbol}.csv")
                data.to_csv(file_path)
                print(f"Saved data for {symbol} to {file_path}")
                logging.info(f"Saved data for {symbol} to {file_path}")
            else:
                print(f"No data found for {symbol}. Skipping.")
                logging.warning(f"No data found for {symbol}. Skipping.")
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
            logging.error(f"Error downloading data for {symbol}: {e}")

    print("Download completed.")
    logging.info("Download completed.")

if __name__ == "__main__":
    def get_args():
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(description="Download historical stock data using yfinance.")
        parser.add_argument(
            "--symbols",
            type=list,
            default=["LDOS", "NFLX", "AMZN", "GOOGL", "AMD", "TSLA", "AAPL", "GE", "BAC", "C"],
            help="List of stock symbols to download (e.g., AAPL MSFT TSLA).",
        )
        parser.add_argument(
            "--start_date",
            type=str,
            required=True,
            help="Start date for historical data (format: YYYY-MM-DD).",
        )
        parser.add_argument(
            "--end_date",
            type=str,
            required=True,
            help="End date for historical data (format: YYYY-MM-DD).",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="data/yfinance",
            help="Directory to save the downloaded data.",
        )
        return parser.parse_args()

    args = get_args()
    download_stock_data(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
    )
