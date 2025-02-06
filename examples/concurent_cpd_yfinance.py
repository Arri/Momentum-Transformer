import multiprocessing
import argparse
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

from settings.default import (
    STOCK_TICKERS,  # Adjusted for Yahoo Finance tickers
    YFINANCE_OUTPUT_FOLDER,  # Updated to the corresponding output folder
    CPD_DEFAULT_LBW,
)

N_WORKERS = len(STOCK_TICKERS)  # Use the updated list of tickers


def main(lookback_window_length: int):
    if not os.path.exists(YFINANCE_OUTPUT_FOLDER(lookback_window_length)):
        os.makedirs(YFINANCE_OUTPUT_FOLDER(lookback_window_length))

    # Adjust the command for the new structure and naming conventions
    print(f"--- Stock Tickers = {STOCK_TICKERS}")
    print("calling examples.cpd_yfinance")
    all_processes = [
        f'python -m examples.cpd_yfinance "{ticker}" "{os.path.join(YFINANCE_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")}" "1990-01-01" "2021-12-31" "{lookback_window_length}"'
        for ticker in STOCK_TICKERS
    ]
    process_pool = multiprocessing.Pool(processes=N_WORKERS)
    process_pool.map(os.system, all_processes)


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(
            description="Run changepoint detection module for all tickers"
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=CPD_DEFAULT_LBW,
            help="CPD lookback window length",
        )
        return [
            parser.parse_known_args()[0].lookback_window_length,
        ]

    main(*get_args())
