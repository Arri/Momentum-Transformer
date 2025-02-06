import os

# Lookback window lengths for Change Point Detection
CPD_LBWS = [10, 21, 63, 126, 256]
CPD_DEFAULT_LBW = 21

# Average basis points for backtesting
BACKTEST_AVERAGE_BASIS_POINTS = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Use Kernel Mean Hypothesis (optional)
USE_KM_HYP_TO_INITIALISE_KC = True

# Output folder for yfinance data
YFINANCE_OUTPUT_FOLDER = lambda lbw=CPD_DEFAULT_LBW: os.path.join(
    "data", f"yfinance_cpd_{(lbw if lbw else 'none')}lbw"
)
if not os.path.exists(YFINANCE_OUTPUT_FOLDER()):
    os.makedirs(YFINANCE_OUTPUT_FOLDER())

# Default path for features and data files
FEATURES_YFINANCE_FILE_PATH = lambda ticker: os.path.join(
    YFINANCE_OUTPUT_FOLDER(), f"{ticker}_features.csv"
)

# List of yfinance tickers (example stock tickers)
STOCK_TICKERS = [
    "LDOS", "NFLX", "AMZN", "GOOGL", "AMD", "TSLA", "AAPL", "GE", "BAC", "C"
]

# Commodities tickers (optional, example)
COMMODITIES_TICKERS = [
    "GC=F",  # Gold
    "CL=F",  # Crude Oil
    "NG=F",  # Natural Gas
    "SI=F",  # Silver
]

# Default start and end dates for yfinance data
YFINANCE_START_DATE = "2010-01-01"
YFINANCE_END_DATE = "2025-01-01"
