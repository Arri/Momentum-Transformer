import os
import numpy as np
import pandas as pd
from mom_trans.classical_strategies import (
    MACDStrategy,
    calc_returns,
    calc_daily_vol,
    calc_vol_scaled_returns,
)

VOL_THRESHOLD = 5  # multiple to winsorize by
HALFLIFE_WINSORISE = 252


def read_changepoint_results_and_fill_na(file_path: str, lookback_window_length: int) -> pd.DataFrame:
    """Read changepoint detection results and fill missing values."""
    # print("^^^^^^^>>>>>>> file_path = ", file_path)
    # df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # print(f"^^^^^^^>>>>>>> Columns in DataFrame: {df.columns}")  # Debugging line
    return (
        pd.read_csv(file_path, index_col=0, parse_dates=True)
        .fillna(method="ffill")
        .dropna()
        .assign(
            cp_location_norm=lambda row: (row["t"] - row["cp_location"]) / lookback_window_length
        )
    )


def prepare_cpd_features(folder_path: str, lookback_window_length: int) -> pd.DataFrame:
    """Prepare changepoint detection features for all tickers."""
    # print(f"------------->> lookback_window_length = {lookback_window_length}")
    # print(f"------------->> folder_path = {folder_path}")
    print(f"------------->> read_changepoint_results_and_fill_na = {read_changepoint_results_and_fill_na(os.path.join(folder_path, 'C.csv'), lookback_window_length)}")
    # exit(0)
    return pd.concat(
        [
            read_changepoint_results_and_fill_na(
                os.path.join(folder_path, f), lookback_window_length
            ).assign(ticker=os.path.splitext(f)[0])
            for f in os.listdir(folder_path)
        ]
    )


def deep_momentum_strategy_features(df_asset: pd.DataFrame) -> pd.DataFrame:
    """Prepare input features for the Momentum Transformer model."""
    required_columns = ["close", "open", "high", "low", "volume"]

    # Ensure required columns exist
    for col in required_columns:
        if col not in df_asset.columns:
            raise ValueError(f"Missing required column: {col}")

    df_asset = df_asset[~df_asset["close"].isna() & (df_asset["close"] > 1e-8)].copy()

    # Winsorize using rolling 5X standard deviations to remove outliers
    df_asset["srs"] = df_asset["close"]
    ewm = df_asset["srs"].ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    df_asset["srs"] = np.minimum(df_asset["srs"], means + VOL_THRESHOLD * stds)
    df_asset["srs"] = np.maximum(df_asset["srs"], means - VOL_THRESHOLD * stds)

    # Calculate returns and volatility
    df_asset["daily_returns"] = calc_returns(df_asset["srs"])
    df_asset["daily_vol"] = calc_daily_vol(df_asset["daily_returns"])
    df_asset["target_returns"] = calc_vol_scaled_returns(df_asset["daily_returns"], df_asset["daily_vol"]).shift(-1)

    # Calculate normalized returns
    def calc_normalized_returns(day_offset):
        return (
            calc_returns(df_asset["srs"], day_offset)
            / df_asset["daily_vol"]
            / np.sqrt(day_offset)
        )

    df_asset["norm_daily_return"] = calc_normalized_returns(1)
    df_asset["norm_monthly_return"] = calc_normalized_returns(21)
    df_asset["norm_quarterly_return"] = calc_normalized_returns(63)
    df_asset["norm_biannual_return"] = calc_normalized_returns(126)
    df_asset["norm_annual_return"] = calc_normalized_returns(252)

    # MACD signals
    trend_combinations = [(8, 24), (16, 48), (32, 96)]
    for short_window, long_window in trend_combinations:
        df_asset[f"macd_{short_window}_{long_window}"] = MACDStrategy.calc_signal(
            df_asset["srs"], short_window, long_window
        )

    # Date-related features
    if len(df_asset):
        df_asset["day_of_week"] = df_asset.index.dayofweek
        df_asset["day_of_month"] = df_asset.index.day
        df_asset["week_of_year"] = df_asset.index.isocalendar().week
        df_asset["month_of_year"] = df_asset.index.month
        df_asset["year"] = df_asset.index.year
        df_asset["date"] = df_asset.index  # duplication but sometimes makes life easier
    else:
        df_asset["day_of_week"] = []
        df_asset["day_of_month"] = []
        df_asset["week_of_year"] = []
        df_asset["month_of_year"] = []
        df_asset["year"] = []
        df_asset["date"] = []

    # print(f"-------->>>>> df_asset.dropna() shape = {df_asset.dropna().shape}")
    # print(f"-------->>>>> df_asset.dropna() = {df_asset.dropna()}")
 
    # Remove from the multi-row column index the rows with "Ticker" as those are not column names
    df_asset.columns = df_asset.columns.droplevel("Ticker")
    # Remove the column Header Name "Price"
    df_asset.columns = df_asset.columns.get_level_values(0)
    # print(f" Columns = {df_asset.columns}")
    # print(df_asset.dropna())
    return df_asset.dropna()


def include_changepoint_features(
    features: pd.DataFrame, cpd_folder_name: pd.DataFrame, lookback_window_length: int
) -> pd.DataFrame:
    """Combine changepoint features with Momentum Transformer features."""
    print(f"..............>>>> features index: {features.index}")
    print(f"..............>>>> cpd_features index: {prepare_cpd_features(cpd_folder_name, lookback_window_length).index}")
    
    features = features.merge(
        prepare_cpd_features(cpd_folder_name, lookback_window_length)[
            ["ticker", "cp_location_norm", "cp_score"]
        ]
        .rename(
            columns={
                "cp_location_norm": f"cp_rl_{lookback_window_length}",
                "cp_score": f"cp_score_{lookback_window_length}",
            }
        )
        .reset_index(),
        on=["date"],
        # on=["Date", "Ticker"],
    )

    features.index = features["date"]
    return features
