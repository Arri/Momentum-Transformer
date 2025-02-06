import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from empyrical import (
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    downside_risk,
    annual_return,
    annual_volatility,
)

VOL_LOOKBACK = 60  # for ex-ante volatility
VOL_TARGET = 0.15  # 15% volatility target


def calc_performance_metrics(data: pd.DataFrame, metric_suffix="", num_identifiers=None) -> dict:
    """Performance metrics for evaluating strategy.

    Args:
        data (pd.DataFrame): dataframe containing captured returns, indexed by date.

    Returns:
        dict: dictionary of performance metrics.
    """
    if num_identifiers is None:
        num_identifiers = len(data.dropna()["ticker"].unique())  # Changed from "identifier"

    srs = data.dropna().groupby(level=0)["captured_returns"].sum() / num_identifiers
    return {
        f"annual_return{metric_suffix}": annual_return(srs),
        f"annual_volatility{metric_suffix}": annual_volatility(srs),
        f"sharpe_ratio{metric_suffix}": sharpe_ratio(srs),
        f"downside_risk{metric_suffix}": downside_risk(srs),
        f"sortino_ratio{metric_suffix}": sortino_ratio(srs),
        f"max_drawdown{metric_suffix}": -max_drawdown(srs),
        f"calmar_ratio{metric_suffix}": calmar_ratio(srs),
        f"perc_pos_return{metric_suffix}": len(srs[srs > 0.0]) / len(srs),
        f"profit_loss_ratio{metric_suffix}": np.mean(srs[srs > 0.0])
        / np.mean(np.abs(srs[srs < 0.0])),
    }


def calc_performance_metrics_subset(srs: pd.Series, metric_suffix="") -> dict:
    """Performance metrics for evaluating a single time series.

    Args:
        srs (pd.Series): Series containing captured returns, aggregated by date.
        metric_suffix (str, optional): Suffix for metric names (default: "").

    Returns:
        dict: Dictionary of performance metrics for the given series.
    """
    return {
        f"annual_return{metric_suffix}": annual_return(srs),
        f"annual_volatility{metric_suffix}": annual_volatility(srs),
        f"downside_risk{metric_suffix}": downside_risk(srs),
        f"max_drawdown{metric_suffix}": -max_drawdown(srs),
    }


def calc_net_returns(data: pd.DataFrame, list_basis_points: List[float], identifiers=None):
    """Calculates net returns adjusted for transaction costs.

    Args:
        data (pd.DataFrame): DataFrame containing trading data.
        list_basis_points (List[float]): List of transaction cost values in basis points.
        identifiers (List[str], optional): List of asset identifiers.

    Returns:
        pd.DataFrame: DataFrame with net returns after deducting transaction costs.
    """
    if identifiers is None:
        identifiers = data["ticker"].unique().tolist()  # Ensure it matches YFinance format

    cost = np.atleast_2d(list_basis_points) * 1e-4  # Convert basis points to fraction

    dfs = []
    for i in identifiers:
        data_slice = data[data["ticker"] == i].reset_index(drop=True)

        # Compute volatility scaling
        annualised_vol = data_slice["daily_vol"] * np.sqrt(252)
        scaled_position = VOL_TARGET * data_slice["position"] / annualised_vol

        # Compute transaction costs
        transaction_costs = (
            scaled_position.diff().abs().fillna(0.0).to_frame().to_numpy() * cost
        )

        # Compute net captured returns
        net_captured_returns = data_slice[["captured_returns"]].to_numpy() - transaction_costs

        # Generate column names
        columns = [f"captured_returns_{str(c).replace('.', '_')}_bps" for c in list_basis_points]

        # Append to final DataFrame
        dfs.append(pd.concat([data_slice, pd.DataFrame(net_captured_returns, columns=columns)], axis=1))

    return pd.concat(dfs).reset_index(drop=True)


def calc_sharpe_by_year(data: pd.DataFrame, suffix: str = None) -> dict:
    """Calculates the Sharpe ratio for each year in the dataframe.

    Args:
        data (pd.DataFrame): DataFrame containing captured returns, indexed by date.
        suffix (str, optional): Suffix for the metric names.

    Returns:
        dict: Dictionary mapping each year to its Sharpe ratio.
    """
    if not suffix:
        suffix = ""

    data = data.copy()
    data["year"] = data.index.year  # Ensure the index is datetime-based

    # Compute the Sharpe ratio per year
    sharpes = (
        data.dropna()[["year", "captured_returns"]]
        .groupby("year")
        .apply(lambda y: sharpe_ratio(y["captured_returns"]))
    )

    sharpes.index = "sharpe_ratio_" + sharpes.index.astype(str) + suffix

    return sharpes.to_dict()


def calc_returns(srs: pd.Series, day_offset: int = 1) -> pd.Series:
    """Calculate returns over the past number of days.

    Args:
        srs (pd.Series): time-series of prices.
        day_offset (int, optional): Number of days to calculate returns over. Defaults to 1.

    Returns:
        pd.Series: series of returns.
    """
    returns = srs / srs.shift(day_offset) - 1.0
    return returns


def calc_daily_vol(daily_returns: pd.Series) -> pd.Series:
    """Calculate daily volatility using exponential moving average.

    Args:
        daily_returns (pd.Series): Series of daily returns.

    Returns:
        pd.Series: Series of daily volatility values.
    """
    return (
        daily_returns.ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK)
        .std()
        .bfill()  # Updated deprecated method
    )


def calc_vol_scaled_returns(daily_returns: pd.Series, daily_vol: pd.Series = None) -> pd.Series:
    """Calculates volatility scaled returns for annualized VOL_TARGET of 15%.

    Args:
        daily_returns (pd.Series): Series of daily returns.
        daily_vol (pd.Series, optional): Series of daily volatilities.

    Returns:
        pd.Series: Volatility-scaled returns.
    """
    if daily_vol is None or daily_vol.isna().all():
        daily_vol = calc_daily_vol(daily_returns)

    annualised_vol = daily_vol * np.sqrt(252)  # Annualized
    return daily_returns * VOL_TARGET / annualised_vol.shift(1)


def calc_trend_intermediate_strategy(
    srs: pd.Series, w: float, volatility_scaling=True
) -> pd.Series:
    """Calculate intermediate strategy.

    Args:
        srs (pd.Series): Series of prices.
        w (float): weight, w=0 is Moskowitz TSMOM.
        volatility_scaling (bool, optional): Apply volatility scaling. Defaults to True.

    Returns:
        pd.Series: Series of captured returns.
    """
    daily_returns = calc_returns(srs)
    monthly_returns = calc_returns(srs, 21)
    annual_returns = calc_returns(srs, 252)

    next_day_returns = (
        calc_vol_scaled_returns(daily_returns).shift(-1)
        if volatility_scaling
        else daily_returns.shift(-1)
    )

    return (
        w * np.sign(monthly_returns) * next_day_returns
        + (1 - w) * np.sign(annual_returns) * next_day_returns
    )


class MACDStrategy:
    def __init__(self, trend_combinations: List[Tuple[float, float]] = None):
        """Used to calculate the combined MACD signal for multiple short/signal combinations.

        Args:
            trend_combinations (List[Tuple[float, float]], optional): Short/long trend combinations.
        """
        if trend_combinations is None:
            self.trend_combinations = [(8, 24), (16, 48), (32, 96)]
        else:
            self.trend_combinations = trend_combinations

    @staticmethod
    def calc_signal(srs: pd.Series, short_timescale: int, long_timescale: int) -> float:
        """Calculate MACD signal for a given short/long timescale combination.

        Args:
            srs (pd.Series): Series of prices.
            short_timescale (int): Short timescale.
            long_timescale (int): Long timescale.

        Returns:
            float: MACD signal.
        """

        def _calc_halflife(timescale):
            return np.log(0.5) / np.log(1 - 1 / timescale)

        macd = (
            srs.ewm(halflife=_calc_halflife(short_timescale)).mean()
            - srs.ewm(halflife=_calc_halflife(long_timescale)).mean()
        )
        q = macd / srs.rolling(63).std().bfill()
        return q / q.rolling(252).std().bfill()

    @staticmethod
    def scale_signal(y):
        """Scale the MACD signal.

        Args:
            y (float): MACD signal.

        Returns:
            float: Scaled MACD signal.
        """
        return y * np.exp(-(y ** 2) / 4) / 0.89

    def calc_combined_signal(self, srs: pd.Series) -> float:
        """Calculate combined MACD signal.

        Args:
            srs (pd.Series): Series of prices.

        Returns:
            float: MACD combined signal.
        """
        return np.sum(
            [self.calc_signal(srs, S, L) for S, L in self.trend_combinations]
        ) / len(self.trend_combinations)
