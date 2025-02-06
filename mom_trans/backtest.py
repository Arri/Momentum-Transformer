import os
from typing import Tuple, List, Dict
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import gc
import copy
import json

from mom_trans.model_inputs import ModelFeatures
from mom_trans.deep_momentum_network import LstmDeepMomentumNetworkModel
from mom_trans.momentum_transformer import TftDeepMomentumNetworkModel
from mom_trans.classical_strategies import (
    VOL_TARGET,
    calc_performance_metrics,
    calc_sharpe_by_year,
    calc_net_returns,
    annual_volatility,
)

from settings.default import BACKTEST_AVERAGE_BASIS_POINTS
from settings.hp_grid import HP_MINIBATCH_SIZE

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def _get_directory_name(experiment_name: str, train_interval: Tuple[int, int, int] = None) -> str:
    """Returns directory name for saving results."""
    if train_interval:
        return os.path.join("results", experiment_name, f"{train_interval[1]}-{train_interval[2]}")
    return os.path.join("results", experiment_name)

def _basis_point_suffix(basis_points: float = None) -> str:
    """Returns suffix for basis points."""
    return "" if not basis_points else "_" + str(basis_points).replace(".", "_") + "_bps"

def _interval_suffix(train_interval: Tuple[int, int, int], basis_points: float = None) -> str:
    """Returns suffix for train interval."""
    return f"_{train_interval[1]}_{train_interval[2]}" + _basis_point_suffix(basis_points)

def _get_asset_classes(asset_class_dictionary: Dict[str, str]):
    """Extracts unique asset classes."""
    return np.unique(list(asset_class_dictionary.values())).tolist()

def save_results(results_sw: pd.DataFrame, output_directory: str, train_interval: Tuple[int, int, int], num_identifiers: int, asset_class_dictionary: Dict[str, str], extra_metrics: dict = {}):
    """Saves results to JSON."""
    asset_classes = ["ALL"]
    results_asset_class = [results_sw]

    if asset_class_dictionary:
        results_sw["asset_class"] = results_sw["identifier"].map(lambda i: asset_class_dictionary.get(i, "Unknown"))
        classes = _get_asset_classes(asset_class_dictionary)
        for ac in classes:
            results_asset_class.append(results_sw[results_sw["asset_class"] == ac])
        asset_classes += classes

    metrics = {}
    for ac, results_ac in zip(asset_classes, results_asset_class):
        suffix = _interval_suffix(train_interval)
        ac_metrics = extra_metrics.copy() if ac == "ALL" else {}
        for basis_points in BACKTEST_AVERAGE_BASIS_POINTS:
            suffix = _interval_suffix(train_interval, basis_points)
            results_ac_bps = results_ac.drop(columns="captured_returns").rename(columns={
                "captured_returns" + _basis_point_suffix(basis_points): "captured_returns"
            }) if basis_points else results_ac

            ac_metrics.update(calc_performance_metrics(results_ac_bps.set_index("time"), suffix, num_identifiers))
            ac_metrics.update(calc_sharpe_by_year(results_ac_bps.set_index("time"), _basis_point_suffix(basis_points)))

        metrics[ac] = ac_metrics

    with open(os.path.join(output_directory, "results.json"), "w") as file:
        json.dump(metrics, file, indent=4)

def run_single_window(
    experiment_name: str,
    features_file_path: str,
    train_interval: Tuple[int, int, int],
    params: dict,
    changepoint_lbws: List[int],
    skip_if_completed: bool = True,
    asset_class_dictionary: Dict[str, str] = None,
    hp_minibatch_size: List[int] = HP_MINIBATCH_SIZE,
):
    """Backtest for a single test window."""
    
    directory = _get_directory_name(experiment_name, train_interval)

    if skip_if_completed and os.path.exists(os.path.join(directory, "results.json")):
        print(f"Skipping {train_interval[1]}-{train_interval[2]} (Already Completed)")
        return

    print(f"Processing: {features_file_path}")

    raw_data = pd.read_csv(features_file_path, index_col=0, parse_dates=True)
    raw_data["date"] = raw_data["date"].astype("datetime64[ns]")

    model_features = ModelFeatures(
        raw_data,
        params["total_time_steps"],
        start_boundary=train_interval[0],
        test_boundary=train_interval[1],
        test_end=train_interval[2],
        changepoint_lbws=changepoint_lbws,
        split_tickers_individually=params["split_tickers_individually"],
        train_valid_ratio=params["train_valid_ratio"],
        add_ticker_as_static=(params["architecture"] == "TFT"),
        time_features=params["time_features"],
    )

    hp_directory = os.path.join(directory, "hp")

    if params["architecture"] == "LSTM":
        dmn = LstmDeepMomentumNetworkModel(
            experiment_name, hp_directory, hp_minibatch_size, **params, **model_features.input_params
        )
    elif params["architecture"] == "TFT":
        dmn = TftDeepMomentumNetworkModel(
            experiment_name, hp_directory, hp_minibatch_size, **params, **model_features.input_params,
            column_definition=model_features.get_column_definition(),
            num_encoder_steps=0, stack_size=1, num_heads=4
        )
    else:
        raise ValueError(f"Invalid architecture: {params['architecture']}")

    best_hp, best_model = dmn.hyperparameter_search(model_features.train, model_features.valid)
    val_loss = dmn.evaluate(model_features.valid, best_model)

    print(f"Best Validation Loss: {val_loss}")

    with open(os.path.join(directory, "best_hyperparameters.json"), "w") as file:
        json.dump(best_hp, file, indent=4)

    print("Predicting on test set...")

    results_sw, performance_sw = dmn.get_positions(
        model_features.test_sliding, best_model, sliding_window=True, years_geq=train_interval[1], years_lt=train_interval[2]
    )

    results_sw = results_sw.merge(
        raw_data.reset_index()[["ticker_x", "date", "daily_vol"]].rename(columns={"ticker_x": "identifier", "date": "time"}),
        on=["identifier", "time"],
    )
    results_sw = calc_net_returns(results_sw, BACKTEST_AVERAGE_BASIS_POINTS[1:], model_features.tickers)
    results_sw.to_csv(os.path.join(directory, "captured_returns_sw.csv"))

    results_fw, performance_fw = dmn.get_positions(
        model_features.test_fixed, best_model, sliding_window=False, years_geq=train_interval[1], years_lt=train_interval[2]
    )

    results_fw = results_fw.merge(
        raw_data.reset_index()[["ticker_x", "date", "daily_vol"]].rename(columns={"ticker_x": "identifier", "date": "time"}),
        on=["identifier", "time"],
    )
    results_fw = calc_net_returns(results_fw, BACKTEST_AVERAGE_BASIS_POINTS[1:], model_features.tickers)
    results_fw.to_csv(os.path.join(directory, "captured_returns_fw.csv"))

    save_results(results_sw, directory, train_interval, model_features.num_tickers, asset_class_dictionary, {
        "performance_sw": performance_sw, "performance_fw": performance_fw, "val_loss": val_loss
    })

    del best_model
    gc.collect()
    tf.keras.backend.clear_session()

def run_all_windows(
    experiment_name: str,
    features_file_path: str,
    train_intervals: List[Tuple[int, int, int]],
    params: dict,
    changepoint_lbws: List[int],
    asset_class_dictionary=Dict[str, str],
    hp_minibatch_size=HP_MINIBATCH_SIZE,
):
    """Runs experiment for multiple test intervals and aggregates results."""
    for interval in train_intervals:
        run_single_window(experiment_name, features_file_path, interval, params, changepoint_lbws, asset_class_dictionary=asset_class_dictionary, hp_minibatch_size=hp_minibatch_size)
