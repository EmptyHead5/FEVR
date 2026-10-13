from itertools import product
from pathlib import Path

import pandas as pd



import trainingModel


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FOLDER = BASE_DIR / "mlp_csv_results"

PARAM_COLUMNS = [
    "hidden_layer_sizes",
    "learning_rate_init",
    "alpha",
    "batch_size",
    "activation",
    "threshold",
]

OUTPUT_COLUMNS = [
    "Return",
    "Prediction",
    "Predicted_Prob",
    "Position",
    "Strategy_Return",
    "Strategy_Equity",
    "BuyHold_Equity",
    "Target",
]


def print_results(name, data):
    print(f"\n===== {name} =====")
    print("Final Strategy Equity:", data["Strategy_Equity"].iloc[-1])
    print("Buy & Hold Equity:", data["BuyHold_Equity"].iloc[-1])
    print("Sharpe Ratio:", trainingModel.calculate_sharpe(data))


def generate_param_list():
    hidden_options = [
        (32, 16),
        (64, 32),
        (128, 64),
        (64, 32, 16),
    ]

    learning_rates = [0.01, 0.001, 0.0005]
    alphas = [0.0001, 0.001, 0.01]
    batch_sizes = [32, 64]
    activations = ["relu"]
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    return [
        {
            "hidden_layer_sizes": hidden,
            "learning_rate_init": learning_rate,
            "alpha": alpha,
            "batch_size": batch_size,
            "activation": activation,
            "threshold": threshold,
        }
        for (
            hidden,
            learning_rate,
            alpha,
            batch_size,
            activation,
            threshold,
        ) in product(
            hidden_options,
            learning_rates,
            alphas,
            batch_sizes,
            activations,
            thresholds,
        )
    ]


def run_backtest(data_mlp, threshold):
    if data_mlp.empty:
        raise ValueError("The model returned no data.")

    train_size = int(len(data_mlp) * 0.8)
    start_date = data_mlp.index[train_size]

    data_backtest = trainingModel.backtestConfidenceFilter(
        data_mlp.copy(),
        start_date,
        min_prob=threshold,
    )

    return start_date, data_backtest


def evaluate_params(data, params):
    result = trainingModel.MLPclassifier(
        data.copy(),
        **params,
    )

    confidence_interval = trainingModel.confidentInterval(
        result["y_test"],
        result["y_pred"],
    )

    _, data_backtest = run_backtest(
        result["data"],
        params["threshold"],
    )

    final_equity = None
    sharpe = None

    if not data_backtest.empty:
        final_equity = data_backtest["Strategy_Equity"].iloc[-1]
        sharpe = trainingModel.calculate_sharpe(data_backtest)

    return {
        **params,
        "accuracy": result["accuracy"],
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["f1"],
        "accuracy_ci": confidence_interval["accuracy_ci"],
        "p_value": confidence_interval["p_value"],
        "ci_low": confidence_interval["ci_low"],
        "ci_high": confidence_interval["ci_high"],
        "n_test": confidence_interval["n_test"],
        "n_correct": confidence_interval["n_correct"],
        "final_equity": final_equity,
        "sharpe": sharpe,
    }


def run_search(symbol, data):
    results = []
    param_list = generate_param_list()

    print(f"\n===== TESTING {symbol} =====")
    print("Total parameter combinations:", len(param_list))

    for index, params in enumerate(param_list, start=1):
        print(
            f"\n[{index}/{len(param_list)}] "
            f"{symbol} params: {params}"
        )

        try:
            output = evaluate_params(data, params)
            results.append(output)

            print("Accuracy:", output["accuracy"])
            print("F1:", output["f1"])
            print("Final Equity:", output["final_equity"])
            print("Sharpe:", output["sharpe"])

        except Exception as error:
            print(f"Parameter combination failed: {error}")

    return save_search_results(symbol, results, data)


def save_search_results(symbol, results, data):
    symbol_folder = OUTPUT_FOLDER / symbol
    symbol_folder.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print(f"No successful results were collected for {symbol}.")
        return results_df

    results_by_f1 = results_df.sort_values(
        "f1",
        ascending=False,
    )

    results_by_equity = results_df.sort_values(
        "final_equity",
        ascending=False,
    )

    results_by_sharpe = results_df.sort_values(
        "sharpe",
        ascending=False,
    )

    results_by_f1.to_csv(
        symbol_folder / "mlp_hyperparameter_results.csv",
        index=False,
    )

    results_by_equity.to_csv(
        symbol_folder / "mlp_results_with_equity.csv",
        index=False,
    )

    results_by_sharpe.to_csv(
        symbol_folder / "mlp_results_by_sharpe.csv",
        index=False,
    )

    save_filtered_results(symbol_folder, results_df)

    save_best_params_and_backtest(
        symbol,
        symbol_folder,
        results_by_sharpe,
        data,
    )

    print("\nCSV files saved in folder:")
    print(symbol_folder)

    return results_df


def save_filtered_results(symbol_folder, results_df):
    filtered_df = results_df[
        (results_df["p_value"] < 0.05)
        & (results_df["recall"] < 0.98)
        & (results_df["accuracy"] > 0.52)
        & (results_df["sharpe"] > 1.0)
    ].copy()

    filtered_df = filtered_df.sort_values(
        "sharpe",
        ascending=False,
    )

    filtered_name_map = {
        "threshold": "Min_Prob_To_Buy",
        "accuracy": "Model_Accuracy",
        "precision": "Model_Precision",
        "recall": "Model_Recall",
        "f1": "Model_F1",
        "accuracy_ci": "Accuracy_CI",
        "p_value": "P_Value",
        "ci_low": "CI_Low",
        "ci_high": "CI_High",
        "n_test": "Test_Data_Count",
        "n_correct": "Correct_Predictions",
        "final_equity": "Final_Strategy_Equity",
        "sharpe": "Sharpe_Ratio",
    }

    filtered_df.rename(columns=filtered_name_map).to_csv(
        symbol_folder / "mlp_filtered_results.csv",
        index=False,
    )


def save_best_params_and_backtest(
    symbol,
    symbol_folder,
    results_by_sharpe,
    data,
):
    valid_results = results_by_sharpe.dropna(subset=["sharpe"])

    if valid_results.empty:
        print(f"No valid Sharpe Ratio was produced for {symbol}.")
        return

    top_params = valid_results.head(1)[PARAM_COLUMNS]
    best_params = top_params.iloc[0].to_dict()

    top_params.to_csv(
        symbol_folder / "best_params_by_sharpe.csv",
        index=False,
    )

    print(f"\n===== {symbol} BEST PARAMS BY SHARPE =====")

    for key, value in best_params.items():
        print(f"{key}: {value}")

    print("Best Sharpe from search:", valid_results.iloc[0]["sharpe"])
    print(
        "Best final equity from search:",
        valid_results.iloc[0]["final_equity"],
    )

    mlp_output = trainingModel.MLPclassifier(
        data.copy(),
        **best_params,
    )

    start_date, data_backtest = run_backtest(
        mlp_output["data"],
        best_params["threshold"],
    )

    if data_backtest.empty:
        print(f"Final backtest returned no rows for {symbol}.")
        return

    print("\nBacktest start date:", start_date)
    print("Backtest end date:", data_backtest.index[-1])

    print_results(
        f"{symbol} MLP Confidence Filter",
        data_backtest,
    )

    output_mlp = data_backtest.loc[
        start_date:,
        OUTPUT_COLUMNS,
    ]

    output_mlp.to_csv(
        symbol_folder / f"{symbol}_result.csv",
        index=True,
    )

if __name__ == "__main__":
    import dataLoad

    data = dataLoad.loadAndCleanData("SPY")
    data = dataLoad.extraDataSet(data)
    data = dataLoad.add_features(data)

    run_search("SPY", data)
