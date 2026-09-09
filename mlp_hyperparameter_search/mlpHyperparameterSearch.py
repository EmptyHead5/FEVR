import os
import pandas as pd
from itertools import product

import trainingModel


OUTPUT_FOLDER = "mlp_csv_results"


def print_results(name, df):
    final_equity = df["Strategy_Equity"].iloc[-1]
    buy_hold = df["BuyHold_Equity"].iloc[-1]
    sharpe = trainingModel.calculate_sharpe(df)

    print(f"===== {name} =====")
    print("Final Strategy Equity:", final_equity)
    print("Buy & Hold Equity:", buy_hold)
    print("Sharpe Ratio:", sharpe)


def generate_param_list():
    hidden_options = [
        (32, 16),
        (64, 32),
    ]

    learning_rates = [0.001, 0.0005]
    alphas = [0.0001, 0.001]
    batch_sizes = [32]
    activations = ["relu"]
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
    #set some hyperparameter options to test
    param_list = []

    for h, lr, alpha, batch, act, th in product(
        hidden_options,
        learning_rates,
        alphas,
        batch_sizes,
        activations,
        thresholds,
    ):
        param_list.append(
            {
                "hidden_layer_sizes": h,
                "learning_rate_init": lr,
                "alpha": alpha,
                "batch_size": batch,
                "activation": act,
                "threshold": th,
            }
        )
    #use for loop to generate all combinations of hyperparameters and store them in a list of dictionaries
    return param_list


def evaluate_params(data, params):
    res = trainingModel.MLPclassifier(data.copy(), **params)
    #call the MLPclassifier function in trainingModel.py to train and evaluate the model with the given hyperparameters
    ci = trainingModel.confidentInterval(res["y_test"], res["y_pred"])
    #calculate confidence interval for the model's accuracy
    data_mlp = res["data"]

    train_size = int(len(data_mlp) * 0.8)
    start_date = data_mlp.index[train_size]
    #80% of the data is used for training, and the remaining 20% is used for testing. 
    data_bt = trainingModel.backtestConfidenceFilter(
        data_mlp.copy(),
        start_date,
        min_prob=params["threshold"],
    )
    

    final_equity = None
    sharpe = None

    if not data_bt.empty:
        final_equity = data_bt["Strategy_Equity"].iloc[-1]
        sharpe = trainingModel.calculate_sharpe(data_bt)

    return {
        **params,
        "accuracy": res["accuracy"],
        "precision": res["precision"],
        "recall": res["recall"],
        "f1": res["f1"],
        "accuracy_ci": ci["accuracy_ci"],
        "p_value": ci["p_value"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "n_test": ci["n_test"],
        "n_correct": ci["n_correct"],
        "final_equity": final_equity,
        "sharpe": sharpe,
    }


def run_search(data):
    results = []
    param_list = generate_param_list()
    #fist generate the list of hyperparameter combinations to test
    #now totaly is 2 hidden layer options * 2 learning rates * 2 alphas * 1 batch size * 1 activation * 5 thresholds = 40 combinations


    print(f"\nTotal parameter combinations: {len(param_list)}")
    #print the total number of combinations to test

    for i, params in enumerate(param_list, start=1):
        #use for loop to iterate through the list of hyperparameter combinations and evaluate each one
        #meanwhile print the current combination being evaluated
        print(f"\n[{i}/{len(param_list)}] params: {params}")

        out = evaluate_params(data, params)

        results.append(out)

        print("Accuracy:", out["accuracy"])
        print("F1:", out["f1"])
        print("Final Equity:", out["final_equity"])
        print("Sharpe:", out["sharpe"])



    save_search_results(results, data)


def save_search_results(results, data):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No successful results were collected.")
        return

    results_by_f1 = results_df.sort_values("f1", ascending=False)
    results_by_equity = results_df.sort_values("final_equity", ascending=False)
    results_by_sharpe = results_df.sort_values("sharpe", ascending=False)

    results_by_f1.to_csv(
        os.path.join(OUTPUT_FOLDER, "mlp_hyperparameter_results.csv"),
        index=False,
    )

    results_by_equity.to_csv(
        os.path.join(OUTPUT_FOLDER, "mlp_results_with_equity.csv"),
        index=False,
    )

    results_by_sharpe.to_csv(
        os.path.join(OUTPUT_FOLDER, "mlp_results_by_sharpe.csv"),
        index=False,
    )

    save_best_params_and_backtest(results_by_sharpe, results_df, data)


def save_best_params_and_backtest(results_by_sharpe, results_df, data):
    param_cols = [
        "hidden_layer_sizes",
        "learning_rate_init",
        "alpha",
        "batch_size",
        "activation",
        "threshold",
    ]

    top_params = results_by_sharpe.head(1)[param_cols]

    top_params.to_csv(
        os.path.join(OUTPUT_FOLDER, "best_params_by_sharpe.csv"),
        index=False,
    )

    best_params = top_params.iloc[0].to_dict()

    print("\n===== BEST PARAMS BY SHARPE =====")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    print("\nBest Sharpe from search:")
    print(results_by_sharpe.iloc[0]["sharpe"])

    print("\nBest final equity from search:")
    print(results_by_sharpe.iloc[0]["final_equity"])

    mlp_output = trainingModel.MLPclassifier(data.copy(), **best_params)

    data_mlp = mlp_output["data"]

    train_size_mlp = int(len(data_mlp) * 0.8)
    start_date_mlp = data_mlp.index[train_size_mlp]

    data_mlp_bt = trainingModel.backtestConfidenceFilter(
        data_mlp.copy(),
        start_date_mlp,
        min_prob=best_params["threshold"],
    )

    if data_mlp_bt.empty:
        print("\nFinal backtest returned no rows.")
        return

    final_equity_check = data_mlp_bt["Strategy_Equity"].iloc[-1]
    sharpe_check = trainingModel.calculate_sharpe(data_mlp_bt)

    print("\n===== FINAL BACKTEST USING BEST PARAMS =====")
    print("Backtest start date:", start_date_mlp)
    print("Backtest end date:", data_mlp_bt.index[-1])
    print("Final equity from rerun:", final_equity_check)
    print("Sharpe from rerun:", sharpe_check)

    print_results("MLP Confidence Filter", data_mlp_bt)

    output_mlp = data_mlp_bt.loc[
        start_date_mlp:,
        [
            "Return",
            "Prediction",
            "Predicted_Prob",
            "Position",
            "Strategy_Equity",
            "BuyHold_Equity",
            "Target",
        ],
    ]

    output_mlp.to_csv(
        os.path.join(OUTPUT_FOLDER, "backtest_mlp.csv"),
        index=True,
    )

    filtered_df = results_df[
        (results_df["p_value"] < 0.05)
        & (results_df["recall"] < 0.98)
        & (results_df["accuracy"] > 0.52)
        & (results_df["sharpe"] > 1.0)
    ].copy()

    filtered_df = filtered_df.sort_values("sharpe", ascending=False)

    filtered_df.to_csv(
        os.path.join(OUTPUT_FOLDER, "mlp_filtered_results.csv"),
        index=False,
    )

    print("\nCSV files saved in folder:")
    print(OUTPUT_FOLDER)