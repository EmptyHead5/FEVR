import pandas as pd
from itertools import product

import noiseReduction
import trainingModel
import dataLoad


def print_results(name, df):
    """
    Print performance metrics for a strategy
    """
    final_equity = df["Strategy_Equity"].iloc[-1]
    buy_hold = df["BuyHold_Equity"].iloc[-1]

    sharpe = (df["Strategy_Return"].mean() / df["Strategy_Return"].std()) * (252 ** 0.5)

    print(f"===== {name} =====")
    print("Final Strategy Equity:", final_equity)
    print("Buy & Hold Equity:", buy_hold)
    print("Sharpe Ratio:", sharpe)


def generate_param_list():
    """
    Generate a large grid of hyperparameter combinations
    """
    hidden_options = [
        (32, 16),
        (64, 32, 16),
        (128, 64, 32),
        (128, 64, 32, 16),
    ]
    learning_rates = [0.01, 0.001, 0.0005, 0.0001]
    alphas = [0.0001, 0.001, 0.01]
    batch_sizes = [16, 32, 64]
    activations = ["relu", "tanh"]
    thresholds = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    param_list = []
    for h, lr, alpha, batch, act, th in product(
        hidden_options, learning_rates, alphas, batch_sizes, activations, thresholds
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
    return param_list


def evaluate_params(data, params):
    """
    Train MLP with given parameters and evaluate performance
    """
    # Train model
    res = trainingModel.MLPclassifier(data.copy(), **params)

    # Compute confidence interval
    ci = trainingModel.confidentInterval(res["y_test"], res["y_pred"])

    # Prepare backtest data
    data_mlp = res["data"]
    train_size = int(len(data_mlp) * 0.8)
    startDate = data_mlp.index[train_size]

    # Run backtest
    data_bt = trainingModel.backtest_from_prediction(data_mlp.copy(), startDate)

    # Extract final equity
    final_equity = data_bt["Strategy_Equity"].iloc[-1] if not data_bt.empty else None

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
        "data_bt": data_bt,
        "raw_result": res,
    }


if __name__ == "__main__":

    # ===============================
    # 1. Load and preprocess data
    # ===============================
    data = dataLoad.loadAndCleanData()
    data = dataLoad.extraDataSet(data)
    data = dataLoad.add_features(data)

    # ===============================
    # 2. MLP model
    # ===============================
    mlp_output = trainingModel.MLPclassifier(data.copy())

    data_mlp = mlp_output["data"]
    train_size_mlp = int(len(data_mlp) * 0.8)
    startDate_mlp = data_mlp.index[train_size_mlp]

    data_mlp_bt = trainingModel.backtest_from_prediction(data_mlp.copy(), startDate_mlp)

    output_mlp = data_mlp_bt.loc[startDate_mlp:, [
        "Return", "Prediction", "Position", "Strategy_Equity", "Target"
    ]]
    output_mlp.to_csv("backtest_mlp.csv", index=True)
    


    # ===============================
    # Hyperparameter search
    # ===============================
    results = []
    param_list = generate_param_list()

    print(f"\nTotal parameter combinations: {len(param_list)}")

    for i, params in enumerate(param_list, start=1):
        try:
            print(f"\n[{i}/{len(param_list)}] params: {params}")

            out = evaluate_params(data, params)

            # Store results (exclude large objects)
            results.append(
                {k: v for k, v in out.items() if k not in ["data_bt", "raw_result"]}
            )

            print("data_bt empty?", out["data_bt"].empty)

        except Exception as e:
            print(f"Failed on params {params}")
            print("Error:", repr(e))
            continue

    # ===============================
    # Save and analyze results
    # ===============================
    results_df = pd.DataFrame(results)

    if not results_df.empty:

        # Sort by F1 score
        results_df = results_df.sort_values("f1", ascending=False)

        print("\n===== RESULTS SORTED BY F1 =====")
        print(results_df.head(20))

        results_df.to_csv("mlp_hyperparameter_results.csv", index=False)

        # Apply filtering conditions
        filtered_df = results_df[
            (results_df["p_value"] < 0.05) &
            (results_df["recall"] < 0.98) &
            (results_df["accuracy"] > 0.52)
        ].copy()

        filtered_df = filtered_df.sort_values("final_equity", ascending=False)

        print("\n===== FILTERED RESULTS SORTED BY FINAL EQUITY =====")
        print(filtered_df.head(20))

        filtered_df.to_csv("mlp_filtered_results.csv", index=False)

        # Sort by final profit
        results_df = results_df.sort_values("final_equity", ascending=False)

        print("\n===== TOP BY PROFIT =====")
        print(results_df.head(5))

        results_df.to_csv("mlp_results_with_equity.csv", index=False)

    else:
        print("No successful results were collected.")