import pandas as pd

import mlp_model as trainingModel
import data_loader


if __name__ == "__main__":

    # ===============================
    # 1. Load and preprocess data
    # ===============================
    data = data_loader.loadAndCleanData()
    data = data_loader.extraDataSet(data)
    data = data_loader.add_features(data)


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
    # 3. Final comparison
    # ===============================
    print("\n===== Final Comparison =====")
    print("MLP Strategy:", data_mlp_bt["Strategy_Equity"].iloc[-1])
    print("MLP Buy & Hold:", data_mlp_bt["BuyHold_Equity"].iloc[-1])

    # ===============================
    # 4. Evaluation metrics
    # ===============================
    print("\n===== MLP Confidence Interval =====")
    trainingModel.confidentInterval(mlp_output["y_test"], mlp_output["y_pred"])

    print("\n===== Neural Network (MLP) =====")
    print("Accuracy:", mlp_output["accuracy"])
    print("Precision:", mlp_output["precision"])
    print("Recall:", mlp_output["recall"])
    print("F1:", mlp_output["f1"])