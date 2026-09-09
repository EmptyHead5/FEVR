import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint


def extractDaysData(data, start, end):
    return data.loc[start:end]


def calculateRSI(close_prices, period=14):
    delta = close_prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def MLPclassifier(
    data,
    hidden_layer_sizes=(64, 32),
    learning_rate_init=0.001,
    alpha=0.0001,
    batch_size=32,
    activation="relu",
    threshold=0.5,
    solver="adam",
    random_state=42,
):
    #set defult parameters for the MLP classifier, but normally will use the hyperparameter search to find the best parameters
    data = data.copy()

    data["RSI"] = calculateRSI(data["Close"])

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Return_1D",
        "Return_5D",
        "SMA_Ratio",
        "Volatility_20",
        "Volume_Change",
        "RSI",
    ]

    data = data.dropna(subset=feature_cols + ["Target"])
    #remove rows with missing values in the feature columns and target column

    X = data[feature_cols]
    y = data["Target"]

    train_size = int(len(data) * 0.8)

    X_train = X[:train_size]
    X_test = X[train_size:]

    y_train = y[:train_size]
    y_test = y[train_size:]
    #general mlp training and testing process, split the data into training and testing sets, scale the features, train the model, make predictions, and return the results
    test_index = X_test.index

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        batch_size=batch_size,
        activation=activation,
        solver=solver,
        max_iter=1000,
        random_state=random_state,
        early_stopping=True,
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    data["Prediction"] = np.nan
    data["Predicted_Prob"] = np.nan

    data.loc[test_index, "Prediction"] = y_pred
    data.loc[test_index, "Predicted_Prob"] = y_prob

    result = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "learning_rate_init": learning_rate_init,
        "alpha": alpha,
        "batch_size": batch_size,
        "activation": activation,
        "solver": solver,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "y_test": y_test,
        "y_pred": y_pred,
        "data": data,
    }

    return result


def backtest_from_prediction(data, startDate):
    data = data.dropna(subset=["Prediction"]).copy()
    data = data.loc[startDate:].copy()

    data["Return"] = data["Close"].pct_change()
    data["Position"] = data["Prediction"].shift(1)
    data["Strategy_Return"] = data["Position"] * data["Return"]

    data = data.dropna().copy()

    initial_capital = 100

    data["Strategy_Equity"] = initial_capital * (1 + data["Strategy_Return"]).cumprod()
    data["BuyHold_Equity"] = initial_capital * (1 + data["Return"]).cumprod()

    return data


def backtestConfidenceFilter(data, startDate, min_prob=0.6):
    data = data.dropna(subset=["Predicted_Prob"]).copy()
    data = data.loc[startDate:].copy()
    #remove all the empty tow, and only keep the data after the startDate
    data["Return"] = data["Close"].pct_change()
    #calculate daily rate of return 
    data["Position"] = 0
    #defult position is 0, which mean no open position
    data.loc[data["Predicted_Prob"] >= min_prob, "Position"] = 1
    #only open position when predicted probability is greater than the minimum probability threshold
    #now set threadhold to 0.6, which is more conservative strategy
    data["Strategy_Return"] = data["Position"].shift(1) * data["Return"]
    #use the position of the previous day to calculate the strategy return of the current day
    data = data.dropna().copy()
    #make sure no empty row
    initial_capital = 100
    #set initial capital to 100, just test, it can be any number
    data["Strategy_Equity"] = initial_capital * (1 + data["Strategy_Return"]).cumprod()
    data["BuyHold_Equity"] = initial_capital * (1 + data["Return"]).cumprod()
    #1.Strategy_Equity,  cumprod will calculate the cumulative product of the strategy return, and than plus initial capital to get the strategy equity 
    #2.BuyHold_Equity,  cumprod will calculate the cumulative product of the buy and hold return, and than plus initial capital to get the buy and hold equity
    return data


def calculate_sharpe(df):
    std = df["Strategy_Return"].std()

    if std == 0 or pd.isna(std):
        return None

    sharpe = (
        df["Strategy_Return"].mean()
        / std
    ) * (252 ** 0.5)

    return sharpe


def confidentInterval(y_test, y_pred):
    correct = (y_test == y_pred).sum()
    n = len(y_test)

    p_value = binomtest(correct, n, p=0.5, alternative="two-sided").pvalue
    ci_low, ci_high = proportion_confint(correct, n, alpha=0.05, method="wilson")

    result = {
        "accuracy_ci": correct / n,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_test": n,
        "n_correct": int(correct),
    }

    print("accuracy =", result["accuracy_ci"])
    print("p-value =", result["p_value"])
    print("95% CI =", (result["ci_low"], result["ci_high"]))

    return result