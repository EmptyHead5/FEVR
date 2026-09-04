import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint


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
    hidden_layer_sizes=(32, 16),
    learning_rate_init=0.0001,
    alpha=0.001,
    batch_size=32,
    activation='relu',
    solver='adam',
    threshold=0.6,
    random_state=42
):
    # hidden_layer_sizes:
    # A tuple representing the number of neurons in each hidden layer
    # if too large → overfitting, too small → underfitting

    # learning_rate_init:
    # controls how fast weights update
    # smaller → more stable but slower convergence

    # alpha:
    # regularization term to prevent overfitting
    # train good / test bad → increase alpha
    # both bad → decrease alpha

    # batch_size:
    # number of samples per update
    # affects stability and generalization

    # activation:
    # function used for neuron output transformation

    # solver:
    # optimization algorithm

    # threshold:
    # probability cutoff for classification
    # higher → more conservative predictions

    # random_state:
    # fixes randomness for reproducibility

    data = data.copy()

    data["RSI"] = calculateRSI(data["Close"])

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Return_1D", "Return_5D", "SMA_Ratio",
        "Volatility_20", "Volume_Change", "RSI"
    ]

    data = data.dropna(subset=feature_cols + ["Target"]).copy()

    X = data[feature_cols]
    y = data["Target"]

    train_size = int(len(data) * 0.8)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    test_index = X_test.index

    # StandardScaler is important for MLP
    # because neural networks are sensitive to feature scale
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
        early_stopping=True
    )

    model.fit(X_train, y_train)

    # predict probability instead of direct class
    y_prob = model.predict_proba(X_test)[:, 1]

    # convert probability → classification
    y_pred = (y_prob >= threshold).astype(int)

    data["Prediction"] = np.nan
    data.loc[test_index, "Prediction"] = y_pred

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
        "data": data
    }

    return result


def backtest_from_prediction(data, startDate):
    data = data.dropna(subset=["Prediction"]).copy()
    data = data.loc[startDate:].copy()

    data["Return"] = data["Close"].pct_change()

    # shift(1) to avoid look-ahead bias
    data["Position"] = data["Prediction"].shift(1)

    data["Strategy_Return"] = data["Position"] * data["Return"]

    data = data.dropna().copy()

    initial_capital = 100

    # cumulative return curve
    data["Strategy_Equity"] = initial_capital * (1 + data["Strategy_Return"]).cumprod()
    data["BuyHold_Equity"] = initial_capital * (1 + data["Return"]).cumprod()

    return data


def confidentInterval(y_test, y_pred):
    correct = (y_test == y_pred).sum()
    n = len(y_test)

    # hypothesis test: H0: p = 0.5
    p_value = binomtest(correct, n, p=0.5, alternative='two-sided').pvalue

    # 95% confidence interval (Wilson)
    ci_low, ci_high = proportion_confint(correct, n, alpha=0.05, method='wilson')

    result = {
        "accuracy_ci": correct / n,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_test": n,
        "n_correct": int(correct)
    }

    print("accuracy =", result["accuracy_ci"])
    print("p-value =", result["p_value"])
    print("95% CI =", (result["ci_low"], result["ci_high"]))

    return result