import pandas as pd
import numpy as np


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import pandas as pd

from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint


#yfinance is a lightweight, open-source Python library for scraping global financial data provided by Yahoo Finance.
# It proves highly practical for quantitative analysis and backtesting, particularly suited for rapidly acquiring historical quotes,
#  dividend information, and stock splits to build foundational datasets.



def extractDaysData(data, start, end):
    return data.loc[start:end]


def backtestBuyLowSellHigh(data, startDate,
                        buy_rsi_threshold=40,
                        sell_rsi_threshold=60,
                        buy_price_window=20,
                        sell_price_window=20):

    data = data.dropna(subset=["Prediction"]).copy()
    data = data.loc[startDate:].copy()

    data["Return"] = data["Close"].pct_change()

    data["SMA_Buy"] = data["Close"].rolling(window=buy_price_window).mean()
    data["SMA_Sell"] = data["Close"].rolling(window=sell_price_window).mean()

    data["Position"] = 0

    in_position = False

    for i in range(len(data)):
        row = data.iloc[i]

        close_price = row["Close"]
        prediction = row["Prediction"]
        rsi = row["RSI"]
        sma_buy = row["SMA_Buy"]
        sma_sell = row["SMA_Sell"]

        if pd.isna(sma_buy) or pd.isna(sma_sell) or pd.isna(rsi):
            continue

        buy_condition = (
            prediction == 1 and
            close_price < sma_buy and
            rsi <= buy_rsi_threshold
        )

        sell_condition = (
            prediction == 0 and
            close_price > sma_sell and
            rsi >= sell_rsi_threshold
        )

        if not in_position and buy_condition:
            data.iloc[i, data.columns.get_loc("Position")] = 1
            in_position = True

        elif in_position and sell_condition:
            data.iloc[i, data.columns.get_loc("Position")] = 0
            in_position = False

        else:
            data.iloc[i, data.columns.get_loc("Position")] = 1 if in_position else 0

    data["Strategy_Return"] = data["Position"].shift(1) * data["Return"]
    data = data.dropna().copy()

    sharpe = (
        data["Strategy_Return"].mean() /
        data["Strategy_Return"].std()
    ) * (252 ** 0.5)

    buy_hold_sharpe = (
        data["Return"].mean() /
        data["Return"].std()
    ) * (252 ** 0.5)



    initial_capital = 100
    data["Strategy_Equity"] = initial_capital * (1 + data["Strategy_Return"]).cumprod()
    data["BuyHold_Equity"] = initial_capital * (1 + data["Return"]).cumprod()



    return data



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
    hidden_layer_sizes=(64, 32, 16),
    learning_rate_init=0.001,
    alpha=0.0001,
    batch_size=32,
    activation='relu',
    solver='adam',
    threshold=0.45,
    random_state=42
):
    #hidden_layer_sizes: A tuple representing the number of neurons in each hidden layer,if 
        #hideen layeer sizes too large, it may lead to overfitting, while too small may result in underfitting.
    #learning_rate_init: The initial learning rate for weight updates, a smaller value may lead
        #inside of MLclassifer, it will gave each feature a weight, and after each iteration, it will update the weight based on the error between the predicted and actual values.
    #Alpha: is a way to pevent overfitting by adding penalty 
        #if train good and test bad, can increase alpha to add more penalty to the model
        #if train and test both bad can decrease alpha to make the model more flexible
    #batch_size: Each time the model is updated, the amount of data used to calculate the error and adjust the parameters affects the stability and generalization ability of the training.
    #activation: use diffent function to manage the output of each neuron
    #sloer: the algorithm used for optimization
    #threshold: the probability bigger than threshold will be classified as 1, otherwise 0. Adjusting the threshold can help balance precision and recall based on the specific requirements of the trading strategy.
    #random_state: fixed all random processes

    data = data.copy()

    data["RSI"] = calculateRSI(data["Close"])

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Return_1D", "Return_5D", "SMA_Ratio",
        "Volatility_20", "Volume_Change", "RSI"
    ]

    data = data.dropna(subset=feature_cols + ["Target"])

    X = data[feature_cols]
    y = data["Target"]

    train_size = int(len(data) * 0.8)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    #new feature
    
    test_index = X_test.index

    scaler= StandardScaler()
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

    y_prob = model.predict_proba(X_test)[:, 1]
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



def bayesClassifier(data, threshold=0.5):
    data = data.copy()
    

    data["RSI"] = calculateRSI(data["Close"])

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Return_1D", "Return_5D", "SMA_Ratio",
        "Volatility_20", "Volume_Change", "RSI"
    ]

    data = data.dropna(subset=feature_cols + ["Target"])

    X = data[feature_cols]
    y = data["Target"]

    train_size = int(len(data) * 0.8)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    model = GaussianNB()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    # this will create a two row matrix where the first row contains the probabilities of 
    # class 0 and the second row contains the probabilities of class 1.

    threshold = np.percentile(proba, 30) 
    pred = (proba > threshold).astype(int)
    # As the percentile increases, the threshold becomes higher,
    # leading to fewer selected trading signals and making the strategy more conservative.
    # Use percentile to compute a threshold,

    # which corresponds to a specific quantile of the proba distribution (e.g., the 30th percentile).

    # This threshold acts as a cutoff point to separate "strong signals" from "weak signals".

    # Then convert probabilities into binary classification results:
    # - proba > threshold -> 1 (predict upward movement)
    # - otherwise -> 0 (predict no upward movement)
    
    # I observed that as the percentile threshold increases, the overall profit decreases.
    # This suggests that higher predicted probabilities do not lead to better performance,
    # indicating that the model may not be reliable for long-term use.

    data["Prediction"] = np.nan
    data.loc[X_test.index, "Prediction"] = pred

    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, zero_division=0))
    print("Recall:", recall_score(y_test, pred, zero_division=0))
    print("F1:", f1_score(y_test, pred, zero_division=0))

    return data

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])

    preds = pd.Series(preds, index=test.index, name="Prediction")
    combine = pd.concat([test["Target"], preds], axis=1)
    return combine

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

def confidentInterval(y_test, y_pred):
    correct = (y_test == y_pred).sum()
    n = len(y_test)

    p_value = binomtest(correct, n, p=0.5, alternative='two-sided').pvalue
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