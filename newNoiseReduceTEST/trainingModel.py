import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

import visualization

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score



#yfinance is a lightweight, open-source Python library for scraping global financial data provided by Yahoo Finance.
# It proves highly practical for quantitative analysis and backtesting, particularly suited for rapidly acquiring historical quotes,
#  dividend information, and stock splits to build foundational datasets.



def extractDaysData(data, start, end):
    return data.loc[start:end]


def XGBClassifierModel(data):

    X = data[["Close", "Volume", "Open", "High", "Low"]]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    baseline = precision_score(y_test, pd.Series(1, index=y_test.index))
    print("Accuracy for XGBoost Classifier:", acc)
    print("Baseline Precision (always up):", baseline)
    


def forestClassifer(data):

    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=42)
    #higer n_estimators means more trees in the forest, which can lead to better performance but also increases computational cost.
    #but it have a limitation, if the number of trees is too high, it can lead to overfitting, where the model performs well on training data but poorly on unseen data.

    #if build tree too deep, it can lead to overfitting
    #min_samples_split=100 means that a node must have at least 100 samples to be considered for splitting.
    #this will help to protect aginst overfitting 

    X = data[["Open","High","Low","Close","Volume"]]
    y = data["Target"]

    train_size = int(len(data) * 0.8)

    X_train = X[:train_size]
    X_test  = X[train_size:]

    y_train = y[:train_size]
    y_test  = y[train_size:]

    
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("Accuracy for Random Forest Classifier:",
          accuracy_score(y_test, pred))

    data.loc[X_test.index, "Prediction"] = pred
    print("Prediction sum:", pred.sum())

    return data


def backtest(data, startDate):

    data = data.dropna(subset=["Prediction"]).copy()

    
    data = data.loc[startDate:]

    data["Return"] = data["Close"].pct_change()

    data["Position"] = data["Prediction"].shift(1)

    data["Strategy_Return"] = data["Position"] * data["Return"]

    data = data.dropna()

    sharpe = (
        data["Strategy_Return"].mean() /
        data["Strategy_Return"].std()
    ) * (252 ** 0.5)

    buy_hold_sharpe = (
        data["Return"].mean() /
        data["Return"].std()
    ) * (252 ** 0.5)

    print("Buy & Hold Sharpe:", buy_hold_sharpe)
    print("My Strategy Sharpe Ratio:", sharpe)

    initial_capital = 1

    data["Strategy_Equity"] = initial_capital * (1 + data["Strategy_Return"]).cumprod()
    data["BuyHold_Equity"] = initial_capital * (1 + data["Return"]).cumprod()

    print("Strategy Final Value:", data["Strategy_Equity"].iloc[-1])
    print("Buy & Hold Final Value:", data["BuyHold_Equity"].iloc[-1])

    return data