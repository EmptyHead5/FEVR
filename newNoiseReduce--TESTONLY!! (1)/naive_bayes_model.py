import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#yfinance is a lightweight, open-source Python library for scraping global financial data provided by Yahoo Finance.
# It proves highly practical for quantitative analysis and backtesting, particularly suited for rapidly acquiring historical quotes,
#  dividend information, and stock splits to build foundational datasets.



def extractDaysData(data, start, end):
    return data.loc[start:end]

def feature_engineering(data):
    data["Return"] = data["Close"].pct_change()
    #.pct_change() will calculate the percentage change between the current and a prior element in the "Close" column,
    #  effectively giving us the daily returns of the SPY stock. 
    # This is a common feature used in financial modeling to capture the price movement of an asset over time.
    #current_return = (current_price - previous_price) / previous_price
    

    data["future_3_return"] = (
    data["Return"]
    .rolling(3)
    .sum()
    .shift(-3)
    )

    data["Direction"] = np.where(data["future_3_return"] > 0, 1, 0)
    #if the return is greater than 0, which mean the stock price has increased compared to the previous day,
    # so assign a value of 1 to the "Direction" column, indicating an upward movement.
    # Conversely, if the return is less than or equal to 0, it indicates a downward movement or no change,
    # and  assign a value of 0 to the "Direction" column.
    data["last_return"] = data["Return"].shift(1)
    data["ret_3"] = data["Return"].shift(1).rolling(3).sum()
    data["ret_5"] = data["Return"].shift(1).rolling(5).sum()
    #useing shift(1) to create a new feature called "last_return" that contains the return from the previous day.


    data = data.dropna()
    return data

def train_model(data):

    X = data[["last_return", "ret_3", "ret_5"]]
    y = data["Direction"]

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
    print("Accuracy:", acc)
    print("Baseline:", y_test.mean())

def walk_forward_validation(data, train_size=756, test_size=63):

    X = data[["last_return", "ret_3", "ret_5"]]
    y = data["Direction"]

    total_samples = len(data)

    all_predictions = []
    all_true = []

    start = 0

    while start + train_size + test_size <= total_samples:

        train_start = start
        train_end = start + train_size
        test_end = train_end + test_size

        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]

        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

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

        all_predictions.extend(predictions)
        all_true.extend(y_test)

        start += test_size  # 滚动窗口

    return np.array(all_true), np.array(all_predictions)