import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np

import visualization

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd



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



    initial_capital = 1
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


def forestClassifer(data, threshold=0.4):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Return_1D", "Return_5D", "SMA_Ratio",
        "Volatility_20", "Volume_Change", "RSI"
    ]
    X = data[feature_cols]
    y = data["Target"]

    train_size = int(len(data) * 0.8)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba > threshold).astype(int)

    data = data.copy()
    data["Prediction"] = np.nan
    data.loc[X_test.index, "Prediction"] = pred

    print("Threshold:", threshold)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, zero_division=0))
    print("Recall:", recall_score(y_test, pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, pred, zero_division=0))

    data["RSI"] = calculateRSI(data["Close"])
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



    initial_capital = 1
    data["Strategy_Equity"] = initial_capital * (1 + data["Strategy_Return"]).cumprod()
    data["BuyHold_Equity"] = initial_capital * (1 + data["Return"]).cumprod()

    

    return data
 

   