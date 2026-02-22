import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import zoneinfo
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#yfinance is a lightweight, open-source Python library for scraping global financial data provided by Yahoo Finance.
# It proves highly practical for quantitative analysis and backtesting, particularly suited for rapidly acquiring historical quotes,
#  dividend information, and stock splits to build foundational datasets.


def lodaData():    
    nowPhiox = datetime.datetime.now(zoneinfo.ZoneInfo("America/Phoenix"))
    print("Today's date:", nowPhiox)
    data = yf.download("SPY", start="2015-01-01", end=nowPhiox)
    print(data.head(), "\n")
    print("Total trading days:", len(data))
    
    return data

def extractDaysData(data, start, end):
    return data.loc[start:end]

def feature_engineering(data):
    data["Return"] = data["Close"].pct_change()
    #.pct_change() will calculate the percentage change between the current and a prior element in the "Close" column,
    #  effectively giving us the daily returns of the SPY stock. 
    # This is a common feature used in financial modeling to capture the price movement of an asset over time.
    #current_return = (current_price - previous_price) / previous_price
    

    data["Direction"] = np.where(data["Return"] > 0, 1, 0)
    #if the return is greater than 0, which mean the stock price has increased compared to the previous day,
    # so assign a value of 1 to the "Direction" column, indicating an upward movement.
    # Conversely, if the return is less than or equal to 0, it indicates a downward movement or no change,
    # and  assign a value of 0 to the "Direction" column.
    data["last_return"] = data["Return"].shift(1)
    #useing shift(1) to create a new feature called "last_return" that contains the return from the previous day.


    data = data.dropna()
    return data

def train_model(data):
    X=data[["last_return"]]
    #using only the "last_return" feature as the input for our Naive Bayes model. 
    # This means that the model will learn to predict the direction of the stock price movement based solely on the return from the previous day.
    y = data["Direction"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print("Row accuracy:", acc)
    print(f"Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    data = lodaData()
    #close -- open price
    #high -- the highest price during the day
    #low -- the lowest price during the day
    #open -- the price at which the stock opened
    #volume -- the number of shares traded during the day

    #print("2026-02-18 data:", extractDaysData(data, "2026-02-18", "2026-02-18"))
    data = feature_engineering(data)
    train_model(data)