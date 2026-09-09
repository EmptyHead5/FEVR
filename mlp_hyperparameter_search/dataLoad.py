import yfinance as yf
import pandas as pd
import trainingModel    

def loadAndCleanData():
    data = yf.Ticker("SPY")
    data = data.history(period="max") #from 2003-01-29 to present,total 5470 rows data
    data = data.drop(columns=["Dividends", "Stock Splits","Capital Gains"])
    # this will remove the the stock company name from the column names, leaving only the price types (e.g., 'open', 'high', 'low', 'close', 'volume')
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)
    return data

def extraDataSet(data):
    data["Tomorrow"] = data["Close"].shift(-1)
    # The shift(-1) function is used to create a new column called "Tomorrow" 
    # that contains the closing price of the next day.

    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    # The expression (data["Tomorrow"] > data["Close"]) creates a boolean Series 
    # where each value is "1" if the closing price of the next day ("Tomorrow") 
    # is greater than the current day's closing price ("Close"), 
    # and "0" otherwise.

    return data


def add_features(data):
    data = data.copy()
    data["Return_1D"] = data["Close"].pct_change()
    data["Return_5D"] = data["Close"].pct_change(5)
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_Ratio"] = data["SMA_10"] / data["SMA_20"]
    data["Volatility_20"] = data["Return_1D"].rolling(20).std()
    data["Volume_Change"] = data["Volume"].pct_change()
    data["RSI"] = trainingModel.calculateRSI(data["Close"])
    return data