import yfinance as yf
import pandas as pd

def loadAndCleanData():
    data = yf.Ticker("SPY")
    data = data.history(period="max") #from 1003-01-29 to present,total 5470 rows data
    data = data.drop(columns=["Dividends", "Stock Splits","Capital Gains"])
    # this will remove the the stock company name from the column names, leaving only the price types (e.g., 'open', 'high', 'low', 'close', 'volume')
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)
    return data

def extraDataSet(data):
    data["Tomorrow"] = data["Close"].shift(-1)
    # The shift(-1) function is used to create a new column called "Tomorrow" that contains the closing price of the next day.

    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    
    # The expression (data["Tomorrow"] > data["Close"]) creates a boolean Series where each value is "1" if the closing price of the next day ("Tomorrow") is greater than the current day's closing price ("Close"), 
    # and "0" otherwise.
    return data

