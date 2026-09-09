import yfinance as yf
import pandas as pd
import numpy as np

# https://www.investopedia.com/ask/answers/difference-between-simple-exponential-moving-average/
def EMA(data, alpha):
    data["EMA"] = data["Close"].ewm(alpha=alpha, adjust=False).mean()
    return data

def SMA(data, window):
    data["SMA"] = data["Close"].rolling(window=window).mean()
    return data