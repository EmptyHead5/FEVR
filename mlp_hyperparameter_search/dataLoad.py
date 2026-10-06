import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

from datetime import datetime, timezone

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
env_path = os.getenv("ENV_PATH")

import trainingModel

print("API key loaded:", bool(API_KEY))
print("Secret key loaded:", bool(SECRET_KEY))
print("ENV path:", env_path)
stock_data_client = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
#study from https://docs.alpaca.markets/us/docs/getting-started-with-alpaca-market-data
#https://alpaca.markets/sdks/python/trading.html#trading

def loadAndCleanData(symbol="TSLA"):
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime(2015, 1, 1),
    )

    data = stock_data_client.get_stock_bars(request_params).df

    if isinstance(data.index, pd.MultiIndex):
        available_symbols = data.index.get_level_values("symbol").unique()
        print("Available symbols:", available_symbols.tolist())
        data = data.xs(symbol, level="symbol")

    data = data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

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