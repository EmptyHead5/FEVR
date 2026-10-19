import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import ast

from datetime import datetime, timezone

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


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

def loadAndCleanData(symbol="SPY", timeframe=TimeFrame.Day):
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=datetime(2015, 1, 1, tzinfo=timezone.utc),
        end=datetime.now(timezone.utc),
        feed=DataFeed.IEX,
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

    return data.sort_index()
def extraDataSet(data):
    data = data.copy()

    data["Tomorrow"] = data["Close"].shift(-1)

    data["Target"] = (
        data["Tomorrow"] > data["Close"]
    ).astype("float")

    data.loc[data["Tomorrow"].isna(), "Target"] = pd.NA

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

def load_best_parameters(symbol):
    csv_path = os.path.join(
    os.path.dirname(__file__),
    "mlp_csv_results",
    symbol,
    "best_params_by_sharpe.csv",
    )
    best_parameters = pd.read_csv(csv_path).iloc[0]
    
    best_parameters = {
        "hidden_layer_sizes": ast.literal_eval(
            best_parameters["hidden_layer_sizes"]
        ),
        "learning_rate_init": float(
            best_parameters["learning_rate_init"]
        ),
        "alpha": float(best_parameters["alpha"]),
        "batch_size": int(best_parameters["batch_size"]),
        "activation": str(best_parameters["activation"]),
        "threshold": float(best_parameters["threshold"]),
    }
    return best_parameters