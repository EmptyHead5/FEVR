from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

import alpacaPaperTrade
import dataLoad


load_dotenv()

symbol = "SPY"

trading_client = TradingClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
    paper=True,
)

clock = trading_client.get_clock()

if clock.is_open:
    csv_path = (
        Path(__file__).resolve().parent
        / "mlp_csv_results"
        / symbol
        / f"{symbol}_result.csv"
    )

    data = pd.read_csv(csv_path, index_col=0)
    probabilities = data["Predicted_Prob"].dropna()

    if probabilities.empty:
        raise ValueError(f"No prediction probabilities found in {csv_path}")

    probability = float(probabilities.iloc[-1])
    threshold = dataLoad.load_best_parameters(symbol)["threshold"]

    print("Prediction data date:", probabilities.index[-1])
    print(f"Up probability: {probability:.2%}")
    print(f"Threshold: {threshold:.2%}")

    alpacaPaperTrade.trade_from_probability(
        symbol=symbol,
        predicted_prob=probability,
        threshold=threshold,
        dollar_amount=100,
        execute=True,
    )

else:
    print("Market is closed.")
