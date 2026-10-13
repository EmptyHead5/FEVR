from pathlib import Path

import pandas as pd

import alpacaPaperTrade
import dataLoad


symbol = "SPY"
clock = alpacaPaperTrade.get_trading_client().get_clock()

if clock.is_open:
    csv_path = (
        Path(__file__).resolve().parent
        / "mlp_csv_results"
        / symbol
        / f"{symbol}_result.csv"
    )

    data = pd.read_csv(csv_path, index_col=0)
    probabilities = data["Predicted_Prob"].dropna()


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