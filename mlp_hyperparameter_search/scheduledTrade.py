from pathlib import Path

import pandas as pd

import alpacaPaperTrade
import dataLoad


symbol = "SPY"
clock = alpacaPaperTrade.trading_client.get_clock()


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
    training_data_end_date = probabilities.index[-1]

    best_parameters = dataLoad.load_best_parameters(symbol)
    threshold = best_parameters["threshold"]

    print(f"Training data end date: {training_data_end_date}")
    print(f"Best parameters: {best_parameters}")
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
