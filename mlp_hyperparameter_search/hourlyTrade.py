import time

from alpaca.data.timeframe import TimeFrame

import alpacaPaperTrade
import dataLoad
import trainingModel


symbol = "SPY"

clock = alpacaPaperTrade.trading_client.get_clock()

print("Market is open:", clock.is_open, flush=True)


if True:  # clock.is_open:

    # 1. Load market data
    start = time.time()
    print("Loading minute data...", flush=True)

    data = dataLoad.loadAndCleanData(
        symbol,
        timeframe=TimeFrame.Minute,
    )

    print(f"Market data loaded: {time.time() - start:.2f}s", flush=True)


    # 2. Feature engineering
    start = time.time()
    print("Creating features...", flush=True)

    data = dataLoad.extraDataSet(data)
    data = dataLoad.add_features(data)

    print(f"Features completed: {time.time() - start:.2f}s", flush=True)


    # 3. Load best parameters
    start = time.time()
    print("Loading best parameters...", flush=True)

    best_parameters = dataLoad.load_best_parameters(symbol)

    print("Best parameters:", best_parameters, flush=True)
    print(f"Parameters loaded: {time.time() - start:.2f}s", flush=True)


    # 4. Prediction
    start = time.time()
    print("Starting prediction...", flush=True)

    prediction = trainingModel.predict_latest(
        data,
        **best_parameters,
    )

    print(f"Prediction completed: {time.time() - start:.2f}s", flush=True)

    print("Prediction result:", prediction)
    print("Latest completed data time:", prediction["data_time"])
    print(f"Up probability: {prediction['probability']:.2%}")
    print(f"Threshold: {prediction['threshold']:.2%}")


    # 5. Paper Trading
    start = time.time()
    print("Starting Paper Trading...", flush=True)

    alpacaPaperTrade.trade_from_probability(
        symbol=symbol,
        predicted_prob=prediction["probability"],
        threshold=prediction["threshold"],
        dollar_amount=100,
        execute=True,
    )

    print(f"Paper Trading completed: {time.time() - start:.2f}s", flush=True)


else:
    print("Market is closed.")
