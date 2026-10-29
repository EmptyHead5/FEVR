from alpaca.data.timeframe import TimeFrame

import alpacaPaperTrade
import dataLoad
import trainingModel


symbol = "SPY"

clock = alpacaPaperTrade.trading_client.get_clock()

print("Market is open:", clock.is_open)


if True:
    data = dataLoad.loadAndCleanData(
        symbol,
        timeframe=TimeFrame.Minute,
    )

    data = dataLoad.extraDataSet(data)
    data = dataLoad.add_features(data)

    best_parameters = dataLoad.load_best_parameters(symbol)

    prediction = trainingModel.predict_latest(
        data,
        **best_parameters,
    )
    print("Prediction result:", prediction)
    print("Latest completed data time:", prediction["data_time"])
    print(f"Up probability: {prediction['probability']:.2%}")
    print(f"Threshold: {prediction['threshold']:.2%}")

    alpacaPaperTrade.trade_from_probability(
        symbol=symbol,
        predicted_prob=prediction["probability"],
        threshold=prediction["threshold"],
        dollar_amount=100,
        execute=True,
    )

else:
    print("Market is closed.")
