import os
import dataLoad
import mlpHyperparameterSearch




def main():
    symbols = [
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "AAPL",
    ]
    #360 x 5 = 1800
    for symbol in symbols:
        print(f"Testing {symbol}...")

        data = dataLoad.loadAndCleanData(symbol)
        data = dataLoad.extraDataSet(data)
        data = dataLoad.add_features(data)

        mlpHyperparameterSearch.run_search(symbol, data)
        print(f"Finished testing {symbol}.")

        #jump to main function in mlpHyperparameterSearch.py to run the hyperparameter search



if __name__ == "__main__":
    main()