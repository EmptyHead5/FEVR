import pandas as pd
import yfinance as yf

import dataLoad
import mlpHyperparameterSearch


def main():
    data = dataLoad.loadAndCleanData()
    data = dataLoad.extraDataSet(data)
    data = dataLoad.add_features(data)
    #load data 

    mlpHyperparameterSearch.run_search(data)
    #jump to main function in mlpHyperparameterSearch.py to run the hyperparameter search

    #following codes are only for meeing with instructor 
    df=pd.read_csv("mlp_csv_results/backtest_mlp.csv")
    print("useing 80% of data for training, 20% for testing")
    print("data total days: ", len(data))
    print("data training days: ",len(data.index) - len(df.index) )
    print("data tesing days: ",len(df.index) )


    # data["Return"] = data["Close"].pct_change()
    # data["Position"] = data["Prediction"].shift(1)
    # data["Strategy_Return"] = data["Position"] * data["Return"]

    print()


if __name__ == "__main__":
    main()