import yfinance as yf
import noiseReduction
import trainingModel 
import visualization 
import dataLoad
import pandas as pd



if __name__ == "__main__":

    USE_BLSH = False
    data = dataLoad.loadAndCleanData()
    data = dataLoad.extraDataSet(data)
    data = dataLoad.add_features(data)

    visualization.closePricePlot(data)

    data = trainingModel.forestClassifer(data)

    train_size = int(len(data) * 0.8)
    startDate = data.index[train_size]

    if USE_BLSH:
        
        data["RSI"] = trainingModel.calculateRSI(data["Close"])
        data = trainingModel.backtestBuyLowSellHigh(data, startDate)
        print("Backtest completed with Buy Low Sell High strategy.")
    else:
       
        data = trainingModel.backtest(data, startDate)
        print("Backtest completed with original strategy.")
    output = data.loc[startDate:, [
        "Return", "Prediction", "Position", "Strategy_Equity", "Target"
    ]]

    output.to_csv("backtest.csv")

    print("Saved to backtest.csv")
    print("Last date:", data.index.max())
    print("Rows:", len(data))
    print("Hash check:", data["Close"].sum())
