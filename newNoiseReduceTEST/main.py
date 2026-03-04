import yfinance as yf
import noiseReduction
import trainingModel 
import visualization 
import dataLoad
import pandas as pd



if __name__ == "__main__":

    data = dataLoad.loadAndCleanData()
    data = dataLoad.extraDataSet(data)

    visualization.closePricePlot(data)

    # Train XGBoost model
    #trainingModel.XGBClassifierModel(data)
    
    # Random Forest baseline
    
    startDate = "2021-12-01"
    data = trainingModel.forestClassifer(data)
    data = trainingModel.backtest(data, startDate)

    print(data.head())
 


    output = data.loc[startDate:,["Return", "Prediction", "Position", "Strategy_Equity", "Target"]]

    output.to_csv("backtest.csv")

    print("Saved to backtest.csv")
    #Return is today's price change
    #Targer is whether the price will go up tomorrow (1) or not (0)

    print("Last date:", data.index.max())
    print("Rows:", len(data))
    print("Hash check:", data["Close"].sum())