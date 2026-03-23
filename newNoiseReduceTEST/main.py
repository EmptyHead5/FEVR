import yfinance as yf
import noiseReduction
import trainingModel 
import visualization 
import dataLoad
import pandas as pd

def print_results(name, df):
    final_equity = df["Strategy_Equity"].iloc[-1]
    buy_hold = df["BuyHold_Equity"].iloc[-1]

    sharpe = (
        df["Strategy_Return"].mean() /
        df["Strategy_Return"].std()
    ) * (252 ** 0.5)

    print(f"\n===== {name} =====")
    print("Final Strategy Equity:", final_equity)
    print("Buy & Hold Equity:", buy_hold)
    print("Sharpe Ratio:", sharpe)


if __name__ == "__main__":

    USE_BLSH = False
    data = dataLoad.loadAndCleanData()
    data = dataLoad.extraDataSet(data)
    data = dataLoad.add_features(data)

    visualization.closePricePlot(data)

    data = trainingModel.forestClassifer(data)

    train_size = int(len(data) * 0.8)
    startDate = data.index[train_size]

# if USE_BLSH:
#     
#     data["RSI"] = trainingModel.calculateRSI(data["Close"])
#     data = trainingModel.backtestBuyLowSellHigh(data, startDate)
#     print("Backtest completed with Buy Low Sell High strategy.")
# else:
#    
#     data = trainingModel.backtest(data, startDate)
#     print("Backtest completed with original strategy.")

# output = data.loc[startDate:, [
#     "Return", "Prediction", "Position", "Strategy_Equity", "Target"
# ]]
# 原始策略

    data_base = trainingModel.backtest(data.copy(), startDate)
    data_blsh = trainingModel.backtestBuyLowSellHigh(data.copy(), startDate)

    output_base = data_base.loc[startDate:, [
        "Return", "Prediction", "Position", "Strategy_Equity", "Target"
    ]]

    output_blsh = data_blsh.loc[startDate:, [
        "Return", "Prediction", "Position", "Strategy_Equity", "Target"
    ]]

    output_base.to_csv("backtest_original.csv")
    output_blsh.to_csv("backtest_blsh.csv")

    print("Saved to backtest.csv")
    print("Last date:", data.index.max())
    print("Rows:", len(data))
    print("Hash check:", data["Close"].sum())

    print_results("Original Strategy", data_base)
    print_results("Buy Low Sell High Strategy", data_blsh)
