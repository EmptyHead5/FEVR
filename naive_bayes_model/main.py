import yfinance as yf
import noiseReduction
import naive_bayes_model

def load_data():
    data = yf.download("SPY", start="2015-01-01")
    return data

if __name__ == "__main__":
    data = load_data()
    #close -- open price
    #high -- the highest price during the day
    #low -- the lowest price during the day
    #open -- the price at which the stock opened
    #volume -- the number of shares traded during the day

    #print("2026-02-18 data:", extractDaysData(data, "2026-02-18", "2026-02-18"))
    data = naive_bayes_model.feature_engineering(data)
    naive_bayes_model.train_model(data)