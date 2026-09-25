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

    print()


if __name__ == "__main__":
    main()
