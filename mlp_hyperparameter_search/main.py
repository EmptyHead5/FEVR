import dataLoad
import mlpHyperparameterSearch


def main():
    data = dataLoad.loadAndCleanData()
    data = dataLoad.extraDataSet(data)
    data = dataLoad.add_features(data)
    #load data 

    mlpHyperparameterSearch.run_search(data)
    #jump to main function in mlpHyperparameterSearch.py to run the hyperparameter search


if __name__ == "__main__":
    main()