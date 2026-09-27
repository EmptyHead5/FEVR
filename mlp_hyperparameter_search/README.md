# Bayes-Model-Trader Updates

## 2026-03-18

### Improvements

- Added `add_features()` to generate technical indicators:
  - `Return_1D`, `Return_5D`
  - `SMA_10`, `SMA_20`, `SMA_Ratio`
  - `Volatility_20`, `Volume_Change`, `RSI`
- Updated model to use new `feature_cols` instead of only OHLCV data
- Replaced `model.predict()` with `predict_proba()` and applied threshold (0.4)
- Added evaluation metrics: accuracy, precision, recall, and F1 score
- Stored predictions in `data["Prediction"]` aligned with test index
- Updated pipeline to run feature engineering before training

### Summary

- Increased capture of upward price movements from ~2.5% to ~99.7%
- Improved backtest performance: $1 → ~$2.4 over ~1000 days (previously ~$1.8)


## 2026-03-23

### Improvements

- Integrated James's `backtestBuyLowSellHigh()` trading strategy into the backtesting module
- Added comparison output between:
  - Original strategy
  - Buy Low Sell High strategy

### Summary

- Enabled direct performance comparison between baseline and enhanced trading strategies
- Introduced more selective entry and exit signals based on combined model and technical conditions


## 2026-04-11

### Improvements

- Implemented MLP classifier in `trainingModel.py`
- Integrated MLP pipeline into `main.py`
- Added hyperparameter search using `generate_param_list()` (1728 combinations)
- Added confidence interval and p-value calculation using `trainingModel.confidentInterval`
- Extended backtesting to support model-based predictions
- Saved hyperparameter search results to CSV for further analysis



## 2026-04-11

### Improvements

- Implemented MLP classifier in `trainingModel.py`
- Integrated MLP pipeline into `main.py`
- Added hyperparameter search using `generate_param_list()` (1728 combinations)
- Added confidence interval and p-value calculation using `trainingModel.confidentInterval`
- Saved hyperparameter search results to CSV for further analysis

  ### Summary

- Identified best MLP configuration: `(32, 16)` with optimized parameters
- Achieved strong performance with high recall (~97%) and improved final equity (~$365)
- Results are statistically significant based on hypothesis testing
  
## 2026-09-08

### Improvements

- Simplified `main.py` to make the main workflow cleaner and easier to read
- Separated the MLP hyperparameter search logic into `mlpHyperparameterSearch.py`
- Added comments to the key MLP hyperparameter search code for better readability
- Added comments to important parts of `trainingModel.py`
- Improved code organization by separating the main execution file from the hyperparameter search module
- Made it easier to pass optimized hyperparameters into the MLP classifier workflow

### Summary

- `main.py` now focuses mainly on running the workflow
- `mlpHyperparameterSearch.py` contains the core hyperparameter search process
- `trainingModel.py` is easier to understand with added explanations
- The project structure is cleaner and easier to maintain for future integration

  ## 2026-09-16

### Improvements

- Added a `stop_loss` parameter to `backtestConfidenceFilter()`
- Added stop-loss logic to limit large daily strategy losses
- Used `data.loc[data["Strategy_Return"] < stop_loss, "Strategy_Return"] = stop_loss` to cap losses when `Strategy_Return` is lower than `-0.02`
- Updated CSV column names to make the output files easier to read
- Renamed important CSV columns such as `Predicted_Prob`, `Position`, `Strategy_Return`, `Strategy_Equity`, and `Target`
- Improved readability of the backtest output for future project explanation
- Recorded the updated final backtest result after applying the stop-loss logic

  ## 2026-09-27

### Improvements

- Integrated Alpaca market data into the project
- Replaced the previous market data source with Alpaca IEX data
- Tested the MLP classifier and backtesting process using the new Alpaca dataset
- Found that the original 49 hyperparameter combinations could no longer meet the performance requirement
- The Sharpe Ratio decreased to approximately `1.2` after changing the data source
- Expanded `generate_param_list()` to test 360 hyperparameter combinations
- Added more hidden-layer sizes, learning rates, alpha values, and batch sizes

### Summary

- Successfully connected the project to Alpaca market data
- Increased the hyperparameter search from 49 to 360 combinations
- Improved the Sharpe Ratio from approximately `1.2` to `1.97`
- The new parameter search helped the MLP classifier adapt to the Alpaca dataset

### Summary

- `backtestConfidenceFilter()` now includes `stop_loss=-0.02`
- The stop-loss rule assumes that during regular U.S. stock market hours, the strategy can sell when the loss is larger than 2%
- The updated stop-loss logic produced:
### Return Conclusion

The strategy's total return was:

```text
($133.80 / $100.00) - 1 = 33.80%
```

The compound annualized return was approximately:

```text
27.21%
```

The buy-and-hold annualized return was approximately:

```text
18.62%
```

The strategy earned approximately `$10.85` more than buy-and-hold:

```text
$133.80 - $122.96 = $10.85
```

Therefore, the strategy achieved a strong return during the backtest period. It outperformed buy-and-hold while maintaining a Sharpe Ratio of approximately `1.97`.

```text
Final equity from rerun: 702.4308675111275
Sharpe from rerun: 1.7930117074616616
