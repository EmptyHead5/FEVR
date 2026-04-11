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
