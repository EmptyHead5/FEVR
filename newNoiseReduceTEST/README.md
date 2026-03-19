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
