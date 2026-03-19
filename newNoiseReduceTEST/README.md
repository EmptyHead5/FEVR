Bayes-Model-Trader Updates
2026-03-18
Improvements

Added add_features() to generate technical indicators (returns, SMA, volatility, RSI)

Updated model to use these features instead of only OHLCV data

Switched from model.predict() to predict_proba() with configurable threshold

Added evaluation metrics: accuracy, precision, recall, and F1 score

Improved prediction behavior by adjusting threshold (0.5 → 0.4)

Updated backtesting results with improved signal coverage

Summary

This version improves the model’s ability to capture upward price movements (from ~2.5% to ~99.7%) and increases backtest performance (from ~$1.8 to ~$2.4 over ~1000 days).
