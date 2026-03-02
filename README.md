# Bayes-Model-Trader

A machine learning model inspired by Bayesian statistics to predict the market direction of SPY (S&P 500 ETF).
Financial markets are highly noisy, complex, and influenced by countless interacting factors.
Precisely predicting future price movements is extremely difficult due to randomness, volatility clustering, and market efficiency.

Instead of attempting to forecast exact future prices, this project focuses on predicting the probable direction of market movement with measurable confidence.

The goal is not merely to improve classification accuracy, but to translate probabilistic directional signals into a systematic trading strategy.

Ultimately, the model aims to provide actionable signals that can be evaluated through real performance metrics such as:

Strategy Return

Market Return

Sharpe Ratio

Maximum Drawdown


---

## 📌 Project Overview

Financial markets are highly noisy and non-stationary systems.  
Predicting short-term price movements is challenging due to randomness, volatility clustering, and market efficiency.

This project explores whether simple Bayesian-inspired statistical models can capture short-term directional signals in SPY.

We begin with a Gaussian Naive Bayes classifier as a baseline probabilistic model.

---
## Data Mining

Historical daily SPY (S&P 500 ETF) data is retrieved using the yfinance library:

Open

High

Low

Close

Volume

---

The dataset spans from 2015 to present and is used to construct a supervised learning problem.
## 📊 Data Source

Historical daily SPY price data is retrieved using the `yfinance` library from Yahoo Finance.

The dataset includes:

- Open price  
- High price  
- Low price  
- Close price  
- Volume  

Data spans from 2015 to present.

---

## ⚙️ Feature Engineering

To transform raw market data into a supervised learning problem, we define:

### 1️⃣ Daily Return

`Return_t = (Close_t - Close_{t-1}) / Close_{t-1}`

This measures the percentage change in closing price from the previous trading day.

This represents the daily percentage price change.

---

### 2️⃣ Direction Label (Target Variable)

`Direction_t = 1 if Return_t > 0`

`Direction_t = 0 if Return_t < 0`
- `1` → Up day  
- `0` → Down day  

---

### 3️⃣ Lagged Feature

`LastReturn_t = Return_{t-1}`

The model uses yesterday’s return to predict today’s direction.

This prevents **data leakage** and ensures only past information is used for prediction.

---

## 🧠 Model

We implement a **Gaussian Naive Bayes classifier** as a baseline probabilistic model.

Dataset splitting strategy:

- First 80% → Training set  
- Final 20% → Testing set  
- `shuffle=False` to preserve chronological order  

This respects the time-series structure of financial data.

---

## 📈 Evaluation

Performance is measured using classification accuracy on the out-of-sample test set.

Baseline Result:

**~57% accuracy**

This suggests the presence of weak short-term momentum effects in SPY.

---

## ⚠️ Risk Considerations

- Financial data is highly noisy and close to random.
- Small improvements over 50% may not translate into profitability.
- Further improvements require hyperparameter tuning and Bayesian Optimization.

---

## 🚀 Future Work

- Optimize lag structure using Bayesian Optimization  
- Introduce additional technical indicators  
- Implement rolling-window backtesting  
- Evaluate risk-adjusted performance metrics (Sharpe Ratio)

---

## 📦 Dependencies

- yfinance  
- pandas  
- numpy  
- scikit-learn
- matplotlib  

Install with:

```bash
pip install -r requirements.txt
