# Bayes-Model-Trader

Bayes-Model-Trader is a machine-learning trading project that predicts the next-day market direction of SPY, the S&P 500 ETF.

The project originally explored a Bayesian classification model. After testing, the Bayesian model did not provide enough flexibility to satisfy the project requirements. The project therefore adopted scikit-learn's [`MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), which supports configurable hidden layers, learning rate, regularization, batch size, activation function, and solver.

The trained model estimates the probability that SPY will close higher on the next trading day. These probability estimates are converted into trading positions and evaluated through historical backtesting.

---

## Project Objectives

- Predict whether SPY will move upward on the next trading day.
- Compare predicted directions with actual market directions.
- Convert model predictions into systematic trading signals.
- Compare the model strategy with a buy-and-hold strategy.
- Evaluate classification performance and risk-adjusted trading performance.

---

## Data Source

Historical SPY data is downloaded with the [`yfinance`](https://pypi.org/project/yfinance/) library:

```python
data = yf.Ticker("SPY")
data = data.history(period="max")
```

The dataset contains:

- Open
- High
- Low
- Close
- Volume

The `Dividends`, `Stock Splits`, and `Capital Gains` columns are removed during data cleaning.

---

## Target Variable

The next trading day's closing price is stored in the `Tomorrow` column:

```python
data["Tomorrow"] = data["Close"].shift(-1)
```

The binary target is then created as follows:

```python
data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
```

- `Target = 1`: the next day's closing price is higher.
- `Target = 0`: the next day's closing price is not higher.

---

## Feature Engineering

The model uses the original OHLCV data together with several technical features.

| Feature | Description |
|---|---|
| `Open` | Daily opening price |
| `High` | Daily highest price |
| `Low` | Daily lowest price |
| `Close` | Daily closing price |
| `Volume` | Daily trading volume |
| `Return_1D` | One-day percentage return |
| `Return_5D` | Five-day percentage return |
| `SMA_10` | 10-day simple moving average |
| `SMA_20` | 20-day simple moving average |
| `SMA_Ratio` | Ratio between the 10-day and 20-day moving averages |
| `Volatility_20` | Rolling 20-day standard deviation of daily returns |
| `Volume_Change` | Daily percentage change in trading volume |
| `RSI` | 14-day Relative Strength Index |

Rows containing missing values in the selected model features or target are removed before training.

---

## MLP Classifier

The project uses scikit-learn's `MLPClassifier` to predict the next-day direction of SPY.

Default model parameters:

```python
hidden_layer_sizes = (64, 32)
learning_rate_init = 0.001
alpha = 0.0001
batch_size = 32
activation = "relu"
threshold = 0.5
solver = "adam"
random_state = 42
max_iter = 1000
early_stopping = True
```

These defaults can be replaced with parameters selected by the project's hyperparameter-search process.

### Time-Based Data Split

The data is divided chronologically:

- First 80%: training set
- Final 20%: testing set

The data is not randomly shuffled. This preserves time order so that earlier observations are used to predict later observations.

### Feature Scaling

The feature values are standardized with `StandardScaler`:

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the testing data.

### Probability-Based Prediction

The classifier generates the probability of an upward movement:

```python
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)
```

The output is stored in:

- `Prediction`: predicted direction, either 0 or 1
- `Predicted_Prob`: estimated probability of an upward movement

---

## Classification Evaluation

The model reports the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

The `confidentInterval()` function also performs statistical evaluation of classification accuracy:

- Two-sided binomial test against a 50% baseline
- 95% Wilson confidence interval
- Number of test observations
- Number of correct predictions
- P-value

---

## Backtesting Methods

The project includes two backtesting methods.

### 1. Prediction-Based Backtest

`backtest_from_prediction()` uses the model's binary prediction as the trading position:

```python
data["Position"] = data["Prediction"].shift(1)
data["Strategy_Return"] = data["Position"] * data["Return"]
```

The position is shifted by one trading day so that today's return uses the prediction available from the previous row. This prevents the strategy from using the current day's completed return before it is available.

### 2. Confidence-Filtered Backtest

`backtestConfidenceFilter()` opens a position only when the predicted upward probability reaches the required confidence level.

Default settings:

```python
min_prob = 0.6
stop_loss = -0.02
transaction_fee = 0.0005
```

The position is created with:

```python
data.loc[data["Predicted_Prob"] >= min_prob, "Position"] = 1
```

Therefore:

- `Position = 1` when `Predicted_Prob >= 0.6`
- `Position = 0` when `Predicted_Prob < 0.6`

The position is shifted by one day before calculating the strategy return.

---

## Stop-Loss and Transaction Fee

The confidence-filtered backtest limits a daily strategy loss when it falls below the stop-loss threshold:

```python
if data.loc[i, "Strategy_Return"] < stop_loss:
    data.loc[i, "Strategy_Return"] = stop_loss - transaction_fee
```

With the default values, a return below `-2%` is replaced with:

```text
-0.02 - 0.0005 = -0.0205
```

In the current implementation, the transaction fee is deducted only when the stop-loss condition is triggered. It is not deducted from every normal buy or sell operation.

The stop-loss reference used by the project is the [CME Group 2% Rule](https://www.cmegroup.com/education/courses/trade-and-risk-management/the-2-percent-rule).

---

## Equity Calculation

Both backtests begin with an initial capital of `$100`:

```python
initial_capital = 100
```

Strategy equity is calculated with compounded strategy returns:

```python
data["Strategy_Equity"] = initial_capital * (1 + data["Strategy_Return"]).cumprod()
```

Buy-and-hold equity is calculated with compounded SPY returns:

```python
data["BuyHold_Equity"] = initial_capital * (1 + data["Return"]).cumprod()
```

---

## Sharpe Ratio

The project calculates the annualized Sharpe Ratio using daily strategy returns:

```python
daily_rf = annual_rf / 252

sharpe = (
    (df["Strategy_Return"].mean() - daily_rf)
    / df["Strategy_Return"].std()
) * (252 ** 0.5)
```

The default annual risk-free rate is:

```python
annual_rf = 0.0376
```

This represents a 3.76% annual risk-free rate based on the [1-Year Treasury Bill Secondary Market Rate from FRED](https://fred.stlouisfed.org/series/RIFSGFSY01NA).

The annual rate is divided by 252 to estimate the daily risk-free rate. The daily Sharpe Ratio is multiplied by `sqrt(252)` to annualize it.

If the standard deviation of the strategy returns is zero or missing, the function returns `None` to avoid invalid division.

---

## Important Output Columns

| Column | Description |
|---|---|
| `Target` | Actual next-day market direction |
| `Prediction` | Direction predicted by the MLP classifier |
| `Predicted_Prob` | Probability of an upward movement |
| `Return` | Daily SPY percentage return |
| `Position` | Strategy position, either 0 or 1 |
| `Strategy_Return` | Daily return generated by the strategy |
| `Strategy_Equity` | Compounded value of the model strategy |
| `BuyHold_Equity` | Compounded value of buying and holding SPY |

---

## Project Workflow

```text
Download Historical SPY Data
              ↓
Clean the Original Dataset
              ↓
Create Tomorrow and Target Columns
              ↓
Create Technical Features
              ↓
Remove Rows with Missing Model Data
              ↓
Split the Data into 80% Training and 20% Testing
              ↓
Standardize the Feature Values
              ↓
Train the MLP Classifier
              ↓
Generate Predictions and Probabilities
              ↓
Run Prediction or Confidence-Filtered Backtest
              ↓
Calculate Strategy Equity and Buy-and-Hold Equity
              ↓
Calculate Classification Metrics and Sharpe Ratio
```

---

## Dependencies

- Python
- pandas
- NumPy
- yfinance
- scikit-learn
- SciPy
- statsmodels

Install the required packages with:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the main program from the project directory:

```bash
python3 main.py
```

The exact results depend on the downloaded SPY data, selected hyperparameters, probability threshold, backtest start date, and chosen backtesting method.

---

## Current Limitations

- Historical performance does not guarantee future results.
- The model is currently tested only on SPY.
- The backtest does not model bid-ask spread or market slippage.
- The transaction fee is currently applied only when the stop-loss condition is triggered.
- The stop-loss implementation caps the daily backtested return; it does not simulate an intraday stop order.
- The model uses one chronological 80/20 split rather than rolling or walk-forward validation.
- Market conditions can change, so model performance may change over time.

---

## Future Work

- Integrate the model and trading strategy with external trading platforms such as Alpaca.
- Use real-time market data to generate trading signals.
- Support paper trading and future live-trading evaluation.

---

