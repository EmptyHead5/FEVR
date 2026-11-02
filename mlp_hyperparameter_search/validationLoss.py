from alpaca.data.timeframe import TimeFrame
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import dataLoad


symbol = "SPY"

feature_cols = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return_1D",
    "Return_5D",
    "SMA_Ratio",
    "Volatility_20",
    "Volume_Change",
    "RSI",
]


data = dataLoad.loadAndCleanData(
    symbol,
    timeframe=TimeFrame.Hour,
)

data = dataLoad.extraDataSet(data)
data = dataLoad.add_features(data)

data = data.dropna(
    subset=feature_cols + ["Target"]
).copy()

X = data[feature_cols]
y = data["Target"].astype(int)


# Use the first 80% for training and the last 20% for validation.
train_size = int(len(data) * 0.8)

X_train = X.iloc[:train_size]
X_validation = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_validation = y.iloc[train_size:]


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)


best_parameters = dataLoad.load_best_parameters(symbol)

model = MLPClassifier(
    hidden_layer_sizes=best_parameters["hidden_layer_sizes"],
    learning_rate_init=best_parameters["learning_rate_init"],
    alpha=best_parameters["alpha"],
    batch_size=best_parameters["batch_size"],
    activation=best_parameters["activation"],
    solver="adam",
    max_iter=1000,
    random_state=42,
    early_stopping=False,
)

model.fit(X_train_scaled, y_train)


training_probability = model.predict_proba(
    X_train_scaled
)[:, 1]

validation_probability = model.predict_proba(
    X_validation_scaled
)[:, 1]


training_loss = log_loss(
    y_train,
    training_probability,
)

validation_loss = log_loss(
    y_validation,
    validation_probability,
)

loss_difference = training_loss - validation_loss


print("Training rows:", len(X_train))
print("Validation rows:", len(X_validation))

print(f"Training Loss: {training_loss:.6f}")
print(f"Validation Loss: {validation_loss:.6f}")
print(
    "Training Loss - Validation Loss: "
    f"{loss_difference:.6f}"
)

if loss_difference > 0:
    print("Acceptance criterion passed.")
else:
    print("Acceptance criterion not passed.")