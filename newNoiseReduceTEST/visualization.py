import matplotlib.pyplot as plt
import pandas as pd

def closePricePlot(data):
    data.plot(y= "Close", use_index=True)
    plt.title(f"Close Price Time Series")
    plt.savefig(f"Close_plot.png")

import matplotlib.pyplot as plt

def plot_prediction(data, test_index, preds):

    plt.figure(figsize=(14,6))

    plt.plot(data.index, data["Close"], label="Close Price", linewidth=2)

    predicted_up = data.loc[test_index][preds == 1]
    predicted_down = data.loc[test_index][preds == 0]

    plt.scatter(predicted_up.index,
                predicted_up["Close"],
                marker="^",
                label="Predicted Up")

    plt.scatter(predicted_down.index,
                predicted_down["Close"],
                marker="v",
                label="Predicted Down")

    plt.title("Random Forest Predictions on SPY")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.show()
    plt.savefig("prediction_plot.png")


