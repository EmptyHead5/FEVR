import os

import dataLoad
import mlpHyperparameterSearch
import trainingModel
import alpacaPaperTrade

from alpaca.trading.stream import TradingStream


async def update_handler(data):
    print("\nReceived trade update:")
    print(data)


def run_trading_stream():
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("Alpaca API keys are missing.")

    trading_stream = TradingStream(api_key, secret_key, paper=True)
    trading_stream.subscribe_trade_updates(update_handler)

    print("Listening for Alpaca trade updates...")
    print("Press Ctrl+C once to stop.")

    trading_stream.run()


def get_user_choice(symbols):
    while True:
        main_choice = input("\n1. Run model and trading menu\n" \
        "2. Run Alpaca trading stream\n" \
        "3. Run Alpaca market data stream\n" \
        "4. Exit\nEnter choice: ").strip()

        if main_choice == "2":
            try:
                run_trading_stream()
            except KeyboardInterrupt:
                print("\nReturning to main menu.")
            continue

        if main_choice == "3":
            print("Running Alpaca market data stream...")
            alpacaPaperTrade.run_market_data_stream(symbols[0])
            return

        if main_choice != "1":
            print("Invalid choice.")
            continue

        search_choice = input("Run Hyperparameter Search? (y/n): ").strip().lower()

        for symbol in symbols:
            print(f"Testing {symbol}...")

            data = dataLoad.loadAndCleanData(symbol)
            data = dataLoad.extraDataSet(data)
            data = dataLoad.add_features(data)

            if search_choice == "y":
                mlpHyperparameterSearch.run_search(symbol, data)

            best_parameters = dataLoad.load_best_parameters(symbol)
            model_result = trainingModel.MLPclassifier(data.copy(), **best_parameters)
            probabilities = model_result["data"]["Predicted_Prob"].dropna()

            latest_probability = float(probabilities.iloc[-1])
            threshold = float(best_parameters["threshold"])

            print(f"Current threshold: {threshold:.2%}")
            print(f"Next-day up probability: {latest_probability:.2%}")

            while True:
                trade_choice = input("\n1. Get all orders\n" \
                "2. Create a new order\n" \
                "3. Back\n" \
                "Enter choice: ").strip()

                if trade_choice == "1":
                    orders = alpacaPaperTrade.return_All_order()
                    print("All orders:", orders)

                elif trade_choice == "2":
                    alpacaPaperTrade.trade_from_probability(symbol, latest_probability, threshold, execute=True)


                elif trade_choice == "3":
                    break
                else:
                    print("Invalid choice.")