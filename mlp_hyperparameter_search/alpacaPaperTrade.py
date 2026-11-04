import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from alpaca.data.live import StockDataStream
from alpaca.trading.enums import PositionSide


load_dotenv()

BASE_URL = "https://paper-api.alpaca.markets"

trading_client = TradingClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    paper=True
)

stock_data_client = StockHistoricalDataClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY")
)


# Study from https://alpaca.markets/sdks/python/trading.html
def return_All_order():

    request_params = GetOrdersRequest(
        status=QueryOrderStatus.ALL
    )

    orders = trading_client.get_orders(
        filter=request_params
    )

    order_data = []

    for order in orders:
        order_data.append({
            "order_id": str(order.id),
            "symbol": order.symbol,
            "side": order.side,
            "type": order.type,
            "status": order.status,
            "qty": order.qty,
            "notional": order.notional,
            "filled_qty": order.filled_qty,
            "filled_avg_price": order.filled_avg_price,
            "submitted_at": order.submitted_at,
            "filled_at": order.filled_at,
        })

    output_path = Path(__file__).resolve().parent / "alpaca_order_history.csv"

    pd.DataFrame(order_data).to_csv(
        output_path,
        index=False
    )

    print("Alpaca orders saved to:", output_path)

    return orders


def trade_from_probability(
    symbol,
    predicted_prob,
    threshold,
    dollar_amount=100,
    execute=False,
    sell_threshold=None,
):
    buy = predicted_prob >= threshold
    sell = sell_threshold is not None and predicted_prob <= sell_threshold
    if buy:

        market_order_data = MarketOrderRequest(
            symbol=symbol,
            notional=dollar_amount,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        market_order = (
            trading_client.submit_order(order_data=market_order_data)
            if execute
            else None
        )
        return market_order
    if sell:
        for position in trading_client.get_all_positions():
            if position.symbol == symbol and position.side == PositionSide.LONG:
                print(f"Sell {symbol}: probability = {predicted_prob:.2%}")

                if execute:
                    return trading_client.close_position(symbol)

        print(f"No position in {symbol}. Nothing to sell.")
        return None
        print(
    f"HOLD {symbol}: "
    f"probability {predicted_prob:.2%}, "
    f"sell threshold {sell_threshold:.2%}, "
    f"buy threshold {threshold:.2%}"
    )
    return None

async def minute_bar_handler(bar):
    change = (bar.close - bar.open) / bar.open

    print("Symbol:", bar.symbol)
    print("Open:", bar.open)
    print("Close:", bar.close)
    print(f"One-minute return: {change:.2%}")


def run_market_data_stream(symbol):

    market_stream = StockDataStream(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY")
    )

    market_stream.subscribe_bars(
        minute_bar_handler,
        symbol
    )

    print(f"Listening for {symbol} one-minute price updates...")

    market_stream.run()

if __name__ == "__main__":
    orders = return_All_order()

    print("Total orders:", len(orders))

    for order in orders:
        print(
            order.symbol,
            order.side,
            order.status,
            order.filled_avg_price
        )
