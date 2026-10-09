import os
from pathlib import Path

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.data.live import StockDataStream

load_dotenv()
os.getenv("ALPACA_API_KEY")
os.getenv("ALPACA_SECRET_KEY")

BASE_URL = "https://paper-api.alpaca.markets"
# initialize the trading client
trading_client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"),  paper=True)
# initialize the historical stock data client
stock_data_client = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))


account = trading_client.get_account()



#study from https://alpaca.markets/sdks/python/trading.html
def return_All_order():
    request_params = GetOrdersRequest(
        status=QueryOrderStatus.ALL
    )

    return trading_client.get_orders(
        filter=request_params
    )


def trade_from_probability(
    symbol,
    predicted_prob,
    threshold,
    dollar_amount=100,
    execute=False,
):
    if predicted_prob >= threshold:
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            notional=dollar_amount,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        market_order = trading_client.submit_order(order_data=market_order_data) if execute else None
        return market_order

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

    market_stream.subscribe_bars(minute_bar_handler, symbol)

    print(f"Listening for {symbol} one-minute price updates...")
    market_stream.run()