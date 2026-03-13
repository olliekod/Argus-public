"""IBKR smoke test (event-loop safe for Python 3.14+)."""

import asyncio
from ib_async import IB
from ib_async.contract import Stock
from ib_async.order import LimitOrder

from src.core.asyncio_compat import ensure_event_loop

async def main():
    ensure_event_loop()
    ib = IB()
    await ib.connectAsync("127.0.0.1", 4002, clientId=19)

    contract = Stock("IBIT", "SMART", "USD", primaryExchange="ARCA")
    qualified = await ib.qualifyContractsAsync(contract)
    print("Qualified:", qualified[0])

    order = LimitOrder("BUY", 1, 1.00)  # absurdly low, won't fill
    order.tif = "DAY"
    trade = ib.placeOrder(contract, order)
    print("Placed orderId:", trade.order.orderId)

    await asyncio.sleep(2)
    status = trade.orderStatus.status
    print("Status after 2s:", status)

    # Only cancel if it actually exists in an active state
    if status not in ("Cancelled", "Inactive", "Rejected", "Filled"):
        ib.cancelOrder(order)
        print("Cancel requested.")
        await asyncio.sleep(2)
        print("Final status:", trade.orderStatus.status)
    else:
        print("No cancel needed.")

    ib.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
