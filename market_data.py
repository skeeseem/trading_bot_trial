import socketio

sio = socketio.Client()

BASE_URL = "wss://ws.coinswitch.co/"
NAMESPACE = "/coinswitchx"


@sio.event(namespace=NAMESPACE)
def connect():
    print("✅ Connected to CoinSwitch!")

    sio.emit(
        "FETCH_ORDER_BOOK_CS_PRO",
        {
            "event": "subscribe",
            "pair": "BTC,INR"
        },
        namespace=NAMESPACE
    )

    print("✅ Subscribed to BTC/INR order book")
    print("⏳ Waiting for market updates...\n")


@sio.on("FETCH_ORDER_BOOK_CS_PRO", namespace=NAMESPACE)
def on_order_book(data):

    # Ignore subscription confirmation
    if "bids" not in data:
        print("Subscription response:", data)
        return

    bids = data["bids"]
    asks = data["asks"]

    print("\n" + "=" * 55)
    print("📊 LIVE BTC/INR ORDER BOOK")
    print("=" * 55)

    if bids:
        print(f"Best Bid : ₹{float(bids[0][0]):,.2f}")
        print(f"Bid Qty  : {float(bids[0][1]):,.6f}")

    if asks:
        print(f"Best Ask : ₹{float(asks[0][0]):,.2f}")
        print(f"Ask Qty  : {float(asks[0][1]):,.6f}")

    if bids and asks:
        spread = float(asks[0][0]) - float(bids[0][0])
        print(f"Spread   : ₹{spread:,.2f}")

    print("=" * 55)


@sio.event(namespace=NAMESPACE)
def disconnect():
    print("❌ Disconnected")


print("🚀 Starting live BTC/INR market feed...\n")

sio.connect(
    BASE_URL,
    namespaces=[NAMESPACE],
    transports=["websocket"],
    socketio_path="/pro/realtime-rates-socket/spot",
    wait=True,
    wait_timeout=3600
)

sio.wait()