import os
import time
import urllib.parse
import requests
import pandas as pd
import ta

from dotenv import load_dotenv
from cryptography.hazmat.primitives.asymmetric import ed25519
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "BTC/INR"
EXCHANGE = "coinswitchx"

STARTING_BALANCE = 100000.0

POSITION_SIZE = 0.25

FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0005

HOLDING_MINUTES = 30

PREDICTION_THRESHOLD = 0.50

CHECK_INTERVAL_SECONDS = 30

TRADE_HISTORY_FILE = "paper_trades.csv"


# ============================================================
# LOAD ENV
# ============================================================

load_dotenv()

API_KEY = os.getenv("COINSWITCH_API_KEY")
SECRET_KEY = os.getenv("COINSWITCH_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        "COINSWITCH_API_KEY or COINSWITCH_SECRET_KEY is missing from .env"
    )


# ============================================================
# PAPER BALANCE
# ============================================================

def load_balance():

    if not os.path.exists(TRADE_HISTORY_FILE):
        return STARTING_BALANCE

    try:

        trades = pd.read_csv(TRADE_HISTORY_FILE)

        if trades.empty:
            return STARTING_BALANCE

        last_balance = float(
            trades["balance_after"].iloc[-1]
        )

        return last_balance

    except Exception:

        print(
            "Could not load previous paper balance."
        )

        return STARTING_BALANCE


balance = load_balance()


# ============================================================
# AUTHENTICATION
# ============================================================

def sign_request(method, path, params=None):

    method = method.upper()

    if params:

        query_string = urllib.parse.urlencode(params)

        separator = "?" if "?" not in path else "&"

        path = path + separator + query_string

    decoded_path = urllib.parse.unquote_plus(path)

    epoch = str(
        int(time.time() * 1000)
    )

    message = (
        method
        + decoded_path
        + epoch
    )

    private_key = (
        ed25519.Ed25519PrivateKey
        .from_private_bytes(
            bytes.fromhex(SECRET_KEY)
        )
    )

    signature = private_key.sign(
        message.encode("utf-8")
    ).hex()

    headers = {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "X-AUTH-EPOCH": epoch,
    }

    return headers, decoded_path


# ============================================================
# GET CANDLES
# ============================================================

def get_candles():

    try:

        end_time = int(
            time.time() * 1000
        )

        start_time = (
            end_time
            - (24 * 60 * 60 * 1000)
        )

        params = {
            "exchange": EXCHANGE,
            "symbol": SYMBOL,
            "interval": "5",
            "start_time": str(start_time),
            "end_time": str(end_time),
        }

        headers, signed_path = sign_request(
            "GET",
            "/trade/api/v2/candles",
            params
        )

        url = (
            "https://coinswitch.co"
            + signed_path
        )

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        print(
            f"API candle request: HTTP {response.status_code}"
        )

        response.raise_for_status()

        result = response.json()

        if isinstance(result, dict):

            candles = result.get(
                "data",
                []
            )

        elif isinstance(result, list):

            candles = result

        else:

            print(
                "Unexpected API response format."
            )

            return pd.DataFrame()

        if isinstance(candles, dict):

            if "data" in candles:

                candles = candles["data"]

            elif "candles" in candles:

                candles = candles["candles"]

            else:

                candles = []

        if not isinstance(candles, list):

            print(
                "Candle data is not a list."
            )

            return pd.DataFrame()

        print(
            f"Raw candles received: {len(candles)}"
        )

        if len(candles) == 0:

            return pd.DataFrame()

        parsed = []

        for candle in candles:

            if not isinstance(candle, dict):
                continue

            try:

                timestamp = candle.get(
                    "end_time"
                )

                if timestamp is None:

                    timestamp = candle.get(
                        "start_time"
                    )

                if timestamp is None:
                    continue

                timestamp = int(timestamp)

                if timestamp < 100000000000:

                    timestamp *= 1000

                row = {
                    "start_time": timestamp,
                    "open": float(candle["o"]),
                    "high": float(candle["h"]),
                    "low": float(candle["l"]),
                    "close": float(candle["c"]),
                    "volume": float(candle["volume"]),
                }

                parsed.append(row)

            except (
                KeyError,
                TypeError,
                ValueError
            ):

                continue

        if not parsed:

            print(
                "No valid candles parsed."
            )

            return pd.DataFrame()

        df = pd.DataFrame(parsed)

        df["start_time"] = pd.to_datetime(
            df["start_time"],
            unit="ms"
        )

        df = (
            df
            .sort_values("start_time")
            .drop_duplicates(
                "start_time"
            )
            .reset_index(drop=True)
        )

        print(
            f"Valid candles parsed: {len(df)}"
        )

        if not df.empty:

            print(
                f"Latest candle: "
                f"{df['start_time'].iloc[-1]}"
            )

            print(
                f"Latest close: "
                f"₹{df['close'].iloc[-1]:,.2f}"
            )

        return df

    except requests.exceptions.HTTPError as e:

        print(
            "HTTP error while getting candles:",
            e
        )

        return pd.DataFrame()

    except Exception as e:

        print(
            "ERROR getting candles:",
            e
        )

        return pd.DataFrame()


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(df):

    df = df.copy()

    if len(df) < 60:

        return pd.DataFrame()

    df["ema_20"] = (
        ta.trend.EMAIndicator(
            close=df["close"],
            window=20
        ).ema_indicator()
    )

    df["ema_50"] = (
        ta.trend.EMAIndicator(
            close=df["close"],
            window=50
        ).ema_indicator()
    )

    df["rsi_14"] = (
        ta.momentum.RSIIndicator(
            close=df["close"],
            window=14
        ).rsi()
    )

    macd = ta.trend.MACD(
        close=df["close"],
        window_fast=12,
        window_slow=26,
        window_sign=9
    )

    df["macd"] = macd.macd()

    df["macd_signal"] = (
        macd.macd_signal()
    )

    df["macd_histogram"] = (
        macd.macd_diff()
    )

    df["atr_14"] = (
        ta.volatility.AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14
        ).average_true_range()
    )

    df["return_1"] = (
        df["close"].pct_change()
    )

    df["return_3"] = (
        df["close"].pct_change(3)
    )

    df["return_6"] = (
        df["close"].pct_change(6)
    )

    df["distance_from_ema20"] = (
        (df["close"] - df["ema_20"])
        / df["ema_20"]
    )

    df["distance_from_ema50"] = (
        (df["close"] - df["ema_50"])
        / df["ema_50"]
    )

    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print()
    print(
        "Loading historical training data..."
    )

    df = pd.read_csv(
        "ai_regression_data_clean.csv"
    )

    print(
        f"Training rows: {len(df)}"
    )

    features = [
        "ema_20",
        "ema_50",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_14",
        "return_1",
        "return_3",
        "return_6",
        "distance_from_ema20",
        "distance_from_ema50",
        "volume",
    ]

    X = df[features]

    y = df["future_return_pct"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    print(
        "Model trained successfully."
    )

    return model, features


# ============================================================
# PAPER TRADE STATE
# ============================================================

open_trade = None

last_processed_candle = None


# ============================================================
# SAVE COMPLETED TRADE
# ============================================================

def save_completed_trade(
    trade,
    exit_price,
    raw_exit_price,
    gross_return,
    net_return,
    pnl,
    balance_after
):

    record = {
        "entry_time": trade["entry_time"],
        "exit_time": trade["exit_time"],
        "signal": trade["signal"],
        "prediction_pct": trade["prediction"],
        "entry_price": trade["entry_price"],
        "exit_price": exit_price,
        "raw_exit_price": raw_exit_price,
        "trade_value": trade["trade_value"],
        "gross_return_pct": gross_return * 100,
        "net_return_pct": net_return * 100,
        "pnl": pnl,
        "balance_after": balance_after,
    }

    new_trade = pd.DataFrame(
        [record]
    )

    if os.path.exists(
        TRADE_HISTORY_FILE
    ):

        new_trade.to_csv(
            TRADE_HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        new_trade.to_csv(
            TRADE_HISTORY_FILE,
            index=False
        )

    print()
    print(
        f"Trade saved to {TRADE_HISTORY_FILE}"
    )


# ============================================================
# OPEN PAPER TRADE
# ============================================================

def open_paper_trade(
    signal,
    prediction,
    candle
):

    global open_trade

    entry_price = float(
        candle["close"]
    )

    trade_value = (
        balance
        * POSITION_SIZE
    )

    if signal == "BUY":

        entry_price *= (
            1 + SLIPPAGE_PER_SIDE
        )

    elif signal == "SELL":

        entry_price *= (
            1 - SLIPPAGE_PER_SIDE
        )

    exit_time = (
        candle["start_time"]
        + pd.Timedelta(
            minutes=HOLDING_MINUTES
        )
    )

    open_trade = {
        "signal": signal,
        "prediction": prediction,
        "entry_price": entry_price,
        "trade_value": trade_value,
        "entry_time": candle["start_time"],
        "exit_time": exit_time,
    }

    print()
    print(
        "========================================"
    )
    print(
        "PAPER TRADE OPENED"
    )
    print(
        "========================================"
    )

    print(
        f"Signal: {signal}"
    )

    print(
        f"Prediction: {prediction:+.4f}%"
    )

    print(
        f"Entry price: ₹{entry_price:,.2f}"
    )

    print(
        f"Trade value: ₹{trade_value:,.2f}"
    )

    print(
        f"Expected exit: "
        f"{exit_time}"
    )

    print(
        "========================================"
    )


# ============================================================
# CHECK OPEN TRADE
# ============================================================

def check_open_trade(df):

    global open_trade
    global balance

    if open_trade is None:

        return

    latest_time = (
        df["start_time"].iloc[-1]
    )

    if (
        latest_time
        < open_trade["exit_time"]
    ):

        return

    future_rows = df[
        df["start_time"]
        >= open_trade["exit_time"]
    ]

    if future_rows.empty:

        return

    exit_candle = (
        future_rows.iloc[0]
    )

    raw_exit_price = float(
        exit_candle["close"]
    )

    signal = open_trade["signal"]

    if signal == "BUY":

        exit_price = (
            raw_exit_price
            * (1 - SLIPPAGE_PER_SIDE)
        )

        gross_return = (
            exit_price
            / open_trade["entry_price"]
        ) - 1

    else:

        exit_price = (
            raw_exit_price
            * (1 + SLIPPAGE_PER_SIDE)
        )

        gross_return = (
            open_trade["entry_price"]
            / exit_price
        ) - 1

    total_fees = (
        FEE_PER_SIDE * 2
    )

    net_return = (
        gross_return
        - total_fees
    )

    pnl = (
        open_trade["trade_value"]
        * net_return
    )

    balance += pnl

    # Save BEFORE clearing the trade
    save_completed_trade(
        open_trade,
        exit_price,
        raw_exit_price,
        gross_return,
        net_return,
        pnl,
        balance
    )

    print()
    print(
        "========================================"
    )
    print(
        "PAPER TRADE CLOSED"
    )
    print(
        "========================================"
    )

    print(
        f"Signal: {signal}"
    )

    print(
        f"Entry: "
        f"₹{open_trade['entry_price']:,.2f}"
    )

    print(
        f"Exit: "
        f"₹{exit_price:,.2f}"
    )

    print(
        f"Prediction: "
        f"{open_trade['prediction']:+.4f}%"
    )

    print(
        f"Gross return: "
        f"{gross_return * 100:+.4f}%"
    )

    print(
        f"Net return: "
        f"{net_return * 100:+.4f}%"
    )

    print(
        f"P&L: ₹{pnl:+,.2f}"
    )

    print(
        f"Paper balance: "
        f"₹{balance:,.2f}"
    )

    print(
        "========================================"
    )

    open_trade = None


# ============================================================
# MAIN
# ============================================================

def main():

    global last_processed_candle

    print()
    print(
        "========================================"
    )
    print(
        "AI CRYPTO PAPER TRADING BOT"
    )
    print(
        "========================================"
    )

    print(
        "REAL ORDERS: DISABLED"
    )

    print(
        f"Paper balance: ₹{balance:,.2f}"
    )

    print(
        f"Prediction threshold: "
        f"±{PREDICTION_THRESHOLD:.2f}%"
    )

    print(
        f"Holding period: "
        f"{HOLDING_MINUTES} minutes"
    )

    print(
        "========================================"
    )

    model, features = train_model()

    print()
    print(
        "Starting live paper-trading loop..."
    )

    while True:

        try:

            df = get_candles()

            if df.empty:

                print(
                    "Not enough candle data."
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            if len(df) < 60:

                print(
                    f"Not enough candle data. "
                    f"Received {len(df)}, "
                    f"need at least 60."
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            features_df = create_features(
                df
            )

            if features_df.empty:

                print(
                    "Could not create indicators."
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            # ------------------------------------------------
            # Check existing trade
            # ------------------------------------------------

            check_open_trade(
                features_df
            )

            # ------------------------------------------------
            # Use latest candle
            # ------------------------------------------------

            candle = (
                features_df.iloc[-1]
            )

            candle_time = (
                candle["start_time"]
            )

            # Don't process the same candle twice
            if (
                last_processed_candle
                is not None
                and candle_time
                <= last_processed_candle
            ):

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            last_processed_candle = (
                candle_time
            )

            # ------------------------------------------------
            # AI prediction
            # ------------------------------------------------

            X_live = pd.DataFrame(
                [
                    candle[
                        features
                    ].values
                ],
                columns=features
            )

            prediction = float(
                model.predict(
                    X_live
                )[0]
            )

            price = float(
                candle["close"]
            )

            # ------------------------------------------------
            # Signal
            # ------------------------------------------------

            if (
                prediction
                >= PREDICTION_THRESHOLD
            ):

                signal = "BUY"

            elif (
                prediction
                <= -PREDICTION_THRESHOLD
            ):

                signal = "SELL"

            else:

                signal = "WAIT"

            print()
            print(
                "----------------------------------------"
            )

            print(
                f"Candle: {candle_time}"
            )

            print(
                f"BTC/INR: "
                f"₹{price:,.2f}"
            )

            print(
                f"EMA20: "
                f"₹{candle['ema_20']:,.2f}"
            )

            print(
                f"EMA50: "
                f"₹{candle['ema_50']:,.2f}"
            )

            print(
                f"RSI: "
                f"{candle['rsi_14']:.2f}"
            )

            print(
                f"AI prediction: "
                f"{prediction:+.4f}%"
            )

            print(
                f"Signal: {signal}"
            )

            print(
                f"Paper balance: "
                f"₹{balance:,.2f}"
            )

            # ------------------------------------------------
            # Open trade
            # ------------------------------------------------

            if open_trade is None:

                if signal in [
                    "BUY",
                    "SELL"
                ]:

                    open_paper_trade(
                        signal,
                        prediction,
                        candle
                    )

            else:

                print(
                    "Existing paper trade is open."
                )

            print(
                "----------------------------------------"
            )

            time.sleep(
                CHECK_INTERVAL_SECONDS
            )

        except KeyboardInterrupt:

            print()
            print(
                "Paper trading stopped."
            )

            print(
                f"Current paper balance: "
                f"₹{balance:,.2f}"
            )

            break

        except Exception as e:

            print()
            print(
                "ERROR in main loop:",
                e
            )

            time.sleep(
                CHECK_INTERVAL_SECONDS
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()