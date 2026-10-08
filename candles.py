import os
import time
import urllib.parse
import requests
import pandas as pd

from dotenv import load_dotenv
from cryptography.hazmat.primitives.asymmetric import ed25519

load_dotenv()

API_KEY = os.getenv("COINSWITCH_API_KEY")
SECRET_KEY = os.getenv("COINSWITCH_SECRET_KEY")

BASE_URL = "https://coinswitch.co"


def sign_request(method, path, params=None):

    method = method.upper()

    if params:
        separator = "&" if "?" in path else "?"
        path += separator + urllib.parse.urlencode(params)

    decoded_path = urllib.parse.unquote_plus(path)

    epoch = str(int(time.time() * 1000))

    message = method + decoded_path + epoch

    secret = ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(SECRET_KEY)
    )

    signature = secret.sign(
        message.encode("utf-8")
    ).hex()

    headers = {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "X-AUTH-EPOCH": epoch,
    }

    return headers, decoded_path


# ==========================================
# SETTINGS
# ==========================================

INTERVAL = 5

days = 30

end_time = int(time.time() * 1000)

start_time = end_time - (
    days * 24 * 60 * 60 * 1000
)


# ==========================================
# DOWNLOAD DATA IN 24-HOUR CHUNKS
# ==========================================

all_candles = []

chunk_size = 24 * 60 * 60 * 1000

current_start = start_time

while current_start < end_time:

    current_end = min(
        current_start + chunk_size,
        end_time
    )

    print(
        f"Downloading: "
        f"{pd.to_datetime(current_start, unit='ms')} "
        f"→ "
        f"{pd.to_datetime(current_end, unit='ms')}"
    )

    params = {
        "exchange": "coinswitchx",
        "symbol": "BTC/INR",
        "interval": INTERVAL,
        "start_time": current_start,
        "end_time": current_end
    }

    headers, path = sign_request(
        "GET",
        "/trade/api/v2/candles",
        params
    )

    response = requests.get(
        BASE_URL + path,
        headers=headers,
        timeout=15
    )

    if response.status_code != 200:

        print("❌ API error:")
        print(response.text)

        break

    result = response.json()

    candles = result.get("data", [])

    print(f"Received {len(candles)} candles")

    all_candles.extend(candles)

    current_start = current_end

    time.sleep(0.5)


# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(all_candles)


if df.empty:

    print("\n❌ No candle data received.")

else:

    # Remove duplicates

    df = df.drop_duplicates(
        subset=["start_time"]
    )

    # Sort chronologically

    df = df.sort_values(
        "start_time"
    ).reset_index(drop=True)

    # Convert numeric columns

    for column in [
        "o",
        "h",
        "l",
        "c",
        "volume"
    ]:

        df[column] = pd.to_numeric(
            df[column]
        )

    # Convert timestamps

    df["start_time"] = pd.to_datetime(
        df["start_time"].astype("int64"),
        unit="ms"
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"].astype("int64"),
        unit="ms"
    )

    # Rename columns

    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close"
        }
    )

    # Save

    df.to_csv(
        "btc_inr_5m.csv",
        index=False
    )

    print("\n================================")
    print("DOWNLOAD COMPLETE")
    print("================================")

    print("Total candles:", len(df))

    print(
        "First:",
        df["start_time"].iloc[0]
    )

    print(
        "Last:",
        df["start_time"].iloc[-1]
    )

    print("\n✅ Saved btc_inr_5m.csv")