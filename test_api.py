import os
import time
import urllib.parse
import requests

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
        path = path + separator + urllib.parse.urlencode(params)

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


# Get BTC/INR ticker
params = {
    "exchange": "coinswitchx",
    "symbol": "BTC/INR"
}

headers, path = sign_request(
    "GET",
    "/trade/api/v2/24hr/ticker",
    params
)

response = requests.get(
    BASE_URL + path,
    headers=headers,
    timeout=10
)

print("Status:", response.status_code)
print("Response:")
print(response.text)