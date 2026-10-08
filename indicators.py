import pandas as pd
import ta


# -----------------------------
# Load candle data
# -----------------------------

df = pd.read_csv("btc_inr_5m.csv")

# Make sure candles are ordered by time
df = df.sort_values("start_time").reset_index(drop=True)


# -----------------------------
# EMA
# -----------------------------

df["ema_20"] = ta.trend.EMAIndicator(
    close=df["close"],
    window=20
).ema_indicator()

df["ema_50"] = ta.trend.EMAIndicator(
    close=df["close"],
    window=50
).ema_indicator()


# -----------------------------
# RSI
# -----------------------------

df["rsi_14"] = ta.momentum.RSIIndicator(
    close=df["close"],
    window=14
).rsi()


# -----------------------------
# MACD
# -----------------------------

macd = ta.trend.MACD(
    close=df["close"],
    window_fast=12,
    window_slow=26,
    window_sign=9
)

df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_histogram"] = macd.macd_diff()


# -----------------------------
# ATR
# -----------------------------

df["atr_14"] = ta.volatility.AverageTrueRange(
    high=df["high"],
    low=df["low"],
    close=df["close"],
    window=14
).average_true_range()


# -----------------------------
# Price returns
# -----------------------------

df["return_1"] = df["close"].pct_change()

df["return_3"] = df["close"].pct_change(3)

df["return_6"] = df["close"].pct_change(6)


# -----------------------------
# Distance from EMA
# -----------------------------

df["distance_from_ema20"] = (
    (df["close"] - df["ema_20"])
    / df["ema_20"]
)

df["distance_from_ema50"] = (
    (df["close"] - df["ema_50"])
    / df["ema_50"]
)


# -----------------------------
# Remove rows where indicators
# aren't available yet
# -----------------------------

df = df.dropna().reset_index(drop=True)


# -----------------------------
# Save AI-ready dataset
# -----------------------------

df.to_csv("ai_features.csv", index=False)


# -----------------------------
# Display result
# -----------------------------

print("\n========================================")
print("AI FEATURE DATASET")
print("========================================")

print("\nRows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nLatest market data:")

columns_to_show = [
    "start_time",
    "close",
    "ema_20",
    "ema_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "atr_14"
]

print(
    df[columns_to_show]
    .tail(10)
    .to_string(index=False)
)

print("\n✅ Saved as: ai_features.csv")