import pandas as pd
import ta

print("=" * 70)
print("CREATING GAP-SAFE TECHNICAL INDICATORS")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD CLEAN DATA
# ---------------------------------------------------------

df = pd.read_csv("btc_inr_5m_clean.csv")

df["start_time"] = pd.to_datetime(df["start_time"])

df = df.sort_values(
    ["sequence_id", "start_time"]
).reset_index(drop=True)

print(f"\nRows loaded: {len(df)}")
print(f"Sequences: {df['sequence_id'].nunique()}")

# ---------------------------------------------------------
# 2. CALCULATE INDICATORS SEQUENCE BY SEQUENCE
# ---------------------------------------------------------
#
# VERY IMPORTANT:
# We calculate indicators separately for every continuous
# sequence so EMA/RSI/etc. never cross a missing-data gap.
# ---------------------------------------------------------

processed_sequences = []

for sequence_id, group in df.groupby("sequence_id", sort=True):

    group = group.copy()
    group = group.sort_values("start_time").reset_index(drop=True)

    # -----------------------------------------------------
    # EMA
    # -----------------------------------------------------

    group["ema_20"] = ta.trend.EMAIndicator(
        close=group["close"],
        window=20
    ).ema_indicator()

    group["ema_50"] = ta.trend.EMAIndicator(
        close=group["close"],
        window=50
    ).ema_indicator()

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    group["rsi_14"] = ta.momentum.RSIIndicator(
        close=group["close"],
        window=14
    ).rsi()

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    macd = ta.trend.MACD(
        close=group["close"],
        window_fast=12,
        window_slow=26,
        window_sign=9
    )

    group["macd"] = macd.macd()
    group["macd_signal"] = macd.macd_signal()
    group["macd_histogram"] = macd.macd_diff()

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    group["atr_14"] = ta.volatility.AverageTrueRange(
        high=group["high"],
        low=group["low"],
        close=group["close"],
        window=14
    ).average_true_range()

    # -----------------------------------------------------
    # PRICE RETURNS
    # -----------------------------------------------------

    group["return_1"] = group["close"].pct_change(1)
    group["return_3"] = group["close"].pct_change(3)
    group["return_6"] = group["close"].pct_change(6)

    # -----------------------------------------------------
    # DISTANCE FROM EMA
    # -----------------------------------------------------

    group["distance_from_ema20"] = (
        (group["close"] - group["ema_20"])
        / group["ema_20"]
    )

    group["distance_from_ema50"] = (
        (group["close"] - group["ema_50"])
        / group["ema_50"]
    )

    processed_sequences.append(group)

# ---------------------------------------------------------
# 3. COMBINE SEQUENCES
# ---------------------------------------------------------

df = pd.concat(
    processed_sequences,
    ignore_index=True
)

df = df.sort_values(
    ["sequence_id", "start_time"]
).reset_index(drop=True)

# ---------------------------------------------------------
# 4. REMOVE ROWS WITHOUT COMPLETE INDICATORS
# ---------------------------------------------------------

indicator_columns = [
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
    "distance_from_ema50"
]

before = len(df)

df = df.dropna(
    subset=indicator_columns
).reset_index(drop=True)

removed = before - len(df)

# ---------------------------------------------------------
# 5. VALIDATE TIME CONTINUITY
# ---------------------------------------------------------

invalid_sequences = []

for sequence_id, group in df.groupby("sequence_id"):

    differences = (
        group["start_time"]
        .diff()
        .dropna()
    )

    if not (differences == pd.Timedelta(minutes=5)).all():
        invalid_sequences.append(sequence_id)

# ---------------------------------------------------------
# 6. SAVE
# ---------------------------------------------------------

df.to_csv(
    "ai_features_clean.csv",
    index=False
)

# ---------------------------------------------------------
# 7. SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("INDICATOR DATA SUMMARY")
print("=" * 70)

print(f"\nRows before indicators : {before}")
print(f"Rows removed           : {removed}")
print(f"Rows after indicators  : {len(df)}")

print(
    f"\nSequences remaining    : "
    f"{df['sequence_id'].nunique()}"
)

print(
    f"Sequences with gaps    : "
    f"{len(invalid_sequences)}"
)

print("\nFeatures created:")

for column in indicator_columns:
    print(f"  ✓ {column}")

print("\nSaved:")
print("ai_features_clean.csv")

print("\nLast 5 rows:")

print(
    df[
        [
            "start_time",
            "close",
            "ema_20",
            "ema_50",
            "rsi_14",
            "macd",
            "atr_14"
        ]
    ].tail(5).to_string(index=False)
)

print("\n" + "=" * 70)
print("INDICATOR CREATION COMPLETE")
print("=" * 70)