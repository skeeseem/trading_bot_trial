import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

# 5-minute candles × 6 = 30 minutes
FUTURE_CANDLES = 6


# ============================================================
# LOAD FEATURES
# ============================================================

df = pd.read_csv("ai_features.csv")

df = df.sort_values("start_time").reset_index(drop=True)


print()
print("=" * 60)
print("PREPARING REGRESSION DATA")
print("=" * 60)

print(f"Input rows: {len(df)}")


# ============================================================
# FUTURE RETURN
# ============================================================

# Price 30 minutes into the future
future_price = df["close"].shift(-FUTURE_CANDLES)

df["future_return"] = (
    future_price / df["close"]
) - 1


# Convert to percentage for easier reading
df["future_return_pct"] = (
    df["future_return"] * 100
)


# Last rows don't have future data
df = df.dropna(
    subset=["future_return"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "ai_regression_data.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print(f"Output rows: {len(df)}")

print()
print("Future return statistics:")
print()

print(
    df["future_return_pct"].describe()
)

print()
print("Latest rows:")
print()

print(
    df[
        [
            "start_time",
            "close",
            "future_return_pct"
        ]
    ].tail(10).to_string(index=False)
)

print()
print("Saved:")
print("ai_regression_data.csv")