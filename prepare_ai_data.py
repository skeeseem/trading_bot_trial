import pandas as pd


# ==========================================
# Load our indicator dataset
# ==========================================

df = pd.read_csv("ai_features.csv")

df = df.sort_values("start_time").reset_index(drop=True)


# ==========================================
# Calculate future price movement
# ==========================================

# We are using 5-minute candles.
# 3 candles = 15 minutes into the future.

future_return = (
    df["close"].shift(-3) / df["close"] - 1
)


# ==========================================
# Create BUY / SELL / WAIT labels
# ==========================================

THRESHOLD = 0.0015     # 0.15%


def create_label(return_value):

    if return_value > THRESHOLD:
        return "BUY"

    elif return_value < -THRESHOLD:
        return "SELL"

    else:
        return "WAIT"


df["future_return"] = future_return

df["target"] = df["future_return"].apply(create_label)


# ==========================================
# Remove rows where future data doesn't exist
# ==========================================

df = df.dropna(subset=["future_return"]).reset_index(drop=True)


# ==========================================
# Show label distribution
# ==========================================

print("\n========================================")
print("AI TARGET DATASET")
print("========================================")

print("\nTotal rows:", len(df))

print("\nTarget distribution:")

print(
    df["target"]
    .value_counts()
)


print("\nPercentage:")

print(
    (df["target"].value_counts(normalize=True) * 100)
    .round(2)
)


# ==========================================
# Show sample
# ==========================================

print("\nLatest rows:")

print(
    df[
        [
            "start_time",
            "close",
            "future_return",
            "target"
        ]
    ].tail(10).to_string(index=False)
)


# ==========================================
# Save dataset
# ==========================================

df.to_csv(
    "ai_training_data.csv",
    index=False
)

print("\n✅ Saved:")
print("ai_training_data.csv")