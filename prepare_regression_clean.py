import pandas as pd

FUTURE_MINUTES = 30
FUTURE_CANDLES = 6

print("=" * 70)
print("CREATING GAP-SAFE 30-MINUTE REGRESSION DATA")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD INDICATOR DATA
# ---------------------------------------------------------

df = pd.read_csv("ai_features_clean.csv")

df["start_time"] = pd.to_datetime(df["start_time"])

df = df.sort_values(
    ["sequence_id", "start_time"]
).reset_index(drop=True)

print(f"\nRows loaded: {len(df)}")
print(f"Sequences: {df['sequence_id'].nunique()}")

# ---------------------------------------------------------
# 2. CREATE FUTURE PRICE WITHIN EACH SEQUENCE
# ---------------------------------------------------------
#
# shift(-6) means 6 candles ahead.
#
# Since every sequence is guaranteed to contain only
# 5-minute candles:
#
# 6 candles = exactly 30 minutes.
#
# We still explicitly verify the timestamp.
# ---------------------------------------------------------

df["future_price"] = (
    df.groupby("sequence_id")["close"]
    .shift(-FUTURE_CANDLES)
)

df["future_time"] = (
    df.groupby("sequence_id")["start_time"]
    .shift(-FUTURE_CANDLES)
)

# ---------------------------------------------------------
# 3. VERIFY THE FUTURE TIME IS EXACTLY 30 MINUTES
# ---------------------------------------------------------

df["future_time_difference"] = (
    df["future_time"] - df["start_time"]
)

valid_target = (
    df["future_time_difference"]
    == pd.Timedelta(minutes=FUTURE_MINUTES)
)

valid_count = int(valid_target.sum())
invalid_count = int(
    df["future_price"].notna().sum() - valid_count
)

print("\nTarget validation:")

print(
    f"Valid 30-minute targets : {valid_count}"
)

print(
    f"Invalid targets         : {invalid_count}"
)

# ---------------------------------------------------------
# 4. KEEP ONLY VALID TARGETS
# ---------------------------------------------------------

df = df[valid_target].copy()

# ---------------------------------------------------------
# 5. CALCULATE FUTURE RETURN
# ---------------------------------------------------------

df["future_return"] = (
    df["future_price"] / df["close"] - 1
)

df["future_return_pct"] = (
    df["future_return"] * 100
)

# ---------------------------------------------------------
# 6. REMOVE HELPER COLUMNS
# ---------------------------------------------------------

df = df.drop(
    columns=[
        "future_time",
        "future_time_difference"
    ]
)

df = df.reset_index(drop=True)

# ---------------------------------------------------------
# 7. BASIC TARGET STATISTICS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET STATISTICS")
print("=" * 70)

print(
    f"\nFinal rows: {len(df)}"
)

print(
    f"Minimum future return: "
    f"{df['future_return_pct'].min():.4f}%"
)

print(
    f"Maximum future return: "
    f"{df['future_return_pct'].max():.4f}%"
)

print(
    f"Average future return: "
    f"{df['future_return_pct'].mean():.4f}%"
)

print(
    f"Median future return: "
    f"{df['future_return_pct'].median():.4f}%"
)

print(
    f"Std deviation: "
    f"{df['future_return_pct'].std():.4f}%"
)

# ---------------------------------------------------------
# 8. CHECK TARGET DISTRIBUTION
# ---------------------------------------------------------

print("\nFuture return distribution:")

print(
    df["future_return_pct"]
    .describe()
    .to_string()
)

# ---------------------------------------------------------
# 9. VERIFY AGAIN
# ---------------------------------------------------------



# Check that every sequence still has valid 5-minute spacing
# around the target.

verification_errors = 0

for sequence_id, group in df.groupby("sequence_id"):

    group = group.sort_values("start_time")

    differences = group["start_time"].diff().dropna()

    if not (differences == pd.Timedelta(minutes=5)).all():

        verification_errors += 1

print(
    f"\nSequences with continuity errors: "
    f"{verification_errors}"
)

# ---------------------------------------------------------
# 10. SAVE
# ---------------------------------------------------------

df.to_csv(
    "ai_regression_data_clean.csv",
    index=False
)

print("\nSaved:")
print("ai_regression_data_clean.csv")

# ---------------------------------------------------------
# 11. SHOW SAMPLE
# ---------------------------------------------------------

print("\nSample rows:")

print(
    df[
        [
            "start_time",
            "close",
            "future_price",
            "future_return_pct"
        ]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 70)
print("REGRESSION DATA CREATION COMPLETE")
print("=" * 70)