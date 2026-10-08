import pandas as pd


df = pd.read_csv("ai_regression_data.csv")

df["start_time"] = pd.to_datetime(df["start_time"])

print()
print("=" * 60)
print("DATA INTEGRITY CHECK")
print("=" * 60)

print()
print(f"Total rows: {len(df)}")

# ------------------------------------------------------------
# Duplicate timestamps
# ------------------------------------------------------------

duplicates = df[
    df["start_time"].duplicated(keep=False)
].sort_values("start_time")

print()
print(f"Duplicate timestamps: {len(duplicates)}")

if len(duplicates) > 0:

    print()
    print("First duplicate timestamps:")
    print()

    print(
        duplicates[
            [
                "start_time",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        ].head(20).to_string(index=False)
    )


# ------------------------------------------------------------
# Time gaps
# ------------------------------------------------------------

df = df.sort_values("start_time").reset_index(drop=True)

time_diff = df["start_time"].diff()

expected = pd.Timedelta(minutes=5)

gaps = df[
    time_diff > expected
].copy()

print()
print(f"Gaps larger than 5 minutes: {len(gaps)}")

if len(gaps) > 0:

    print()
    print("First gaps:")

    gap_info = pd.DataFrame({
        "timestamp": df.loc[gaps.index, "start_time"],
        "previous_timestamp": df.loc[gaps.index - 1, "start_time"],
        "gap": time_diff.loc[gaps.index]
    })

    print(
        gap_info.head(20).to_string(index=False)
    )


# ------------------------------------------------------------
# Basic chronological check
# ------------------------------------------------------------

is_sorted = df["start_time"].is_monotonic_increasing

print()
print(f"Chronologically sorted: {is_sorted}")


# ------------------------------------------------------------
# Price sanity
# ------------------------------------------------------------

bad_prices = df[
    (df["open"] <= 0)
    | (df["high"] <= 0)
    | (df["low"] <= 0)
    | (df["close"] <= 0)
]

print()
print(f"Invalid price rows: {len(bad_prices)}")


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print()
print("=" * 60)
print("CHECK COMPLETE")
print("=" * 60)