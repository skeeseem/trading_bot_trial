import pandas as pd

EXPECTED_INTERVAL = pd.Timedelta(minutes=5)
MIN_SEQUENCE_LENGTH = 100

print("=" * 70)
print("CLEANING BTC/INR 5-MINUTE DATA")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("btc_inr_5m.csv")

print(f"\nRaw rows: {len(df)}")

# ---------------------------------------------------------
# 2. CONVERT TIMESTAMP
# ---------------------------------------------------------

df["start_time"] = pd.to_datetime(
    df["start_time"],
    errors="coerce"
)

df = df.dropna(subset=["start_time"])

# Sort chronologically
df = df.sort_values("start_time").reset_index(drop=True)

# ---------------------------------------------------------
# 3. REMOVE DUPLICATES
# ---------------------------------------------------------

duplicate_count = df["start_time"].duplicated().sum()

print(f"\nDuplicate rows found: {duplicate_count}")

df = df.drop_duplicates(
    subset=["start_time"],
    keep="first"
).reset_index(drop=True)

# ---------------------------------------------------------
# 4. FIND TIME GAPS
# ---------------------------------------------------------

time_difference = df["start_time"].diff()

gap_mask = time_difference != EXPECTED_INTERVAL

# First row isn't a gap
gap_mask.iloc[0] = False

gap_count = int(gap_mask.sum())

print(f"\nTime gaps found: {gap_count}")

# ---------------------------------------------------------
# 5. SHOW LARGEST GAPS
# ---------------------------------------------------------

if gap_count > 0:

    gap_rows = df.loc[gap_mask].copy()

    gap_rows["previous_time"] = (
        df["start_time"].shift(1).loc[gap_rows.index]
    )

    gap_rows["gap_duration"] = (
        df["start_time"].loc[gap_rows.index]
        - gap_rows["previous_time"]
    )

    gap_rows = gap_rows.sort_values(
        "gap_duration",
        ascending=False
    )

    print("\nLargest gaps:\n")

    print(
        gap_rows[
            [
                "previous_time",
                "start_time",
                "gap_duration"
            ]
        ].head(10).to_string(index=False)
    )

# ---------------------------------------------------------
# 6. CREATE SEQUENCE IDs
# ---------------------------------------------------------
#
# Every time there is a gap, start a new sequence.
#
# Example:
#
# Sequence 0
# 10:00
# 10:05
# 10:10
#
# GAP
#
# Sequence 1
# 11:00
# 11:05
# 11:10
#
# ---------------------------------------------------------

df["sequence_id"] = gap_mask.cumsum()

# ---------------------------------------------------------
# 7. CHECK SEQUENCE SIZES
# ---------------------------------------------------------

sequence_sizes = (
    df.groupby("sequence_id")
    .size()
    .sort_values(ascending=False)
)

print("\nTotal sequences:", len(sequence_sizes))

print("\nLargest sequences:")

print(sequence_sizes.head(10).to_string())

# ---------------------------------------------------------
# 8. KEEP ONLY REASONABLY LARGE SEQUENCES
# ---------------------------------------------------------

valid_sequences = sequence_sizes[
    sequence_sizes >= MIN_SEQUENCE_LENGTH
].index

clean_df = df[
    df["sequence_id"].isin(valid_sequences)
].copy()

clean_df = clean_df.reset_index(drop=True)

# ---------------------------------------------------------
# 9. FINAL VALIDATION
# ---------------------------------------------------------

remaining_sequence_count = (
    clean_df["sequence_id"].nunique()
)

print("\n" + "=" * 70)
print("CLEAN DATA SUMMARY")
print("=" * 70)

print(f"\nOriginal rows : {len(df)}")
print(f"Clean rows    : {len(clean_df)}")
print(f"Removed rows  : {len(df) - len(clean_df)}")

print(
    f"\nValid continuous sequences: "
    f"{remaining_sequence_count}"
)

# ---------------------------------------------------------
# 10. VERIFY EACH SEQUENCE
# ---------------------------------------------------------

invalid_sequences = []

for sequence_id, group in clean_df.groupby("sequence_id"):

    differences = group["start_time"].diff().dropna()

    bad_gaps = differences[
        differences != EXPECTED_INTERVAL
    ]

    if len(bad_gaps) > 0:
        invalid_sequences.append(sequence_id)

print(
    f"Sequences with internal gaps: "
    f"{len(invalid_sequences)}"
)

# ---------------------------------------------------------
# 11. SAVE
# ---------------------------------------------------------

clean_df.to_csv(
    "btc_inr_5m_clean.csv",
    index=False
)

print("\nFirst candle:")
print(clean_df["start_time"].iloc[0])

print("\nLast candle:")
print(clean_df["start_time"].iloc[-1])

print("\nSaved:")
print("btc_inr_5m_clean.csv")

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)