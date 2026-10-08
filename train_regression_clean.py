import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 70)
print("TRAINING GAP-SAFE REGRESSION MODEL")
print("=" * 70)


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("ai_regression_data_clean.csv")

df["start_time"] = pd.to_datetime(df["start_time"])

df = df.sort_values("start_time").reset_index(drop=True)

print(f"\nTotal rows: {len(df)}")


# =========================================================
# 2. FEATURES
# =========================================================

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
    "volume"
]

target = "future_return_pct"


# =========================================================
# 3. CHECK FEATURES
# =========================================================

missing_features = [
    column
    for column in features
    if column not in df.columns
]

if missing_features:

    print("\nERROR: Missing features:")

    for column in missing_features:
        print("  -", column)

    raise SystemExit


# =========================================================
# 4. REMOVE INVALID ROWS
# =========================================================

df = df.dropna(
    subset=features + [target]
).reset_index(drop=True)

print(f"Rows after NaN removal: {len(df)}")


# =========================================================
# 5. TIME-ORDERED SPLIT
# =========================================================
#
# 70% training
# 15% validation
# 15% final test
#
# We NEVER shuffle time-series data.
# =========================================================

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()

validation_df = df.iloc[
    train_end:validation_end
].copy()

test_df = df.iloc[
    validation_end:
].copy()


print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    f"\nTraining   : {len(train_df)} rows"
)

print(
    f"Validation : {len(validation_df)} rows"
)

print(
    f"Final test : {len(test_df)} rows"
)

print("\nTraining period:")

print(
    train_df["start_time"].iloc[0],
    "→",
    train_df["start_time"].iloc[-1]
)

print("\nValidation period:")

print(
    validation_df["start_time"].iloc[0],
    "→",
    validation_df["start_time"].iloc[-1]
)

print("\nFinal test period:")

print(
    test_df["start_time"].iloc[0],
    "→",
    test_df["start_time"].iloc[-1]
)


# =========================================================
# 6. PREPARE X / Y
# =========================================================

X_train = train_df[features]
y_train = train_df[target]

X_validation = validation_df[features]
y_validation = validation_df[target]

X_test = test_df[features]
y_test = test_df[target]


# =========================================================
# 7. TRAIN MODEL
# =========================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("\nModel training complete.")


# =========================================================
# 8. EVALUATION FUNCTION
# =========================================================

def evaluate_model(name, X, y):

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    directional_accuracy = (
        np.sign(predictions)
        == np.sign(y)
    ).mean() * 100

    print("\n" + "-" * 70)

    print(name)

    print("-" * 70)

    print(
        f"MAE                  : {mae:.4f}%"
    )

    print(
        f"RMSE                 : {rmse:.4f}%"
    )

    print(
        f"R²                   : {r2:.4f}"
    )

    print(
        f"Directional accuracy : "
        f"{directional_accuracy:.2f}%"
    )

    return predictions


# =========================================================
# 9. EVALUATE
# =========================================================

validation_predictions = evaluate_model(
    "VALIDATION RESULTS",
    X_validation,
    y_validation
)

test_predictions = evaluate_model(
    "FINAL TEST RESULTS",
    X_test,
    y_test
)


# =========================================================
# 10. FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(index=False)
)


# =========================================================
# 11. LATEST PREDICTION
# =========================================================

latest_row = df.iloc[-1:]

latest_features = latest_row[features]

latest_prediction = model.predict(
    latest_features
)[0]

latest_actual = latest_row[target].iloc[0]

print("\n" + "=" * 70)
print("LATEST MODEL OUTPUT")
print("=" * 70)

print(
    f"\nTime: "
    f"{latest_row['start_time'].iloc[0]}"
)

print(
    f"Predicted 30-min return: "
    f"{latest_prediction:.4f}%"
)

print(
    f"Actual 30-min return: "
    f"{latest_actual:.4f}%"
)

if latest_prediction > 0:

    print("Predicted direction: UP")

elif latest_prediction < 0:

    print("Predicted direction: DOWN")

else:

    print("Predicted direction: FLAT")


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)