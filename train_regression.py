import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("ai_regression_data.csv")

df = df.sort_values("start_time").reset_index(drop=True)


print()
print("=" * 65)
print("AI REGRESSION MODEL")
print("=" * 65)

print(f"Total rows: {len(df)}")


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
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

TARGET = "future_return"


X = df[FEATURES]
y = df[TARGET]


# ============================================================
# 70 / 15 / 15 SPLIT
# ============================================================

total = len(df)

train_end = int(total * 0.70)
validation_end = int(total * 0.85)


X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_validation = X.iloc[train_end:validation_end]
y_validation = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]


print()
print("DATA SPLIT")
print("-" * 65)

print(f"Training rows   : {len(X_train)}")
print(f"Validation rows : {len(X_validation)}")
print(f"Final test rows : {len(X_test)}")


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("Training Random Forest Regressor...")


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


print("Training complete")


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

validation_predictions = model.predict(
    X_validation
)


# ============================================================
# VALIDATION METRICS
# ============================================================

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        validation_predictions
    )
)

validation_r2 = r2_score(
    y_validation,
    validation_predictions
)


print()
print("=" * 65)
print("VALIDATION RESULTS")
print("=" * 65)

print()
print(
    f"MAE  : {validation_mae * 100:.4f}%"
)

print(
    f"RMSE : {validation_rmse * 100:.4f}%"
)

print(
    f"R²   : {validation_r2:.4f}"
)


# ============================================================
# DIRECTIONAL ACCURACY
# ============================================================

actual_direction = np.sign(
    y_validation
)

predicted_direction = np.sign(
    validation_predictions
)

directional_accuracy = (
    actual_direction == predicted_direction
).mean() * 100


print()
print(
    f"Directional accuracy: "
    f"{directional_accuracy:.2f}%"
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print()
print("SAMPLE VALIDATION PREDICTIONS")
print("-" * 65)

sample = df.iloc[
    train_end:validation_end
].copy()

sample["predicted_return"] = validation_predictions

sample["actual_return_pct"] = (
    sample["future_return"] * 100
)

sample["predicted_return_pct"] = (
    sample["predicted_return"] * 100
)


print(
    sample[
        [
            "start_time",
            "close",
            "actual_return_pct",
            "predicted_return_pct"
        ]
    ].head(15).to_string(index=False)
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)


print()
print("FEATURE IMPORTANCE")
print("-" * 65)

print(
    importance.to_string(index=False)
)


# ============================================================
# LATEST PREDICTION
# ============================================================

latest_features = df[FEATURES].iloc[[-1]]

latest_prediction = model.predict(
    latest_features
)[0]


print()
print("=" * 65)
print("LATEST PREDICTION")
print("=" * 65)

print()
print(
    f"Current BTC price : "
    f"₹{df['close'].iloc[-1]:,.2f}"
)

print(
    f"Predicted 30-min return : "
    f"{latest_prediction * 100:.4f}%"
)


if latest_prediction > 0:

    print("Direction: UP")

elif latest_prediction < 0:

    print("Direction: DOWN")

else:

    print("Direction: FLAT")


print()
print("Training script complete.")