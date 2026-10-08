import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# ==========================================
# Load dataset
# ==========================================

df = pd.read_csv("ai_training_data.csv")

df = df.sort_values("start_time").reset_index(drop=True)


# ==========================================
# Features
# ==========================================

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


X = df[features]
y = df["target"]


# ==========================================
# TIME-BASED TRAIN / TEST SPLIT
# ==========================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\n========================================")
print("AI MODEL TRAINING")
print("========================================")

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ==========================================
# Random Forest
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)


print("\nTraining model...")

model.fit(X_train, y_train)

print("✅ Training complete")


# ==========================================
# Predictions
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# Evaluation
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========================================")
print("MODEL RESULTS")
print("========================================")

print(f"\nAccuracy: {accuracy:.2%}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# Prediction probabilities
# ==========================================

probabilities = model.predict_proba(X_test)

classes = model.classes_

print("\nLatest prediction:")

latest_probabilities = probabilities[-1]

for class_name, probability in zip(
    classes,
    latest_probabilities
):
    print(
        f"{class_name}: {probability:.2%}"
    )


print(
    "\nAI prediction:",
    predictions[-1]
)