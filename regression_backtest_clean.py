import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor


# =========================================================
# SETTINGS
# =========================================================

STARTING_BALANCE = 100000.0

FEE_PER_SIDE = 0.0010
SLIPPAGE_PER_SIDE = 0.0005

POSITION_SIZE = 0.25

HOLDING_MINUTES = 30

THRESHOLDS = [0.50, 0.40, 0.35, 0.30]


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
    "volume",
]


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 70)
print("CORRECTED GAP-SAFE REGRESSION BACKTEST")
print("=" * 70)

df = pd.read_csv("ai_regression_data_clean.csv")

df["start_time"] = pd.to_datetime(df["start_time"])

df = df.sort_values(
    ["sequence_id", "start_time"]
).reset_index(drop=True)

print(f"\nRows: {len(df)}")
print(f"Sequences: {df['sequence_id'].nunique()}")


# =========================================================
# TIME SPLIT
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


print("\nTraining rows:", len(train_df))
print("Validation rows:", len(validation_df))
print("Test rows:", len(test_df))


# =========================================================
# TRAIN MODEL
# =========================================================

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(
    train_df[FEATURES],
    train_df["future_return_pct"]
)

print("Model trained.")


# =========================================================
# PREDICTIONS
# =========================================================

validation_df["prediction"] = model.predict(
    validation_df[FEATURES]
)

test_df["prediction"] = model.predict(
    test_df[FEATURES]
)


# =========================================================
# BACKTEST
# =========================================================

def run_backtest(data, threshold):

    balance = STARTING_BALANCE

    trades = []

    data = data.sort_values(
        ["sequence_id", "start_time"]
    ).reset_index(drop=True)

    for i in range(len(data)):

        row = data.iloc[i]

        prediction = row["prediction"]

        # -------------------------------------------------
        # Prediction must exceed threshold
        # -------------------------------------------------

        if abs(prediction) < threshold:
            continue

        sequence_id = row["sequence_id"]

        prediction_time = row["start_time"]

        # -------------------------------------------------
        # ENTRY = NEXT 5-MINUTE CANDLE
        # -------------------------------------------------

        if i + 1 >= len(data):
            continue

        entry_row = data.iloc[i + 1]

        if entry_row["sequence_id"] != sequence_id:
            continue

        expected_entry_time = (
            prediction_time
            + pd.Timedelta(minutes=5)
        )

        if entry_row["start_time"] != expected_entry_time:
            continue

        entry_time = entry_row["start_time"]

        entry_price = entry_row["open"]

        # -------------------------------------------------
        # EXIT = EXACTLY 30 MINUTES AFTER ENTRY
        # -------------------------------------------------

        exit_time = (
            entry_time
            + pd.Timedelta(minutes=HOLDING_MINUTES)
        )

        matching_exit = data[
            (data["sequence_id"] == sequence_id)
            &
            (data["start_time"] == exit_time)
        ]

        if matching_exit.empty:
            continue

        exit_row = matching_exit.iloc[0]

        exit_price = exit_row["close"]

        # -------------------------------------------------
        # ACTUAL RAW RETURN
        # -------------------------------------------------

        actual_return = (
            exit_price / entry_price - 1
        ) * 100

        # -------------------------------------------------
        # DIRECTION
        # -------------------------------------------------

        if prediction > 0:
            direction = "LONG"
        else:
            direction = "SHORT"

        # -------------------------------------------------
        # POSITION SIZE
        # -------------------------------------------------

        position_value = (
            balance * POSITION_SIZE
        )

        quantity = (
            position_value / entry_price
        )

        # -------------------------------------------------
        # SLIPPAGE
        # -------------------------------------------------

        if direction == "LONG":

            effective_entry = (
                entry_price
                * (1 + SLIPPAGE_PER_SIDE)
            )

            effective_exit = (
                exit_price
                * (1 - SLIPPAGE_PER_SIDE)
            )

            gross_pnl = (
                effective_exit
                - effective_entry
            ) * quantity

        else:

            effective_entry = (
                entry_price
                * (1 - SLIPPAGE_PER_SIDE)
            )

            effective_exit = (
                exit_price
                * (1 + SLIPPAGE_PER_SIDE)
            )

            gross_pnl = (
                effective_entry
                - effective_exit
            ) * quantity

        # -------------------------------------------------
        # FEES
        # -------------------------------------------------

        entry_fee = (
            entry_price
            * quantity
            * FEE_PER_SIDE
        )

        exit_fee = (
            exit_price
            * quantity
            * FEE_PER_SIDE
        )

        total_fees = (
            entry_fee + exit_fee
        )

        net_pnl = (
            gross_pnl - total_fees
        )

        # -------------------------------------------------
        # UPDATE BALANCE
        # -------------------------------------------------

        balance_before = balance

        balance += net_pnl

        trades.append({
            "sequence_id": sequence_id,
            "prediction_time": prediction_time,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": direction,
            "prediction": prediction,
            "actual_return_pct": actual_return,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_pnl": gross_pnl,
            "fees": total_fees,
            "net_pnl": net_pnl,
            "balance_before": balance_before,
            "balance_after": balance,
        })

    # =====================================================
    # NO TRADES
    # =====================================================

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:

        return {
            "return_pct": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "trades": 0,
            "max_drawdown": 0.0,
            "final_balance": STARTING_BALANCE,
            "trades_df": trades_df,
        }

    # =====================================================
    # METRICS
    # =====================================================

    wins = trades_df[
        trades_df["net_pnl"] > 0
    ]

    losses = trades_df[
        trades_df["net_pnl"] < 0
    ]

    win_rate = (
        len(wins)
        / len(trades_df)
        * 100
    )

    gross_profit = wins["net_pnl"].sum()

    gross_loss = abs(
        losses["net_pnl"].sum()
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    else:
        profit_factor = np.inf

    equity = trades_df["balance_after"]

    running_max = equity.cummax()

    drawdown = (
        equity / running_max - 1
    ) * 100

    max_drawdown = drawdown.min()

    final_balance = balance

    return_pct = (
        final_balance
        / STARTING_BALANCE
        - 1
    ) * 100

    return {
        "return_pct": return_pct,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "trades": len(trades_df),
        "max_drawdown": max_drawdown,
        "final_balance": final_balance,
        "trades_df": trades_df,
    }


# =========================================================
# VALIDATION
# =========================================================

print("\n" + "=" * 70)
print("VALIDATION THRESHOLD TEST")
print("=" * 70)

results = []

for threshold in THRESHOLDS:

    result = run_backtest(
        validation_df,
        threshold
    )

    results.append({
        "threshold": threshold,
        "return_pct": result["return_pct"],
        "profit_factor": result["profit_factor"],
        "win_rate": result["win_rate"],
        "trades": result["trades"],
        "max_drawdown": result["max_drawdown"],
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "return_pct",
    ascending=False
).reset_index(drop=True)

print(
    results_df.to_string(
        index=False
    )
)


# =========================================================
# SELECT BEST VALIDATION THRESHOLD
# =========================================================

best_threshold = results_df.iloc[0]["threshold"]

print(
    f"\nSelected threshold: "
    f"{best_threshold:.2f}%"
)


# =========================================================
# FINAL UNSEEN TEST
# =========================================================

print("\n" + "=" * 70)
print("FINAL UNSEEN TEST")
print("=" * 70)

final_result = run_backtest(
    test_df,
    best_threshold
)

print(
    f"\nStarting balance : "
    f"₹{STARTING_BALANCE:,.2f}"
)

print(
    f"Final balance    : "
    f"₹{final_result['final_balance']:,.2f}"
)

print(
    f"Return           : "
    f"{final_result['return_pct']:.2f}%"
)

print(
    f"Profit factor    : "
    f"{final_result['profit_factor']:.2f}"
)

print(
    f"Win rate         : "
    f"{final_result['win_rate']:.2f}%"
)

print(
    f"Trades           : "
    f"{final_result['trades']}"
)

print(
    f"Max drawdown     : "
    f"{final_result['max_drawdown']:.2f}%"
)


# =========================================================
# TRADE BREAKDOWN
# =========================================================

trades_df = final_result["trades_df"]

if not trades_df.empty:

    long_count = (
        trades_df["direction"] == "LONG"
    ).sum()

    short_count = (
        trades_df["direction"] == "SHORT"
    ).sum()

    print(
        f"\nLong trades : {long_count}"
    )

    print(
        f"Short trades: {short_count}"
    )

    print("\nFirst 10 trades:")

    print(
        trades_df.head(10).to_string(
            index=False
        )
    )

    trades_df.to_csv(
        "clean_final_test_trades.csv",
        index=False
    )

    print(
        "\nUpdated:"
        "\nclean_final_test_trades.csv"
    )


# =========================================================
# SAVE VALIDATION RESULTS
# =========================================================

results_df.to_csv(
    "clean_validation_results.csv",
    index=False
)

print(
    "\nUpdated:"
    "\nclean_validation_results.csv"
)

print("\n" + "=" * 70)
print("BACKTEST COMPLETE")
print("=" * 70)