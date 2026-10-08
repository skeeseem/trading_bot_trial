import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# SETTINGS
# ============================================================

STARTING_BALANCE = 100000.0

# Trading costs
FEE_RATE = 0.001          # 0.10% per side
SLIPPAGE_RATE = 0.0005    # 0.05% per side

# Use only part of the account for each trade
POSITION_SIZE = 0.25       # 25% of current equity

# Possible parameters to test on VALIDATION data
CONFIDENCE_LEVELS = [0.55, 0.60, 0.65, 0.70]

HOLDING_PERIODS = [
    3,      # 15 minutes
    6,      # 30 minutes
    12      # 60 minutes
]

USE_TREND_FILTER_OPTIONS = [
    False,
    True
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("ai_training_data.csv")

df = df.sort_values("start_time").reset_index(drop=True)

print()
print("=" * 70)
print("AI PAPER TRADING - VALIDATION BASED BACKTEST")
print("=" * 70)

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


X = df[FEATURES]
y = df["target"]


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
print("-" * 70)

print(f"Training rows   : {len(X_train)}")
print(f"Validation rows : {len(X_validation)}")
print(f"Final test rows : {len(X_test)}")


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Training complete")


# ============================================================
# GET PROBABILITIES
# ============================================================

classes = model.classes_

class_to_index = {
    name: i
    for i, name in enumerate(classes)
}


def probability_array(model, X_data):

    probabilities = model.predict_proba(X_data)

    result = {
        "BUY": np.zeros(len(X_data)),
        "SELL": np.zeros(len(X_data)),
        "WAIT": np.zeros(len(X_data))
    }

    for class_name, index in class_to_index.items():

        if class_name in result:
            result[class_name] = probabilities[:, index]

    return result


validation_probabilities = probability_array(
    model,
    X_validation
)


# ============================================================
# STRATEGY BACKTEST FUNCTION
# ============================================================

def run_backtest(
    data,
    probabilities,
    confidence_threshold,
    holding_period,
    use_trend_filter
):

    data = data.reset_index(drop=True)

    balance = STARTING_BALANCE

    position = None
    entry_price = None
    entry_time = None
    entry_index = None
    position_value = None

    trades = []

    equity_curve = []

    for i in range(len(data)):

        row = data.iloc[i]

        current_price = float(row["close"])
        current_time = row["start_time"]


        # ====================================================
        # EXIT EXISTING POSITION
        # ====================================================

        if position is not None:

            candles_held = i - entry_index

            should_exit = False
            exit_reason = ""

            buy_probability = probabilities["BUY"][i]
            sell_probability = probabilities["SELL"][i]


            # Maximum holding period
            if candles_held >= holding_period:

                should_exit = True
                exit_reason = "MAX_HOLD"


            # Opposite signal
            elif position == "LONG":

                if (
                    sell_probability >= confidence_threshold
                    and sell_probability > buy_probability
                ):

                    should_exit = True
                    exit_reason = "OPPOSITE_SIGNAL"


            elif position == "SHORT":

                if (
                    buy_probability >= confidence_threshold
                    and buy_probability > sell_probability
                ):

                    should_exit = True
                    exit_reason = "OPPOSITE_SIGNAL"


            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            if should_exit:

                raw_exit = current_price

                if position == "LONG":

                    exit_price = (
                        raw_exit * (1 - SLIPPAGE_RATE)
                    )

                    price_return = (
                        exit_price - entry_price
                    ) / entry_price

                else:

                    exit_price = (
                        raw_exit * (1 + SLIPPAGE_RATE)
                    )

                    price_return = (
                        entry_price - exit_price
                    ) / entry_price


                gross_pnl = position_value * price_return

                entry_fee = position_value * FEE_RATE
                exit_fee = position_value * FEE_RATE

                net_pnl = (
                    gross_pnl
                    - entry_fee
                    - exit_fee
                )

                balance += net_pnl


                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "direction": position,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "position_value": position_value,
                    "pnl": net_pnl,
                    "return_pct": (
                        net_pnl / position_value
                    ) * 100,
                    "reason": exit_reason,
                    "holding_candles": candles_held
                })


                position = None
                entry_price = None
                entry_time = None
                entry_index = None
                position_value = None


        # ====================================================
        # OPEN NEW POSITION
        # ====================================================

        if position is None and i < len(data) - 1:

            buy_probability = probabilities["BUY"][i]
            sell_probability = probabilities["SELL"][i]
            wait_probability = probabilities["WAIT"][i]


            signal = "WAIT"


            # -----------------------------------------------
            # BUY
            # -----------------------------------------------

            if (
                buy_probability >= confidence_threshold
                and buy_probability > sell_probability
                and buy_probability > wait_probability
            ):

                if use_trend_filter:

                    # BUY only when short EMA is above long EMA
                    if row["ema_20"] > row["ema_50"]:
                        signal = "BUY"

                else:

                    signal = "BUY"


            # -----------------------------------------------
            # SELL
            # -----------------------------------------------

            elif (
                sell_probability >= confidence_threshold
                and sell_probability > buy_probability
                and sell_probability > wait_probability
            ):

                if use_trend_filter:

                    # SELL only when short EMA is below long EMA
                    if row["ema_20"] < row["ema_50"]:
                        signal = "SELL"

                else:

                    signal = "SELL"


            # -----------------------------------------------
            # ENTER NEXT CANDLE
            # -----------------------------------------------

            if signal != "WAIT":

                next_row = data.iloc[i + 1]

                next_open = float(next_row["open"])

                position_value = (
                    balance * POSITION_SIZE
                )

                if signal == "BUY":

                    position = "LONG"

                    entry_price = (
                        next_open * (1 + SLIPPAGE_RATE)
                    )

                elif signal == "SELL":

                    position = "SHORT"

                    entry_price = (
                        next_open * (1 - SLIPPAGE_RATE)
                    )

                entry_time = next_row["start_time"]

                entry_index = i + 1


        # ====================================================
        # EQUITY CURVE
        # ====================================================

        equity = balance

        if position is not None:

            if position == "LONG":

                unrealized_return = (
                    current_price - entry_price
                ) / entry_price

            else:

                unrealized_return = (
                    entry_price - current_price
                ) / entry_price


            unrealized_pnl = (
                position_value
                * unrealized_return
            )

            equity += unrealized_pnl


        equity_curve.append(equity)


    # ========================================================
    # CLOSE FINAL POSITION
    # ========================================================

    if position is not None:

        final_row = data.iloc[-1]

        final_price = float(final_row["close"])

        if position == "LONG":

            exit_price = (
                final_price * (1 - SLIPPAGE_RATE)
            )

            price_return = (
                exit_price - entry_price
            ) / entry_price

        else:

            exit_price = (
                final_price * (1 + SLIPPAGE_RATE)
            )

            price_return = (
                entry_price - exit_price
            ) / entry_price


        gross_pnl = position_value * price_return

        fees = (
            position_value
            * FEE_RATE
            * 2
        )

        net_pnl = gross_pnl - fees

        balance += net_pnl


        trades.append({
            "entry_time": entry_time,
            "exit_time": final_row["start_time"],
            "direction": position,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_value": position_value,
            "pnl": net_pnl,
            "return_pct": (
                net_pnl / position_value
            ) * 100,
            "reason": "END_OF_DATA",
            "holding_candles": len(data) - entry_index
        })


    # ========================================================
    # STATISTICS
    # ========================================================

    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:

        return {
            "balance": STARTING_BALANCE,
            "pnl": 0,
            "return_pct": 0,
            "trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "trades_df": trades_df
        }


    winning = trades_df[
        trades_df["pnl"] > 0
    ]

    losing = trades_df[
        trades_df["pnl"] < 0
    ]


    total_profit = winning["pnl"].sum()

    total_loss = abs(
        losing["pnl"].sum()
    )


    if total_loss > 0:

        profit_factor = (
            total_profit / total_loss
        )

    else:

        profit_factor = float("inf")


    win_rate = (
        len(winning)
        / len(trades_df)
    ) * 100


    total_pnl = balance - STARTING_BALANCE

    total_return = (
        total_pnl
        / STARTING_BALANCE
    ) * 100


    equity_series = pd.Series(
        equity_curve
    )

    running_max = (
        equity_series.cummax()
    )

    drawdown = (
        (equity_series - running_max)
        / running_max
    ) * 100

    max_drawdown = drawdown.min()


    return {
        "balance": balance,
        "pnl": total_pnl,
        "return_pct": total_return,
        "trades": len(trades_df),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "trades_df": trades_df
    }


# ============================================================
# VALIDATION DATA
# ============================================================

validation_df = df.iloc[
    train_end:validation_end
].copy().reset_index(drop=True)


print()
print("=" * 70)
print("VALIDATION PHASE")
print("=" * 70)

print()
print("Testing strategy combinations...")


validation_results = []


for confidence in CONFIDENCE_LEVELS:

    for holding in HOLDING_PERIODS:

        for trend_filter in USE_TREND_FILTER_OPTIONS:

            result = run_backtest(
                validation_df,
                validation_probabilities,
                confidence,
                holding,
                trend_filter
            )


            validation_results.append({
                "confidence": confidence,
                "holding": holding,
                "trend_filter": trend_filter,
                "return_pct": result["return_pct"],
                "profit_factor": result["profit_factor"],
                "win_rate": result["win_rate"],
                "trades": result["trades"],
                "max_drawdown": result["max_drawdown"]
            })


validation_results_df = pd.DataFrame(
    validation_results
)


# ============================================================
# SELECT STRATEGY
# ============================================================

# Prefer strategies that:
# 1. Make money
# 2. Have profit factor > 1
# 3. Have enough trades
#
# If none qualify, select the highest validation return.

profitable = validation_results_df[
    (validation_results_df["profit_factor"] > 1.0)
    & (validation_results_df["trades"] >= 10)
]


if len(profitable) > 0:

    best = profitable.sort_values(
        ["return_pct", "profit_factor"],
        ascending=False
    ).iloc[0]

else:

    best = validation_results_df.sort_values(
        "return_pct",
        ascending=False
    ).iloc[0]


best_confidence = float(best["confidence"])
best_holding = int(best["holding"])
best_trend_filter = bool(best["trend_filter"])


print()
print("BEST VALIDATION STRATEGY")
print("-" * 70)

print(
    f"Confidence threshold : {best_confidence:.2f}"
)

print(
    f"Holding period       : "
    f"{best_holding * 5} minutes"
)

print(
    f"Trend filter         : "
    f"{best_trend_filter}"
)

print(
    f"Validation return    : "
    f"{best['return_pct']:.2f}%"
)

print(
    f"Validation PF        : "
    f"{best['profit_factor']:.2f}"
)

print(
    f"Validation win rate  : "
    f"{best['win_rate']:.2f}%"
)

print(
    f"Validation trades    : "
    f"{int(best['trades'])}"
)

print(
    f"Validation drawdown  : "
    f"{best['max_drawdown']:.2f}%"
)


# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

validation_results_df.to_csv(
    "validation_results.csv",
    index=False
)

print()
print("Saved validation results to:")
print("validation_results.csv")


# ============================================================
# RETRAIN MODEL
# ============================================================

print()
print("=" * 70)
print("FINAL TEST")
print("=" * 70)

print()
print("Retraining model using TRAIN + VALIDATION data...")


X_train_final = X.iloc[:validation_end]

y_train_final = y.iloc[:validation_end]


final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)


final_model.fit(
    X_train_final,
    y_train_final
)


print("Final model training complete")


# ============================================================
# FINAL TEST PROBABILITIES
# ============================================================

final_classes = final_model.classes_

final_class_to_index = {
    name: i
    for i, name in enumerate(final_classes)
}


test_probabilities_raw = (
    final_model.predict_proba(X_test)
)


test_probabilities = {
    "BUY": np.zeros(len(X_test)),
    "SELL": np.zeros(len(X_test)),
    "WAIT": np.zeros(len(X_test))
}


for class_name, index in final_class_to_index.items():

    if class_name in test_probabilities:

        test_probabilities[class_name] = (
            test_probabilities_raw[:, index]
        )


final_test_df = df.iloc[
    validation_end:
].copy().reset_index(drop=True)


# ============================================================
# FINAL BACKTEST
# ============================================================

final_result = run_backtest(
    final_test_df,
    test_probabilities,
    best_confidence,
    best_holding,
    best_trend_filter
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL UNSEEN TEST RESULTS")
print("=" * 70)

print()
print(
    f"Starting balance : "
    f"₹{STARTING_BALANCE:,.2f}"
)

print(
    f"Final balance    : "
    f"₹{final_result['balance']:,.2f}"
)

print()
print(
    f"Total P&L        : "
    f"₹{final_result['pnl']:,.2f}"
)

print(
    f"Return           : "
    f"{final_result['return_pct']:.2f}%"
)

print()
print(
    f"Total trades     : "
    f"{final_result['trades']}"
)

print(
    f"Win rate         : "
    f"{final_result['win_rate']:.2f}%"
)

print(
    f"Profit factor    : "
    f"{final_result['profit_factor']:.2f}"
)

print(
    f"Max drawdown     : "
    f"{final_result['max_drawdown']:.2f}%"
)


# ============================================================
# TRADE BREAKDOWN
# ============================================================

trades_df = final_result["trades_df"]


if len(trades_df) > 0:

    print()
    print("FINAL TEST TRADE SUMMARY")
    print("-" * 70)

    long_trades = trades_df[
        trades_df["direction"] == "LONG"
    ]

    short_trades = trades_df[
        trades_df["direction"] == "SHORT"
    ]


    print(
        f"Long trades  : {len(long_trades)}"
    )

    print(
        f"Short trades : {len(short_trades)}"
    )


    print()
    print("First 10 trades:")
    print()

    print(
        trades_df[
            [
                "entry_time",
                "exit_time",
                "direction",
                "entry_price",
                "exit_price",
                "position_value",
                "pnl",
                "reason"
            ]
        ].head(10).to_string(index=False)
    )


    trades_df.to_csv(
        "final_test_trades.csv",
        index=False
    )


print()
print("=" * 70)
print("BACKTEST COMPLETE")
print("=" * 70)

print()
print("Files created:")
print("  validation_results.csv")
print("  final_test_trades.csv")