import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# SETTINGS
# ============================================================

STARTING_BALANCE = 100000.0

FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005

POSITION_SIZE = 0.25

# Prediction threshold
THRESHOLDS = [
    0.0015,   # 0.15%
    0.0020,   # 0.20%
    0.0025,   # 0.25%
    0.0030    # 0.30%
]

# 5-minute candles
HOLDING_PERIODS = [
    3,        # 15 minutes
    6,        # 30 minutes
    12        # 60 minutes
]

# Initial training / validation / test sizes
INITIAL_TRAIN = 3200
VALIDATION_SIZE = 800
TEST_SIZE = 800

# Move forward this many candles after every test
STEP_SIZE = 800


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "ai_regression_data.csv"
)

df["start_time"] = pd.to_datetime(
    df["start_time"],
    errors="coerce"
)

df = df.dropna(
    subset=["start_time"]
)

df = df.sort_values(
    "start_time"
).reset_index(drop=True)


print()
print("=" * 70)
print("WALK-FORWARD AI BACKTEST")
print("=" * 70)

print(
    f"Total rows: {len(df)}"
)


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


# ============================================================
# BACKTEST FUNCTION
# ============================================================

def run_backtest(
    data,
    predictions,
    threshold,
    holding_period
):

    data = data.reset_index(
        drop=True
    )

    balance = STARTING_BALANCE

    position = None

    entry_price = None
    entry_index = None
    entry_time = None
    position_value = None

    trades = []

    equity_curve = []


    for i in range(len(data)):

        row = data.iloc[i]

        current_price = float(
            row["close"]
        )

        current_time = row[
            "start_time"
        ]


        # ====================================================
        # EXIT EXISTING POSITION
        # ====================================================

        if position is not None:

            candles_held = (
                i - entry_index
            )


            # Never exit on entry candle
            if candles_held == 0:

                equity = balance


                if position == "LONG":

                    unrealized_return = (
                        current_price
                        - entry_price
                    ) / entry_price

                else:

                    unrealized_return = (
                        entry_price
                        - current_price
                    ) / entry_price


                equity += (
                    position_value
                    * unrealized_return
                )


                equity_curve.append(
                    equity
                )

                continue


            should_exit = False

            exit_reason = ""


            # Maximum holding time
            if candles_held >= holding_period:

                should_exit = True

                exit_reason = "MAX_HOLD"


            # Opposite signal
            elif position == "LONG":

                if predictions[i] <= -threshold:

                    should_exit = True

                    exit_reason = (
                        "OPPOSITE_SIGNAL"
                    )


            elif position == "SHORT":

                if predictions[i] >= threshold:

                    should_exit = True

                    exit_reason = (
                        "OPPOSITE_SIGNAL"
                    )


            # =================================================
            # EXECUTE EXIT
            # =================================================

            if should_exit:

                if position == "LONG":

                    exit_price = (
                        current_price
                        * (1 - SLIPPAGE_RATE)
                    )

                    price_return = (
                        exit_price
                        - entry_price
                    ) / entry_price

                else:

                    exit_price = (
                        current_price
                        * (1 + SLIPPAGE_RATE)
                    )

                    price_return = (
                        entry_price
                        - exit_price
                    ) / entry_price


                gross_pnl = (
                    position_value
                    * price_return
                )


                fees = (
                    position_value
                    * FEE_RATE
                    * 2
                )


                net_pnl = (
                    gross_pnl
                    - fees
                )


                balance += net_pnl


                trades.append({

                    "entry_time":
                        entry_time,

                    "exit_time":
                        current_time,

                    "direction":
                        position,

                    "entry_price":
                        entry_price,

                    "exit_price":
                        exit_price,

                    "position_value":
                        position_value,

                    "pnl":
                        net_pnl,

                    "reason":
                        exit_reason
                })


                position = None

                entry_price = None
                entry_index = None
                entry_time = None
                position_value = None


        # ====================================================
        # OPEN NEW POSITION
        # ====================================================

        if (
            position is None
            and i < len(data) - 1
        ):

            prediction = predictions[i]

            signal = "WAIT"


            if prediction >= threshold:

                signal = "LONG"


            elif prediction <= -threshold:

                signal = "SHORT"


            if signal != "WAIT":

                next_row = data.iloc[
                    i + 1
                ]

                next_open = float(
                    next_row["open"]
                )


                position_value = (
                    balance
                    * POSITION_SIZE
                )


                if signal == "LONG":

                    position = "LONG"

                    entry_price = (
                        next_open
                        * (1 + SLIPPAGE_RATE)
                    )

                else:

                    position = "SHORT"

                    entry_price = (
                        next_open
                        * (1 - SLIPPAGE_RATE)
                    )


                entry_time = (
                    next_row["start_time"]
                )

                entry_index = (
                    i + 1
                )


        # ====================================================
        # EQUITY
        # ====================================================

        equity = balance


        if position is not None:

            if position == "LONG":

                unrealized_return = (
                    current_price
                    - entry_price
                ) / entry_price

            else:

                unrealized_return = (
                    entry_price
                    - current_price
                ) / entry_price


            equity += (
                position_value
                * unrealized_return
            )


        equity_curve.append(
            equity
        )


    # ========================================================
    # CLOSE FINAL POSITION
    # ========================================================

    if position is not None:

        final_row = data.iloc[-1]

        final_price = float(
            final_row["close"]
        )


        if position == "LONG":

            exit_price = (
                final_price
                * (1 - SLIPPAGE_RATE)
            )

            price_return = (
                exit_price
                - entry_price
            ) / entry_price

        else:

            exit_price = (
                final_price
                * (1 + SLIPPAGE_RATE)
            )

            price_return = (
                entry_price
                - exit_price
            ) / entry_price


        gross_pnl = (
            position_value
            * price_return
        )


        fees = (
            position_value
            * FEE_RATE
            * 2
        )


        net_pnl = (
            gross_pnl
            - fees
        )


        balance += net_pnl


        trades.append({

            "entry_time":
                entry_time,

            "exit_time":
                final_row["start_time"],

            "direction":
                position,

            "entry_price":
                entry_price,

            "exit_price":
                exit_price,

            "position_value":
                position_value,

            "pnl":
                net_pnl,

            "reason":
                "END_OF_DATA"
        })


    # ========================================================
    # STATISTICS
    # ========================================================

    trades_df = pd.DataFrame(
        trades
    )


    if len(trades_df) == 0:

        return {
            "balance": STARTING_BALANCE,
            "return_pct": 0,
            "trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown": 0
        }


    winning = trades_df[
        trades_df["pnl"] > 0
    ]

    losing = trades_df[
        trades_df["pnl"] < 0
    ]


    total_profit = (
        winning["pnl"].sum()
    )

    total_loss = abs(
        losing["pnl"].sum()
    )


    if total_loss > 0:

        profit_factor = (
            total_profit
            / total_loss
        )

    else:

        profit_factor = float(
            "inf"
        )


    win_rate = (
        len(winning)
        / len(trades_df)
    ) * 100


    total_return = (
        balance
        - STARTING_BALANCE
    ) / STARTING_BALANCE * 100


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

        "balance":
            balance,

        "return_pct":
            total_return,

        "trades":
            len(trades_df),

        "win_rate":
            win_rate,

        "profit_factor":
            profit_factor,

        "max_drawdown":
            max_drawdown
    }


# ============================================================
# WALK-FORWARD LOOP
# ============================================================

fold_results = []

fold_number = 1

test_start = (
    INITIAL_TRAIN
    + VALIDATION_SIZE
)


while test_start < len(df):

    train_start = 0

    train_end = (
        test_start
        - VALIDATION_SIZE
    )


    validation_start = train_end

    validation_end = test_start


    test_end = min(
        test_start + TEST_SIZE,
        len(df)
    )


    # Need enough test data
    if (
        test_end
        - test_start
        < 200
    ):

        break


    print()
    print("=" * 70)

    print(
        f"FOLD {fold_number}"
    )

    print("=" * 70)


    print()

    print(
        f"Train      : "
        f"{df.iloc[train_start]['start_time']} "
        f"→ "
        f"{df.iloc[train_end - 1]['start_time']}"
    )


    print(
        f"Validation : "
        f"{df.iloc[validation_start]['start_time']} "
        f"→ "
        f"{df.iloc[validation_end - 1]['start_time']}"
    )


    print(
        f"Test       : "
        f"{df.iloc[test_start]['start_time']} "
        f"→ "
        f"{df.iloc[test_end - 1]['start_time']}"
    )


    # ========================================================
    # TRAIN
    # ========================================================

    X_train = df.iloc[
        train_start:train_end
    ][FEATURES]

    y_train = df.iloc[
        train_start:train_end
    ]["future_return"]


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


    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    validation_df = df.iloc[
        validation_start:validation_end
    ].copy().reset_index(
        drop=True
    )


    validation_predictions = (
        model.predict(
            validation_df[FEATURES]
        )
    )


    # ========================================================
    # SELECT BEST PARAMETERS
    # ========================================================

    validation_results = []


    for threshold in THRESHOLDS:

        for holding in HOLDING_PERIODS:

            result = run_backtest(

                validation_df,

                validation_predictions,

                threshold,

                holding
            )


            validation_results.append({

                "threshold":
                    threshold,

                "holding":
                    holding,

                "return_pct":
                    result["return_pct"],

                "profit_factor":
                    result["profit_factor"],

                "win_rate":
                    result["win_rate"],

                "trades":
                    result["trades"],

                "max_drawdown":
                    result["max_drawdown"]
            })


    validation_results_df = pd.DataFrame(
        validation_results
    )


    # Prefer positive PF with enough trades
    candidates = validation_results_df[
        (validation_results_df["profit_factor"] > 1.0)
        &
        (validation_results_df["trades"] >= 20)
    ]


    if len(candidates) > 0:

        best = (
            candidates
            .sort_values(
                [
                    "return_pct",
                    "profit_factor"
                ],
                ascending=False
            )
            .iloc[0]
        )

    else:

        best = (
            validation_results_df
            .sort_values(
                "return_pct",
                ascending=False
            )
            .iloc[0]
        )


    best_threshold = float(
        best["threshold"]
    )

    best_holding = int(
        best["holding"]
    )


    print()

    print(
        f"Selected threshold : "
        f"{best_threshold * 100:.2f}%"
    )

    print(
        f"Selected holding   : "
        f"{best_holding * 5} minutes"
    )

    print(
        f"Validation return  : "
        f"{best['return_pct']:.2f}%"
    )

    print(
        f"Validation PF      : "
        f"{best['profit_factor']:.2f}"
    )


    # ========================================================
    # FINAL TEST FOR THIS FOLD
    # ========================================================

    test_df = df.iloc[
        test_start:test_end
    ].copy().reset_index(
        drop=True
    )


    test_predictions = (
        model.predict(
            test_df[FEATURES]
        )
    )


    test_result = run_backtest(

        test_df,

        test_predictions,

        best_threshold,

        best_holding
    )


    print()

    print(
        f"TEST return        : "
        f"{test_result['return_pct']:.2f}%"
    )

    print(
        f"TEST profit factor : "
        f"{test_result['profit_factor']:.2f}"
    )

    print(
        f"TEST win rate      : "
        f"{test_result['win_rate']:.2f}%"
    )

    print(
        f"TEST trades        : "
        f"{test_result['trades']}"
    )

    print(
        f"TEST drawdown      : "
        f"{test_result['max_drawdown']:.2f}%"
    )


    # ========================================================
    # SAVE FOLD
    # ========================================================

    fold_results.append({

        "fold":
            fold_number,

        "train_end":
            df.iloc[
                train_end - 1
            ]["start_time"],

        "validation_start":
            df.iloc[
                validation_start
            ]["start_time"],

        "validation_end":
            df.iloc[
                validation_end - 1
            ]["start_time"],

        "test_start":
            df.iloc[
                test_start
            ]["start_time"],

        "test_end":
            df.iloc[
                test_end - 1
            ]["start_time"],

        "threshold":
            best_threshold,

        "holding_minutes":
            best_holding * 5,

        "validation_return_pct":
            best["return_pct"],

        "validation_profit_factor":
            best["profit_factor"],

        "test_return_pct":
            test_result["return_pct"],

        "test_profit_factor":
            test_result["profit_factor"],

        "test_win_rate":
            test_result["win_rate"],

        "test_trades":
            test_result["trades"],

        "test_max_drawdown":
            test_result["max_drawdown"]
    })


    # ========================================================
    # MOVE FORWARD
    # ========================================================

    fold_number += 1

    test_start += STEP_SIZE


# ============================================================
# FINAL SUMMARY
# ============================================================

results_df = pd.DataFrame(
    fold_results
)


print()
print()
print("=" * 70)
print("WALK-FORWARD SUMMARY")
print("=" * 70)


if len(results_df) == 0:

    print()
    print("No folds were completed.")

else:

    print()

    print(
        results_df[
            [
                "fold",
                "threshold",
                "holding_minutes",
                "validation_return_pct",
                "test_return_pct",
                "test_profit_factor",
                "test_win_rate",
                "test_trades",
                "test_max_drawdown"
            ]
        ].to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Aggregate statistics
    # --------------------------------------------------------

    total_test_return = (
        results_df["test_return_pct"].sum()
    )


    average_test_return = (
        results_df["test_return_pct"].mean()
    )


    median_test_return = (
        results_df["test_return_pct"].median()
    )


    positive_folds = (
        results_df["test_return_pct"] > 0
    ).sum()


    total_folds = len(
        results_df
    )


    average_profit_factor = (
        results_df["test_profit_factor"].mean()
    )


    total_trades = (
        results_df["test_trades"].sum()
    )


    print()
    print("-" * 70)

    print(
        f"Completed folds       : "
        f"{total_folds}"
    )

    print(
        f"Positive test folds   : "
        f"{positive_folds}/{total_folds}"
    )

    print(
        f"Average test return   : "
        f"{average_test_return:.2f}%"
    )

    print(
        f"Median test return    : "
        f"{median_test_return:.2f}%"
    )

    print(
        f"Sum of fold returns   : "
        f"{total_test_return:.2f}%"
    )

    print(
        f"Average profit factor : "
        f"{average_profit_factor:.2f}"
    )

    print(
        f"Total test trades     : "
        f"{total_trades}"
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df.to_csv(
        "walk_forward_results.csv",
        index=False
    )


    print()
    print(
        "Saved: walk_forward_results.csv"
    )


print()
print("=" * 70)
print("WALK-FORWARD TEST COMPLETE")
print("=" * 70)