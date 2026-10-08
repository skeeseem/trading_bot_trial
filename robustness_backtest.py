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

THRESHOLDS = [
    0.0015,   # 0.15%
    0.0020,   # 0.20%
    0.0025,   # 0.25%
    0.0030,   # 0.30%
    0.0040    # 0.40%
]

HOLDING_PERIODS = [
    3,        # 15 minutes
    6,        # 30 minutes
    12        # 60 minutes
]

COOLDOWN_CANDLES = [
    0,        # No cooldown
    3         # 15 minute cooldown
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("ai_regression_data.csv")

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
print("REGRESSION ROBUSTNESS BACKTEST")
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

y = df["future_return"]


# ============================================================
# 70 / 15 / 15 SPLIT
# ============================================================

total = len(df)

train_end = int(total * 0.70)

validation_end = int(total * 0.85)


X_train = X.iloc[:train_end]

y_train = y.iloc[:train_end]


X_validation = X.iloc[
    train_end:validation_end
]


X_test = X.iloc[
    validation_end:
]


print()
print("DATA SPLIT")
print("-" * 70)

print(f"Training   : {len(X_train)}")

print(f"Validation : {len(X_validation)}")

print(f"Final test : {len(X_test)}")


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("Training regression model...")


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
# PREDICTIONS
# ============================================================

validation_predictions = model.predict(
    X_validation
)


# ============================================================
# BACKTEST FUNCTION
# ============================================================

def run_backtest(
    data,
    predictions,
    threshold,
    holding_period,
    strategy_mode,
    cooldown
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

    cooldown_until = -1

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


            # ------------------------------------------------
            # IMPORTANT FIX
            #
            # Do NOT allow a position to exit on the same
            # candle on which it was entered.
            # ------------------------------------------------

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


            # =================================================
            # FLIP STRATEGY
            # =================================================

            if strategy_mode == "FLIP":

                if candles_held >= holding_period:

                    should_exit = True

                    exit_reason = (
                        "MAX_HOLD"
                    )


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
            # FIXED HOLD STRATEGY
            # =================================================

            elif strategy_mode == "FIXED":

                if candles_held >= holding_period:

                    should_exit = True

                    exit_reason = (
                        "FIXED_HOLD"
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
                        exit_reason,

                    "holding_candles":
                        candles_held
                })


                cooldown_until = (
                    i + cooldown
                )


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
            and i >= cooldown_until
        ):

            prediction = predictions[i]

            signal = "WAIT"


            # ------------------------------------------------
            # LONG
            # ------------------------------------------------

            if prediction >= threshold:

                signal = "LONG"


            # ------------------------------------------------
            # SHORT
            # ------------------------------------------------

            elif prediction <= -threshold:

                signal = "SHORT"


            # ------------------------------------------------
            # ENTER AT NEXT CANDLE OPEN
            # ------------------------------------------------

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
                "END_OF_DATA",

            "holding_candles":
                len(data) - entry_index
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


    total_pnl = (
        balance
        - STARTING_BALANCE
    )


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


    max_drawdown = (
        drawdown.min()
    )


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
].copy().reset_index(
    drop=True
)


print()
print("=" * 70)
print("VALIDATION ROBUSTNESS TEST")
print("=" * 70)


results = []


for threshold in THRESHOLDS:

    for holding in HOLDING_PERIODS:


        # ====================================================
        # FLIP
        # ====================================================

        result = run_backtest(
            validation_df,
            validation_predictions,
            threshold,
            holding,
            "FLIP",
            0
        )


        results.append({

            "strategy":
                "FLIP",

            "threshold":
                threshold,

            "holding":
                holding,

            "cooldown":
                0,

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


        # ====================================================
        # FIXED HOLD - NO COOLDOWN
        # ====================================================

        result = run_backtest(
            validation_df,
            validation_predictions,
            threshold,
            holding,
            "FIXED",
            0
        )


        results.append({

            "strategy":
                "FIXED",

            "threshold":
                threshold,

            "holding":
                holding,

            "cooldown":
                0,

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


        # ====================================================
        # FIXED HOLD + COOLDOWN
        # ====================================================

        result = run_backtest(
            validation_df,
            validation_predictions,
            threshold,
            holding,
            "FIXED",
            3
        )


        results.append({

            "strategy":
                "FIXED",

            "threshold":
                threshold,

            "holding":
                holding,

            "cooldown":
                3,

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


# ============================================================
# VALIDATION RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


print()
print("TOP VALIDATION RESULTS")
print("-" * 70)


display_columns = [
    "strategy",
    "threshold",
    "holding",
    "cooldown",
    "return_pct",
    "profit_factor",
    "win_rate",
    "trades",
    "max_drawdown"
]


top_results = (
    results_df
    .sort_values(
        "return_pct",
        ascending=False
    )
    .head(10)
)


print(
    top_results[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST STRATEGY
# ============================================================

valid_candidates = results_df[
    (results_df["return_pct"] > 0)
    &
    (results_df["profit_factor"] > 1.0)
    &
    (results_df["trades"] >= 20)
]


if len(valid_candidates) > 0:

    best = (
        valid_candidates
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
        results_df
        .sort_values(
            "return_pct",
            ascending=False
        )
        .iloc[0]
    )


best_strategy = best[
    "strategy"
]

best_threshold = float(
    best["threshold"]
)

best_holding = int(
    best["holding"]
)

best_cooldown = int(
    best["cooldown"]
)


print()
print("=" * 70)
print("SELECTED VALIDATION STRATEGY")
print("=" * 70)

print(
    f"Strategy       : {best_strategy}"
)

print(
    f"Threshold      : "
    f"{best_threshold * 100:.2f}%"
)

print(
    f"Holding        : "
    f"{best_holding * 5} minutes"
)

print(
    f"Cooldown       : "
    f"{best_cooldown * 5} minutes"
)

print(
    f"Return         : "
    f"{best['return_pct']:.2f}%"
)

print(
    f"Profit factor  : "
    f"{best['profit_factor']:.2f}"
)

print(
    f"Win rate       : "
    f"{best['win_rate']:.2f}%"
)

print(
    f"Trades         : "
    f"{int(best['trades'])}"
)

print(
    f"Drawdown       : "
    f"{best['max_drawdown']:.2f}%"
)


# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

results_df.to_csv(
    "robustness_validation_results.csv",
    index=False
)


# ============================================================
# FINAL MODEL
# ============================================================

print()
print("=" * 70)
print("FINAL UNSEEN TEST")
print("=" * 70)

print()
print(
    "Retraining model on "
    "TRAIN + VALIDATION..."
)


X_final_train = X.iloc[
    :validation_end
]


y_final_train = y.iloc[
    :validation_end
]


final_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


final_model.fit(
    X_final_train,
    y_final_train
)


final_test_predictions = (
    final_model.predict(
        X_test
    )
)


final_test_df = df.iloc[
    validation_end:
].copy().reset_index(
    drop=True
)


# ============================================================
# FINAL TEST
# ============================================================

final_result = run_backtest(

    final_test_df,

    final_test_predictions,

    best_threshold,

    best_holding,

    best_strategy,

    best_cooldown
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("ROBUSTNESS FINAL TEST RESULTS")
print("=" * 70)

print()
print(
    f"Strategy       : "
    f"{best_strategy}"
)

print(
    f"Threshold      : "
    f"{best_threshold * 100:.2f}%"
)

print(
    f"Holding        : "
    f"{best_holding * 5} minutes"
)

print(
    f"Cooldown       : "
    f"{best_cooldown * 5} minutes"
)

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

trades_df = final_result[
    "trades_df"
]


if len(trades_df) > 0:

    long_trades = trades_df[
        trades_df["direction"]
        == "LONG"
    ]


    short_trades = trades_df[
        trades_df["direction"]
        == "SHORT"
    ]


    print()
    print("TRADE BREAKDOWN")
    print("-" * 70)

    print(
        f"Long trades  : "
        f"{len(long_trades)}"
    )

    print(
        f"Short trades : "
        f"{len(short_trades)}"
    )


    print()
    print("First 10 trades:")


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
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


    trades_df.to_csv(
        "robustness_final_test_trades.csv",
        index=False
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("ROBUSTNESS TEST COMPLETE")
print("=" * 70)