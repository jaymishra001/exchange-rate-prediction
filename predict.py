"""
Load the trained XGBoost model and predict next-day EUR/USD Close.

Usage:
    python predict.py
"""

import joblib
import pandas as pd

MODEL_PATH = "models/xgb_eurusd_model.pkl"

FEATURE_ORDER = [
    "momentum_rsi",
    "trend_macd_diff",
    "momentum_stoch",
    "momentum_stoch_signal",
    "Close_Lag1",
    "bb_percent_b",
    "bb_width",
]

model = joblib.load(MODEL_PATH)


def predict_next_close(features: dict) -> float:
    """
    features: dict containing all keys in FEATURE_ORDER (any order).
    Returns the model's predicted next-day Close price.
    """
    missing = [f for f in FEATURE_ORDER if f not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = pd.DataFrame([features])[FEATURE_ORDER]
    return float(model.predict(X)[0])


if __name__ == "__main__":
    example = {
        "momentum_rsi": 58.6,
        "trend_macd_diff": 0.00013,
        "momentum_stoch": 50.9,
        "momentum_stoch_signal": 70.9,
        "Close_Lag1": 1.1773,
        "bb_percent_b": 0.637,
        "bb_width": 0.0094,
    }
    prediction = predict_next_close(example)
    print(f"Predicted next Close: {prediction:.5f}")
