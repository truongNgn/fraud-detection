"""
Prediction and Inference Test Module for Fraud Detection.
Uses the trained XGBoost model and MLflow Production Registry for instant risk scoring.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import io
import time
import yaml
import joblib
import pandas as pd
import numpy as np

# Reconfigure stdout/stderr for UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def predict_transaction(sample: dict):
    """
    Predicts fraud probability for an individual transaction.
    """
    config = load_config()
    model_path = "models/model.joblib"
    encoders_path = "models/encoders.joblib"

    if not os.path.exists(model_path) or not os.path.exists(encoders_path):
        raise FileNotFoundError("Model or encoders not found in 'models/'. Please run training pipeline first!")

    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)

    # Feature transformation
    data = sample.copy()
    amt = data.get("TransactionAmt", 0)
    data["log_transaction_amt"] = float(np.log1p(amt))

    dt = data.get("TransactionDT", 0)
    data["hour_of_day"] = int((dt // 3600) % 24)
    data["day_of_week"] = int((dt // 86400) % 7)

    p_email = str(data.get("P_emaildomain") or "missing")
    data["p_email_provider"] = p_email.split(".")[0] if "." in p_email else "missing"

    r_email = str(data.get("R_emaildomain") or "missing")
    data["r_email_provider"] = r_email.split(".")[0] if "." in r_email else "missing"

    df = pd.DataFrame([data])

    # Encode categoricals
    for col, mapping in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

    df = df.fillna(0)

    # Align columns with model feature names
    expected_features = model.get_booster().feature_names
    for f in expected_features:
        if f not in df.columns:
            df[f] = 0
    df = df[expected_features]

    # Predict
    prob = float(model.predict_proba(df)[:, 1][0])
    is_fraud = bool(prob >= 0.5)

    if prob >= 0.70:
        risk_level = "CRITICAL / HIGH RISK"
        decision = "DECLINE_TRANSACTION"
    elif prob >= 0.35:
        risk_level = "MEDIUM RISK"
        decision = "REQUIRE_OTP_OR_MANUAL_REVIEW"
    else:
        risk_level = "LOW RISK"
        decision = "APPROVE_TRANSACTION"

    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(prob, 4),
        "risk_level": risk_level,
        "decision": decision
    }

if __name__ == "__main__":
    print("=== [TEST: REAL-TIME FRAUD PROBABILITY PREDICTION] ===")

    # 1. Normal legitimate transaction
    sample_legit = {
        "TransactionAmt": 45.0,
        "TransactionDT": 86400,
        "ProductCD": "W",
        "card1": 13926,
        "card2": 111.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 142.0,
        "card6": "debit",
        "addr1": 315.0,
        "addr2": 87.0,
        "dist1": 5.0,
        "P_emaildomain": "gmail.com",
        "DeviceType": "desktop",
        "DeviceInfo": "Windows",
        "C1": 1.0, "C2": 1.0, "C13": 1.0, "D1": 14.0
    }

    print("\n--- Transaction 1: Typical E-commerce Purchase (Legitimate) ---")
    res1 = predict_transaction(sample_legit)
    for k, v in res1.items():
        print(f" - {k:20}: {v}")

    # 2. Suspicious fraudulent transaction
    sample_fraud = {
        "TransactionAmt": 3500.0,
        "TransactionDT": 180000,
        "ProductCD": "C",
        "card1": 9999,
        "card2": 555.0,
        "card3": 185.0,
        "card4": "discover",
        "card5": 137.0,
        "card6": "credit",
        "addr1": 441.0,
        "addr2": 87.0,
        "dist1": 2500.0,
        "P_emaildomain": "protonmail.com",
        "DeviceType": "mobile",
        "DeviceInfo": "SM-G950F",
        "C1": 35.0, "C2": 40.0, "C13": 80.0, "D1": 0.0
    }

    print("\n--- Transaction 2: Large Amount, Distant IP/Address (Suspicious) ---")
    res2 = predict_transaction(sample_fraud)
    for k, v in res2.items():
        print(f" - {k:20}: {v}")
