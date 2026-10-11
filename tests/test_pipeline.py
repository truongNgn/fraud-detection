"""
Unit tests for Fraud Detection MLOps Pipeline.
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.data.feature_engineering import engineer_features, load_config
from src.serving.app import TransactionPayload

def test_config():
    config = load_config()
    assert "data" in config
    assert "mlflow" in config
    assert "model" in config
    assert config["model"]["min_auc_threshold"] > 0

def test_engineer_features():
    sample_df = pd.DataFrame({
        "TransactionID": [1001, 1002],
        "isFraud": [0, 1],
        "TransactionAmt": [50.0, 150.0],
        "TransactionDT": [86400, 172800],
        "ProductCD": ["W", "C"],
        "card1": [1000, 2000],
        "card4": ["visa", "mastercard"],
        "P_emaildomain": ["gmail.com", "yahoo.com"]
    })
    
    transformed = engineer_features(sample_df)
    assert "log_transaction_amt" in transformed.columns
    assert "hour_of_day" in transformed.columns
    assert "day_of_week" in transformed.columns
    assert "p_email_provider" in transformed.columns
    assert transformed["p_email_provider"].iloc[0] == "gmail"

def test_transaction_payload_validation():
    valid_payload = {
        "TransactionAmt": 100.5,
        "TransactionDT": 86400,
        "ProductCD": "W",
        "card1": 13926
    }
    payload = TransactionPayload(**valid_payload)
    assert payload.TransactionAmt == 100.5
    assert payload.ProductCD == "W"

def test_fastapi_endpoints():
    from fastapi.testclient import TestClient
    from src.serving.app import app
    
    with TestClient(app) as client:
        # Test Health
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"
        
        # Test Predict
        predict_resp = client.post("/predict", json={
            "TransactionAmt": 75.0,
            "TransactionDT": 86400,
            "ProductCD": "W",
            "card1": 13926,
            "card4": "visa",
            "card6": "debit"
        })
        assert predict_resp.status_code == 200
        data = predict_resp.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert "latency_ms" in data

