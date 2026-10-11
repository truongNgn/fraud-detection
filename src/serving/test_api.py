"""
Client Test Script to simulate incoming transactions against the Fraud Detection Inference Service.
Tests legitimate and suspicious transaction scenarios.
"""

import sys
import io
import requests
import json

# Reconfigure stdout/stderr for UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

API_URL = "http://127.0.0.1:8000"

def test_api():
    print("=== TEST REAL-TIME FRAUD DETECTION INFERENCE SERVICE ===")

    # 1. Health check endpoint
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=3)
        print(f"Health Check: {health_resp.json()}")
    except Exception as e:
        print(f"⚠️ Unable to connect to {API_URL}. Is 'python src/serving/app.py' running?")
        print(f"Detail: {e}")
        return

    # Scenario 1: Normal legitimate transaction
    legit_payload = {
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

    print("\n--- Transaction 1: Regular Grocery Purchase (Legitimate) ---")
    resp1 = requests.post(f"{API_URL}/predict", json=legit_payload)
    print(json.dumps(resp1.json(), indent=2))

    # Scenario 2: Suspicious / Fraudulent transaction
    fraud_payload = {
        "TransactionAmt": 2899.99,
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
        "C1": 25.0, "C2": 30.0, "C13": 50.0, "D1": 0.0
    }

    print("\n--- Transaction 2: High Amount, Unusual Location (Suspicious) ---")
    resp2 = requests.post(f"{API_URL}/predict", json=fraud_payload)
    print(json.dumps(resp2.json(), indent=2))

if __name__ == "__main__":
    test_api()
