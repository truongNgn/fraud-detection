"""
FastAPI Model Serving Application for Real-Time Fraud Detection.
Exposes RESTful endpoints for healthchecks, model metadata, and transaction fraud scoring.
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
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
import mlflow
from mlflow.tracking import MlflowClient

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

# Global variables to store model in memory for ultra-fast serving
MODEL = None
ENCODERS = None
MODEL_VERSION = "local-v1"
MODEL_STAGE = "Production"

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_active_model()
    yield

# Initialize FastAPI App
app = FastAPI(
    title="Fraud Detection Inference Service",
    description="Real-time transaction fraud scoring API powered by XGBoost & MLflow",
    version="1.0.0",
    lifespan=lifespan
)

class TransactionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    TransactionAmt: float = Field(..., json_schema_extra={"example": 150.50}, description="Transaction amount")
    TransactionDT: Optional[int] = Field(86400, description="Transaction timestamp in seconds")
    ProductCD: str = Field("W", description="Product code (W, C, R, H, S)")
    card1: int = Field(13926, description="Card code 1")
    card2: Optional[float] = Field(111.0, description="Card code 2")
    card3: Optional[float] = Field(150.0, description="Card code 3")
    card4: Optional[str] = Field("discover", description="Card brand (visa, mastercard, discover, etc.)")
    card5: Optional[float] = Field(142.0, description="Card code 5")
    card6: Optional[str] = Field("credit", description="Card type (credit, debit)")
    addr1: Optional[float] = Field(315.0, description="Address code 1")
    addr2: Optional[float] = Field(87.0, description="Address code 2")
    dist1: Optional[float] = Field(19.0, description="Estimated distance")
    P_emaildomain: Optional[str] = Field("gmail.com", description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    DeviceType: Optional[str] = Field("desktop", description="Device type (desktop, mobile)")
    DeviceInfo: Optional[str] = Field("Windows", description="Device information")

    # Optional numerical features
    C1: Optional[float] = 1.0
    C2: Optional[float] = 1.0
    C13: Optional[float] = 1.0
    D1: Optional[float] = 14.0
    D2: Optional[float] = 0.0

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    action_recommended: str
    model_version: str
    latency_ms: float

def load_active_model():
    """
    Load active model: Prefers model in 'Production' stage from MLflow,
    otherwise falls back to local artifact 'models/model.joblib'.
    """
    global MODEL, ENCODERS, MODEL_VERSION, MODEL_STAGE
    config = load_config()
    mlflow_cfg = config.get("mlflow", {})
    registered_model_name = mlflow_cfg.get("registered_model_name", "fraud_detector_xgb")
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")

    # 1. Load encoders
    encoders_path = "models/encoders.joblib"
    if os.path.exists(encoders_path):
        ENCODERS = joblib.load(encoders_path)
    else:
        ENCODERS = {}

    # 2. Attempt loading from MLflow
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        prod_models = client.get_latest_versions(registered_model_name, stages=["Production"])
        if prod_models:
            latest = prod_models[0]
            model_uri = f"models:/{registered_model_name}/Production"
            MODEL = mlflow.xgboost.load_model(model_uri)
            MODEL_VERSION = f"v{latest.version}"
            MODEL_STAGE = "Production"
            print(f"✅ Successfully loaded Production Model: {model_uri}")
            return
    except Exception as e:
        print(f"Notice while loading model from MLflow: {e}")

    # Fallback to local artifact
    local_path = "models/model.joblib"
    if os.path.exists(local_path):
        MODEL = joblib.load(local_path)
        MODEL_VERSION = "local-artifact"
        MODEL_STAGE = "Local"
        print(f"✅ Loaded fallback model from '{local_path}'")
    else:
        print("⚠️ No model artifact found. Please run training pipeline before calling predict!")

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION,
        "model_stage": MODEL_STAGE
    }

@app.get("/model-info")
def model_info():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded in memory.")
    config = load_config()
    return {
        "model_name": config["mlflow"].get("registered_model_name", "fraud_detector_xgb"),
        "version": MODEL_VERSION,
        "stage": MODEL_STAGE,
        "framework": "XGBoost",
        "threshold": config["model"].get("min_auc_threshold", 0.80)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(payload: TransactionPayload):
    if MODEL is None:
        load_active_model()
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model is not ready. Please run training first!")

    start_time = time.time()
    data = payload.model_dump()

    # Preprocessing identical to training pipeline
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

    # Encode categoricals using saved encoders
    if ENCODERS:
        for col, mapping in ENCODERS.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

    # Impute missing values
    df = df.fillna(0)

    # Align columns with model features
    try:
        expected_features = MODEL.get_booster().feature_names
        for f in expected_features:
            if f not in df.columns:
                df[f] = 0
        df = df[expected_features]
    except Exception:
        pass

    # Predict
    prob = float(MODEL.predict_proba(df)[:, 1][0])
    is_fraud = bool(prob >= 0.5)

    if prob >= 0.70:
        risk_level = "CRITICAL / HIGH RISK"
        action = "DECLINE_TRANSACTION"
    elif prob >= 0.35:
        risk_level = "MEDIUM RISK"
        action = "REQUIRE_OTP_OR_MANUAL_REVIEW"
    else:
        risk_level = "LOW RISK"
        action = "APPROVE_TRANSACTION"

    latency = round((time.time() - start_time) * 1000, 2)

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(prob, 4),
        risk_level=risk_level,
        action_recommended=action,
        model_version=MODEL_VERSION,
        latency_ms=latency
    )

if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    host = cfg.get("serving", {}).get("host", "127.0.0.1")
    port = cfg.get("serving", {}).get("port", 8000)
    print(f"🚀 Starting Fraud Detection API Server on http://{host}:{port} ...")
    uvicorn.run("src.serving.app:app", host=host, port=port, reload=False)
