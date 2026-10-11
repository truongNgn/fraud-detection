"""
Model Evaluation & Quality Gate Module for IEEE-CIS Fraud Detection.
Evaluates the model on the independent TEST set.
If the model exceeds the quality threshold, transitions the model to 'Production' in MLflow Model Registry.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import io
import yaml
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import mlflow
from mlflow.tracking import MlflowClient
from src.models.train import preprocess_features

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

def evaluate_and_promote():
    config = load_config()
    mlflow_cfg = config["mlflow"]
    model_cfg = config["model"]
    data_dir = config["data"].get("processed_dir", "data/processed")
    min_auc = model_cfg.get("min_auc_threshold", 0.80)

    test_path = os.path.join(data_dir, "test.parquet")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file '{test_path}' not found. Please run feature engineering first!")

    print("=== [STAGE 3: MODEL EVALUATION & QUALITY GATE] ===")
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    registered_model_name = mlflow_cfg.get("registered_model_name", "fraud_detector_xgb")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Load model and encoders
    encoders_path = "models/encoders.joblib"
    if not os.path.exists(encoders_path):
        raise FileNotFoundError("Local encoders file not found. Please run train.py first!")
    encoders = joblib.load(encoders_path)

    # Attempt to load model from MLflow, or fallback to local artifact
    model = None
    latest_version = None
    try:
        versions = client.search_model_versions(f"name='{registered_model_name}'")
        if versions:
            latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            model_uri = f"models:/{registered_model_name}/{latest_version.version}"
            print(f"-> Loading model version v{latest_version.version} from MLflow: {model_uri}")
            model = mlflow.xgboost.load_model(model_uri)
    except Exception as e:
        print(f"Notice while loading from MLflow Registry: {e}")

    if model is None:
        local_model_path = "models/model.joblib"
        print(f"-> Fallback: Loading model from local file: {local_model_path}")
        model = joblib.load(local_model_path)

    # 2. Read TEST split
    print(f"-> Reading TEST split from '{test_path}'...")
    test_df = pd.read_parquet(test_path)

    # 3. Preprocess
    test_processed, _ = preprocess_features(test_df, is_train=False, encoders=encoders)
    target_col = model_cfg.get("target_column", "isFraud")

    X_test = test_processed.drop(columns=[target_col])
    y_test = test_processed[target_col].astype(int)

    # 4. Predict and evaluate
    test_preds_proba = model.predict_proba(X_test)[:, 1]
    test_preds_binary = (test_preds_proba >= 0.5).astype(int)

    test_roc_auc = roc_auc_score(y_test, test_preds_proba)
    test_pr_auc = average_precision_score(y_test, test_preds_proba)

    print("\n--- INDEPENDENT TEST SPLIT (HOLDOUT) EVALUATION ---")
    print(f" - Test ROC-AUC Score : {test_roc_auc:.4f} (Required Threshold: {min_auc})")
    print(f" - Test PR-AUC Score  : {test_pr_auc:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, test_preds_binary, target_names=["Legit", "Fraud"]))

    # 5. Quality Gate validation
    if test_roc_auc < min_auc:
        print(f"❌ WARNING: Model DID NOT MEET quality threshold ({test_roc_auc:.4f} < {min_auc}).")
        print("-> Promotion to Production rejected.")
        return False

    print(f"✅ CONFIRMED: Model MEETS QUALITY THRESHOLD ({test_roc_auc:.4f} >= {min_auc})!")

    # Promote stage in MLflow Model Registry
    if latest_version:
        try:
            print(f"-> Promoting Model v{latest_version.version} to 'Production' in MLflow...")
            client.transition_model_version_stage(
                name=registered_model_name,
                version=latest_version.version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"🎉 Model v{latest_version.version} successfully promoted to 'Production'!")
        except Exception as e:
            print(f"Notice while updating MLflow stage: {e}")

    return True

if __name__ == "__main__":
    evaluate_and_promote()
