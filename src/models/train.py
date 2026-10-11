"""
Model Training Module with MLflow Tracking & Model Registry.
Trains an XGBoost model, tracks experiments, logs metrics, generates visualization artifacts,
and automatically registers top-performing models in the MLflow Model Registry.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import io
import argparse
import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Reconfigure stdout/stderr for UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import xgboost as xgb
import mlflow
import mlflow.xgboost

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def preprocess_features(df, is_train=True, encoders=None):
    """
    Data preprocessing: categorical encoding and missing value imputation.
    """
    df = df.copy()
    if "TransactionID" in df.columns:
        df = df.drop(columns=["TransactionID"])

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if encoders is None:
        encoders = {}

    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna("missing")
        if is_train:
            categories = {val: idx for idx, val in enumerate(df[col].unique())}
            encoders[col] = categories
            df[col] = df[col].map(categories).fillna(0).astype(int)
        else:
            categories = encoders.get(col, {})
            df[col] = df[col].map(categories).fillna(-1).astype(int)

    # Impute missing numericals with 0
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if col != "isFraud":
            df[col] = df[col].fillna(0)

    return df, encoders

def generate_evaluation_plots(y_true, y_pred_proba, model, feature_names, output_dir="models/plots"):
    """
    Generates evaluation plots (ROC Curve, Confusion Matrix, Feature Importance) for MLflow Artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    plots = {}

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Validation ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    roc_plot_path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(roc_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots["roc_curve"] = roc_plot_path

    # 2. Confusion Matrix
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Legit (0)", "Fraud (1)"],
                yticklabels=["Legit (0)", "Fraud (1)"])
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Validation Confusion Matrix (Threshold=0.5)", fontsize=13, fontweight="bold")
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots["confusion_matrix"] = cm_plot_path

    # 3. Top 15 Feature Importance
    try:
        importances = model.feature_importances_
        feat_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        feat_df = feat_df.sort_values("importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=feat_df, x="importance", y="feature", hue="feature", palette="viridis", legend=False, ax=ax)
        ax.set_title("Top 15 Most Important Features", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_ylabel("Feature", fontsize=11)
        feat_plot_path = os.path.join(output_dir, "feature_importance.png")
        fig.savefig(feat_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots["feature_importance"] = feat_plot_path
    except Exception as e:
        print(f"Skipping feature importance plot: {e}")

    return plots

def train_and_track():
    config = load_config()
    mlflow_cfg = config["mlflow"]
    model_cfg = config["model"]
    data_dir = config["data"].get("processed_dir", "data/processed")

    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "val.parquet")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Processed datasets not found in '{data_dir}'. Run 'python src/data/feature_engineering.py' first!")

    print("=== [STAGE 2: MODEL TRAINING & MLFLOW TRACKING] ===")
    print(f"-> Reading data from '{data_dir}'...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # 1. Preprocessing
    train_processed, encoders = preprocess_features(train_df, is_train=True)
    val_processed, _ = preprocess_features(val_df, is_train=False, encoders=encoders)

    target_col = model_cfg.get("target_column", "isFraud")
    X_train = train_processed.drop(columns=[target_col])
    y_train = train_processed[target_col].astype(int)

    X_val = val_processed.drop(columns=[target_col])
    y_val = val_processed[target_col].astype(int)

    feature_names = X_train.columns.tolist()

    # 2. Configure MLflow Tracking
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_cfg.get("experiment_name", "ieee-fraud-detection")
    registered_model_name = mlflow_cfg.get("registered_model_name", "fraud_detector_xgb")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    hyperparams = model_cfg.get("hyperparameters", {})
    print(f"-> MLflow Tracking URI: {tracking_uri}")
    print(f"-> MLflow Experiment  : {experiment_name}")
    print(f"-> Hyperparameters    : {hyperparams}")

    with mlflow.start_run(run_name="xgboost-baseline") as run:
        run_id = run.info.run_id
        print(f"\n🚀 Running MLflow Run ID: {run_id}")

        # Log parameters
        mlflow.log_params(hyperparams)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("num_features", len(feature_names))

        # 3. Train XGBoost Model
        model = xgb.XGBClassifier(
            max_depth=hyperparams.get("max_depth", 6),
            learning_rate=hyperparams.get("learning_rate", 0.05),
            n_estimators=hyperparams.get("n_estimators", 150),
            subsample=hyperparams.get("subsample", 0.8),
            colsample_bytree=hyperparams.get("colsample_bytree", 0.8),
            scale_pos_weight=hyperparams.get("scale_pos_weight", 27.0),
            random_state=hyperparams.get("random_state", 42),
            eval_metric="auc",
            early_stopping_rounds=20
        )

        print("-> Training XGBoost Classifier...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=30
        )

        # 4. Compute Validation Metrics
        val_preds_proba = model.predict_proba(X_val)[:, 1]
        val_preds_binary = (val_preds_proba >= 0.5).astype(int)

        val_roc_auc = roc_auc_score(y_val, val_preds_proba)
        val_pr_auc = average_precision_score(y_val, val_preds_proba)
        val_f1 = f1_score(y_val, val_preds_binary)

        metrics = {
            "val_roc_auc": float(val_roc_auc),
            "val_pr_auc": float(val_pr_auc),
            "val_f1_score": float(val_f1)
        }
        mlflow.log_metrics(metrics)

        print("\n--- VALIDATION EVALUATION RESULTS ---")
        print(f" - ROC-AUC Score : {val_roc_auc:.4f}")
        print(f" - PR-AUC Score  : {val_pr_auc:.4f}")
        print(f" - F1-Score      : {val_f1:.4f}")

        # 5. Generate and Log Visual Artifacts
        print("-> Generating evaluation plots and logging artifacts to MLflow...")
        plot_files = generate_evaluation_plots(y_val, val_preds_proba, model, feature_names)
        for _, path in plot_files.items():
            mlflow.log_artifact(path, artifact_path="evaluation_plots")

        # 6. Save Preprocessor and Model Artifacts
        os.makedirs("models", exist_ok=True)
        local_model_path = "models/model.joblib"
        local_encoders_path = "models/encoders.joblib"
        joblib.dump(model, local_model_path)
        joblib.dump(encoders, local_encoders_path)

        mlflow.log_artifact(local_encoders_path, artifact_path="preprocessors")

        # 7. Register in MLflow Model Registry
        min_auc = model_cfg.get("min_auc_threshold", 0.80)
        if val_roc_auc >= min_auc:
            print(f"\n✅ Model MET CRITERIA (ROC-AUC {val_roc_auc:.4f} >= {min_auc})!")
            print(f"-> Registering into MLflow Model Registry as '{registered_model_name}'...")
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                registered_model_name=registered_model_name
            )
            print(f"✅ Successfully registered Model in MLflow Model Registry!")
        else:
            print(f"\n⚠️ Model below threshold (ROC-AUC {val_roc_auc:.4f} < {min_auc}). Logging run without registry promotion.")
            mlflow.xgboost.log_model(xgb_model=model, artifact_path="model")

    print("\n✅ Model Training & Tracking Stage Completed!")
    return run_id, metrics

if __name__ == "__main__":
    train_and_track()
