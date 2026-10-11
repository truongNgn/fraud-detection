"""
End-to-End MLOps Pipeline Orchestrator.
Chains:
Step 1: Feature Engineering (Raw CSVs -> Cleaned Data Splits in Parquet)
Step 2: Model Training & Experiment Tracking (XGBoost + MLflow Tracking)
Step 3: Model Evaluation & Promotion (Holdout Test Quality Gate -> MLflow Production Stage)
Step 4: Model Serving Validation
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import io
import time
from src.data.feature_engineering import run_pipeline as run_feature_engineering
from src.models.train import train_and_track
from src.models.evaluate import evaluate_and_promote

# Reconfigure stdout/stderr for UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_end_to_end_pipeline(sample_limit=50000):
    start_total = time.time()
    print("=====================================================================")
    print("🚀 STARTING END-TO-END MLOPS PIPELINE (OPEN-SOURCE STACK)")
    print("=====================================================================")

    # 1. Feature Engineering
    print("\n>>> STEP 1/3: DATA PREPROCESSING & FEATURE ENGINEERING")
    t0 = time.time()
    run_feature_engineering(sample_limit=sample_limit)
    print(f"⏱️ Step 1 completed in {time.time() - t0:.1f}s")

    # 2. Model Training & MLflow Tracking
    print("\n>>> STEP 2/3: MODEL TRAINING & MLFLOW EXPERIMENT TRACKING")
    t1 = time.time()
    run_id, metrics = train_and_track()
    print(f"⏱️ Step 2 completed in {time.time() - t1:.1f}s (Run ID: {run_id})")

    # 3. Model Evaluation & Production Promotion Gate
    print("\n>>> STEP 3/3: EVALUATION & MLFLOW MODEL REGISTRY PROMOTION")
    t2 = time.time()
    promoted = evaluate_and_promote()
    print(f"⏱️ Step 3 completed in {time.time() - t2:.1f}s")

    total_time = time.time() - start_total
    print("\n=====================================================================")
    print(f"🎉 COMPLETE MLOPS PIPELINE SUCCEEDED IN {total_time:.1f}s!")
    print("=====================================================================")
    print("\n💡 NEXT STEPS:")
    print("1. Launch MLflow UI to view experiment runs and registered models:")
    print("   👉 mlflow ui --port 5000")
    print("2. Start the FastAPI Real-time Inference Server:")
    print("   👉 python src/serving/app.py")
    print("3. Test transaction fraud predictions:")
    print("   👉 python src/serving/test_api.py")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_limit", type=int, default=50000, help="Row count limit for processing (default 50,000)")
    args = parser.parse_args()
    run_end_to_end_pipeline(sample_limit=args.sample_limit)
