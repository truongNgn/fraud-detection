# 🛡️ IEEE-CIS Fraud Detection - End-to-End Open-Source MLOps Pipeline

An end-to-end, production-ready MLOps pipeline for IEEE-CIS Fraud Detection built with an industry-standard open-source technology stack: **MLflow + FastAPI + XGBoost + Docker + GitHub Actions**, completely free from cloud vendor lock-in.

---

## 🏛️ Architecture Overview

```
[Kaggle Raw Data / BigQuery Sandbox]
       │
       ▼ (Stage 1: Feature Engineering)
[data/processed: train.parquet, val.parquet, test.parquet]
       │
       ▼ (Stage 2: Model Training & Tracking)
[XGBoost Classifier + Class Imbalance Handling]
       │
       ├── Log Hyperparameters, Metrics & Artifacts ──> [MLflow Tracking Server]
       │
       ▼ (Stage 3: Evaluation & Quality Gate)
[Test ROC-AUC >= 0.80]
       │
       ├── PASS ──> Promote Model to 'Production' ──> [MLflow Model Registry]
       │
       ▼ (Stage 4: Serving & Inference)
[FastAPI REST API Service (Swagger UI)] ──> Real-time Fraud Scoring
       │
       ▲
       │
[CI/CD: GitHub Actions] & [Containerization: Docker / Docker Compose]
```

---

## 📂 Project Directory Structure

```
fraud-detection/
├── config/
│   └── config.yaml               # Central configuration (data, mlflow, model, serving)
├── data/
│   └── processed/                # Processed train/val/test data splits in Parquet format
├── ieee-fraud-detection/         # Original raw Kaggle CSV dataset
├── models/
│   ├── model.joblib              # Serialized trained model artifact
│   ├── encoders.joblib           # Preprocessing encoders
│   └── plots/                    # Evaluation plots (ROC curve, Confusion Matrix, Feature Importance)
├── src/
│   ├── data/
│   │   └── feature_engineering.py# Feature engineering & data splitting
│   ├── models/
│   │   ├── train.py              # XGBoost training & MLflow tracking
│   │   └── evaluate.py           # Evaluation on holdout test set & production promotion
│   ├── pipelines/
│   │   └── run_pipeline.py       # Pipeline orchestrator running all stages end-to-end
│   └── serving/
│       ├── app.py                # FastAPI REST API serving service
│       ├── predict.py            # Standalone inference verification script
│       └── test_api.py           # HTTP client test simulation
├── tests/
│   └── test_pipeline.py          # Unit & integration tests for transformations and schemas
├── .github/workflows/
│   └── mlops-ci-cd.yaml          # Automated CI/CD test workflow via GitHub Actions
├── Dockerfile                    # Container definition for FastAPI inference app
├── docker-compose.yaml           # Runs MLflow Tracking Server + FastAPI together
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Pytest configuration
├── conftest.py                   # Pytest test configuration and path setup
├── UPDATE_REPORT.md              # Detailed architecture migration report
└── README.md                     # Project documentation
```

---

## 🚀 Step-by-Step Walkthrough

### Step 1: Environment Setup
Activate your Python virtual environment and install dependencies:
```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

### Step 2: Execute the End-to-End Pipeline
Run the complete pipeline with a single command:
1. Feature extraction and deterministic data splitting (Train/Val/Test).
2. XGBoost model training and logging metrics/artifacts to MLflow.
3. Model evaluation on holdout test data and automated promotion to `Production` in MLflow Model Registry.

```powershell
python src/pipelines/run_pipeline.py
```
*(Default runs on 50,000 samples for optimal speed. To train on the entire dataset, pass `--sample_limit 0`)*.

---

### Step 3: Explore the MLflow Dashboard
Launch the MLflow Tracking Server and Model Registry UI:
```powershell
mlflow ui --port 5000
```
👉 Open your browser at: **http://127.0.0.1:5000**
- Inspect individual experiment runs and compare metrics (ROC-AUC, PR-AUC, F1-Score).
- View evaluation plots: ROC curves, Confusion Matrices, and Top 15 Feature Importances in the **Artifacts** tab.
- Inspect models registered in the **Model Registry** under the `Production` stage.

---

### Step 4: Start the Real-Time Inference Server (FastAPI)
Launch the FastAPI REST API service:
```powershell
python src/serving/app.py
```
👉 Access the interactive Swagger UI at: **http://127.0.0.1:8000/docs**

---

### Step 5: Test Real-Time Fraud Predictions
In a new terminal, simulate incoming transaction requests (normal vs. suspicious scenarios):
```powershell
python src/serving/test_api.py
# Or run standalone:
python src/serving/predict.py
```

---

### Step 6: Run Automated Tests (Pytest)
Execute unit and integration tests:
```powershell
pytest -v
```

---

### Step 7: Run with Docker Compose (Optional)
If Docker is installed on your machine:
```powershell
docker compose up --build
```
This spins up both the **MLflow Tracking Server** (port 5000) and the **FastAPI Inference Service** (port 8000) simultaneously.
