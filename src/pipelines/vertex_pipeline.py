"""
Vertex AI Pipeline Definition using Kubeflow Pipelines (KFP).
Orchestrates:
1. Feature Engineering (BigQuery)
2. Model Training (XGBoost)
3. Model Evaluation & Threshold Validation
4. Model Registry and Endpoint Deployment
"""

from kfp import dsl
from kfp.dsl import component, Output, Artifact, Metrics, Model, Input

@component(
    packages_to_install=["google-cloud-bigquery>=3.25.0", "pyyaml>=6.0.1"],
    base_image="python:3.10-slim"
)
def feature_engineering_component(
    project_id: str,
    dataset_id: str,
    features_table: str
) -> str:
    from google.cloud import bigquery
    client = bigquery.Client(project=project_id)
    
    query = f"""
    CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{features_table}` AS
    SELECT
        t.TransactionID,
        t.isFraud,
        t.TransactionAmt,
        LOG(t.TransactionAmt + 1.0) AS log_transaction_amt,
        MOD(CAST(FLOOR(t.TransactionDT / 3600) AS INT64), 24) AS hour_of_day,
        MOD(CAST(FLOOR(t.TransactionDT / 86400) AS INT64), 7) AS day_of_week,
        t.ProductCD, t.card1, t.card2, t.card3, t.card4, t.card5, t.card6,
        t.addr1, t.addr2, t.dist1,
        SPLIT(COALESCE(t.P_emaildomain, 'missing'), '.')[SAFE_OFFSET(0)] AS p_email_provider,
        t.C1, t.C2, t.C3, t.C4, t.C5, t.C6, t.C7, t.C8, t.C9, t.C10, t.C11, t.C12, t.C13, t.C14,
        t.D1, t.D2, t.D3, t.D4, t.D5, t.D10, t.D11, t.D15,
        i.DeviceType, i.DeviceInfo,
        CASE 
            WHEN ABS(MOD(FARM_FINGERPRINT(CAST(t.TransactionID AS STRING)), 10)) < 7 THEN 'TRAIN'
            WHEN ABS(MOD(FARM_FINGERPRINT(CAST(t.TransactionID AS STRING)), 10)) < 9 THEN 'VALIDATION'
            ELSE 'TEST'
        END AS data_split
    FROM `{project_id}.{dataset_id}.raw_transaction` t
    LEFT JOIN `{project_id}.{dataset_id}.raw_identity` i
        ON t.TransactionID = i.TransactionID;
    """
    client.query(query).result()
    return f"{project_id}.{dataset_id}.{features_table}"

@component(
    packages_to_install=[
        "google-cloud-bigquery>=3.25.0",
        "xgboost>=2.0.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0"
    ],
    base_image="python:3.10-slim"
)
def train_model_component(
    project_id: str,
    dataset_id: str,
    features_table: str,
    metrics_output: Output[Metrics],
    model_output: Output[Model]
):
    import joblib
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, average_precision_score
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id)
    
    query = f"""
    SELECT * EXCEPT(data_split, TransactionID)
    FROM `{project_id}.{dataset_id}.{features_table}`
    WHERE data_split = 'TRAIN'
    LIMIT 50000;
    """
    train_df = client.query(query).to_dataframe()
    
    val_query = f"""
    SELECT * EXCEPT(data_split, TransactionID)
    FROM `{project_id}.{dataset_id}.{features_table}`
    WHERE data_split = 'VALIDATION'
    LIMIT 15000;
    """
    val_df = client.query(val_query).to_dataframe()
    
    # Preprocessing
    for col in train_df.select_dtypes(include=["object"]).columns:
        cats = {val: i for i, val in enumerate(train_df[col].unique())}
        train_df[col] = train_df[col].map(cats).fillna(0)
        val_df[col] = val_df[col].map(cats).fillna(-1)
        
    train_df = train_df.fillna(0)
    val_df = val_df.fillna(0)
    
    X_train = train_df.drop(columns=["isFraud"])
    y_train = train_df["isFraud"].astype(int)
    X_val = val_df.drop(columns=["isFraud"])
    y_val = val_df["isFraud"].astype(int)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=27.0,
        eval_metric="auc"
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    pr_auc = average_precision_score(y_val, preds)
    
    metrics_output.log_metric("val_roc_auc", float(auc))
    metrics_output.log_metric("val_pr_auc", float(pr_auc))
    
    # Save model artifact
    joblib.dump(model, model_output.path)

@component(
    packages_to_install=[
        "google-cloud-bigquery>=3.25.0",
        "google-cloud-aiplatform>=1.50.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "pandas>=2.0.0",
        "joblib>=1.3.0"
    ],
    base_image="python:3.10-slim"
)
def evaluate_and_deploy_component(
    project_id: str,
    region: str,
    dataset_id: str,
    features_table: str,
    model_input: Input[Model],
    min_auc_threshold: float
) -> str:
    import joblib
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from google.cloud import bigquery, aiplatform
    
    model = joblib.load(model_input.path)
    client = bigquery.Client(project=project_id)
    
    test_query = f"""
    SELECT * EXCEPT(data_split, TransactionID)
    FROM `{project_id}.{dataset_id}.{features_table}`
    WHERE data_split = 'TEST'
    LIMIT 10000;
    """
    test_df = client.query(test_query).to_dataframe()
    for col in test_df.select_dtypes(include=["object"]).columns:
        test_df[col] = test_df[col].astype("category").cat.codes
    test_df = test_df.fillna(0)
    
    X_test = test_df.drop(columns=["isFraud"])
    y_test = test_df["isFraud"].astype(int)
    
    preds = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, preds)
    print(f"Test AUC: {test_auc} (Required threshold: {min_auc_threshold})")
    
    if test_auc < min_auc_threshold:
        return f"REJECTED: Test AUC {test_auc:.4f} < {min_auc_threshold}"
        
    aiplatform.init(project=project_id, location=region)
    uploaded_model = aiplatform.Model.upload(
        display_name="fraud_detector_xgb_pipeline",
        artifact_uri=model_input.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"
    )
    return f"DEPLOYED: {uploaded_model.resource_name}"

@dsl.pipeline(
    name="fraud-detection-end-to-end-pipeline",
    description="Full MLOps Pipeline: Feature Engineering -> Training -> Evaluation -> Deployment"
)
def fraud_detection_pipeline(
    project_id: str = "mlops-fraud-pipeline-dev",
    region: str = "us-central1",
    dataset_id: str = "fraud_detection_dwh",
    features_table: str = "fraud_features",
    min_auc_threshold: float = 0.80
):
    # Stage 1: Feature Engineering
    feat_step = feature_engineering_component(
        project_id=project_id,
        dataset_id=dataset_id,
        features_table=features_table
    )
    
    # Stage 2: Train Model
    train_step = train_model_component(
        project_id=project_id,
        dataset_id=dataset_id,
        features_table=features_table
    )
    train_step.after(feat_step)
    
    # Stage 3: Evaluate & Deploy
    eval_step = evaluate_and_deploy_component(
        project_id=project_id,
        region=region,
        dataset_id=dataset_id,
        features_table=features_table,
        model_input=train_step.outputs["model_output"],
        min_auc_threshold=min_auc_threshold
    )
    eval_step.after(train_step)
