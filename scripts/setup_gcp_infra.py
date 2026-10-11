"""
Setup Google Cloud Infrastructure for MLOps Fraud Detection Project.
- Enables required Google Cloud APIs (Vertex AI, Cloud Storage, Artifact Registry, Cloud Build)
- Creates Cloud Storage Bucket for pipeline artifacts and model staging
- Validates BigQuery dataset availability
"""

import os
import sys
import io
import yaml
import subprocess

# Reconfigure stdout/stderr for UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from google.cloud import storage, bigquery

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_command(cmd):
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print(f"Warning / Error: {result.stderr.strip()}")
    else:
        print(f"Success: {result.stdout.strip()}")
    return result

def enable_apis(project_id):
    apis = [
        "aiplatform.googleapis.com",
        "storage.googleapis.com",
        "artifactregistry.googleapis.com",
        "cloudbuild.googleapis.com",
        "bigquery.googleapis.com"
    ]
    print(f"\n--- [1/3] Enabling Google Cloud APIs for project '{project_id}' ---")
    cmd = ["gcloud", "services", "enable"] + apis + [f"--project={project_id}"]
    run_command(cmd)

def create_gcs_bucket(project_id, bucket_name, region):
    print(f"\n--- [2/3] Checking and creating GCS Bucket: gs://{bucket_name} ---")
    client = storage.Client(project=project_id)
    try:
        bucket = client.lookup_bucket(bucket_name)
        if bucket is None:
            print(f"Bucket 'gs://{bucket_name}' does not exist. Creating new bucket in location '{region}'...")
            new_bucket = client.create_bucket(bucket_name, location=region)
            print(f"✅ Successfully created bucket: gs://{new_bucket.name}")
        else:
            print(f"✅ Bucket 'gs://{bucket_name}' already exists and is ready for use.")
    except Exception as e:
        print(f"Notice while creating bucket: {e}")
        print("If the bucket name is globally taken, please update the bucket name in 'config/config.yaml'.")

def verify_bigquery(project_id, dataset_id):
    print(f"\n--- [3/3] Checking BigQuery Dataset: {dataset_id} ---")
    client = bigquery.Client(project=project_id)
    dataset_ref = f"{project_id}.{dataset_id}"
    try:
        dataset = client.get_dataset(dataset_ref)
        print(f"✅ Dataset '{dataset_ref}' is valid and ready.")
        tables = list(client.list_tables(dataset))
        print(f"Existing tables list ({len(tables)} tables):")
        for t in tables:
            print(f" - {t.table_id}")
    except Exception as e:
        print(f"BigQuery verification error: {e}")

def main():
    config = load_config()
    bq_cfg = config.get("data", {}).get("bigquery", {})
    project_id = bq_cfg.get("project_id", "mlops-fraud-pipeline-dev")
    region = "us-central1"
    bucket_name = f"{project_id}-artifacts"
    dataset_id = bq_cfg.get("dataset_id", "fraud_detection_dwh")

    print(f"=== INITIALIZING GCP MLOPS INFRASTRUCTURE ===")
    print(f"Project ID: {project_id}")
    print(f"Region    : {region}")

    enable_apis(project_id)
    create_gcs_bucket(project_id, bucket_name, region)
    verify_bigquery(project_id, dataset_id)

    print("\n✅ GCP MLOps infrastructure setup completed!")

if __name__ == "__main__":
    main()
