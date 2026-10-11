import sys
import io
from google.cloud import bigquery
import pandas as pd

# Reconfigure stdout/stderr for UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Initialize client (Ensure you are logged in via CLI: gcloud auth application-default login)
project_id = "mlops-fraud-pipeline-dev"
dataset_id = "fraud_detection_dwh"
client = bigquery.Client(project=project_id)

# List of CSV files and destination BigQuery tables
files_to_load = [
    {
        "csv_path": "ieee-fraud-detection/train_identity.csv",
        "table_id": f"{project_id}.{dataset_id}.raw_identity",
    },
    {
        "csv_path": "ieee-fraud-detection/train_transaction.csv",
        "table_id": f"{project_id}.{dataset_id}.raw_transaction",
    },
]

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # Overwrite if table exists
)

for item in files_to_load:
    csv_path = item["csv_path"]
    table_id = item["table_id"]
    
    print(f"\n--------------------------------------------------")
    print(f"Reading CSV file: {csv_path} ...")
    df = pd.read_csv(csv_path)
    
    print(f"Loading {len(df)} rows ({df.shape[1]} columns) to table {table_id}...")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # Wait for job completion
    
    print(f"✅ Successfully loaded {job.output_rows} rows into table {table_id}")
