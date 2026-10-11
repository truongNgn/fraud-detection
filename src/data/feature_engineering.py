"""
Feature Engineering Module for IEEE-CIS Fraud Detection.
Supports both Local (CSV files) and BigQuery Sandbox data sources.
Transforms raw transactions & identities, creates features, and exports train/val/test splits.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import io
import yaml
import numpy as np
import pandas as pd

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

def load_raw_data(config, sample_limit=None):
    """
    Loads raw data from the local directory or BigQuery Sandbox.
    """
    data_cfg = config["data"]
    source_mode = data_cfg.get("source_mode", "local")

    if source_mode == "local":
        raw_dir = data_cfg.get("raw_dir", "ieee-fraud-detection")
        tx_path = os.path.join(raw_dir, data_cfg.get("train_transaction_file", "train_transaction.csv"))
        id_path = os.path.join(raw_dir, data_cfg.get("train_identity_file", "train_identity.csv"))

        print(f"-> [Data Source: Local] Reading: {tx_path}...")
        df_tx = pd.read_csv(tx_path, nrows=sample_limit)
        print(f"-> Successfully loaded {len(df_tx)} transaction rows.")

        print(f"-> Reading: {id_path}...")
        df_id = pd.read_csv(id_path)
        print(f"-> Successfully loaded {len(df_id)} identity rows.")

        print("-> Merging transaction and identity tables on TransactionID...")
        df = pd.merge(df_tx, df_id, on="TransactionID", how="left")
        return df

    elif source_mode == "bigquery":
        from google.cloud import bigquery
        bq_cfg = data_cfg["bigquery"]
        client = bigquery.Client(project=bq_cfg["project_id"])
        limit_clause = f"LIMIT {sample_limit}" if sample_limit else ""
        query = f"""
        SELECT t.*, i.DeviceType, i.DeviceInfo, i.id_01, i.id_02, i.id_11, i.id_12, i.id_15, i.id_28, i.id_29, i.id_31, i.id_35, i.id_36, i.id_37, i.id_38
        FROM `{bq_cfg['project_id']}.{bq_cfg['dataset_id']}.raw_transaction` t
        LEFT JOIN `{bq_cfg['project_id']}.{bq_cfg['dataset_id']}.raw_identity` i
            ON t.TransactionID = i.TransactionID
        {limit_clause};
        """
        print(f"-> [Data Source: BigQuery] Querying from BigQuery Sandbox...")
        return client.query(query).to_dataframe()

    else:
        raise ValueError(f"Unsupported source_mode: {source_mode}")

def engineer_features(df):
    """
    Performs Feature Engineering:
    - Log transforms transaction amount
    - Extracts cyclical/time features (hour, day)
    - Extracts email domain provider
    - Selects core features
    """
    print("-> Performing Feature Engineering...")
    df = df.copy()

    # 1. Log Transaction Amount
    df["log_transaction_amt"] = np.log1p(df["TransactionAmt"].fillna(0))

    # 2. Time Features from TransactionDT (seconds)
    df["hour_of_day"] = (np.floor(df["TransactionDT"] / 3600) % 24).astype(int)
    df["day_of_week"] = (np.floor(df["TransactionDT"] / 86400) % 7).astype(int)

    # 3. Email domain features
    if "P_emaildomain" in df.columns:
        df["p_email_provider"] = df["P_emaildomain"].astype(str).apply(lambda x: x.split(".")[0] if "." in x else "missing")
    else:
        df["p_email_provider"] = "missing"

    if "R_emaildomain" in df.columns:
        df["r_email_provider"] = df["R_emaildomain"].astype(str).apply(lambda x: x.split(".")[0] if "." in x else "missing")
    else:
        df["r_email_provider"] = "missing"

    # 4. Select core feature set
    core_cols = [
        "TransactionID", "isFraud", "TransactionAmt", "log_transaction_amt",
        "hour_of_day", "day_of_week", "ProductCD",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "dist1", "p_email_provider", "r_email_provider"
    ]

    optional_cols = [
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
        "D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15",
        "M1", "M2", "M3", "M4", "M5", "M6",
        "V1", "V2", "V3", "V4", "V5", "V12", "V13", "V14", "V29", "V30", "V53", "V54", "V75", "V82", "V86", "V90",
        "DeviceType", "DeviceInfo", "id_01", "id_02", "id_11", "id_12", "id_15", "id_28", "id_29", "id_31", "id_35", "id_36", "id_37", "id_38"
    ]

    selected_cols = [c for c in core_cols if c in df.columns]
    for c in optional_cols:
        if c in df.columns:
            selected_cols.append(c)

    return df[selected_cols]

def split_and_save_data(df, output_dir="data/processed"):
    """
    Splits dataset into Train (70%), Validation (15%), Test (15%) and saves as Parquet files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"-> Splitting data and saving to '{output_dir}'...")

    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df))

    train_end = int(0.70 * len(df))
    val_end = int(0.85 * len(df))

    train_df = df.iloc[shuffled_indices[:train_end]]
    val_df = df.iloc[shuffled_indices[train_end:val_end]]
    test_df = df.iloc[shuffled_indices[val_end:]]

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print("\n--- DATA SPLITS SUMMARY ---")
    for name, split in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        fraud_cnt = split["isFraud"].sum()
        pct = (fraud_cnt / len(split)) * 100
        print(f" - {name:10} Split: {len(split):,} rows | Fraud: {fraud_cnt:,} ({pct:.2f}%)")

    return train_path, val_path, test_path

def run_pipeline(sample_limit=None):
    config = load_config()
    print("=== [STAGE 1: FEATURE ENGINEERING] STARTING DATA PROCESSING ===")
    df = load_raw_data(config, sample_limit=sample_limit)
    df_features = engineer_features(df)
    processed_dir = config["data"].get("processed_dir", "data/processed")
    paths = split_and_save_data(df_features, output_dir=processed_dir)
    print("\n✅ Feature Engineering Stage Completed!")
    return paths

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_limit", type=int, default=50000, help="Row count limit for rapid processing (default 50,000)")
    args = parser.parse_args()
    run_pipeline(sample_limit=args.sample_limit)
