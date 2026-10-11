-- Feature Engineering Query for IEEE-CIS Fraud Detection
-- Joins raw_transaction and raw_identity, creates time/amount features and deterministic train/val/test split

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{features_table}` AS
SELECT
    t.TransactionID,
    t.isFraud,
    
    -- Transaction Amount Features
    t.TransactionAmt,
    LOG(t.TransactionAmt + 1.0) AS log_transaction_amt,
    
    -- Time-based Features from TransactionDT (seconds)
    MOD(CAST(FLOOR(t.TransactionDT / 3600) AS INT64), 24) AS hour_of_day,
    MOD(CAST(FLOOR(t.TransactionDT / 86400) AS INT64), 7) AS day_of_week,
    
    -- Categorical & Domain Features
    t.ProductCD,
    t.card1,
    t.card2,
    t.card3,
    t.card4,
    t.card5,
    t.card6,
    t.addr1,
    t.addr2,
    t.dist1,
    SPLIT(COALESCE(t.P_emaildomain, 'missing'), '.')[SAFE_OFFSET(0)] AS p_email_provider,
    SPLIT(COALESCE(t.R_emaildomain, 'missing'), '.')[SAFE_OFFSET(0)] AS r_email_provider,
    
    -- Count / Frequency numeric features (C1 - C14)
    t.C1, t.C2, t.C3, t.C4, t.C5, t.C6, t.C7, t.C8, t.C9, t.C10, t.C11, t.C12, t.C13, t.C14,
    
    -- Time delta numeric features (D1 - D15)
    t.D1, t.D2, t.D3, t.D4, t.D5, t.D10, t.D11, t.D15,
    
    -- Match flags (M1 - M9)
    t.M1, t.M2, t.M3, t.M4, t.M5, t.M6, t.M7, t.M8, t.M9,
    
    -- Selected Vesta Engineered Features
    t.V1, t.V2, t.V3, t.V4, t.V5, t.V12, t.V13, t.V14, t.V29, t.V30,
    t.V53, t.V54, t.V56, t.V75, t.V76, t.V82, t.V86, t.V87, t.V90, t.V91,
    t.V107, t.V108, t.V111, t.V115, t.V130, t.V131, t.V281, t.V282, t.V283,
    
    -- Identity Features
    i.DeviceType,
    i.DeviceInfo,
    i.id_01, i.id_02, i.id_03, i.id_04, i.id_05, i.id_06, i.id_11, i.id_12,
    i.id_15, i.id_16, i.id_28, i.id_29, i.id_31, i.id_35, i.id_36, i.id_37, i.id_38,
    
    -- Deterministic Data Split using FarmFingerprint (70% Train, 20% Validation, 10% Test)
    CASE 
        WHEN ABS(MOD(FARM_FINGERPRINT(CAST(t.TransactionID AS STRING)), 10)) < 7 THEN 'TRAIN'
        WHEN ABS(MOD(FARM_FINGERPRINT(CAST(t.TransactionID AS STRING)), 10)) < 9 THEN 'VALIDATION'
        ELSE 'TEST'
    END AS data_split

FROM `{project_id}.{dataset_id}.raw_transaction` t
LEFT JOIN `{project_id}.{dataset_id}.raw_identity` i
    ON t.TransactionID = i.TransactionID;
