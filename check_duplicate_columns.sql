#Count duplicate columns 
SELECT COUNT(*) AS duplicate_columns_number
FROM `mlops-fraud-pipeline-dev.fraud_detection_dwh.INFORMATION_SCHEMA.COLUMNS` c1
JOIN `mlops-fraud-pipeline-dev.fraud_detection_dwh.INFORMATION_SCHEMA.COLUMNS` c2
    ON c1.column_name = c2.column_name
WHERE c1.table_name = 'raw_identity'
    AND c2.table_name = 'raw_transaction';

#Find specific duplicate columns name
SELECT c1.column_name,
        c1.data_type
FROM `mlops-fraud-pipeline-dev.fraud_detection_dwh.INFORMATION_SCHEMA.COLUMNS` c1
JOIN `mlops-fraud-pipeline-dev.fraud_detection_dwh.INFORMATION_SCHEMA.COLUMNS` c2
    ON c1.column_name=c2.column_name
WHERE c1.table_name='raw_identity'
  AND c2.table_name='raw_transaction';

