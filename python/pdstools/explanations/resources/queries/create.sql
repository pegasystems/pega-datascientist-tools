SET memory_limit='{MEMORY_LIMIT}GB';
SET enable_progress_bar = {ENABLE_PROGRESS_BAR};

CREATE TABLE {TABLE_NAME} AS
WITH data AS (
    SELECT * FROM read_parquet([{SELECTED_FILES}])
    WHERE predictor_type = '{PREDICTOR_TYPE}'
),
selected_partitions AS (
    SELECT 
        partition, 
        COUNT(DISTINCT pySubjectID || '-' || pyInteractionID) AS nb_samples
    FROM data
    GROUP BY partition
    ORDER BY nb_samples DESC, partition
    LIMIT {MODEL_CONTEXT_LIMIT}
)
SELECT main.*
FROM data AS main
JOIN selected_partitions AS partitions ON main.partition = partitions.partition;