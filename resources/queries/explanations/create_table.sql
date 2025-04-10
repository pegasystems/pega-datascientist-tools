SET memory_limit='{MEMORY_LIMIT}GB';
SET enable_progress_bar = {ENABLE_PROGRESS_BAR};

CREATE TABLE {TABLE_N} as
SELECT *
FROM read_parquet([{SELECTED_FILES}])
WHERE predictor_type = '{PREDICTOR_TYPE}';