SET memory_limit='{MEMORY_LIMIT}GB';
SET threads TO {THREAD_COUNT};
SET enable_progress_bar = {ENABLE_PROGRESS_BAR};

SELECT 
    t.partition, 
    COUNT(DISTINCT(t.pyInteractionID)) AS nb_samples
FROM {TABLE_NAME} as t
GROUP BY partition
ORDER BY nb_samples DESC
LIMIT {MODEL_CONTEXT_LIMIT};