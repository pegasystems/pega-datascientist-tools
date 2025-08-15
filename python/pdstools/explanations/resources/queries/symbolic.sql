SET threads TO {THREAD_COUNT};
SET memory_limit = '{MEMORY_LIMIT}GB';
SET enable_progress_bar = {ENABLE_PROGRESS_BAR};

WITH sym_grp AS (
  SELECT 
        {LEFT_PREFIX}.partition
      , {LEFT_PREFIX}.predictor_name
      , {LEFT_PREFIX}.predictor_type
      , IFNULL(NULLIF(trim({LEFT_PREFIX}.symbolic_value), ''), 'MISSING') AS bin_contents
      , AVG(ABS({LEFT_PREFIX}.shap_coeff)) AS contribution_abs
      , AVG({LEFT_PREFIX}.shap_coeff) AS contribution
      , MIN({LEFT_PREFIX}.shap_coeff) AS contribution_min
      , MAX({LEFT_PREFIX}.shap_coeff) AS contribution_max
      , COUNT(*) AS frequency
  FROM {TABLE_NAME} AS {LEFT_PREFIX} 
  WHERE {WHERE_CONDITION}
  GROUP BY 
      {LEFT_PREFIX}.partition
    , {LEFT_PREFIX}.predictor_name
    , {LEFT_PREFIX}.predictor_type
    , {LEFT_PREFIX}.symbolic_value
)
SELECT
  {LEFT_PREFIX}.partition
, predictor_name
, predictor_type
, bin_contents
, ROW_NUMBER() OVER(PARTITION BY {LEFT_PREFIX}.partition, predictor_name ORDER BY frequency DESC) AS bin_order
, contribution_abs
, contribution
, contribution_min
, contribution_max
, frequency
FROM sym_grp AS {LEFT_PREFIX}
