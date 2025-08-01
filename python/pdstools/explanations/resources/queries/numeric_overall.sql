SET threads TO {THREAD_COUNT};
SET memory_limit = '{MEMORY_LIMIT}GB';
SET enable_progress_bar = {ENABLE_PROGRESS_BAR};

WITH
    quantiles AS (
        SELECT
            *
            , NTILE(10) OVER (PARTITION BY predictor_name ORDER BY numeric_value ASC) AS decile
            FROM {TABLE_NAME} as {LEFT_PREFIX}
            WHERE {WHERE_CONDITION} AND numeric_value IS NOT NULL
    ),
    grouped_data AS (
        SELECT
            'whole_model' AS 'partition'
            , {LEFT_PREFIX}.predictor_name
            , {LEFT_PREFIX}.predictor_type
            , {LEFT_PREFIX}.decile
            , AVG(ABS({LEFT_PREFIX}.shap_coeff)) AS contribution_abs
            , AVG({LEFT_PREFIX}.shap_coeff) AS contribution
            , MIN({LEFT_PREFIX}.shap_coeff) AS contribution_min
            , MAX({LEFT_PREFIX}.shap_coeff) AS contribution_max
            , COUNT(*) AS frequency
            , MIN({LEFT_PREFIX}.numeric_value) AS minimum
            , MAX({LEFT_PREFIX}.numeric_value) AS maximum
        FROM quantiles AS {LEFT_PREFIX}
        GROUP BY {LEFT_PREFIX}.predictor_name, {LEFT_PREFIX}.predictor_type, {LEFT_PREFIX}.decile
    ),
    re_grouped_data AS (
        SELECT
            {LEFT_PREFIX}.partition
            , {LEFT_PREFIX}.predictor_name
            , {LEFT_PREFIX}.predictor_type
            , MIN({LEFT_PREFIX}.decile) AS decile
            , AVG({LEFT_PREFIX}.contribution_abs) AS contribution_abs
            , AVG({LEFT_PREFIX}.contribution) AS contribution
            , MIN({LEFT_PREFIX}.contribution_min) AS contribution_min
            , MAX({LEFT_PREFIX}.contribution_max) AS contribution_max
            , SUM(frequency)::INT64 AS frequency
            , MIN({LEFT_PREFIX}.minimum) AS minimum
            , MAX({LEFT_PREFIX}.maximum) AS maximum
        FROM grouped_data AS {LEFT_PREFIX}
        GROUP BY {LEFT_PREFIX}.predictor_name, {LEFT_PREFIX}.predictor_type, {LEFT_PREFIX}.minimum, {LEFT_PREFIX}.partition, 
    ),
    intervals AS (
        SELECT
            {LEFT_PREFIX}.predictor_name
            , {LEFT_PREFIX}.decile
            , LAG({LEFT_PREFIX}.maximum) OVER (PARTITION BY {LEFT_PREFIX}.predictor_name ORDER BY {LEFT_PREFIX}.decile) AS min_interval
            , LEAD({LEFT_PREFIX}.minimum) OVER (PARTITION BY {LEFT_PREFIX}.predictor_name ORDER BY {LEFT_PREFIX}.decile) AS max_interval
        FROM re_grouped_data as {LEFT_PREFIX}
    ),
    result AS (
        SELECT
            {LEFT_PREFIX}.partition
            , {LEFT_PREFIX}.predictor_name
            , {LEFT_PREFIX}.predictor_type
            , CASE 
                WHEN {RIGHT_PREFIX}.min_interval IS NULL AND {RIGHT_PREFIX}.max_interval IS NOT NULL
                    THEN '<=' || CAST(CAST(({LEFT_PREFIX}.maximum + {RIGHT_PREFIX}.max_interval) / 2.0 AS DECIMAL) AS VARCHAR)
                WHEN {RIGHT_PREFIX}.max_interval IS NULL AND {RIGHT_PREFIX}.min_interval IS NOT NULL
                    THEN '>' || CAST(CAST(({LEFT_PREFIX}.minimum + {RIGHT_PREFIX}.min_interval) / 2.0 AS DECIMAL) AS VARCHAR)
                WHEN {RIGHT_PREFIX}.max_interval IS NULL AND {RIGHT_PREFIX}.min_interval IS NULL
                    THEN '[' || CAST({LEFT_PREFIX}.minimum AS VARCHAR) || ':' || CAST({LEFT_PREFIX}.maximum AS VARCHAR) || ']'
                ELSE '[' || CAST(CAST(({LEFT_PREFIX}.minimum + {RIGHT_PREFIX}.min_interval) / 2.0 AS DECIMAL) AS VARCHAR) || ':' || CAST(CAST(({LEFT_PREFIX}.maximum + {RIGHT_PREFIX}.max_interval) / 2.0 AS DECIMAL) AS VARCHAR) || ']'
            END AS bin_contents
            , {LEFT_PREFIX}.decile AS bin_order
            , {LEFT_PREFIX}.contribution_abs
            , {LEFT_PREFIX}.contribution
            , {LEFT_PREFIX}.contribution_min
            , {LEFT_PREFIX}.contribution_max
            , {LEFT_PREFIX}.frequency
        FROM re_grouped_data AS {LEFT_PREFIX}
        JOIN intervals AS {RIGHT_PREFIX}
        ON {LEFT_PREFIX}.predictor_name={RIGHT_PREFIX}.predictor_name AND {LEFT_PREFIX}.decile={RIGHT_PREFIX}.decile
    ),
    result_missing AS (
        SELECT
            'whole_model' AS 'partition'
            , {LEFT_PREFIX}.predictor_name
            , {LEFT_PREFIX}.predictor_type
            , 'MISSING' AS bin_contents
            , 0 AS bin_order
            , AVG(ABS({LEFT_PREFIX}.shap_coeff)) AS contribution_abs
            , AVG({LEFT_PREFIX}.shap_coeff) AS contribution
            , MIN({LEFT_PREFIX}.shap_coeff) AS contribution_min
            , MAX({LEFT_PREFIX}.shap_coeff) AS contribution_max
            , COUNT(*) AS frequency
        FROM {TABLE_NAME} AS {LEFT_PREFIX} WHERE {WHERE_CONDITION} AND {LEFT_PREFIX}.numeric_value IS NULL
        GROUP BY {LEFT_PREFIX}.predictor_name, {LEFT_PREFIX}.predictor_type
    )
SELECT
    *
FROM result
UNION
SELECT
    *
FROM result_missing

