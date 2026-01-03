"""Module for handling metric limits configuration."""

from functools import lru_cache
from typing import Callable, Optional, Union

import polars as pl

from ..resources import get_metric_limits_path


def _convert_excel_csv_value(value: str) -> Union[float, bool, None]:
    """Convert percentage string, boolean string, or parse string number.

    Parameters
    ----------
    value : str
        The string value from the CSV to convert.

    Returns
    -------
    Union[float, bool, None]
        - None if the value is empty or cannot be parsed
        - True/False for boolean strings
        - float for numeric values (percentages converted to decimals)
    """
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value_upper = value.upper().strip()
        # Handle boolean strings
        if value_upper == "TRUE":
            return True
        if value_upper == "FALSE":
            return False
        # Handle percentages
        if "%" in value:
            try:
                return float(value.replace("%", "").strip()) / 100.0
            except (ValueError, TypeError):
                return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


class MetricLimits:
    """Singleton class for accessing metric limits configuration.

    This class provides lazy-loaded, cached access to the metric limits
    defined in MetricLimits.csv. The data is loaded only once on first access.

    Examples
    --------
    >>> limits_df = MetricLimits.get_limits()
    >>> limits_df.filter(pl.col("Category") == "ML Models")
    """

    _instance: Optional["MetricLimits"] = None
    _limits_df: Optional[pl.DataFrame] = None

    def __new__(cls) -> "MetricLimits":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    @lru_cache(maxsize=1)
    def get_limits(cls) -> pl.DataFrame:
        """Get the metric limits as a Polars DataFrame.

        The threshold columns (Minimum, Best Practice Min, Best Practice Max,
        Maximum) are converted from Excel/CSV format to proper numeric or
        boolean values. Percentages are converted to decimals (e.g., "50%" -> 0.5).

        Returns
        -------
        pl.DataFrame
            DataFrame containing metric limits with columns:
            - Category: str
            - MetricID: str
            - Minimum: float | bool | None
            - Best Practice Min: float | bool | None
            - Best Practice Max: float | bool | None
            - Maximum: float | bool | None
            - Notes (won't be used): str

        Examples
        --------
        >>> df = MetricLimits.get_limits()
        >>> df.head()
        """
        csv_path = get_metric_limits_path()
        raw_csv = pl.read_csv(source=csv_path)

        # Filter out empty rows (where MetricID is null or empty)
        raw_csv = raw_csv.filter(
            pl.col("MetricID").is_not_null() & (pl.col("MetricID") != "")
        )

        threshold_columns = [
            "Minimum",
            "Best Practice Min",
            "Best Practice Max",
            "Maximum",
        ]
        existing_columns = [col for col in threshold_columns if col in raw_csv.columns]

        limits_df = raw_csv.with_columns(
            [
                pl.col(col)
                .map_elements(_convert_excel_csv_value, return_dtype=pl.Object)
                .alias(col)
                for col in existing_columns
            ]
        )

        # Add is_boolean column based on whether best practice values are boolean
        limits_df = limits_df.with_columns(
            pl.col("Best Practice Min")
            .map_elements(lambda x: isinstance(x, bool), return_dtype=pl.Boolean)
            .alias("is_boolean")
        )

        return limits_df

    @classmethod
    def get_limit_for_metric(
        cls,
        metric_id: str,
    ) -> dict:
        """Get the limits for a specific metric.

        Parameters
        ----------
        metric_id : str
            The MetricID to look up.
        category : str, optional
            The category to filter by. If not provided, returns the first
            matching metric.

        Returns
        -------
        dict
            Dictionary containing the limit values for the metric, with keys:
            - minimum
            - best_practice_min
            - best_practice_max
            - maximum
            Returns empty dict if metric not found.

        Examples
        --------
        >>> limits = MetricLimits.get_limit_for_metric("ModelPerformance", "ML Models")
        >>> limits["best_practice_min"]
        55.0
        """
        df = cls.get_limits()

        filter_expr = pl.col("MetricID") == metric_id

        result = df.filter(filter_expr)

        if result.is_empty():
            return {}

        row = result.row(0, named=True)
        return {
            "minimum": row.get("Minimum"),
            "best_practice_min": row.get("Best Practice Min"),
            "best_practice_max": row.get("Best Practice Max"),
            "maximum": row.get("Maximum"),
            "is_boolean": row.get("is_boolean", False),
        }

    @classmethod
    def get_metric_RAG_code(
        cls,
        column: str,
        metric_id: str,
    ) -> pl.Expr:
        """Generate a Polars expression that evaluates metric status as RED/AMBER/GREEN/null."""
        limits = cls.get_limit_for_metric(metric_id)
        if not limits:
            return pl.lit(None).alias(f"{column}_RAG")

        min_val = limits.get("minimum")
        bp_min = limits.get("best_practice_min")
        bp_max = limits.get("best_practice_max")
        max_val = limits.get("maximum")
        is_bool = limits.get("is_boolean", False)

        col = pl.col(column)

        if is_bool:
            expected = bp_min if bp_min is not None else bp_max
            return (
                pl.when(col == expected).then(pl.lit("GREEN")).otherwise(pl.lit("RED"))
            ).alias(f"{column}_RAG")

        # Convert limits to Polars literals for clean expression
        min_lit = pl.lit(min_val)
        max_lit = pl.lit(max_val)
        bp_min_lit = pl.lit(bp_min)
        bp_max_lit = pl.lit(bp_max)

        return (
            pl.when((min_lit.is_not_null()) & (col < min_lit))
            .then(pl.lit("RED"))
            .when((max_lit.is_not_null()) & (col > max_lit))
            .then(pl.lit("RED"))
            .when(
                (bp_min_lit.is_not_null())
                & (bp_max_lit.is_not_null())
                & (col >= bp_min_lit)
                & (col <= bp_max_lit)
            )
            .then(pl.lit("GREEN"))
            .when(
                (bp_min_lit.is_not_null())
                & (bp_max_lit.is_null())
                & (col >= bp_min_lit)
            )
            .then(pl.lit("GREEN"))
            .when(
                (bp_min_lit.is_null())
                & (bp_max_lit.is_not_null())
                & (col <= bp_max_lit)
            )
            .then(pl.lit("GREEN"))
            .when((bp_min_lit.is_not_null()) & (col < bp_min_lit))
            .then(pl.lit("AMBER"))
            .when((bp_max_lit.is_not_null()) & (col > bp_max_lit))
            .then(pl.lit("AMBER"))
            .otherwise(pl.lit(None))
        ).alias(f"{column}_RAG")


def standard_NBAD_configurations_rag(value: str) -> Optional[str]:
    """RAG for NBAD model configuration names.

    - None: empty
    - GREEN: all single-channel configs
    - YELLOW: includes potentially multi-channel configs (default/other/omni)
    - AMBER: unrecognized config

    Matching is case-insensitive and allows for:
    - Optional prefix ending with underscore (e.g., "MyApp_Web_Click_Through_Rate")
    - Optional "_GB" suffix (e.g., "Web_Click_Through_Rate_GB")
    """
    import re

    SINGLE_CHANNEL_NBAD_CONFIGURATIONS = {
        "Web_Click_Through_Rate",
        "WebTreatmentClickModel",
        "Mobile_Click_Through_Rate",
        "Email_Click_Through_Rate",
        "Push_Click_Through_Rate",
        "SMS_Click_Through_Rate",
        "Retail_Click_Through_Rate",
        "Retail_Click_Through_Rate_Outbound",
        "CallCenter_Click_Through_Rate",
        "CallCenterAcceptRateOutbound",
        "Assisted_Click_Through_Rate",
        "Assisted_Click_Through_Rate_Outbound",
    }

    POTENTIALLY_MULTI_CHANNEL_NBAD_CONFIGURATIONS = {
        "Default_Inbound_Model",
        "Default_Outbound_Model",
        "Default_Click_Through_Rate",
        "Other_Inbound_Click_Through_Rate",
        "OmniAdaptiveModel",
    }

    def matches_config(item: str, config_set: set) -> bool:
        for config in config_set:
            pattern = rf"^(?:\w+_)?{re.escape(config)}(?:_GB)?$"
            if re.match(pattern, item, re.IGNORECASE):
                return True
        return False

    if not value:
        return None

    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        return None

    has_multi_channel = False
    for item in items:
        if matches_config(item, POTENTIALLY_MULTI_CHANNEL_NBAD_CONFIGURATIONS):
            has_multi_channel = True
        elif not matches_config(item, SINGLE_CHANNEL_NBAD_CONFIGURATIONS):
            return "AMBER"

    return "YELLOW" if has_multi_channel else "GREEN"


def _normalize_name(name: str) -> str:
    """Normalize a name by removing whitespace, dashes, underscores and lowercasing."""
    import re

    return re.sub(r"[\s\-_]", "", name).lower()


def standard_NBAD_channels_rag(value: str) -> Optional[str]:
    """RAG for NBAD channel names.

    - None: empty
    - GREEN: standard single-channel (Web, Mobile, E-mail, Push, SMS, Retail, Call Center, Assisted)
    - YELLOW: Other
    - AMBER: Multi-channel or unrecognized

    Matching is case-insensitive and ignores whitespace, dashes, underscores.
    """
    STANDARD_CHANNELS = {
        "web",
        "mobile",
        "email",
        "push",
        "sms",
        "retail",
        "callcenter",
        "assisted",
    }
    OTHER_CHANNELS = {"other"}
    MULTI_CHANNELS = {"multichannel"}

    if not value:
        return None

    normalized = _normalize_name(value)
    if not normalized:
        return None

    if normalized in STANDARD_CHANNELS:
        return "GREEN"
    if normalized in OTHER_CHANNELS:
        return "YELLOW"
    if normalized in MULTI_CHANNELS:
        return "AMBER"
    return "AMBER"


def standard_NBAD_directions_rag(value: str) -> Optional[str]:
    """RAG for NBAD direction names.

    - None: empty
    - GREEN: Inbound or Outbound
    - AMBER: Multi-channel or unrecognized

    Matching is case-insensitive and ignores whitespace, dashes, underscores.
    """
    STANDARD_DIRECTIONS = {"inbound", "outbound"}

    if not value:
        return None

    normalized = _normalize_name(value)
    if not normalized:
        return None

    if normalized in STANDARD_DIRECTIONS:
        return "GREEN"
    return "AMBER"


def standard_NBAD_predictions_rag(value: str) -> Optional[str]:
    """RAG for NBAD prediction names.

    - None: empty
    - GREEN: all single-channel predictions
    - YELLOW: includes potentially multi-channel predictions (default/other/omni)
    - AMBER: unrecognized prediction

    Matching is case-insensitive and allows for:
    - Optional prefix ending with underscore (e.g., "MyApp_PredictWebPropensity")
    """
    import re

    SINGLE_CHANNEL_NBAD_PREDICTIONS = {
        "PredictWebPropensity",
        "PredictMobilePropensity",
        "PredictOutboundEmailPropensity",
        "PredictOutboundPushPropensity",
        "PredictOutboundSMSPropensity",
        "PredictInboundRetailPropensity",
        "PredictOutboundRetailPropensity",
        "PredictInboundCallCenterPropensity",
        "PredictOutboundCallCenterPropensity",
    }

    POTENTIALLY_MULTI_CHANNEL_NBAD_PREDICTIONS = {
        "PredictInboundDefaultPropensity",
        "PredictOutboundDefaultPropensity",
        "PredictInboundOtherPropensity",
        "PredictActionPropensity",
        "PredictTreatmentPropensity",
    }

    def matches_prediction(item: str, prediction_set: set) -> bool:
        for pred in prediction_set:
            pattern = rf"^(?:\w+_)?{re.escape(pred)}$"
            if re.match(pattern, item, re.IGNORECASE):
                return True
        return False

    if not value:
        return None

    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        return None

    has_multi_channel = False
    for item in items:
        if matches_prediction(item, POTENTIALLY_MULTI_CHANNEL_NBAD_PREDICTIONS):
            has_multi_channel = True
        elif not matches_prediction(item, SINGLE_CHANNEL_NBAD_PREDICTIONS):
            return "AMBER"

    return "YELLOW" if has_multi_channel else "GREEN"


def create_RAG_table(
    gt: "GT",
    df: pl.DataFrame,
    column_to_metric: Optional[dict[str, Union[str, Callable]]] = None,
    color_background: bool = True,
    strict_metric_validation: bool = True,
    highlight_issues_only: bool = True,
):
    """Apply RAG coloring to a great_tables display for metric columns.

    Parameters
    ----------
    gt : great_tables.GT
        The GT instance to apply coloring to.
    df : pl.DataFrame
        DataFrame containing data columns to be colored.
    column_to_metric : dict, optional
        Mapping from column names to either:
        - str: metric ID to look up in MetricLimits.csv
        - callable: function(value) -> "RED"|"AMBER"|"YELLOW"|"GREEN"|None
        If a column is not in this dict, its name is used as the metric ID.
    color_background : bool, default True
        If True, colors the cell background. If False, colors the text.
    strict_metric_validation : bool, default True
        If True, raises an exception if a metric ID in column_to_metric
        is not found in MetricLimits.csv. Set to False to skip validation.
    highlight_issues_only : bool, default True
        If True, only RED/AMBER/YELLOW values are styled (GREEN is not highlighted).
        Set to False to also highlight GREEN values.

    Returns
    -------
    great_tables.GT
        A great_tables instance with RAG coloring applied.
    """
    from great_tables import style, loc

    RAG_COLORS = {
        "RED": "orangered",
        "AMBER": "orange",
        "YELLOW": "yellow",
    }
    if not highlight_issues_only:
        RAG_COLORS["GREEN"] = "green"

    # Expand tuple keys in column_to_metric to individual columns
    expanded_mapping = {}
    for key, value in (column_to_metric or {}).items():
        if isinstance(key, tuple):
            for col in key:
                expanded_mapping[col] = value
        else:
            expanded_mapping[key] = value

    # Validate metric IDs if strict validation is enabled
    if strict_metric_validation:
        limits_df = MetricLimits.get_limits()
        known_metrics = set(limits_df["MetricID"].to_list())
        for col, mapping in expanded_mapping.items():
            if isinstance(mapping, str) and mapping not in known_metrics:
                raise ValueError(
                    f"Unknown metric ID '{mapping}' for column '{col}'. "
                    f"Add it to MetricLimits.csv or use a callable for custom RAG logic."
                )

    data_columns = list(df.columns)

    rag_expressions = []
    for col in data_columns:
        mapping = expanded_mapping.get(col, col)
        if callable(mapping):
            rag_expressions.append(
                pl.col(col)
                .map_elements(mapping, return_dtype=pl.Utf8)
                .alias(f"{col}_RAG")
            )
        else:
            rag_expressions.append(MetricLimits.get_metric_RAG_code(col, mapping))

    df_with_rag = df.with_columns(rag_expressions)

    # Only hide RAG columns if they exist in the GT's data
    # (They may not if GT was created from original df without RAG columns)

    for col in data_columns:
        rag_col = f"{col}_RAG"
        if rag_col not in df_with_rag.columns or df_with_rag[rag_col].dtype == pl.Null:
            continue

        for rag_value, color in RAG_COLORS.items():
            row_indices = (
                df_with_rag.with_row_index()
                .filter(pl.col(rag_col) == rag_value)["index"]
                .to_list()
            )
            if row_indices:
                if color_background:
                    gt = gt.tab_style(
                        style=style.fill(color=color),
                        locations=loc.body(columns=col, rows=row_indices),
                    )
                else:
                    gt = gt.tab_style(
                        style=style.text(color=color, weight="bold"),
                        locations=loc.body(columns=col, rows=row_indices),
                    )

    return gt


if __name__ == "__main__":
    print("=== MetricLimits Demo ===\n")

    limits_df = MetricLimits.get_limits()
    print(f"Loaded {len(limits_df)} metric limits from CSV\n")

    def configurations_rag(value: str) -> Optional[str]:
        """Custom RAG: RED if empty, GREEN if 1-2 items, AMBER if >2."""
        if value is None or value == "":
            return "RED"
        count = len([v.strip() for v in value.split(",") if v.strip()])
        if count == 0:
            return "RED"
        elif count <= 2:
            return "GREEN"
        else:
            return "AMBER"

    def config_names_rag(value: str) -> Optional[str]:
        """Custom RAG based on allowed values A, B, C.

        - Empty → None (no coloring)
        - All values are A, B, or C → GREEN
        - Mix of allowed and other → AMBER
        - All values are NOT A, B, or C → RED
        """
        if value is None or value == "":
            return None
        allowed = {"A", "B", "C"}
        items = [v.strip() for v in value.split(",") if v.strip()]
        if not items:
            return None
        allowed_count = sum(1 for v in items if v in allowed)
        if allowed_count == len(items):
            return "GREEN"
        elif allowed_count == 0:
            return "RED"
        else:
            return "AMBER"

    def priority_rag(value: float) -> Optional[str]:
        """Custom RAG with YELLOW for priority values.

        - < 0 → None (no coloring)
        - 0 → RED (no priority)
        - 1-2 → YELLOW (low priority)
        - 3-4 → AMBER (medium priority)
        - 5+ → GREEN (high priority)
        """
        if value is None or value < 0:
            return None
        if value == 0:
            return "RED"
        elif value <= 2:
            return "YELLOW"
        elif value <= 4:
            return "AMBER"
        else:
            return "GREEN"

    sample_data = pl.DataFrame(
        {
            "NBADConfigs": [
                "",  # None - empty
                "Web_Click_Through_Rate",  # GREEN - standard
                "MyApp_Web_Click_Through_Rate_GB",  # GREEN - with prefix and suffix
                "OmniAdaptiveModel",  # YELLOW - omni config
                "Default_Inbound_Model",  # YELLOW - default config
                "Web_Click_Through_Rate,InvalidConfig",  # AMBER - one invalid
            ],
            "NBADPredictions": [
                "",  # None - empty
                "PredictWebPropensity",  # GREEN - standard
                "MyApp_PredictMobilePropensity",  # GREEN - with prefix
                "PredictActionPropensity",  # YELLOW - multi-channel
                "PredictInboundDefaultPropensity",  # YELLOW - default
                "PredictWebPropensity,InvalidPrediction",  # AMBER - one invalid
            ],
        }
    )

    print("Sample data:")
    print(sample_data)
    print()

    print("NBADConfigs expected:")
    print('  "" → None (no color)')
    print('  "Web_Click_Through_Rate" → GREEN')
    print('  "MyApp_Web_Click_Through_Rate_GB" → GREEN (with prefix/suffix)')
    print('  "OmniAdaptiveModel" → YELLOW')
    print('  "Default_Inbound_Model" → YELLOW')
    print('  "Web_Click_Through_Rate,InvalidConfig" → AMBER')
    print()

    print("NBADPredictions expected:")
    print('  "" → None (no color)')
    print('  "PredictWebPropensity" → GREEN')
    print('  "MyApp_PredictMobilePropensity" → GREEN (with prefix)')
    print('  "PredictActionPropensity" → YELLOW')
    print('  "PredictInboundDefaultPropensity" → YELLOW')
    print('  "PredictWebPropensity,InvalidPrediction" → AMBER')
    print()

    print("Creating great_tables display with RAG coloring...")
    from great_tables import GT

    gt = GT(sample_data)
    gt = create_RAG_table(
        gt,
        sample_data,
        column_to_metric={
            "NBADConfigs": standard_NBAD_configurations_rag,
            "NBADPredictions": standard_NBAD_predictions_rag,
        },
        color_background=True,
    )

    html_path = "/tmp/rag_table_demo.html"
    with open(html_path, "w") as f:
        f.write(gt.as_raw_html())
    print(f"Saved to {html_path}")
