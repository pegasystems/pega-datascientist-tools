"""Metric limits and NBAD configuration utilities.

The MetricLimits.csv in resources defines min/max and best practice values
for CDH/DSM metrics. It currently is sourced from an Excel file that gets
exported to CSV (no special options, just straight export) and copied into
this library.

This module provides access methods to this data and validation functions
that turn metric values into "RAG" indicators that can be used to highlight
values in tables.
"""

import difflib
import re
from functools import lru_cache
from typing import Any, Callable, Dict, Literal, Optional, Union
from ..resources import get_metric_limits_path

import polars as pl

from .number_format import NumberFormat

# Type alias for RAG status values
RAGValue = Literal["RED", "AMBER", "YELLOW", "GREEN"]

# Type alias for value mappings: maps column values to metric values
# e.g., {"Yes": True, "No": False} or {("Yes", "yes"): True, "No": False}
ValueMapping = dict[Union[str, tuple], Any]

# Type for metric specification with optional value mapping
# Can be: "MetricID" or ("MetricID", ValueMapping) or callable
MetricSpec = Union[str, tuple[str, ValueMapping], Callable]


def _convert_excel_csv_value(value: str) -> Union[float, bool, None]:
    """Convert Excel/CSV value to Python type (float, bool, or None)."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value_upper = value.upper().strip()
        if value_upper == "TRUE":
            return True
        if value_upper == "FALSE":
            return False
        if "%" in value:
            try:
                return float(value.replace("%", "").strip()) / 100.0
            except (ValueError, TypeError):
                return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _normalize_name(name: str) -> str:
    """Normalize a name by removing whitespace, dashes, underscores and lowercasing."""
    return re.sub(r"[\s\-_]", "", name).lower()


class MetricLimits:
    """Singleton for accessing metric limits from MetricLimits.csv."""

    _instance: Optional["MetricLimits"] = None

    def __new__(cls) -> "MetricLimits":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    @lru_cache(maxsize=1)
    def get_limits(cls) -> pl.DataFrame:
        """Get all metric limits as a DataFrame."""
        raw_csv = pl.read_csv(source=get_metric_limits_path()).filter(
            pl.col("MetricID").is_not_null() & (pl.col("MetricID") != "")
        )

        limits_df = raw_csv.with_columns(
            [
                pl.col(col)
                .map_elements(_convert_excel_csv_value, return_dtype=pl.Object)
                .alias(col)
                for col in [
                    "Minimum",
                    "Best Practice Min",
                    "Best Practice Max",
                    "Maximum",
                ]
            ]
        )

        return limits_df.with_columns(
            pl.col("Best Practice Min")
            .map_elements(lambda x: isinstance(x, bool), return_dtype=pl.Boolean)
            .alias("is_boolean")
        )

    @classmethod
    def get_limit_for_metric(cls, metric_id: str) -> dict:
        """Get limits for a specific metric. Returns empty dict if not found."""
        df = cls.get_limits()
        result = df.filter(pl.col("MetricID") == metric_id)

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
    def _get_limit_or_raise(cls, metric_id: str) -> dict:
        """Get limits for a metric, raising KeyError if not found."""
        limits = cls.get_limit_for_metric(metric_id)
        if not limits:
            known_metrics = cls.get_limits()["MetricID"].to_list()
            close_matches = difflib.get_close_matches(
                metric_id, known_metrics, n=1, cutoff=0.6
            )
            suggestion = f" Did you mean '{close_matches[0]}'?" if close_matches else ""
            raise KeyError(f"Unknown metric ID '{metric_id}'.{suggestion}")
        return limits

    @classmethod
    def minimum(cls, metric_id: str) -> Optional[float]:
        """Get the minimum (hard limit) for a metric.

        Raises KeyError if metric_id is not found in MetricLimits.csv.
        """
        return cls._get_limit_or_raise(metric_id).get("minimum")

    @classmethod
    def maximum(cls, metric_id: str) -> Optional[float]:
        """Get the maximum (hard limit) for a metric.

        Raises KeyError if metric_id is not found in MetricLimits.csv.
        """
        return cls._get_limit_or_raise(metric_id).get("maximum")

    @classmethod
    def best_practice_min(cls, metric_id: str) -> Optional[Union[float, bool]]:
        """Get the best practice minimum for a metric.

        Raises KeyError if metric_id is not found in MetricLimits.csv.
        """
        return cls._get_limit_or_raise(metric_id).get("best_practice_min")

    @classmethod
    def best_practice_max(cls, metric_id: str) -> Optional[Union[float, bool]]:
        """Get the best practice maximum for a metric.

        Raises KeyError if metric_id is not found in MetricLimits.csv.
        """
        return cls._get_limit_or_raise(metric_id).get("best_practice_max")

    @classmethod
    def evaluate_metric_rag(cls, metric_id: str, value) -> Optional[RAGValue]:
        """Evaluate RAG status for a metric value.

        Parameters
        ----------
        metric_id : str
            The metric identifier from MetricLimits.csv.
        value : Any
            The value to evaluate (numeric or boolean).

        Returns
        -------
        Optional[RAGValue]
            "RED", "AMBER", "GREEN", or None if metric not found or value is None.

        Raises
        ------
        TypeError
            If the value is not a valid type for the metric (e.g., string instead of numeric).

        Notes
        -----
        For boolean metrics:
        - If TRUE is in Minimum or Maximum (hard limit): TRUE → GREEN, FALSE → RED
        - If TRUE is in Best Practice Min or Max (soft limit): TRUE → GREEN, FALSE → AMBER
        """
        if value is None:
            return None

        limits = cls.get_limit_for_metric(metric_id)
        if not limits:
            return None

        min_val = limits.get("minimum")
        bp_min = limits.get("best_practice_min")
        bp_max = limits.get("best_practice_max")
        max_val = limits.get("maximum")
        is_bool = limits.get("is_boolean", False)

        if is_bool:
            # Check hard limits first (Minimum/Maximum) - violations are RED
            if min_val is True or max_val is True:
                return "GREEN" if value is True else "RED"
            # Check soft limits (Best Practice Min/Max) - violations are AMBER
            if bp_min is True or bp_max is True:
                return "GREEN" if value is True else "AMBER"
            # No boolean limits defined, default to GREEN
            return "GREEN"

        # Validate that value is numeric for non-boolean metrics
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Metric '{metric_id}' requires a numeric value, but received "
                f"{type(value).__name__}: {value!r}. "
                f"Check that the column contains numeric data, not strings. "
            )

        # Check RED conditions (outside hard limits)
        if min_val is not None and value < min_val:
            return "RED"
        if max_val is not None and value > max_val:
            return "RED"

        # Check AMBER conditions (between hard limit and best practice)
        if bp_min is not None and value < bp_min:
            return "AMBER"
        if bp_max is not None and value > bp_max:
            return "AMBER"

        return "GREEN"

    @classmethod
    def get_metric_RAG_code(cls, column: str, metric_id: str) -> pl.Expr:
        """Generate a Polars expression that evaluates metric status as RED/AMBER/GREEN.

        Uses evaluate_metric_rag internally via map_elements to ensure consistent
        RAG logic between Python and Polars approaches.
        """
        return (
            pl.col(column)
            .map_elements(
                lambda v: cls.evaluate_metric_rag(metric_id, v),
                return_dtype=pl.Utf8,
            )
            .alias(f"{column}_RAG")
        )


# =============================================================================
# NBAD Configuration Names
# =============================================================================

SINGLE_CHANNEL_NBAD_CONFIGURATIONS = [
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
]

POTENTIALLY_MULTI_CHANNEL_NBAD_CONFIGURATIONS = [
    "Default_Inbound_Model",
    "Default_Outbound_Model",
    "Default_Click_Through_Rate",
    "Other_Inbound_Click_Through_Rate",
    "OmniAdaptiveModel",
]

ALL_NBAD_CONFIGURATIONS = (
    SINGLE_CHANNEL_NBAD_CONFIGURATIONS + POTENTIALLY_MULTI_CHANNEL_NBAD_CONFIGURATIONS
)


def _matches_NBAD_configuration(item: str, config_list: list) -> bool:
    r"""Check if item matches any config in the list.

    Matches with optional prefix (e.g., MyApp_) and postfix (e.g., _GB).
    Pattern: ^(?:\w+_)?{config}(?:_GB)?$
    """
    for config in config_list:
        pattern = rf"^(?:\w+_)?{re.escape(config)}(?:_GB)?$"
        if re.match(pattern, item, re.IGNORECASE):
            return True
    return False


def is_standard_NBAD_configuration(field: str = "Configuration") -> pl.Expr:
    """Polars expression to check if a configuration is a known NBAD config."""

    def check_config(value: str) -> bool:
        if not value:
            return False
        items = [v.strip() for v in value.split(",") if v.strip()]
        return all(
            _matches_NBAD_configuration(item, ALL_NBAD_CONFIGURATIONS) for item in items
        )

    return (
        pl.col(field)
        .cast(pl.String)
        .map_elements(check_config, return_dtype=pl.Boolean)
    )


def standard_NBAD_configurations_rag(value: str) -> Optional[RAGValue]:
    """RAG status for NBAD configuration names.

    Returns AMBER if any is a default/other/multi-channel or a non-standard configuration,
    GREEN if all are standard single-channel configurations.
    """
    if not value:
        return None

    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        return None

    for item in items:
        if _matches_NBAD_configuration(
            item, POTENTIALLY_MULTI_CHANNEL_NBAD_CONFIGURATIONS
        ):
            return "AMBER"  # Multi-channel/default config
        if not _matches_NBAD_configuration(item, SINGLE_CHANNEL_NBAD_CONFIGURATIONS):
            return "AMBER"  # Unknown/non-standard config

    return "GREEN"  # All are standard single-channel configs


# =============================================================================
# NBAD Channel Names
# =============================================================================

STANDARD_NBAD_CHANNELS = [
    "Web",
    "Mobile",
    "E-mail",
    "Push",
    "SMS",
    "Retail",
    "Call Center",
    "Assisted",
]


def get_standard_NBAD_channels() -> list[str]:
    """Get the list of standard NBAD channel names."""
    return STANDARD_NBAD_CHANNELS.copy()


def standard_NBAD_channels_rag(value: str) -> Optional[RAGValue]:
    """RAG status for NBAD channel names.

    Returns GREEN for standard channels, YELLOW for Other, AMBER for multi-channel/unknown.
    """
    _STANDARD_CHANNELS_NORMALIZED = {_normalize_name(c) for c in STANDARD_NBAD_CHANNELS}

    if not value:
        return None

    normalized = _normalize_name(value)
    if not normalized:
        return None

    if normalized in _STANDARD_CHANNELS_NORMALIZED:
        return "GREEN"
    if normalized == "other":
        return "YELLOW"
    return "AMBER"


def standard_NBAD_directions_rag(value: str) -> Optional[RAGValue]:
    """RAG status for NBAD direction names. GREEN for Inbound/Outbound, AMBER otherwise."""
    if not value:
        return None

    normalized = _normalize_name(value)
    if normalized in {"inbound", "outbound"}:
        return "GREEN"
    return "AMBER"


# =============================================================================
# NBAD Prediction Names and Channel Mapping
# =============================================================================

_NBAD_PREDICTION_DATA = [
    ["PredictWebPropensity", "Web", "Inbound", False],
    ["PredictMobilePropensity", "Mobile", "Inbound", False],
    ["PredictOutboundEmailPropensity", "E-mail", "Outbound", False],
    ["PredictOutboundPushPropensity", "Push", "Outbound", False],
    ["PredictOutboundSMSPropensity", "SMS", "Outbound", False],
    ["PredictInboundRetailPropensity", "Retail", "Inbound", False],
    ["PredictOutboundRetailPropensity", "Retail", "Outbound", False],
    ["PredictInboundCallCenterPropensity", "Call Center", "Inbound", False],
    ["PredictOutboundCallCenterPropensity", "Call Center", "Outbound", False],
    ["PredictInboundDefaultPropensity", "Other", "Inbound", False],
    ["PredictOutboundDefaultPropensity", "Other", "Outbound", False],
    ["PredictInboundOtherPropensity", "Other", "Inbound", False],
    ["PredictActionPropensity", "Multi-channel", "Multi-channel", True],
    ["PredictTreatmentPropensity", "Multi-channel", "Multi-channel", True],
]

SINGLE_CHANNEL_NBAD_PREDICTIONS = [p[0] for p in _NBAD_PREDICTION_DATA if not p[3]]
MULTI_CHANNEL_NBAD_PREDICTIONS = [p[0] for p in _NBAD_PREDICTION_DATA if p[3]]
ALL_NBAD_PREDICTIONS = [p[0] for p in _NBAD_PREDICTION_DATA]


def get_predictions_channel_mapping(
    custom_predictions: Optional[list] = None,
) -> pl.DataFrame:
    """Get prediction to channel/direction mapping as a DataFrame."""
    custom_predictions = custom_predictions or []
    all_predictions = _NBAD_PREDICTION_DATA + [
        p
        for p in custom_predictions
        if p[0].upper() not in {x[0].upper() for x in _NBAD_PREDICTION_DATA}
    ]

    df = (
        pl.DataFrame(data=all_predictions, orient="row")
        .with_columns(pl.col("column_0").str.to_uppercase())
        .unique()
    )
    df.columns = ["Prediction", "Channel", "Direction", "isMultiChannel"]
    return df


def is_standard_NBAD_prediction(field: str = "Prediction") -> pl.Expr:
    """Polars expression to check if a prediction is a known NBAD prediction."""
    return (
        pl.col(field)
        .cast(pl.String)
        .str.contains_any(ALL_NBAD_PREDICTIONS, ascii_case_insensitive=True)
    )


def standard_NBAD_predictions_rag(value: str) -> Optional[RAGValue]:
    """RAG status for NBAD prediction names.

    Returns GREEN for single-channel, YELLOW for multi-channel, AMBER for unknown.
    """

    def matches_prediction(item: str, prediction_list: list) -> bool:
        for pred in prediction_list:
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
        if matches_prediction(item, MULTI_CHANNEL_NBAD_PREDICTIONS):
            has_multi_channel = True
        elif not matches_prediction(item, SINGLE_CHANNEL_NBAD_PREDICTIONS):
            return "AMBER"

    return "YELLOW" if has_multi_channel else "GREEN"


# =============================================================================
# Utility RAG Functions
# =============================================================================


def exclusive_0_1_range_rag(value: float) -> Optional[RAGValue]:
    """RAG for percentage values. GREEN if 0 < value < 1, RED otherwise."""
    if value is None:
        return None
    return "GREEN" if 0 < value < 1 else "RED"


def positive_values(value: float) -> Optional[RAGValue]:
    if value is None:
        return None
    return "GREEN" if value >= 0 else "RED"


def strict_positive_values(value: float) -> Optional[RAGValue]:
    if value is None:
        return None
    return "GREEN" if value > 0 else "RED"


# =============================================================================
# DataFrame RAG Functions
# =============================================================================


def add_rag_columns(
    df: pl.DataFrame,
    column_to_metric: Optional[dict[str, MetricSpec]] = None,
    strict_metric_validation: bool = True,
) -> pl.DataFrame:
    """Add RAG status columns to a DataFrame.

    For each column, adds a new column with suffix '_RAG' containing the
    RAG status (RED/AMBER/YELLOW/GREEN or None).

    Parameters
    ----------
    df : pl.DataFrame
        The source DataFrame.
    column_to_metric : dict, optional
        Mapping from column names (or tuples of column names) to one of:

        - **str**: metric ID to look up in MetricLimits.csv
        - **callable**: function(value) -> "RED"|"AMBER"|"YELLOW"|"GREEN"|None
        - **tuple**: (metric_id, value_mapping) where value_mapping is a dict
          that maps column values to metric values before evaluation.
          Supports tuple keys: {("Yes", "yes"): True, "No": False}

        If a column is not in this dict, its name is used as the metric ID.
    strict_metric_validation : bool, default True
        If True, raises ValueError if a metric ID is not found in MetricLimits.csv.

    Returns
    -------
    pl.DataFrame
        DataFrame with additional _RAG columns.

    Examples
    --------
    >>> from pdstools.utils.metric_limits import add_rag_columns
    >>> df_with_rag = add_rag_columns(
    ...     df,
    ...     column_to_metric={
    ...         "Performance": "ModelPerformance",
    ...         "AGB": ("UsingAGB", {"Yes": True, "No": False}),
    ...     }
    ... )
    """
    # Expand tuple column keys to individual columns
    expanded_mapping: dict[str, MetricSpec] = {}
    for key, value in (column_to_metric or {}).items():
        if isinstance(key, tuple) and all(isinstance(k, str) for k in key):
            # Tuple of column names -> same metric for all
            for col in key:
                expanded_mapping[col] = value
        else:
            expanded_mapping[key] = value

    # Validate metric IDs
    if strict_metric_validation:
        limits_df = MetricLimits.get_limits()
        known_metrics = set(limits_df["MetricID"].to_list())
        for col, mapping in expanded_mapping.items():
            metric_id = mapping[0] if isinstance(mapping, tuple) else mapping
            if isinstance(metric_id, str) and metric_id not in known_metrics:
                # Suggest close matches like git does
                close_matches = difflib.get_close_matches(
                    metric_id, known_metrics, n=1, cutoff=0.6
                )
                suggestion = (
                    f" Did you mean '{close_matches[0]}'?" if close_matches else ""
                )
                raise ValueError(
                    f"Unknown metric ID '{metric_id}' for column '{col}'.{suggestion} "
                    f"If it is spelled correctly, add it to MetricLimits.csv or use a callable."
                )

    def build_rag_expr(col: str, spec: MetricSpec) -> pl.Expr:
        """Build a Polars expression for RAG evaluation."""
        if callable(spec):
            return (
                pl.col(col).map_elements(spec, return_dtype=pl.Utf8).alias(f"{col}_RAG")
            )
        elif isinstance(spec, tuple) and len(spec) == 2:
            # (metric_id, value_mapping)
            metric_id, value_mapping = spec

            def mapped_rag(v):
                # Map column value to metric value using the provided mapping
                mapped_v = v
                for key, mapped_value in value_mapping.items():
                    if isinstance(key, tuple):
                        if v in key:
                            mapped_v = mapped_value
                            break
                    elif v == key:
                        mapped_v = mapped_value
                        break
                return MetricLimits.evaluate_metric_rag(metric_id, mapped_v)

            return (
                pl.col(col)
                .map_elements(mapped_rag, return_dtype=pl.Utf8)
                .alias(f"{col}_RAG")
            )
        else:
            # Simple metric ID string
            return MetricLimits.get_metric_RAG_code(col, spec)

    rag_expressions = []
    for col in df.columns:
        spec = expanded_mapping.get(col, col)
        rag_expressions.append(build_rag_expr(col, spec))

    return df.with_columns(rag_expressions)


# =============================================================================
# Metric Format Definitions
# =============================================================================


class MetricFormats:
    """Registry of predefined number formats for common metrics.

    Provides centralized format definitions for use across table rendering
    backends (great_tables, itables, etc.).

    Examples
    --------
    >>> MetricFormats.get("ModelPerformance").format_value(0.875)
    '0.88'
    >>> MetricFormats.has_format("CTR")
    True
    >>> MetricFormats.register("Custom", NumberFormat(decimals=4))
    """

    _FORMATS: Dict[str, NumberFormat] = {
        "ModelPerformance": NumberFormat(decimals=2),
        "EngagementLift": NumberFormat(decimals=0, scale_by=100, suffix="%"),
        "OmniChannelPercentage": NumberFormat(decimals=1, scale_by=100, suffix="%"),
        "InboundNoActionRatio": NumberFormat(decimals=0, scale_by=100, suffix="%"),
        "OutboundNoActionRatio": NumberFormat(decimals=0, scale_by=100, suffix="%"),
        "CTR": NumberFormat(decimals=3, scale_by=100, suffix="%"),
    }

    DEFAULT_FORMAT = NumberFormat(decimals=0, compact=True)

    @classmethod
    def get(cls, metric_id: str) -> Optional[NumberFormat]:
        """Get format for a metric, or None if not defined."""
        return cls._FORMATS.get(metric_id)

    @classmethod
    def get_or_default(cls, metric_id: str) -> NumberFormat:
        """Get format for a metric, falling back to DEFAULT_FORMAT."""
        return cls._FORMATS.get(metric_id, cls.DEFAULT_FORMAT)

    @classmethod
    def has_format(cls, metric_id: str) -> bool:
        """Check if a metric has a defined format."""
        return metric_id in cls._FORMATS

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List all metric IDs with defined formats."""
        return list(cls._FORMATS.keys())

    @classmethod
    def register(cls, metric_id: str, format_spec: NumberFormat) -> None:
        """Register a custom format for a metric."""
        cls._FORMATS[metric_id] = format_spec

    @classmethod
    def all_formats(cls) -> Dict[str, NumberFormat]:
        """Get a copy of all defined metric formats."""
        return cls._FORMATS.copy()
