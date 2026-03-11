"""Generic color mapping utilities for categorical data in Streamlit apps.

This module provides utilities for creating consistent color mappings for
categorical dimensions across Streamlit applications. Color consistency is
critical for user experience when filtering/interacting with visualizations.
"""

import polars as pl


def create_categorical_color_mappings(
    data: pl.LazyFrame,
    columns: list[str],
    colorway: list[str],
    column_mapping: dict[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Create consistent color mappings for categorical columns.

    Computes color assignments for categorical dimensions based on all unique
    values in the dataset, sorted alphabetically. This ensures colors remain
    consistent throughout a session regardless of filtering.

    Parameters
    ----------
    data : pl.LazyFrame
        Source data containing the categorical columns.
    columns : list[str]
        List of column names to create color mappings for. Can be display names
        if column_mapping is provided.
    colorway : list[str]
        List of color hex codes to assign (e.g., ["#001F5F", "#10A5AC", ...]).
        Colors are assigned using modulo indexing if there are more unique
        values than colors.
    column_mapping : dict[str, str], optional
        Optional mapping from display names to internal column names.
        Example: {"Issue": "pyIssue", "Group": "pyGroup"}

    Returns
    -------
    dict[str, dict[str, str]]
        Nested dictionary mapping column names to color dictionaries.
        Example: {
            "Issue": {"Retention": "#001F5F", "Sales": "#10A5AC"},
            "Group": {"Cards": "#001F5F", "Loans": "#10A5AC"},
        }

    Examples
    --------
    >>> data = pl.LazyFrame({"Category": ["A", "B", "C"]})
    >>> colors = ["#FF0000", "#00FF00"]
    >>> mappings = create_categorical_color_mappings(data, ["Category"], colors)
    >>> mappings["Category"]["A"]
    '#FF0000'
    >>> mappings["Category"]["C"]  # Wraps around
    '#FF0000'

    Notes
    -----
    - Missing columns are skipped silently
    - Null values are dropped from unique value computation
    - Empty columns return empty color dictionaries
    - Values are sorted alphabetically for deterministic color assignment
    - Colors wrap using modulo indexing when categories exceed colorway length
    """
    result = {}

    # Get actual column names in the data
    schema_columns = set(data.collect_schema().names())

    for col in columns:
        # Resolve display name to internal name if mapping provided
        internal_col = column_mapping.get(col, col) if column_mapping else col

        # Skip if column doesn't exist in data
        if internal_col not in schema_columns:
            continue

        # Get unique values, drop nulls, sort alphabetically
        unique_values = data.select(internal_col).unique().drop_nulls().collect().get_column(internal_col).to_list()

        # Sort alphabetically for deterministic assignment
        unique_values = sorted([str(v) for v in unique_values])

        # Assign colors using modulo indexing
        color_map = {value: colorway[i % len(colorway)] for i, value in enumerate(unique_values)}

        # Use display name (original col) as key in result
        result[col] = color_map

    return result
