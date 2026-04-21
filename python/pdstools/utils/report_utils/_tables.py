"""RAG-coloured metric table builders (itables and great_tables)."""

import html
from typing import Any

import polars as pl

from ..metric_limits import MetricFormats


def create_metric_itable(
    source_table: pl.DataFrame,
    column_to_metric: dict | None = None,
    column_descriptions: dict[str, str] | None = None,
    color_background: bool = False,
    strict_metric_validation: bool = True,
    highlight_issues_only: bool = False,
    rag_source: pl.DataFrame | None = None,
    **itable_kwargs,
):
    """Create an interactive table with RAG coloring for metric columns.

    Displays the table using itables with cells colored based on RAG
    (Red/Amber/Green) status derived from metric thresholds.

    Parameters
    ----------
    source_table : pl.DataFrame
        DataFrame containing data columns to be colored.
    column_to_metric : dict, optional
        Mapping from column names (or tuples of column names) to one of:

        - **str**: metric ID to look up in MetricLimits.csv
        - **callable**: function(value) -> "RED"|"AMBER"|"YELLOW"|"GREEN"|None
        - **tuple**: (metric_id, value_mapping) where value_mapping is a dict
          that maps column values to metric values before evaluation.
          Supports tuple keys for multiple values: {("Yes", "yes"): True}

        If a column is not in this dict, its name is used as the metric ID.
    column_descriptions : dict, optional
        Mapping from column names to tooltip descriptions. When provided,
        column headers will display the description as a tooltip on hover.
        Example: {"Performance": "Model AUC performance metric"}
    color_background : bool, default False
        If True, colors the cell background. If False, colors the text (foreground).
    strict_metric_validation : bool, default True
        If True, raises an exception if a metric ID in column_to_metric
        is not found in MetricLimits.csv. Set to False to skip validation.
    highlight_issues_only : bool, default False
        If True, only RED/AMBER/YELLOW values are styled (GREEN is not highlighted).
        Set to False to also highlight GREEN values.
    rag_source : pl.DataFrame, optional
        If provided, RAG thresholds are evaluated against this DataFrame
        instead of ``source_table``. Use this when ``source_table`` contains
        non-numeric display values (e.g. HTML strings) but you still want
        RAG coloring based on the original numeric data. Must have the same
        columns and row order as ``source_table``.
    **itable_kwargs
        Additional keyword arguments passed to itables.show().
        Common options include: lengthMenu, paging, searching, ordering.

    Returns
    -------
    itables HTML display
        An itables display object that will render in Jupyter/Quarto.

    Examples
    --------
    >>> from pdstools.utils.report_utils import create_metric_itable
    >>> create_metric_itable(
    ...     df,
    ...     column_to_metric={
    ...         # Simple metric ID
    ...         "Performance": "ModelPerformance",
    ...         # Custom RAG function
    ...         "Channel": standard_NBAD_channels_rag,
    ...         # Value mapping: column values -> metric values
    ...         "AGB": ("UsingAGB", {"Yes": True, "No": False}),
    ...         # Multiple column values to same metric value
    ...         "AGB": ("UsingAGB", {("Yes", "yes", "YES"): True, "No": False}),
    ...     },
    ...     column_descriptions={
    ...         "Performance": "Model AUC performance metric",
    ...         "Channel": "Communication channel for the action",
    ...     },
    ...     paging=False
    ... )

    """
    from itables import show

    RAG_COLORS = {
        "RED": "orangered",
        "AMBER": "orange",
        "YELLOW": "yellow",
    }
    if not highlight_issues_only:
        RAG_COLORS["GREEN"] = "green"

    # Get RAG values using the shared function from metric_limits
    from ..metric_limits import add_rag_columns

    df_with_rag = add_rag_columns(
        rag_source if rag_source is not None else source_table,
        column_to_metric=column_to_metric,
        strict_metric_validation=strict_metric_validation,
    )

    # Convert to pandas for styling
    pdf = source_table.to_pandas()
    pdf_rag = df_with_rag.to_pandas()

    # Rename columns to include HTML tooltips if column_descriptions provided
    # Must be done before styling since Styler doesn't have rename method
    column_rename = {}
    if column_descriptions:
        for col in source_table.columns:
            if col in column_descriptions:
                escaped_desc = html.escape(column_descriptions[col], quote=True)
                column_rename[col] = f'<span title="{escaped_desc}">{col}</span>'
        if column_rename:
            pdf = pdf.rename(columns=column_rename)

    def style_row(row):
        styles = []
        for col in pdf.columns:
            rag_col = f"{col}_RAG"
            if rag_col in pdf_rag.columns:
                rag_val = pdf_rag.loc[row.name, rag_col]
                if rag_val in RAG_COLORS:
                    color = RAG_COLORS[rag_val]
                    if color_background:
                        styles.append(f"background-color: {color}")
                    else:
                        styles.append(f"color: {color}; font-weight: bold")
                else:
                    styles.append("")
            else:
                styles.append("")
        return styles

    # Build format dict from centralized config using MetricFormats
    expanded_mapping = {}
    for key, value in (column_to_metric or {}).items():
        if isinstance(key, tuple):
            for col in key:
                expanded_mapping[col] = value
        else:
            expanded_mapping[key] = value

    format_dict = {}
    # When rag_source is provided, source_table may contain pre-formatted strings;
    # use the numeric source for column detection and sort values, but only
    # populate format_dict for columns that are still numeric in source_table.
    numeric_source = rag_source if rag_source is not None else source_table
    numeric_cols = []
    for col in source_table.columns:
        if not numeric_source[col].dtype.is_numeric():
            continue
        numeric_cols.append(col)
        metric_id = expanded_mapping.get(col, col)
        if isinstance(metric_id, tuple):
            metric_id = metric_id[0]
        if isinstance(metric_id, str):
            metric_fmt = MetricFormats.get(metric_id) or MetricFormats.DEFAULT_FORMAT
            if source_table[col].dtype.is_numeric():
                format_dict[col] = metric_fmt.to_pandas_format()
        else:
            if source_table[col].dtype.is_numeric():
                format_dict[col] = MetricFormats.DEFAULT_FORMAT.to_pandas_format()

    # Add hidden sort columns for all formatted numeric columns so that
    # DataTables can sort by the raw numeric value instead of the display string
    # (e.g. "4K" < "500" or "100%" < "2%" lexicographically without this).
    sort_col_map = {col: f"__sort__{col}" for col in numeric_cols}
    for col, sort_col in sort_col_map.items():
        pdf[sort_col] = numeric_source[col].to_pandas()

    styled_df = pdf.style.apply(style_row, axis=1).format(format_dict, na_rep="")

    # Set default itable options
    default_kwargs: dict[str, Any] = {
        "paging": False,
        "allow_html": True,  # Required for styled DataFrames
        "classes": "compact",
        # "maxBytes":0,
        # "maxRows":0,
        "connected": True,
        "show_dtypes": False,
    }
    default_kwargs.update(itable_kwargs)

    if sort_col_map:
        final_cols = list(pdf.columns)
        sort_defs = []
        for orig_col, sort_col in sort_col_map.items():
            display_name = column_rename.get(orig_col, orig_col)
            display_idx = final_cols.index(display_name)
            sort_idx = final_cols.index(sort_col)
            sort_defs.extend(
                [
                    {"targets": display_idx, "orderData": [sort_idx]},
                    {"targets": sort_idx, "visible": False, "searchable": False},
                ]
            )
        default_kwargs["columnDefs"] = default_kwargs.get("columnDefs", []) + sort_defs

    return show(styled_df, **default_kwargs)


def create_metric_gttable(
    source_table: pl.DataFrame,
    title: str | None = None,
    subtitle: str | None = None,
    column_to_metric: dict | None = None,
    column_descriptions: dict[str, str] | None = None,
    color_background: bool = True,
    strict_metric_validation: bool = True,
    highlight_issues_only: bool = True,
    **gt_kwargs,
):
    """Create a great_tables table with RAG coloring for metric columns.

    Displays the table using great_tables with cells colored based on RAG
    (Red/Amber/Green) status derived from metric thresholds.

    Parameters
    ----------
    source_table : pl.DataFrame
        DataFrame containing data columns to be colored.
    title : str, optional
        Table title.
    subtitle : str, optional
        Table subtitle.
    column_to_metric : dict, optional
        Mapping from column names (or tuples of column names) to one of:

        - **str**: metric ID to look up in MetricLimits.csv
        - **callable**: function(value) -> "RED"|"AMBER"|"YELLOW"|"GREEN"|None
        - **tuple**: (metric_id, value_mapping) where value_mapping is a dict
          that maps column values to metric values before evaluation.
          Supports tuple keys for multiple values: {("Yes", "yes"): True}

        If a column is not in this dict, its name is used as the metric ID.
    column_descriptions : dict, optional
        Mapping from column names to tooltip descriptions. When provided,
        column headers will display the description as a tooltip on hover.
        Example: {"Performance": "Model AUC performance metric"}
    color_background : bool, default True
        If True, colors the cell background. If False, colors the text.
    strict_metric_validation : bool, default True
        If True, raises an exception if a metric ID in column_to_metric
        is not found in MetricLimits.csv. Set to False to skip validation.
    highlight_issues_only : bool, default True
        If True, only RED/AMBER/YELLOW values are styled (GREEN is not highlighted).
        Set to False to also highlight GREEN values.
    **gt_kwargs
        Additional keyword arguments passed to great_tables.GT constructor.
        Common options include: rowname_col, groupname_col.

    Returns
    -------
    great_tables.GT
        A great_tables instance with RAG coloring applied.

    Examples
    --------
    >>> from pdstools.utils.report_utils import create_metric_gttable
    >>> create_metric_gttable(
    ...     df,
    ...     title="Model Overview",
    ...     column_to_metric={
    ...         # Simple metric ID
    ...         "Performance": "ModelPerformance",
    ...         # Custom RAG function
    ...         "Channel": standard_NBAD_channels_rag,
    ...         # Value mapping: column values -> metric values
    ...         "AGB": ("UsingAGB", {"Yes": True, "No": False}),
    ...         # Multiple column values to same metric value
    ...         "AGB": ("UsingAGB", {("Yes", "yes", "YES"): True, "No": False}),
    ...     },
    ...     column_descriptions={
    ...         "Performance": "Model AUC performance metric",
    ...         "Channel": "Communication channel for the action",
    ...     },
    ...     rowname_col="Name",
    ... )

    """
    import html as html_module

    from great_tables import GT, html, loc, style

    from ..metric_limits import add_rag_columns

    RAG_COLORS = {"RED": "orangered", "AMBER": "orange", "YELLOW": "yellow"}
    if not highlight_issues_only:
        RAG_COLORS["GREEN"] = "green"

    gt = GT(source_table, **gt_kwargs)
    gt = gt.tab_options(
        table_font_size="12px",
        column_labels_font_size="12px",
    )
    gt = gt.sub_missing(missing_text="")

    if title is not None:
        gt = gt.tab_header(title=title, subtitle=subtitle)

    # Apply column label tooltips if column_descriptions provided
    if column_descriptions:
        label_kwargs = {}
        for col in source_table.columns:
            if col in column_descriptions:
                escaped_desc = html_module.escape(column_descriptions[col], quote=True)
                # Wrap column label in span with title attribute for tooltip
                label_kwargs[col] = html(f'<span title="{escaped_desc}">{col}</span>')
        if label_kwargs:
            gt = gt.cols_label(**label_kwargs)  # type: ignore[arg-type]  # great_tables overload routes **kwargs to first positional

    # Expand tuple keys to individual columns
    expanded_mapping = {}
    for key, value in (column_to_metric or {}).items():
        if isinstance(key, tuple):
            for col in key:
                expanded_mapping[col] = value
        else:
            expanded_mapping[key] = value

    # Apply formatting based on metric type using MetricFormats
    formatted_cols = set()
    for col in source_table.columns:
        metric_id = expanded_mapping.get(col, col)
        if isinstance(metric_id, tuple):
            metric_id = metric_id[0]
        if isinstance(metric_id, str):
            fmt = MetricFormats.get(metric_id)
            if fmt is not None:
                gt = fmt.apply_to_gt(gt, [col])
                formatted_cols.add(col)
        elif callable(metric_id):
            # When a callable is used for RAG, still apply formatting if
            # the column name itself has a format defined in MetricFormats
            fmt = MetricFormats.get(col)
            if fmt is not None:
                gt = fmt.apply_to_gt(gt, [col])
                formatted_cols.add(col)

    # Apply default number formatting to numeric columns not yet formatted
    # Exclude columns used as row/group names from numeric formatting
    rowname_col = gt_kwargs.get("rowname_col")
    groupname_col = gt_kwargs.get("groupname_col")
    numeric_cols = [
        col
        for col in source_table.columns
        if col not in formatted_cols
        and col != rowname_col
        and col != groupname_col
        and source_table[col].dtype.is_numeric()
    ]
    if numeric_cols:
        gt = MetricFormats.DEFAULT_FORMAT.apply_to_gt(gt, numeric_cols)

    # Apply RAG coloring
    df_with_rag = add_rag_columns(
        source_table,
        column_to_metric=column_to_metric,
        strict_metric_validation=strict_metric_validation,
    )

    for col in source_table.columns:
        rag_col = f"{col}_RAG"
        if rag_col not in df_with_rag.columns or df_with_rag[rag_col].dtype == pl.Null:
            continue

        for rag_value, color in RAG_COLORS.items():
            row_indices = df_with_rag.with_row_index().filter(pl.col(rag_col) == rag_value)["index"].to_list()
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
