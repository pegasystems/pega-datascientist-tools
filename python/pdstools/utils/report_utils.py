import os
import datetime
import io
import json
import logging
import re
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import polars as pl

from ..utils.types import QUERY

# Re-export RAG functions from metric_limits for convenience in Quarto reports.
# noqa: F401 prevents ruff from removing these "unused" imports during pre-commit.
from .metric_limits import (  # noqa: F401
    exclusive_0_1_range_rag,
    positive_values,
    strict_positive_values,
    standard_NBAD_channels_rag,
    standard_NBAD_configurations_rag,
    standard_NBAD_directions_rag,
    standard_NBAD_predictions_rag,
)

# Re-export NumberFormat, MetricFormats, and MetricLimits for external use
from .number_format import NumberFormat  # noqa: F401
from .metric_limits import MetricFormats, MetricLimits  # noqa: F401

logger = logging.getLogger(__name__)


def get_output_filename(
    name: Optional[str],  # going to be the full file name
    report_type: str,
    model_id: Optional[str] = None,
    output_type: str = "html",
) -> str:
    """Generate the output filename based on the report parameters."""
    name = name.replace(" ", "_") if name else None
    if report_type == "ModelReport":
        return (
            f"{report_type}_{name}_{model_id}.{output_type}"
            if name
            else f"{report_type}_{model_id}.{output_type}"
        )
    return (
        f"{report_type}_{name}.{output_type}"
        if name
        else f"{report_type}.{output_type}"
    )


def copy_quarto_file(qmd_file: str, temp_dir: Path) -> None:
    """Copy the report quarto file to the temporary directory.

    Parameters
    ----------
    qmd_file : str
        Name of the Quarto markdown file to copy
    temp_dir : Path
        Destination directory to copy files to

    Returns
    -------
    None
    """
    from pdstools import __reports__

    shutil.copy(__reports__ / qmd_file, temp_dir)
    shutil.copytree(__reports__ / "assets", temp_dir / "assets", dirs_exist_ok=True)


def _write_params_files(
    temp_dir: Path,
    params: Optional[Dict] = None,
    project: Dict = {"type": "default"},
    analysis: Optional[Dict] = None,
    size_reduction_method: Optional[Literal["strip", "cdn"]] = None,
) -> None:
    """Write parameters to YAML files for Quarto processing.

    Parameters
    ----------
    temp_dir : Path
        Directory where YAML files will be written
    params : dict, optional
        Parameters to write to params.yml, by default None
    project : dict, optional
        Project configuration to write to _quarto.yml, by default {"type": "default"}
    analysis : dict, optional
        Analysis configuration to write to _quarto.yml, by default None
    size_reduction_method : Optional[Literal["strip", "cdn"]], default=None
        When "cdn", sets plotly-connected to false so Plotly.js loads from CDN
        (resulting in smaller files ~8MB vs ~110MB).
        When None or "strip", sets plotly-connected to true for fully embedded Plotly.

    Returns
    -------
    None
    """
    import yaml

    params = params or {}
    analysis = analysis or {}

    # Parameters to python code
    with open(temp_dir / "params.yml", "w") as f:
        yaml.dump(
            params,
            f,
        )

    # Always embed resources for standalone HTML
    # plotly-connected: false = load Plotly from CDN (smaller file ~8MB)
    # plotly-connected: true = embed Plotly (larger file ~110MB)
    html_format: Dict = {
        "embed-resources": True,
        "plotly-connected": size_reduction_method != "cdn",
    }

    quarto_config: Dict = {
        "project": project,
        "analysis": analysis,
        "format": {
            "html": html_format,
        },
    }

    with open(temp_dir / "_quarto.yml", "w") as f:
        yaml.dump(quarto_config, f)


def run_quarto(
    qmd_file: Optional[str] = None,
    output_filename: Optional[str] = None,
    output_type: Optional[str] = "html",
    params: Optional[Dict] = None,
    project: Dict = {"type": "default"},
    analysis: Optional[Dict] = None,
    temp_dir: Path = Path("."),
    verbose: bool = False,
    *,
    size_reduction_method: Optional[Literal["strip", "cdn"]] = None,
) -> int:
    """Run the Quarto command to generate the report.

    Parameters
    ----------
    qmd_file : str, optional
        Path to the Quarto markdown file to render, by default None
    output_filename : str, optional
        Name of the output file, by default None
    output_type : str, optional
        Type of output format (html, pdf, etc.), by default "html"
    params : dict, optional
        Parameters to pass to Quarto execution, by default None
    project : dict, optional
        Project configuration settings, by default {"type": "default"}
    analysis : dict, optional
        Analysis configuration settings, by default None
    temp_dir : Path, optional
        Temporary directory for processing, by default Path(".")
    verbose : bool, optional
        Whether to print detailed execution logs, by default False
    size_reduction_method : Optional[Literal["strip", "cdn"]], default=None
        When None will fully embed all resources into the HTML output.
        When "cdn" will pass this on to Quarto and Plotly so Javascript libraries will be loaded from the internet.
        When "strip" the HTML will be post-processed to remove duplicate Javascript that would otherwise get embedded multiple times.

    Returns
    -------
    int
        Return code from the Quarto process (0 for success)

    Raises
    ------
    RuntimeError
        If the Quarto process fails (non-zero return code), includes captured output
    subprocess.SubprocessError
        If the Quarto command fails to execute
    FileNotFoundError
        If required files are not found
    """

    def get_command() -> List[str]:
        quarto_exec, _ = get_quarto_with_version(verbose)
        _command = [str(quarto_exec), "render"]

        if qmd_file is not None:
            _command.append(qmd_file)

        options = _set_command_options(
            output_type=output_type,
            output_filename=output_filename,
            execute_params=params is not None,
        )

        _command.extend(options)
        return _command

    if params is not None:
        _write_params_files(
            temp_dir,
            params=params,
            project=project,
            analysis=analysis,
            size_reduction_method=size_reduction_method,
        )

    # render file or render project with options
    command = get_command()

    if verbose:
        print(f"Executing: {' '.join(command)} in temp directory {temp_dir}")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        cwd=temp_dir,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Capture all output for potential error reporting
    output_lines: List[str] = []
    if process.stdout is not None:
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            output_lines.append(line)
            if verbose:
                print(line)
            logger.info(line)
    else:  # pragma: no cover
        logger.warning("subprocess.stdout is None, unable to read output")

    return_code = process.wait()
    message = f"Quarto process exited with return code {return_code}"
    logger.info(message)

    # Raise an exception with captured output if Quarto failed
    if return_code != 0:
        captured_output = "\n".join(output_lines)
        raise RuntimeError(
            f"Quarto rendering failed with return code {return_code}.\n"
            f"Output:\n{captured_output}"
        )

    # Post-process HTML files to deduplicate JavaScript libraries
    if (
        return_code == 0
        and output_type == "html"
        and size_reduction_method == "strip"
        and output_filename is not None
    ):
        try:
            html_file_path = temp_dir / output_filename
            if html_file_path.exists():
                html_content = html_file_path.read_text(encoding="utf-8")
                deduplicated_content = remove_duplicate_html_scripts(
                    html_content, verbose
                )
                html_file_path.write_text(deduplicated_content, encoding="utf-8")
        except Exception as e:
            logger.warning(f"HTML post-processing failed: {e}")

    return return_code


def _set_command_options(
    output_type: Optional[str] = None,
    output_filename: Optional[str] = None,
    execute_params: bool = False,
) -> List[str]:
    """Set the options for the Quarto command.

    Parameters
    ----------
    output_type : str, optional
        Output format type (html, pdf, etc.), by default None
    output_filename : str, optional
        Name of the output file, by default None
    execute_params : bool, optional
        Whether to include parameter execution flag, by default False

    Returns
    -------
    List[str]
        List of command line options for Quarto
    """

    options = []
    if output_type is not None:
        options.append("--to")
        options.append(output_type)
    if output_filename is not None:
        options.append("--output")
        options.append(output_filename)
    if execute_params:
        options.append("--execute-params")
        options.append("params.yml")
    return options


def copy_report_resources(resource_dict: list[tuple[str, str]]):
    """Copy report resources from the reports directory to specified destinations.

    Parameters
    ----------
    resource_dict : list[tuple[str, str]]
        List of tuples containing (source_path, destination_path) pairs

    Returns
    -------
    None
    """
    from pdstools import __reports__

    for src, dest in resource_dict:
        source_path = __reports__ / src
        destination_path = dest

        if destination_path == "":
            destination_path = "./"

        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy(source_path, destination_path)


def generate_zipped_report(output_filename: str, folder_to_zip: str):
    """Generate a zipped archive of a directory.

    This is a general-purpose utility function that can compress any directory
    into a zip archive. While named for report generation, it works with any
    directory structure.

    Parameters
    ----------
    output_filename : str
        Name of the output file (extension will be replaced with .zip)
    folder_to_zip : str
        Path to the directory to be compressed

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the folder to zip does not exist or is not a directory

    Examples
    --------
    >>> generate_zipped_report("my_archive.zip", "/path/to/directory")
    >>> generate_zipped_report("report_2023", "/tmp/report_output")
    """
    if not os.path.isdir(folder_to_zip):
        logger.error(f"The output path {folder_to_zip} is not a directory.")
        return

    if not os.path.exists(folder_to_zip):
        logger.warning(
            f"The {folder_to_zip} directory does not exist. Skipping zip creation."
        )
        return

    base_filename = os.path.splitext(output_filename)[0]
    zippy = shutil.make_archive(base_filename, "zip", folder_to_zip)
    logger.info(f"created zip file...{zippy}")


def _get_cmd_output(args: List[str]) -> List[str]:
    """Get command output in an OS-agnostic way."""
    try:
        result = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return result.stdout.split("\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run command {' '.join(args)}: {e}")
        raise FileNotFoundError(
            f"Command failed. Make sure {args[0]} is installed and in the system PATH."
        )


def _get_version_only(versionstr: str) -> str:
    """Extract version number from version string."""
    match = re.search(r"(\d+(?:\.\d+)*)", versionstr)
    return match.group(1) if match else ""


def get_quarto_with_version(verbose: bool = True) -> Tuple[Path, str]:
    """Get Quarto executable path and version."""
    try:
        executable = Path(shutil.which("quarto"))
        if not executable:
            raise FileNotFoundError(
                "Quarto executable not found. Please ensure Quarto is installed and in the system PATH."
            )

        version_string = _get_version_only(_get_cmd_output(["quarto", "--version"])[0])
        message = f"quarto version: {version_string}"
        logger.info(message)
        if verbose:
            print(message)
        return executable, version_string
    except Exception as e:
        logger.error(f"Error getting quarto version: {e}")
        raise


def get_pandoc_with_version(verbose: bool = True) -> Tuple[Path, str]:
    """Get Pandoc executable path and version."""
    try:
        executable = Path(shutil.which("pandoc"))
        if not executable:
            raise FileNotFoundError(
                "Pandoc executable not found. Please ensure Pandoc is installed and in the system PATH."
            )

        version_string = _get_version_only(_get_cmd_output(["pandoc", "--version"])[0])
        message = f"pandoc version: {version_string}"
        logger.info(message)
        if verbose:
            print(message)
        return executable, version_string
    except Exception as e:
        logger.error(f"Error getting pandoc version: {e}")
        raise


def quarto_print(text):
    from IPython.display import Markdown, display

    display(Markdown(text))


def quarto_callout_info(info):
    quarto_print(
        """
::: {.callout-note}
%s
:::
"""
        % info
    )


def quarto_callout_important(info):
    quarto_print(
        """
::: {.callout-important}
%s
:::
"""
        % info
    )


def quarto_plot_exception(plot_name: str, e: Exception):
    quarto_print(
        """
::: {.callout-important collapse="true"}
## Error rendering %s plot: %s

%s
:::
"""
        % (plot_name, e, traceback.format_exc())
    )


def quarto_callout_no_prediction_data_warning(extra=""):
    quarto_callout_important(f"Prediction Data is not available. {extra}")


def quarto_callout_no_predictor_data_warning(extra=""):
    quarto_callout_important(f"Predictor Data is not available. {extra}")


def polars_col_exists(df, col):
    return col in df.collect_schema().names() and df.collect_schema()[col] != pl.Null


def polars_subset_to_existing_cols(all_columns, cols):
    return [col for col in cols if col in all_columns]


def create_metric_itable(
    source_table: pl.DataFrame,
    column_to_metric: Optional[Dict] = None,
    color_background: bool = False,
    strict_metric_validation: bool = True,
    highlight_issues_only: bool = False,
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
    color_background : bool, default False
        If True, colors the cell background. If False, colors the text (foreground).
    strict_metric_validation : bool, default True
        If True, raises an exception if a metric ID in column_to_metric
        is not found in MetricLimits.csv. Set to False to skip validation.
    highlight_issues_only : bool, default False
        If True, only RED/AMBER/YELLOW values are styled (GREEN is not highlighted).
        Set to False to also highlight GREEN values.
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
    from .metric_limits import add_rag_columns

    df_with_rag = add_rag_columns(
        source_table,
        column_to_metric=column_to_metric,
        strict_metric_validation=strict_metric_validation,
    )

    # Convert to pandas for styling
    pdf = source_table.to_pandas()
    pdf_rag = df_with_rag.to_pandas()

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
    for col in source_table.columns:
        metric_id = expanded_mapping.get(col, col)
        if isinstance(metric_id, tuple):
            metric_id = metric_id[0]
        if isinstance(metric_id, str):
            fmt = MetricFormats.get(metric_id)
            if fmt is not None:
                format_dict[col] = fmt.to_pandas_format()
            elif source_table[col].dtype.is_numeric():
                # Use default compact formatting for numeric columns
                format_dict[col] = MetricFormats.DEFAULT_FORMAT.to_pandas_format()
        elif source_table[col].dtype.is_numeric():
            # Use default compact formatting for numeric columns
            format_dict[col] = MetricFormats.DEFAULT_FORMAT.to_pandas_format()

    styled_df = pdf.style.apply(style_row, axis=1).format(format_dict, na_rep="")

    # Set default itable options
    default_kwargs = {
        "paging": False,
        "allow_html": True,  # Required for styled DataFrames
        "classes": "compact",
        # "maxBytes":0,
        # "maxRows":0,
        "connected": True,
        "show_dtypes": False,
    }
    default_kwargs.update(itable_kwargs)

    return show(styled_df, **default_kwargs)


def create_metric_gttable(
    source_table: pl.DataFrame,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    column_to_metric: Optional[Dict] = None,
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
    ...     rowname_col="Name",
    ... )
    """
    from great_tables import GT, loc, style
    from .metric_limits import add_rag_columns

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


def n_unique_values(dm, all_dm_cols, fld):
    if not isinstance(fld, list):
        fld = [fld]
    fld = polars_subset_to_existing_cols(all_dm_cols, fld)
    if len(fld) == 0:
        return 0
    return dm.model_data.select(pl.col(fld)).drop_nulls().collect().n_unique()


def max_by_hierarchy(dm, all_dm_cols, fld, grouping):
    if not isinstance(fld, list):
        fld = [fld]
    fld = polars_subset_to_existing_cols(all_dm_cols, fld)
    if len(fld) == 0:
        return 0
    grouping = polars_subset_to_existing_cols(all_dm_cols, grouping)
    if len(grouping) == 0:
        return 0
    return (
        dm.model_data.group_by(grouping)
        .agg(pl.col(fld).drop_nulls().n_unique())
        .select(fld)
        .drop_nulls()
        .max()
        .collect()
        .item()
    )


def avg_by_hierarchy(dm, all_dm_cols, fld, grouping):
    if not isinstance(fld, list):
        fld = [fld]
    fld = polars_subset_to_existing_cols(all_dm_cols, fld)
    if len(fld) == 0:
        return 0
    grouping = polars_subset_to_existing_cols(all_dm_cols, grouping)
    if len(grouping) == 0:
        return 0
    return (
        dm.model_data.group_by(grouping)
        .agg(pl.col(fld).drop_nulls().n_unique())
        .select(fld)
        .drop_nulls()
        .mean()
        .collect()
        .item()
    )


def sample_values(dm, all_dm_cols, fld, n=6):
    if not isinstance(fld, list):
        fld = [fld]
    fld = polars_subset_to_existing_cols(all_dm_cols, fld)
    if len(fld) == 0:
        return "-"
    return (
        dm.model_data.select(
            pl.concat_str(fld, separator="/").alias("__SampleValues__")
        )
        .drop_nulls()
        .collect()
        .to_series()
        .unique()
        .sort()
        .to_list()[:n]
    )


def show_credits(quarto_source: str):
    _, quarto_version = get_quarto_with_version(verbose=False)
    _, pandoc_version = get_pandoc_with_version(verbose=False)

    timestamp_str = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")

    quarto_print(
        f"""

    Document created at: {timestamp_str}

    This notebook: {quarto_source}

    Quarto runtime: {quarto_version}
    Pandoc: {pandoc_version}

    """
    )


def serialize_query(query: Optional[QUERY]) -> Optional[Dict]:
    if query is None:
        return None

    if isinstance(query, pl.Expr):
        query = [query]

    if isinstance(query, (list, tuple)):
        serialized_exprs = {}
        for i, expr in enumerate(query):
            if not isinstance(expr, pl.Expr):
                raise ValueError("All items in query list must be Expressions")
            serialized_exprs[str(i)] = json.loads(expr.meta.serialize(format="json"))
        return {"type": "expr_list", "expressions": serialized_exprs}

    elif isinstance(query, dict):
        return {"type": "dict", "data": query}

    raise ValueError(f"Unsupported query type: {type(query)}")


def deserialize_query(serialized_query: Optional[Dict]) -> Optional[QUERY]:
    """Deserialize a query that was previously serialized with serialize_query.

    Parameters
    ----------
    serialized_query : Optional[Dict]
        A serialized query dictionary created by serialize_query

    Returns
    -------
    Optional[QUERY]
        The deserialized query
    """
    if serialized_query is None:
        return None

    if serialized_query["type"] == "expr_list":
        expr_list = []
        for _, val in serialized_query["expressions"].items():
            json_str = json.dumps(val)
            str_io = io.StringIO(json_str)
            expr_list.append(pl.Expr.deserialize(str_io, format="json"))
        return expr_list

    elif serialized_query["type"] == "dict":
        return serialized_query["data"]

    raise ValueError(f"Unknown query type: {serialized_query['type']}")


def remove_duplicate_html_scripts(html_content: str, verbose: bool = False) -> str:
    """Remove duplicate script tags from HTML to reduce file size.

    Specifically targets large JavaScript libraries (like Plotly.js) that get
    embedded multiple times in HTML reports, while preserving all unique
    plot data and initialization scripts.
    """
    try:
        script_pattern = r"(?i)<script[^>]*?>(.*?)</script>"
        matches = list(re.finditer(script_pattern, html_content, re.DOTALL))

        seen_hashes = set()
        to_remove = []

        for match in matches:
            content = match.group(1)

            # Only target large scripts that are likely libraries
            if len(content) < 1000000:  # 1MB - target the 4.61MB Plotly duplicates
                continue

            content_hash = hash(content)
            if content_hash in seen_hashes:
                to_remove.append(match)
            else:
                seen_hashes.add(content_hash)

        # Remove duplicates (reverse order to preserve indices)
        result = html_content
        for match in reversed(to_remove):
            start, end = match.span()
            result = (
                result[:start] + "<!-- Duplicate script removed -->\n" + result[end:]
            )

        if verbose and to_remove:
            size_reduction = 1 - len(result) / len(html_content)
            logger.info(
                f"Removed {len(to_remove)} duplicate scripts ({size_reduction:.1%} reduction)"
            )

        return result

    except Exception as e:
        logger.warning(f"Script deduplication failed: {e}")
        return html_content
