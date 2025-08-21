import os
import datetime
import json
import logging
import re
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import polars as pl

from ..adm.CDH_Guidelines import CDHGuidelines
from ..utils.show_versions import show_versions
from ..utils.types import QUERY

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
    """Copy the report quarto file to the temporary directory."""
    from pdstools import __reports__

    shutil.copy(__reports__ / qmd_file, temp_dir)
    shutil.copytree(__reports__ / "assets", temp_dir / "assets", dirs_exist_ok=True)

def _write_params_files(
    temp_dir: Path,
    params: Optional[Dict] = None,
    project: Dict = {"type": "default"},
    analysis: Optional[Dict] = None,
) -> None:
    """Write parameters to a YAML file."""
    import yaml

    params = params or {}
    analysis = analysis or {}

    # Parameters to python code
    with open(temp_dir / "params.yml", "w") as f:
        yaml.dump(
            params,
            f,
        )
    
    # Project/rendering options to quarto
    with open(temp_dir / "_quarto.yml", "w") as f:
        yaml.dump(
            {
                "project": project,
                "analysis": analysis,
            },
            f,
        )

def run_quarto(
    qmd_file: Optional[str] = None,
    output_filename: Optional[str] = None,
    output_type: Optional[str] = "html",
    params: Optional[Dict] = None,
    project: Dict = {"type": "default"},
    analysis: Optional[Dict] = None,
    temp_dir: Path = Path("."),
    verbose: bool = False,
) -> int:
    """Run the Quarto command to generate the report."""

    def get_command() -> List[str]:
        quarto_exec, _ = get_quarto_with_version(verbose)
        _command = [str(quarto_exec), "render"]

        if qmd_file is not None:
            _command.append(qmd_file)

        options = _set_command_options(
            output_type=output_type,
            output_filename=output_filename,
            execute_params=params is not None)

        _command.extend(options)
        return _command

    if params is not None:
        _write_params_files(
            temp_dir,
            params=params,
            project=project,
            analysis=analysis,
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

    if process.stdout is not None:
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if verbose:
                print(line)
            logger.info(line)
    else:  # pragma: no cover
        logger.warning("subprocess.stdout is None, unable to read output")

    return_code = process.wait()
    message = f"Quarto process exited with return code {return_code}"
    logger.info(message)

    return return_code

def _set_command_options(
    output_type: Optional[str] = None,
    output_filename: Optional[str] = None,
    execute_params: bool = False,
) -> List[str]:
    """Set the options for the Quarto command."""

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
    # Match version numbers in the format X.Y.Z (ignoring any pre-release or build metadata)
    match = re.search(r'(\d+(?:\.\d+)*)', versionstr)
    return match.group(1) if match else ""
    # Match version numbers in the format X.Y.Z (ignoring any pre-release or build metadata)
    match = re.search(r'(\d+(?:\.\d+)*)', versionstr)
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


def rag_background_styler(rag: Optional[str] = None):
    from great_tables import style

    if rag is not None and len(rag) > 0:
        rag_upper = rag[0].upper()
        if rag_upper == "R":
            return style.fill(color="orangered")
        elif rag_upper == "A":
            return style.fill(color="orange")
        elif rag_upper == "Y":
            return style.fill(color="yellow")
        elif rag_upper == "G":
            return None  # no green background to keep it light
    raise ValueError(f"Not a supported RAG value: {rag}")


def rag_background_styler_dense(rag: Optional[str] = None):
    from great_tables import style

    if rag is not None and len(rag) > 0:
        rag_upper = rag[0].upper()
        if rag_upper == "R":
            return style.fill(color="orangered")
        elif rag_upper == "A":
            return style.fill(color="orange")
        elif rag_upper == "Y":
            return style.fill(color="yellow")
        elif rag_upper == "G":
            return style.fill(color="green")
    raise ValueError(f"Not a supported RAG value: {rag}")


def rag_textcolor_styler(rag: Optional[str] = None):
    from great_tables import style

    if rag is not None and len(rag) > 0:
        rag_upper = rag[0].upper()
        if rag_upper == "R":
            return style.text(color="orangered")
        elif rag_upper == "A":
            return style.text(color="orange")
        elif rag_upper == "Y":
            return style.text(color="yellow")
        elif rag_upper == "G":
            return style.text(color="green")
    raise ValueError(f"Not a supported RAG value: {rag}")


def table_standard_formatting(
    source_table,
    title=None,
    subtitle=None,
    rowname_col=None,
    groupname_col=None,
    cdh_guidelines=CDHGuidelines(),
    highlight_limits: Dict[str, Union[str, List[str]]] = {},
    highlight_lists: Dict[str, List[str]] = {},
    highlight_configurations: List[str] = [],
    rag_styler: callable = rag_background_styler,
):
    from great_tables import GT, loc

    def apply_style(gt, rag, rows):
        style = rag_styler(rag)
        if style is not None:
            gt = gt.tab_style(
                style=style,
                locations=loc.body(columns=col_name, rows=rows),
            )
        return gt

    def apply_rag_styling(gt, col_name, metric):
        if col_name in source_table.collect_schema().names():
            min_val = cdh_guidelines.min(metric)
            max_val = cdh_guidelines.max(metric)
            best_practice_min = cdh_guidelines.best_practice_min(metric)
            best_practice_max = cdh_guidelines.best_practice_max(metric)

            values = source_table[col_name].to_list()
            bad_rows = [
                i
                for i, v in enumerate(values)
                if v is not None
                and (
                    (min_val is not None and v < min_val)
                    or (max_val is not None and v > max_val)
                )
            ]
            warning_rows = [
                i
                for i, v in enumerate(values)
                if v is not None
                and (
                    (
                        min_val is not None
                        and best_practice_min is not None
                        and v >= min_val
                        and v < best_practice_min
                    )
                    or (
                        max_val is not None
                        and best_practice_max is not None
                        and v > best_practice_max
                        and v <= max_val
                    )
                )
            ]
            good_rows = [
                i
                for i, v in enumerate(values)
                if v is not None
                and (best_practice_min is None or v >= best_practice_min)
                and (best_practice_max is None or v <= best_practice_max)
            ]

            gt = apply_style(gt, "green", good_rows)
            gt = apply_style(gt, "amber", warning_rows)
            gt = apply_style(gt, "red", bad_rows)
        return gt

    gt = (
        GT(source_table, rowname_col=rowname_col, groupname_col=groupname_col)
        .tab_options(table_font_size=8)
        .sub_missing(missing_text="")
    )

    if title is not None:
        gt = gt.tab_header(title=title, subtitle=subtitle)

    for metric in highlight_limits.keys():
        cols = highlight_limits[metric]
        if isinstance(cols, str):
            cols = [cols]
        # Highlight colors
        for col_name in cols:
            gt = apply_rag_styling(gt, col_name=col_name, metric=metric)

        # Value formatting
        if metric == "Model Performance":
            gt = gt.fmt_number(
                decimals=2,
                columns=cols,
            )
        elif metric == "Engagement Lift":
            gt = gt.fmt_percent(
                decimals=0,
                columns=cols,
            )
        elif metric == "OmniChannel":
            gt = gt.fmt_percent(
                decimals=1,
                columns=cols,
            )
        elif metric == "CTR":
            gt = gt.fmt_percent(
                decimals=3,
                columns=cols,
            )
        else:
            gt = gt.fmt_number(
                decimals=0,
                compact=True,
                columns=cols,
            )

    # Highlight columns with non-standard values
    def simplify_name(x: str) -> str:
        if x is None:
            return x
        return re.sub("\\W", "", x, flags=re.IGNORECASE).upper()

    for col_name in highlight_lists.keys():
        if col_name in source_table.collect_schema().names():
            simplified_names = [simplify_name(x) for x in highlight_lists[col_name]]
            values = source_table[col_name].to_list()
            non_standard_rows = [
                i
                for i, v in enumerate(values)
                if simplify_name(v) not in simplified_names
            ]
            gt = apply_style(gt, "yellow", non_standard_rows)

    # Highlight column with more than one element (assuming its a comma-separated string)
    for col_name in highlight_configurations:
        if col_name in source_table.collect_schema().names():
            values = source_table[col_name].to_list()
            multiple_config_rows = [i for i, v in enumerate(values) if v.count(",") > 1]
            gt = apply_style(gt, "yellow", multiple_config_rows)

    return gt


def table_style_predictor_count(
    gt, flds, cdh_guidelines=CDHGuidelines(), rag_styler=rag_textcolor_styler
):
    from great_tables import GT, loc

    if not isinstance(gt, GT):
        raise ValueError("gt argument should be a Great Table")
    for col in flds:
        gt = gt.tab_style(
            style=rag_styler("amber"),
            locations=loc.body(
                columns=col,
                rows=(pl.col(col) < 200) | (pl.col(col) > 700) & (pl.col(col) > 0),
            ),
        ).tab_style(
            style=rag_styler("red"),
            locations=loc.body(
                columns=col,
                rows=(pl.col(col) == 0),
            ),
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


def _serialize_query(query: Optional[QUERY]) -> Optional[Dict]:
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
