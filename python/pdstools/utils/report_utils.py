import re
import traceback
from typing import Dict, List, Literal, Optional, Union
from IPython.display import display, Markdown
from great_tables import GT, style, loc
from ..adm.CDH_Guidelines import CDHGuidelines
from ..utils.show_versions import show_versions
from ..adm.Reports import Reports
import polars as pl
import datetime


def quarto_print(text):
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
    return col in df.collect_schema().names() and df.schema[col] != pl.Null


def polars_subset_to_existing_cols(all_columns, cols):
    return [col for col in cols if col in all_columns]


def rag_background_styler(
    rag: Optional[Literal["Red", "Amber", "Yellow", "Green"]] = None
):
    match rag[0].upper() if len(rag) > 0 else None:
        case "R":
            return style.fill(color="orangered")
        case "A":
            return style.fill(color="orange")
        case "Y":
            return style.fill(color="yellow")
        case "G":
            return None  # no green background to keep it light
        case _:
            raise ValueError(f"Not a supported RAG value: {rag}")


def rag_background_styler_dense(
    rag: Optional[Literal["Red", "Amber", "Yellow", "Green"]] = None
):
    match rag[0].upper() if len(rag) > 0 else None:
        case "R":
            return style.fill(color="orangered")
        case "A":
            return style.fill(color="orange")
        case "Y":
            return style.fill(color="yellow")
        case "G":
            return style.fill(color="green")
        case _:
            raise ValueError(f"Not a supported RAG value: {rag}")


def rag_textcolor_styler(
    rag: Optional[Literal["Red", "Amber", "Yellow", "Green"]] = None
):
    match rag[0].upper() if len(rag) > 0 else None:
        case "R":
            return style.text(color="orangered")
        case "A":
            return style.text(color="orange")
        case "Y":
            return style.text(color="yellow")
        case "G":
            return style.text(color="green")
        case _:
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
            # TODO consider that bad / warning rows are exclusive

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
        match metric:
            case "Model Performance":
                gt = gt.fmt_number(
                    decimals=2,
                    columns=cols,
                )
            case "Engagement Lift":
                gt = gt.fmt_percent(
                    decimals=0,
                    columns=cols,
                )
            case "OmniChannel":
                gt = gt.fmt_percent(
                    decimals=0,
                    columns=cols,
                )
            case "CTR":
                gt = gt.fmt_percent(
                    decimals=3,
                    columns=cols,
                )
            case _:
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
    gt: GT, flds, cdh_guidelines=CDHGuidelines(), rag_styler=rag_textcolor_styler
):
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
    _, quarto_version = Reports.get_quarto_with_version(verbose=False)
    _, pandoc_version = Reports.get_pandoc_with_version(verbose=False)

    timestamp_str = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")

    quarto_print(
        f"""

    Document created at: {timestamp_str}

    This notebook: {quarto_source}
    
    Quarto runtime: {quarto_version}
    Pandoc: {pandoc_version}
    
    Additional details from 'pdstools.show_versions()':

    """
    )

    show_versions()

    quarto_print(
        "For more information please see the [Pega Data Scientist Tools](https://github.com/pegasystems/pega-datascientist-tools)."
    )
