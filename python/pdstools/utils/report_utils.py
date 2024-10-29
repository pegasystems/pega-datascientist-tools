from typing import Dict, List
from IPython.display import display, Markdown
from great_tables import GT, style, loc
from ..adm.CDH_Guidelines import CDHGuidelines
import polars as pl


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


def polars_col_exists(df, col):
    return col in df.columns and df.schema[col] != pl.Null


def polars_subset_to_existing_cols(all_columns, cols):
    return [col for col in cols if col in all_columns]


def table_standard_formatting(
    source_table,
    title=None,
    rowname_col=None,
    groupname_col=None,
    cdh_guidelines=CDHGuidelines(),
    highlight_limits: Dict[str, str] = {},
    highlight_lists: Dict[str, List[str]] = {},
    highlight_configurations: List[str] = [],
):
    def apply_metric_style(gt, col_name, metric):
        if col_name in source_table.columns:
            min_val = cdh_guidelines.min(metric)
            max_val = cdh_guidelines.max(metric)
            best_practice_min = cdh_guidelines.best_practice_min(metric)
            best_practice_max = cdh_guidelines.best_practice_max(metric)

            values = source_table[col_name].to_list()
            bad_rows = [
                i
                for i, v in enumerate(values)
                if v < min_val or (max_val is not None and v > max_val)
            ]
            warning_rows = [
                i
                for i, v in enumerate(values)
                if (v >= min_val and v < best_practice_min)
                or (
                    best_practice_max is not None
                    and max_val is not None
                    and v > best_practice_max
                    and v <= max_val
                )
            ]

            gt = gt.tab_style(
                style=style.fill(color="orangered"),
                locations=loc.body(columns=col_name, rows=bad_rows),
            )
            gt = gt.tab_style(
                style=style.fill(color="orange"),
                locations=loc.body(columns=col_name, rows=warning_rows),
            )
        return gt

    def apply_standard_name_style(gt, col_name, standard_list):
        if col_name in source_table.columns:
            values = source_table[col_name].to_list()
            non_standard_rows = [
                i for i, v in enumerate(values) if v not in standard_list
            ]
            gt = gt.tab_style(
                style=style.fill(color="yellow"),
                locations=loc.body(columns=col_name, rows=non_standard_rows),
            )
        return gt

    def apply_configuration_style(gt, col_name):
        if col_name in source_table.columns:
            values = source_table[col_name].to_list()
            multiple_config_rows = [i for i, v in enumerate(values) if v.count(",") > 1]
            gt = gt.tab_style(
                style=style.fill(color="yellow"),
                locations=loc.body(columns=col_name, rows=multiple_config_rows),
            )
        return gt

    gt = GT(
        source_table, rowname_col=rowname_col, groupname_col=groupname_col
    ).tab_options(table_font_size=8)

    if title is not None:
        gt = gt.tab_header(title=title)

    for c in highlight_limits.keys():
        gt = apply_metric_style(gt, c, highlight_limits[c])
        gt = gt.fmt_number(
            columns=c, decimals=0, compact=True
        )  # default number formatting

    for c in highlight_lists.keys():
        gt = apply_standard_name_style(gt, c, highlight_lists[c])

    for c in highlight_configurations:
        gt = apply_configuration_style(gt, c)

    return gt


def table_style_predictor_count(gt: GT, flds, cdh_guidelines=CDHGuidelines()):
    for col in flds:
        gt = gt.tab_style(
            style=style.fill(color="orange"),
            locations=loc.body(
                columns=col,
                rows=(pl.col(col) < 200) | (pl.col(col) > 700) & (pl.col(col) > 0),
            ),
        ).tab_style(
            style=style.fill(color="orangered"),
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
