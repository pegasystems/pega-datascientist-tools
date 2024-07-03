import os
from pathlib import Path
from typing import List

import polars as pl
import streamlit as st
from st_pages import Page

from pdstools.decision_analyzer.data_read_utils import (
    get_da_data_path,
    read_data,
    read_nested_zip_files,
)


# from ...decision_analyzer.plots import (
#     plot_propensity_vs_optionality,
#     plot_optionality_per_stage,
#     plot_offer_quality_piecharts,
#     plot_action_variation,
#     plot_optionality_trend,
#     plot_prio_factor_boxplots,
#     plot_rank_boxplot,
#     plot_value_distribution,
#     plot_trend_chart,
# )
def ensure_data():
    if "decision_data" not in st.session_state:
        st.warning("Please upload your data in the Home page")
        st.stop()


def ensure_getFilterComponentData():
    return "pxComponentName" in st.session_state.decision_data.decision_data.columns


st.elements.utils._shown_default_value_warning = (
    True  # to suppress default val+key warning in date filter
)
polars_lazyframe_hashing = {
    pl.LazyFrame: lambda x: hash(x.explain(optimized=False)),
    pl.Expr: lambda x: str(x.inspect()),
    # datetime.datetime: lambda x: x.strftime("%Y%m%d%H%M%S")
}


def get_current_index(options, session_state_key, default=0):
    if (session_state_key in st.session_state.keys()) and (
        st.session_state[session_state_key] in options
    ):
        return options.index(st.session_state[session_state_key])
    else:
        return default


def get_current_stage_index(options):
    """Picks up the last stage from session state if it is set and it is applicable to
    current visual. Otherwise, picks the first option in the list"""
    return get_current_index(options, "stage")


def get_current_scope_index(options):
    """Picks up the last scope from session state if it is set and it is applicable to
    current visual. Otherwise, picks the first option in the list"""
    return get_current_index(options, "scope")


def show_filtered_counts(statsBefore, statsAfter):
    keys = set().union(statsBefore.keys(), statsAfter.keys())
    for k in keys:
        st.progress(
            (statsAfter[k] / statsBefore[k]),
            text=f"{statsAfter[k]} {k} remaining from {statsBefore[k]}",
        )


def _clean_unselected_filters(to_filter_columns, filter_type):
    keys_to_remove = []
    for key in st.session_state.keys():
        if key.__contains__("selected_"):
            column_name = key.split("selected_", 1)[1]
            if column_name not in to_filter_columns:
                keys_to_remove.append(column_name)
    for column in keys_to_remove:
        selected_key = f"{filter_type}selected_{column}"
        _selected_key = f"{filter_type}_selected_{column}"
        regexselected_key = f"{filter_type}regexselected_{column}"
        regex_selected_key = f"{filter_type}regex_selected_{column}"
        categories_key = f"{filter_type}categories_{column}"
        for val in [
            selected_key,
            _selected_key,
            categories_key,
            regexselected_key,
            regex_selected_key,
        ]:
            if val in st.session_state:
                del st.session_state[val]


def get_data_filters(
    df: pl.LazyFrame, columns=[], queries=[], filter_type="local"
) -> List[
    pl.Expr
]:  # this one is way too complex, should be split up into probably 5 functions
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Parameters
    ----------
    df : pl.DataFrame
        Original dataframe
    """

    def _save_selected(
        filter_type, column, regex=""
    ):  ## see the issue on why we need to save a different session.state variable https://discuss.streamlit.io/t/session-state-is-not-preserved-when-navigating-pages/48787
        st.session_state[f"{filter_type}{regex}selected_{column}"] = st.session_state[
            f"{filter_type}{regex}_selected_{column}"
        ]

    def _save_multiselect():
        st.session_state[f"{filter_type}multiselect"] = st.session_state[
            f"{filter_type}_multiselect"
        ]

    columns = df.columns if columns == [] else columns

    st.session_state[f"{filter_type}_multiselect"] = (
        st.session_state[f"{filter_type}multiselect"]
        if f"{filter_type}multiselect" in st.session_state
        else []
    )
    to_filter_columns = st.multiselect(
        "Filter data on",
        columns,
        key=f"{filter_type}_multiselect",
        format_func=lambda x: (
            x.lstrip("py")
            if x.startswith("py")
            else (x.lstrip("px") if x.startswith("px") else x)
        ),
        on_change=_save_multiselect,
    )
    for column in to_filter_columns:
        left, right = st.columns((1, 20))
        left.write("## ↳")

        # Treat columns with < 20 unique values as categorical
        if (df.schema[column] == pl.Categorical) or (df.schema[column] == pl.Utf8):
            if f"{filter_type}categories_{column}" not in st.session_state.keys():
                st.session_state[f"{filter_type}categories_{column}"] = (
                    df.select(pl.col(column).unique()).collect().to_series().to_list()
                )
            if f"{filter_type}_selected_{column}" not in st.session_state.keys():
                st.session_state[f"{filter_type}_selected_{column}"] = (
                    st.session_state[f"{filter_type}categories_{column}"]
                    if f"{filter_type}selected_{column}" not in st.session_state
                    else st.session_state[f"{filter_type}selected_{column}"]
                )
            if len(st.session_state[f"{filter_type}categories_{column}"]) < 200:
                options = st.session_state[f"{filter_type}categories_{column}"]
                default_selected = (
                    st.session_state[f"{filter_type}selected_{column}"]
                    if f"{filter_type}selected_{column}" in st.session_state
                    else options
                )
                st.session_state[f"{filter_type}_selected_{column}"] = default_selected
                selected = right.multiselect(
                    f"Values for {column}",
                    options=options,
                    key=f"{filter_type}_selected_{column}",
                    on_change=_save_selected,
                    kwargs={"filter_type": filter_type, "column": column},
                )
                if selected != st.session_state[f"{filter_type}categories_{column}"]:
                    queries.append(
                        pl.col(column)
                        .cast(pl.Utf8)
                        .is_in(st.session_state[f"{filter_type}selected_{column}"])
                    )

            else:
                del st.session_state[f"{filter_type}_selected_pyName"]
                default_selected = (
                    st.session_state[f"{filter_type}regexselected_{column}"]
                    if f"{filter_type}regexselected_{column}" in st.session_state
                    else ""
                )
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    value=default_selected,
                    key=f"{filter_type}regex_selected_{column}",
                    on_change=_save_selected,
                    kwargs={
                        "filter_type": filter_type,
                        "column": column,
                        "regex": "regex",
                    },
                )
                if user_text_input:
                    queries.append(pl.col(column).str.contains(user_text_input))

        elif df.schema[column] in pl.NUMERIC_DTYPES:
            min_col, max_col = right.columns((1, 1))
            _min = float(df.select(pl.min(column)).collect().item())
            _max = float(df.select(pl.max(column)).collect().item())
            if f"{filter_type}selected_{column}" not in st.session_state:
                default_min, default_max = _min, _max
            else:
                default_min, default_max = st.session_state[
                    f"{filter_type}selected_{column}"
                ]
            if _max - _min <= 200:
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(default_min, default_max),
                )
            else:
                user_min = min_col.number_input(
                    label=f"Min value for {column} (Min:{_min})",
                    min_value=_min,
                    max_value=_max,
                    value=default_min,
                )
                user_max = max_col.number_input(
                    label=f"Max value for {column} (Max:{_max})",
                    min_value=_min,
                    max_value=_max,
                    value=default_max,
                )
                user_num_input = [user_min, user_max]
            st.session_state[f"{filter_type}selected_{column}"] = user_num_input
            if user_num_input[0] != _min or user_num_input[1] != _max:
                queries.append(pl.col(column).is_between(*user_num_input))
        elif df.schema[column] in pl.TEMPORAL_DTYPES:
            value = (
                df.select(pl.min(column)).collect().item(),
                df.select(pl.max(column)).collect().item(),
            )
            st.session_state[f"{filter_type}_selected_{column}"] = (
                (st.session_state[f"{filter_type}selected_{column}"])
                if f"{filter_type}selected_{column}" in st.session_state
                else value
            )
            user_date_input = right.date_input(
                f"Values for {column}",
                value=value,
                key=f"{filter_type}_selected_{column}",
                on_change=_save_selected,
                kwargs={"filter_type": filter_type, "column": column},
            )
            if len(user_date_input) == 2:
                queries.append(pl.col(column).is_between(*user_date_input))
    _clean_unselected_filters(to_filter_columns, filter_type)
    return queries


def _create_page(relative_path, name):
    current_dir = os.path.dirname(__file__)
    return Page(os.path.join(current_dir, relative_path), name)


def get_pages(extract_type):
    if extract_type == "explainability_extract":
        pages = [
            _create_page("home.py", "Home"),
            _create_page("pages/1-Global_Filters.py", "Global Filters"),
            _create_page("pages/2-Global_Dashboard.py", "Global Dashboard"),
            _create_page("pages/3-action_Distribution.py", "Action Distribution"),
            _create_page("pages/5-Global_Sensitivity.py", "Global Sensitivity"),
            _create_page("pages/6-Win_Loss_Analysis.py", "Win Loss Analysis"),
            _create_page(
                "pages/7-Personalization_Analysis.py", "Personalization Analysis"
            ),
        ]
    elif extract_type == "decision_analyzer":
        pages = [
            _create_page("Home.py", "Home"),
            _create_page("pages/1-Global_Filters.py", "Global Filters"),
            _create_page("pages/2-Global_Dashboard.py", "Global Dashboard"),
            _create_page("pages/3-Action_Distribution.py", "Action Distribution"),
            _create_page("pages/4-Action_Funnel.py", "Action Funnel"),
            _create_page("pages/5-Global_Sensitivity.py", "Global Sensitivity"),
            _create_page("pages/6-Win_Loss_Analysis.py", "Win Loss Analysis"),
            _create_page(
                "pages/7-Personalization_Analysis.py", "Personalization Analysis"
            ),
        ]
        # Page("pages/8-Offer_Quality_Analysis.py", "Offer Quality Analysis"),
        # Page("pages/9-Thresholding_Analysis.py", "Thresholding Analysis"),
        # Page("pages/10-Business_Value_Analysis.py", "Business Value Analysis"),
        # Page("pages/11-Business_Lever_Analysis.py", "Business Lever Analysis"),
        # Page("pages/12-Impact_Analysis.py", "Impact Analysis"),
    return pages


def get_options():
    is_ec2 = os.getcwd() == "/app"
    if is_ec2:
        return ["Sample Data", "File Upload"]
    else:
        return ["Sample Data", "File Upload", "Direct File Path"]


def handle_sample_data(is_ec2):
    if is_ec2:
        path = Path("/s3-files/anonymized/anonymized")
    else:
        path = Path(get_da_data_path(), "sample_data/anonymized")
    return read_data(path)


def handle_file_upload():
    uploaded_file = st.file_uploader("Choose your zipped file", type="zip")
    if uploaded_file is not None:
        return read_nested_zip_files(uploaded_file)


def handle_direct_file_path():
    st.write(
        """You can import the data simply by pointing the app to the directory  
        where the original files are located."""
    )
    dir = st.text_input(
        "File or partitioned folder path",
        placeholder="/Users/Downloads",
    )
    if dir:
        return read_data(dir)
    return None


# @st.cache_data(hash_funcs=polars_lazyframe_hashing)
# def st_plot_propensity_vs_optionality(df: pl.LazyFrame):
#     return plot_propensity_vs_optionality(df)


# @st.cache_data(hash_funcs=polars_lazyframe_hashing)
# def st_plot_optionality_per_stage(df: pl.LazyFrame, NBADStages_Mapping):
#     return plot_optionality_per_stage(df, NBADStages_Mapping)


# @st.cache_data(hash_funcs=polars_lazyframe_hashing)
# def st_plot_offer_quality_piecharts(
#     df: pl.LazyFrame, propensityTH, NBADStages_FilterView, NBADStages_Mapping
# ):
#     return plot_offer_quality_piecharts(
#         df, propensityTH, NBADStages_FilterView, NBADStages_Mapping
#     )


# @st.cache_data(hash_funcs=polars_lazyframe_hashing)
# def st_plot_action_variation(df: pl.LazyFrame):
#     return plot_action_variation(df)


# @st.cache_data(hash_funcs=polars_lazyframe_hashing)
# def st_plot_optionality_trend(df: pl.LazyFrame, NBADStages_Mapping):
#     return plot_optionality_trend(df, NBADStages_Mapping)


# def st_plot_prio_factor_boxplots(
#     df: pl.LazyFrame,
#     reference: Optional[Union[pl.Expr, List[pl.Expr]]] = None,
#     sample_size=10000,
# ) -> Optional[go.Figure]:
#     # Call the core function to generate the plot and check for warnings
#     fig, warning_message = plot_prio_factor_boxplots(df, reference, sample_size)

#     if warning_message:
#         st.warning(warning_message)
#         st.stop()

#     return fig


# @st.cache_data(hash_funcs=polars_lazyframe_hashing)
# def st_plot_rank_boxplot(
#     df: pl.LazyFrame, reference: Optional[Union[pl.Expr, List[pl.Expr]]] = None
# ):
#     return plot_rank_boxplot(df, reference)


# @st.cache_data(hash_funcs=polars_lazyframe_hashing)
# def st_plot_value_distribution(value_data: pl.LazyFrame, scope: str):
#     return plot_value_distribution(value_data, scope)


# def st_plot_trend_chart(df: pl.LazyFrame, scope: str) -> Optional[go.Figure]:
#     fig, warning_message = plot_trend_chart(df, scope)

#     if warning_message:
#         st.warning(warning_message)

#     return fig
