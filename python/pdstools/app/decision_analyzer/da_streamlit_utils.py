import os
from pathlib import Path
from typing import List

import polars as pl
import streamlit as st

from pdstools.decision_analyzer.data_read_utils import (
    get_da_data_path,
    read_data,
    read_nested_zip_files,
)


from pdstools.decision_analyzer.plots import (
    #     propensity_vs_optionality,
    #     optionality_per_stage,
    #     offer_quality_piecharts,
    #     action_variation,
    #     optionality_trend,
    #     prio_factor_boxplots,
    #     rank_boxplot,
    plot_priority_component_distribution,
    #     trend_chart,
)


def ensure_data():
    if "decision_data" not in st.session_state:
        st.warning("Please upload your data in the Home page")
        st.stop()


def ensure_funnel():
    if st.session_state.decision_data.extract_type == "explainability_extract":
        st.warning("You can only view this page with EEV2 dataset")
        st.stop()


def ensure_getFilterComponentData():
    return (
        "pxComponentName"
        in st.session_state.decision_data.decision_data.collect_schema().names()
    )


# st.elements.utils._shown_default_value_warning = (
#     True  # to suppress default val+key warning in date filter
# )
polars_lazyframe_hashing = {
    pl.LazyFrame: lambda x: hash(x.explain(optimized=False)),
    pl.Expr: lambda x: str(x.inspect()),
    # datetime.datetime: lambda x: x.strftime("%Y%m%d%H%M%S")
}


def get_current_index(options, key, default=0):
    """Get index from session state if key exists and value is in options, else return default"""
    return (
        options.index(st.session_state[key])
        if key in st.session_state and st.session_state[key] in options
        else default
    )


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

    columns = df.collect_schema().names() if columns == [] else columns

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
        path = Path(get_da_data_path(), "sample_data/rb_sample/data")
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
# def st_propensity_vs_optionality(df: pl.LazyFrame):
#     return propensity_vs_optionality(df)


# @st.cache_data(hash_funcs=polars_lazyframe_hashing)
# def st_optionality_per_stage(df: pl.LazyFrame, NBADStages_Mapping):
#     return optionality_per_stage(df, NBADStages_Mapping)


# @st.cache_data(hash_funcs=polars_lazyframe_hashing)
# def st_offer_quality_piecharts(
#     df: pl.LazyFrame, propensityTH, NBADStages_FilterView, NBADStages_Mapping
# ):
#     return offer_quality_piecharts(
#         df, propensityTH, NBADStages_FilterView, NBADStages_Mapping
#     )


# @st.cache_data(hash_funcs=polars_lazyframe_hashing)
# def st_action_variation(df: pl.LazyFrame):
#     return action_variation(df)


# @st.cache_data(hash_funcs=polars_lazyframe_hashing)
# def st_optionality_trend(df: pl.LazyFrame, NBADStages_Mapping):
#     return optionality_trend(df, NBADStages_Mapping)


# def st_prio_factor_boxplots(
#     df: pl.LazyFrame,
#     reference: Optional[Union[pl.Expr, List[pl.Expr]]] = None,
# ) -> Optional[go.Figure]:
#     # Call the core function to generate the plot and check for warnings
#     fig, warning_message = prio_factor_boxplots(df, reference)

#     if warning_message:
#         st.warning(warning_message)
#         st.stop()

#     return fig


# @st.cache_data(hash_funcs=polars_lazyframe_hashing)
# def st_rank_boxplot(
#     df: pl.LazyFrame, reference: Optional[Union[pl.Expr, List[pl.Expr]]] = None
# ):
#     return rank_boxplot(df, reference)


@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def st_priority_component_distribution(
    value_data: pl.LazyFrame, component, granularity
):
    return plot_priority_component_distribution(value_data, component, granularity)


# def st_trend_chart(df: pl.LazyFrame, scope: str) -> Optional[go.Figure]:
#     fig, warning_message = trend_chart(df, scope)

#     if warning_message:
#         st.warning(warning_message)

#     return fig
