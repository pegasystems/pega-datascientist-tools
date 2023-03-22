from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import streamlit as st
from pdstools import ADMDatamart
from pdstools.utils import datasets
import polars as pl


def multiFilter(df, q):
    df = df.lazy()
    with pl.StringCache():
        for filter in q:
            df = df.filter(filter)
    return df


def remove_duplicate_expressions(exprs):
    expr_dict = {}
    for expr in exprs:
        try:
            expr_dict[expr.meta.root_names()[0]] = expr
        except:
            pass

    return list(expr_dict.values())


def import_data(params, default=0, **kwargs):
    config_type = st.selectbox(
        "Data import type",
        options=[
            "Sample dataset",
            "Filepath",
            "File upload",
            "S3 Bucket",
        ],
        index=default,
    )
    if config_type == "File upload":
        with st.expander("Upload files", expanded=True):
            model_file = st.file_uploader(
                "Upload Model Snapshot", type=["json", "zip", "parquet", "csv", "arrow"]
            )
            predictor_file = st.file_uploader(
                "Upload Predictor Binning snapshot",
                type=["json", "zip", "parquet", "csv", "arrow"],
            )
            params["kwargs"] = ADMDatamart_options()
            if st.checkbox("Use this data source"):
                if model_file is not None:
                    data = import_datamart(
                        ADMDatamart,
                        model_df=model_file,
                        predictor_df=predictor_file,
                        **params["kwargs"],
                    )
                else:
                    "Please upload your datamart files."
    elif config_type == "Sample dataset":
        data = import_datamart(datasets.CDHSample)
    elif config_type == "Filepath":
        with st.expander("Configure data", expanded=True):
            path = st.text_input("Path to files")
            params["kwargs"] = ADMDatamart_options()
            if st.checkbox("Use this data source"):
                data = import_datamart(ADMDatamart, path, **params["kwargs"])
    elif config_type == "S3 Bucket":
        with st.expander("S3 settings", expanded=True):
            bucket = st.text_input("S3 bucket name")
            key = st.text_input("File location")
            sql_query = st.text_input("Optional SQL query", value="")
            if st.checkbox("Use this data source"):
                cols = [
                    "pyModelID",
                    "pyPerformance",
                    "pyResponseCount",
                    "pyNegatives",
                    "pyPositives",
                    "pyGroup",
                    "pyIssue",
                    "pyChannel",
                    "pyDirection",
                    "pyName",
                    "pySnapshotTime",
                ]
                import sys

                try:
                    from boto3 import client
                except ModuleNotFoundError:
                    raise ImportError("To use an S3 connection, please install boto3.")

                @st.cache(show_spinner=True)
                def get_from_s3(bucket, key, sql_query):
                    conn = client("s3")
                    resp = conn.select_object_content(
                        Bucket=bucket,
                        Key=key,
                        ExpressionType="SQL",
                        Expression=f"SELECT s.{', s.'.join(cols)} FROM s3object s",
                        InputSerialization={
                            "CompressionType": "GZIP",
                            "JSON": {"Type": "LINES"},
                        },
                        OutputSerialization={"JSON": {"RecordDelimiter": "\n"}},
                    )
                    return [
                        event["Records"]["Payload"].decode("utf-8")
                        for event in resp["Payload"]
                        if "Records" in event
                    ]

                response = get_from_s3(bucket, key, sql_query)
                data = import_datamart(
                    ADMDatamart(
                        model_df=pd.read_json(
                            str("".join(response)), lines=True, **kwargs
                        )
                    )
                )
    try:
        return params, data
    except UnboundLocalError:
        return params, None


@st.cache_resource(show_spinner=True)
def import_datamart(_fun, *args, **kwargs):
    dm = _fun(*args, **kwargs)
    dm.modelData = dm.modelData.collect()
    dm.predictorData = dm.predictorData.collect()
    return dm


def ADMDatamart_options():
    params = dict()
    if st.checkbox("Edit Parameters"):
        extract_treatment = st.checkbox("Extract treatments", False)
        if extract_treatment:
            params["extract_treatment"] = "pyName"
        context_keys = ["Issue", "Group", "Channel", "Direction"]
        if extract_treatment:
            context_keys.append("Treatment")
        params["context_keys"] = st.multiselect(
            "Select context keys", context_keys, default=context_keys
        )
        params["plotting_engine"] = "plotly"
        params["timestamp_fmt"] = st.text_input(
            "Timestamp format", value="%Y%m%dT%H%M%S.%f %Z"
        )
    return params


def generate_modeldata_filters(data: ADMDatamart, params: dict) -> List[pl.Expr]:
    """Generates and applies filters to our dataframe.

    Parameters
    ----------
    data : ADMDatamart
        Our original, unfiltered datamart class
    params : dict
        A dictionary of all configurations,
        Not really used here except for keeping track of what we do


    Returns
    -------
    dict
        Our original params file, updated with
        whatever queries we've added.

    """

    df = data.modelData.with_column(pl.col(pl.Categorical).cast(pl.Utf8))
    filtereddf = df
    st.session_state["filters"] = Filters(df=df)
    filters = []

    if st.checkbox("Add filters"):
        filters = st.multiselect(label="add filters.", options=df.columns)
        if len(filters) > 0:
            for col in filters:
                st.session_state["filters"].add_filter(col)

        if len(st.session_state["filters"].exprs) > 0:
            filters = remove_duplicate_expressions(st.session_state["filters"].exprs)

    with st.expander("Preview", expanded=False):
        if len(filters) > 0:
            st.write("#### Preview of the filtered dataset:")
            filtereddf = multiFilter(df, filters)
            st.dataframe(filtereddf.collect().to_pandas().head(10))
        else:
            st.write("#### Preview of the dataset")
            st.dataframe(df.collect().to_pandas().head(10))
    params["filters"] = filters

    return filtereddf, params


@dataclass
class Filters:
    df: pl.DataFrame
    exprs = []

    def add_filter(self, column):
        if self.df.schema[column] in [pl.Utf8, pl.Categorical]:
            self.exprs.append(self.CategoryFilter(self.df, column))
        else:
            self.exprs.append(self.ValueFilter(df=self.df, column_name=column))

    @staticmethod
    def CategoryFilter(df, column_name):
        with st.expander(column_name, expanded=True):
            filter, selectall = st.columns([3, 1])
            with filter:
                container = st.container()
            with selectall:
                all = st.checkbox("Select all", key=f"select_all_{column_name}")
                options = (
                    df.select(column_name).unique().collect().to_series().to_list()
                )
                if all:
                    opts = list(
                        container.multiselect(
                            f"Filter {column_name}",
                            options,
                            options,
                            key=f"{column_name}_all",
                        )
                    )

                else:
                    opts = list(
                        container.multiselect(
                            f"Filter {column_name}", options, key=column_name
                        )
                    )

            def format(val):
                return "Include selection" if val else "Exclude selection"

            include = st.selectbox(
                "Filter method",
                [True, False],
                key=f"include_{column_name}",
                format_func=format,
            )

            expr = pl.col(column_name).is_in(opts)
            if not include:
                expr = expr.is_not()
        return expr

    @staticmethod
    def ValueFilter(df, column_name):
        with st.expander(column_name, expanded=True):
            st.write("this is a value filter")
            valrange = (
                df.select(
                    pl.col(column_name).min().alias("min"), pl.col(column_name).max()
                )
                .collect()
                .row(0)
            )
            min, max = tuple(
                st.slider(
                    f"Filter on {column_name}",
                    min_value=valrange[0],
                    max_value=valrange[1],
                    value=(valrange),
                    key=column_name,
                )
            )
            st.write(f"key:{column_name}, min:{min}, max:{max}")
            st.write()
            return pl.col(column_name).is_between(min, max)
