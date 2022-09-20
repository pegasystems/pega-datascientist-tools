from dataclasses import dataclass
from typing import Dict
import pandas as pd
import streamlit as st
from pdstools import ADMDatamart, datasets


def import_data(params, default=1):
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
                "Upload Model Snapshot", type=["json", "zip", "parquet", "csv"]
            )
            predictor_file = st.file_uploader(
                "Upload Predictor Binning snapshot",
                type=["json", "zip", "parquet", "csv"],
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

                if "boto3" not in sys.modules:
                    raise ImportError("To use an S3 connection, please install boto3.")
                from boto3 import client

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
                        model_df=pd.read_json(str("".join(response)), lines=True)
                    )
                )
    try:
        return params, data
    except UnboundLocalError:
        return params, None


@st.cache(show_spinner=True)
def import_datamart(fun, *args, **kwargs):
    return fun(*args, **kwargs)


def ADMDatamart_options():
    params = dict()
    extract_treatment = st.checkbox("Extract treatments", False)
    if extract_treatment:
        params["extract_treatment"] = "pyName"
    context_keys = ["Issue", "Group", "Channel", "Direction"]
    if extract_treatment:
        context_keys.append("Treatment")
    params["context_keys"] = st.multiselect(
        "Select context keys", context_keys, default=context_keys
    )
    params["plotting_engine"] = st.selectbox(
        "Select engine for plots", ("plotly", "mpl")
    )
    return params


def generate_modeldata_filters(df, params):
    filters, query = None, None

    if st.checkbox("Add filters"):
        filters = Filters(df=df, params=params)
        to_filter = st.multiselect(label="Add filters.", options=filters.all_fields)
        if len(to_filter) > 0:
            for col in to_filter:
                filters.add_filter(col)

    with st.expander("Preview", expanded=False):
        if filters is not None:
            query = filters.generatePandasFilters()
            params["pandasquery"] = str(query)
        if query != "" and query is not None:
            st.write("Query to filter dataset:")
            st.write(query)
            st.write("#### Preview of the filtered dataset:")
            filtereddf = df.query(query)
            st.dataframe(filtereddf.sample(10))
        else:
            st.write("#### Preview of the dataset")
            st.dataframe(df.sample(10))
        return params


@dataclass
class Filters:
    params: Dict
    df: pd.DataFrame = None
    categoricals = {
        "Channel",
        "Issue",
        "Group",
        "Direction",
        "Treatment",
        "Configuration",
    }
    numericals = {"Positives", "ResponseCount"}
    dateFields = {"SnapshotTime"}
    pandasquery = ""

    def __post_init__(self):
        self.categoricals = self.categoricals.intersection(self.df.columns)
        self.all_fields = set().union(
            *[self.categoricals, self.numericals, self.dateFields]
        )

    def add_filter(self, key):
        if not "filters" in self.params.keys():
            self.params["filters"] = dict()

        if key in self.categoricals:
            self.CategoryFilter(key)
        elif key in self.numericals:
            self.ValueFilter(key)
        elif key in self.dateFields:
            self.DateFilter()
        else:
            raise ValueError(f"{key} not known.")
        self.generatePandasFilters()

    def CategoryFilter(self, key):
        with st.expander(key, expanded=True):
            filter, selectall = st.columns([3, 1])
            with filter:
                container = st.container()
            with selectall:
                all = st.checkbox("Select all", key=f"select_all_{key}")
                df = (
                    self.df
                    if self.pandasquery == ""
                    else self.df.query(self.pandasquery)
                )
                options = df[key].unique()

                if all:
                    self.params["filters"][key] = list(
                        container.multiselect(
                            f"Filter {key}", options, options, key=f"{key}_all"
                        )
                    )
                else:
                    self.params["filters"][key] = list(
                        container.multiselect(f"Filter {key}", options, key=key)
                    )

            def format(val):
                return "Include selection" if val else "Exclude selection"

            include = st.selectbox(
                "Filter method", [True, False], key=f"include_{key}", format_func=format
            )
            if not include:
                self.params["filters"][key] = list(
                    set(options) - set(self.params["filters"][key])
                )
            if len(self.params["filters"][key]) == 0:
                del self.params["filters"][key]

    def ValueFilter(self, key):
        df = self.df if self.pandasquery == "" else self.df.query(self.pandasquery)
        valrange = (int(df[key].min()), int(df[key].max()))
        self.params["filters"][key] = list(
            st.slider(
                f"Filter on {key}",
                min_value=valrange[0],
                max_value=valrange[1],
                value=(valrange),
                key=key,
            )
        )

    def DateFilter(self):
        last = st.checkbox("Last snapshot only", 0)
        df = self.df if self.pandasquery == "" else self.df.query(self.pandasquery)
        time_range = list(
            {time.strftime("%Y-%m-%d %H:%m:%S") for time in df.SnapshotTime.unique()}
        )
        if len(time_range) > 1 and not last:
            self.params["filters"]["SnapshotTime"] = st.select_slider(
                "Time range", time_range, value=(time_range[0], time_range[-1])
            )

    def CustomFilter(self):
        """Custom query
        You can also supply a custom query.
        This query should be formatted according to Pandas' `query` functionality."""
        raise NotImplementedError()

    def generatePandasFilters(self):
        pandasquery = []
        if "filters" not in self.params.keys() or len(self.params["filters"]) == 0:
            return ""
        for name, filter in self.params["filters"].items():
            if name in self.categoricals:
                pandasquery.append(f"{name} in {list(filter)}")
            if name in self.numericals:
                pandasquery.append(f"{filter[0]} <= {name} <= {filter[1]}")
        self.pandasquery = " and ".join(pandasquery)
        return self.pandasquery
