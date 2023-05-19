import streamlit as st
from typing import Optional
import polars as pl
from pathlib import Path
from . import cdh_utils
from ..adm.ADMDatamart import ADMDatamart
from ..utils import datasets
import plotly.express as px
from .. import pega_io


@st.cache_resource
def cachedSample():
    return datasets.CDHSample()


@st.cache_resource
def cachedDatamart(*args, **kwargs):
    print("Importing datamart.")
    return ADMDatamart(*args, **kwargs)


def import_datamart(**opts):
    st.session_state["params"] = {}
    st.session_state["modelhc"] = False
    st.write("### Data import")

    source = st.selectbox(
        "Select data source",
        options=[
            "Direct file path",
            "Direct file upload",
            "CDH Sample",
            "Download from S3",
        ],
    )
    if source == "CDH Sample":
        st.session_state["dm"] = cachedSample()
    elif source == "Download from S3":
        raise NotImplementedError("Want to do this soon.")
    elif source == "Direct file upload":
        return fromUploadedFile(**opts)
    elif source == "Direct file path":
        return fromFilePath(**opts)


def fromUploadedFile(**opts):
    model_file = st.file_uploader(
        "Upload Model Snapshot", type=["json", "zip", "parquet", "csv", "arrow"]
    )
    predictor_file = st.file_uploader(
        "Upload Predictor Binning snapshot",
        type=["json", "zip", "parquet", "csv", "arrow"],
    )
    if model_file and predictor_file:
        try:
            st.session_state["dm"] = cachedDatamart(
                model_df=model_file, predictor_df=predictor_file, **opts
            )
        except Exception as e:
            st.write("Oh oh.", e)
    elif model_file is not None and predictor_file is None:
        st.warning("""Please also upload the Predictor Binning file. 
                If you don't have access to a predictor binning file
                and want to run the Health Check only on the model snapshot, check the
                checkbox below.
                """)
        model_analysis = st.checkbox("Only run model-based Health Check")
        if model_analysis:
            try:
                st.session_state["dm"] = cachedDatamart(
                    model_df=model_file, predictor_filename=None, **opts
                )
                st.session_state["modelhc"] = True
            except Exception as e:
                st.write("Oh oh.", e)


def fromFilePath(**opts):
    st.write(
        """If you've followed the instructions on how to get the ADMDatamart data,
    you can import the data simply by pointing the app to the directory
    where the original files are located, and we can find it automatically."""
    )
    dir = st.text_input(
        "The folder of the Model Snapshot and Predictor Binning files:",
        placeholder="/Users/Downloads",
    )
    import_strategy = "eager" if opts["extract_keys"] else "lazy"
    if dir != "":
        try:
            model_matches = pega_io.get_latest_file(dir, target="modelData")
        except FileNotFoundError:
            st.error(f"**Directory not found:** {dir}")
            st.stop()
        except NotADirectoryError:
            st.error(
                f"""**Not a directory**:  
            It looks like {dir} is a file.  
            Please supply the path to the **folder** the files are in."""
            )
            st.stop()

        box, data = st.columns([1, 15])
        if model_matches is not None:
            box.write("## √")
            data.write(f"Model snapshot found: {model_matches}")
        else:
            box.write("## X")
            data.write("Could not find a model snapshot in the given folder.   ")

        predictor_matches = pega_io.get_latest_file(dir, target="predictorData")
        box, data = st.columns([1, 15])
        if predictor_matches is not None:
            box.write("## √")
            data.write(f"Predictor binning found: {predictor_matches}")
        else:
            box.write("## X")
            data.write(
                "Could not find the predicting binning snapshot in the given folder."
            )

        if model_matches is None:
            st.write(
                """If you can't get the files to automatically be detected, 
    try uploading the files manually using a different data source."""
            )

        elif predictor_matches is None:
            st.warning(
                """No predictor binning file found, please also upload the Predictor
                Binning file. If you have a predictor binning snapshot but we can't
                detect it, use the **Direct file upload** option in the dropdown above.
                If you don't have access to a predictor binning file
                and want to run the Health Check only on the model snapshot, check the
                checkbox below.
                """
            )
            model_analysis = st.checkbox("Only run model-based Health Check")
            if model_analysis:
                st.session_state["dm"] = cachedDatamart(
                    path=dir,
                    model_filename=Path(model_matches).name,
                    predictor_filename=None,
                    import_strategy=import_strategy,
                    **opts,
                )
                st.session_state["modelhc"] = True

        else:
            st.session_state["dm"] = cachedDatamart(
                path=dir,
                model_filename=Path(model_matches).name,
                predictor_filename=Path(predictor_matches).name,
                import_strategy=import_strategy,
                **opts,
            )


def filter_dataframe(df: pl.LazyFrame, schema: Optional[dict] = None) -> pl.LazyFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Parameters
    ----------
    df : pl.DataFrame
        Original dataframe

    Returns
    -------
    pl.LazyFrame
        The filtered LazyFrame

    """

    queries = []

    to_filter_columns = st.multiselect(
        "Filter dataframe on", df.columns, key="multiselect"
    )
    if "uniques" not in st.session_state:
        st.session_state["uniques"] = {}
        st.session_state["selected"] = {}
    for column in to_filter_columns:
        left, right = st.columns((1, 20))
        left.write("## ↳")

        # Treat columns with < 20 unique values as categorical
        if (df.schema[column] == pl.Categorical) or (df.schema[column] == pl.Utf8):
            if column not in st.session_state.uniques.keys():
                st.session_state.uniques[column] = (
                    df.select(pl.col(column).unique()).collect().to_series().to_list()
                )
            if column not in st.session_state.selected.keys():
                st.session_state.selected[column] = st.session_state.uniques[column]
            if len(st.session_state.uniques[column]) < 20:
                options = st.session_state.uniques[column]
                previously_selected = st.session_state["selected"][column]
                selected = right.multiselect(
                    f"Values for {column}",
                    options,
                    default=previously_selected,
                    key=f"{column}",
                )
                st.session_state["selected"][column] = selected
                if (
                    st.session_state["selected"][column]
                    != st.session_state.uniques[column]
                ):
                    queries.append(
                        pl.col(column)
                        .cast(pl.Utf8)
                        .is_in(st.session_state["selected"][column])
                    )
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    queries.append(pl.col(column).str.contains(user_text_input))

        elif df.schema[column] in pl.NUMERIC_DTYPES:
            _min = float(df.select(pl.min(column)).collect().item())
            _max = float(df.select(pl.max(column)).collect().item())
            step = (_max - _min) / 100
            user_num_input = right.slider(
                f"Values for {column}",
                min_value=_min,
                max_value=_max,
                value=(_min, _max),
                step=step,
            )
            if user_num_input[0] != _min or user_num_input[1] != _max:
                queries.append(pl.col(column).is_between(*user_num_input))
        elif df.schema[column] in pl.TEMPORAL_DTYPES:
            user_date_input = right.date_input(
                f"Values for {column}",
                value=(
                    df.select(pl.min(column)).collect().item(),
                    df.select(pl.max(column)).collect().item(),
                ),
            )
            if len(user_date_input) == 2:
                queries.append(pl.col(column).is_between(*user_date_input))

    return queries


def configure_predictor_categorization():
    df = st.session_state["dm"].combinedData
    if len(st.session_state["filters"]) > 0:
        for filter in st.session_state["filters"]:
            df = df.filter(filter)
    df = (
        df.filter(pl.col("PredictorName") != "Classifier")
        .with_columns((pl.col("PerformanceBin") - 0.5) * 2)
        .groupby("PredictorCategory")
        .agg(
            Performance=cdh_utils.weighed_average_polars(
                "PerformanceBin", "BinResponseCount"
            )
        )
        .with_columns(
            Contribution=((pl.col("Performance") / (pl.sum("Performance"))) * 100)
        )
        .collect()
    )
    color = "PredictorCategory"
    fig = px.bar(
        df.sort(color).to_pandas(),
        x="Contribution",
        color=color,
        orientation="h",
        template="pega",
        title="Contribution of different sources",
    )
    st.plotly_chart(fig)


# def newPredictorCategorizationFunc():

#     def conditions():
#         from polars.internals.expr.string import ExprStringNameSpace
#         funcs = {
#             "Starts with": ExprStringNameSpace.starts_with,
#             "Ends with": ExprStringNameSpace.ends_with,
#             "Contains": ExprStringNameSpace.contains,
#             "Regex": ExprStringNameSpace.extract,
#         }
#         x = pl.col("PredictorName")
#         func = st.selectbox('When', self.funcs.keys())
#         # func = funcs['Starts with']
#         st.write(func(self.x, "test"))

#     class Condition:
#         def __new__(self):


#     conditions()


#     st.write("COMON")

#     """
#     [{'func':'starts_with', 'value':'IH', 'then_func':'lit', 'then_value':'IHPredictor'},
#     {'func':'contains','value':'customer', 'then_func', 'head', 'then_value':[2]},
#     {'func':'regex', 'value':'IH^', 'then_func':'regex', 'then_value':'IH^.^'}]
#     """
