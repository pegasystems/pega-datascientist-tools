import streamlit as st
import json
import polars as pl
from pdstools.utils.streamlit_utils import filter_dataframe, model_and_row_counts
from pdstools.utils.cdh_utils import _apply_query

"""# Add custom filters"""

"""You can easily add custom filters to specify the Health Check to your needs.
In the selectbox below, simply select the columns you wish to filter.
For each column, a new configuration screen will be added in which you can specify
what values you want to keep in the Health Check.
"""

if "dm" in st.session_state:
    expr_list = []
    uploaded_file = st.file_uploader(
        "Upload Filters You Downloaded Earlier", type=["json"]
    )
    if uploaded_file:
        import io

        imported_filters = json.load(uploaded_file)
        for key, val in imported_filters.items():
            # Convert the JSON string to a StringIO object and specify the format as 'json'
            json_str = json.dumps(val)
            str_io = io.StringIO(json_str)
            expr_list.append(pl.Expr.deserialize(str_io, format="json"))

    st.session_state["filters"] = filter_dataframe(
        st.session_state["dm"].model_data, queries=expr_list
    )
    if "unfiltered_counts" not in st.session_state.keys():
        st.session_state["unfiltered_counts"] = model_and_row_counts(
            st.session_state["dm"].model_data
        )
    if st.session_state["filters"] != []:
        filtered_modelid_count, filtered_row_count = model_and_row_counts(
            _apply_query(st.session_state["dm"].model_data, st.session_state["filters"])
        )
        serialized_exprs = {}
        for i, expr in enumerate(st.session_state["filters"]):
            serialized = expr.meta.serialize(format="json")
            serialized_exprs[i] = json.loads(serialized)
        data = json.dumps(serialized_exprs)
        st.download_button(
            label="Download Filters",
            data=data,
            file_name="datamart_filters.json",
        )
        row_count_bar = st.progress(
            (filtered_row_count / st.session_state["unfiltered_counts"][1]),
            text=f"{filtered_row_count} rows are left out of {st.session_state['unfiltered_counts'][1]}",
        )
        model_count_bar = st.progress(
            (filtered_modelid_count / st.session_state["unfiltered_counts"][0]),
            text=f"{filtered_modelid_count} models are left out of {st.session_state['unfiltered_counts'][0]}",
        )

else:
    st.warning("Please configure your files in the `data import` tab.")
