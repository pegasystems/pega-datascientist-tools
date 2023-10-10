import streamlit as st
import json
import polars as pl
from pdstools.utils import streamlit_utils

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
        imported_filters = json.load(uploaded_file)
        for key, val in imported_filters.items():
            expr_list.append(pl.Expr.from_json(json.dumps(val)))

    st.session_state["filters"] = streamlit_utils.filter_dataframe(
        st.session_state["dm"].modelData, queries=expr_list
    )

    if st.session_state["filters"] != []:
        deserialize_exprs = {}
        for i, expr in enumerate(st.session_state["filters"]):
            deserialize_exprs[i] = json.loads(expr.meta.write_json())
        data = json.dumps(deserialize_exprs)
        st.download_button(
            label="Download Filters",
            data=data,
            file_name="datamart_filters.json",
        )

else:
    st.warning("Please configure your files in the `data import` tab.")
