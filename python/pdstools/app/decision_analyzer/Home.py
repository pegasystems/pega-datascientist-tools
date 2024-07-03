import os

import polars as pl
import streamlit as st
from st_pages import show_pages

from pdstools.app.decision_analyzer.da_streamlit_utils import (
    get_options,
    get_pages,
    handle_direct_file_path,
    handle_file_upload,
    handle_sample_data,
)
from pdstools.decision_analyzer.decision_data import DecisionData

st.set_page_config(layout="wide")
pl.enable_string_cache()  # Done here, but also put in ensure_data()

app_version, tag_date = "V0.1", "18 Jun 2024"

# TODO: the caching is not optimal yet, sometimes caches too much, sometimes too little - need to clean this up
# TODO: fix up the commentary, labels, annotations everywhere, align with Dennis' designs and NBA designer terminology
# TODO: support multiple physical formats: hives partioned folders, parquet, Pega dataset exports etc. NOTE: we can delete if the read_data func is sufficient.
# TODO: support different content types: not only "Decision Analyzer" but also Interaction History (similar for
# TODO: the most part but lacking filter component names etc, and wont have pxEngagementStage but is implicitly for
# TODO: the last stage), or the Explainability Extract (no filter component names either, and has a Remaining perspective
# TODO: with stages from Arbitration onwards - but no details of what gets filtered where).


f"""
# Decision Analyzer App

Welcome to the early alpha version ({app_version}) of the Decision Analyzer Streamlit app, released on {tag_date}.

This is a supplementary tool to visualize and analyze data from **Decision Analyzer**. Once the
feature is available, you will be able to point this app to the Decision Analyzer data generated
from Pega. We also plan to support the "Explainability Extract" for prior releases: this is a similar
but smaller dataset with just the "Arbitration" stage. For now the only data available is
the demo dataset pre-loaded in the app.

* Decision Analyzer tracks every action, returning all of them for each decision with details where and how that action got filtered out (or not).

* DA data does not contain outcomes, it is about decisions made at runtime (no simulations).

* DA will have propensities only for the arbitration/final offers, not for earlier stages as it just follows the exact runtime of NBAD.

The charts are built with [Plotly](https://plotly.com/graphing-libraries/), an interactive charting library with user controls for panning, zooming etc.
For best results, familiarize yourself with the many ways you can [interact with the charts](
https://plotly.com/chart-studio-help/zoom-pan-hover-controls/).

### Data import
"""

st.session_state.clear()
st.cache_data.clear()
st.session_state["filters"] = []

# st.write(st.session_state.to_dict().keys())  # just for debugging

file_upload_options = get_options()
source = st.selectbox(
    "Select folder with the Decision Analyzer data or select the anonymized sample data",
    options=file_upload_options,
    index=0,
)

if source == "Sample Data":
    raw_data = handle_sample_data(os.getcwd() == "/app")
elif source == "File Upload":
    raw_data = handle_file_upload()
elif source == "Direct File Path":
    raw_data = handle_direct_file_path()

if raw_data is not None:
    with st.spinner("Reading Data"):
        st.session_state.decision_data = DecisionData(raw_data)
        del raw_data

        if "decision_data" in st.session_state:
            extract_type = st.session_state.decision_data.extract_type
            pages = get_pages(extract_type)
            show_pages(pages)
            st.success("Data reading is complete. You can proceed now.")
