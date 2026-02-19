import os

import polars as pl
import streamlit as st

from pdstools import __version__ as pdstools_version
from pdstools.app.decision_analyzer.da_streamlit_utils import (
    get_options,
    handle_direct_file_path,
    handle_file_upload,
    handle_sample_data,
    load_decision_analyzer,
)

st.set_page_config(layout="wide", page_title="Decision Analysis")
pl.enable_string_cache()

st.caption(f"pdstools {pdstools_version}")

"""
# Decision Analysis

Visualize and analyze NBA decision data. This app supports **two data formats**:

| | **Explainability Extract (v1)** | **Decision Analyzer / EEV2 (v2)** |
|---|---|---|
| Scope | Arbitration stage only | Full decision pipeline |
| Stages | Synthetic (Arbitration → Output) | Real stages from strategy framework |
| Filter components | Not available | Available (which rule filtered what) |
| Use case | Quick arbitration analysis | Deep pipeline diagnostics |

Upload your data below. The format is auto-detected.

The charts use [Plotly](https://plotly.com/graphing-libraries/) — you can pan, zoom, and hover for details.

### Data import
"""

if "decision_data" in st.session_state:
    del st.session_state["decision_data"]
st.session_state["filters"] = []

# level = st.selectbox(
#     "Select Stage Granularity",
#     ("StageGroup", "Stage"),
# )
level = "StageGroup"  # TODO: Allow user to select the level once "Stage" level is implemented

sample_size = st.number_input(
    "Sample Size",
    min_value=1000,
    value=50000,
    step=1000,
    help="Number of interactions used in resource-intensive analysis. Reduce the sample size if the app runs too slowly. Note that statistical significance decreases as sample size decreases.",
)

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
        da = load_decision_analyzer(raw_data, level=level, sample_size=sample_size)
        st.session_state.decision_data = da
        del raw_data

        if da.validation_error:
            st.warning(da.validation_error)
        extract_type = da.extract_type
        format_label = (
            "**Explainability Extract (v1)** — arbitration stage only"
            if extract_type == "explainability_extract"
            else "**Decision Analyzer / EEV2 (v2)** — full pipeline"
        )
        st.success(f"Data loaded successfully. Detected format: {format_label}")
