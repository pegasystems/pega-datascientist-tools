# python/pdstools/app/decision_analyzer/Home.py
import polars as pl
import streamlit as st

from pdstools.app.decision_analyzer.da_streamlit_utils import (
    handle_file_path,
    handle_file_upload,
    handle_sample_data,
    is_managed_deployment,
    load_decision_analyzer,
)
from pdstools.decision_analyzer.DecisionAnalyzer import DEFAULT_SAMPLE_SIZE
from pdstools.utils.streamlit_utils import (
    show_sidebar_branding,
    show_version_header,
    standard_page_config,
)

standard_page_config(page_title="Decision Analysis")
show_sidebar_branding("Decision Analysis")
pl.enable_string_cache()

show_version_header()

"""
# Decision Analysis

Visualize and diagnose your NBA decision pipeline. Two data formats are supported — the
format is auto-detected on upload:

| | **Explainability Extract (v1)** | **Action Analysis / EEV2 (v2)** |
|---|---|---|
| Scope | Arbitration stage only | Full decision pipeline |
| Stages | Synthetic (Arbitration → Output) | Real stages from strategy framework |
| Filter components | Not available | Available (which rule filtered what) |
| Use case | Quick arbitration analysis | Deep pipeline diagnostics |

All charts are interactive ([Plotly](https://plotly.com/graphing-libraries/)) — pan,
zoom, and hover for details.

### Data import
"""

if "decision_data" in st.session_state:
    del st.session_state["decision_data"]
st.session_state["filters"] = []

# TODO: Allow user to select the level once "Stage" level is implemented
level = "Stage Group"

sample_size = st.number_input(
    "Sample Size",
    min_value=1000,
    value=DEFAULT_SAMPLE_SIZE,
    step=1000,
    help=(
        "Number of interactions used in resource-intensive analysis. "
        "Reduce the sample size if the app runs too slowly. "
        "Note that statistical significance decreases as sample size decreases."
    ),
)

# File upload — always visible. Drag-and-drop or use the Browse button.
raw_data = handle_file_upload()

# For managed deployments, also show a server-side file path input
if is_managed_deployment():
    if raw_data is None:
        raw_data = handle_file_path()

# If no file uploaded, load sample data automatically
if raw_data is None:
    with st.spinner("Loading sample data"):
        raw_data = handle_sample_data()
    st.info(
        "No file uploaded — using built-in sample data. "
        "Upload your own data above to analyze it."
    )

if raw_data is not None:
    with st.spinner("Reading Data"):
        data_id = str(hash(raw_data.explain(optimized=False)))
        da = load_decision_analyzer(
            raw_data, level=level, sample_size=sample_size, data_fingerprint=data_id
        )
        st.session_state.decision_data = da
        del raw_data

        if da.validation_error:
            st.warning(da.validation_error)
        extract_type = da.extract_type
        format_label = (
            "**Explainability Extract (v1)** — arbitration stage only"
            if extract_type == "explainability_extract"
            else "**Action Analysis / EEV2 (v2)** — full pipeline"
        )

        overview = da.get_overview_stats
        rows = da.decision_data.select(pl.len()).collect().item()
        summary = (
            f"Data loaded successfully. Detected format: {format_label}\n\n"
            f"**{rows:,}** rows · "
            f"**{overview['Decisions']:,}** decisions · "
            f"**{overview['Customers']:,}** customers · "
            f"**{overview['Actions']}** actions · "
            f"**{overview['Channels']}** channels · "
            f"**{overview['Duration'].days}** days "
            f"(from {overview['StartDate']})"
        )
        st.success(summary)
