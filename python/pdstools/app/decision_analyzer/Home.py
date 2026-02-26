# python/pdstools/app/decision_analyzer/Home.py
import polars as pl
import streamlit as st

from pdstools.app.decision_analyzer.da_streamlit_utils import (
    handle_data_path,
    handle_file_path,
    handle_file_upload,
    handle_sample_data,
    is_managed_deployment,
    load_decision_analyzer,
)
from pdstools.decision_analyzer.DecisionAnalyzer import DEFAULT_SAMPLE_SIZE
from pdstools.decision_analyzer.utils import parse_sample_flag, sample_and_save
from pdstools.utils.streamlit_utils import (
    get_data_path,
    get_sample_limit,
    get_temp_dir,
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

level = "Stage Group"

sample_size = st.number_input(
    "Sample Size",
    min_value=1000,
    value=DEFAULT_SAMPLE_SIZE,
    step=1000,
    help=(
        "Maximum number of interactions sampled for resource-intensive pages "
        "(Global Data Filters, Win/Loss Analysis, Optionality Analysis, "
        "Business Lever Analysis). Other pages use the full dataset. "
        "This is *not* the same as the `--sample` CLI flag, which reduces "
        "the data before ingestion. "
        "Reduce this value if those pages run too slowly; increase it for "
        "higher statistical significance."
    ),
)

# File upload — always visible. Drag-and-drop or use the Browse button.
raw_data = handle_file_upload()

# For managed deployments, also show a server-side file path input
if is_managed_deployment():
    if raw_data is None:
        raw_data = handle_file_path()

# If --data-path was provided, load from that path (takes priority over sample data)
configured_path = get_data_path()
if raw_data is None and configured_path:
    with st.spinner(f"Loading data from configured path: {configured_path}"):
        raw_data = handle_data_path()
    if raw_data is not None:
        st.info(f"📂 Loaded data from configured path: `{configured_path}`")

# Fall back to sample data only when nothing else provided
if raw_data is None:
    with st.spinner("Loading sample data"):
        raw_data = handle_sample_data()
    st.info(
        "No file uploaded — using built-in sample data. Upload your own data above to analyze it.",
    )

# Pre-ingestion sampling (--sample CLI flag)
sample_limit_raw = get_sample_limit()
if raw_data is not None and sample_limit_raw:
    try:
        sample_kwargs = parse_sample_flag(sample_limit_raw)
    except ValueError as e:
        st.error(f"Invalid --sample value: {e}")
        st.stop()

    with st.spinner("Sampling interactions…"):
        raw_data = sample_and_save(
            raw_data,
            n=sample_kwargs.get("n"),  # type: ignore[arg-type]
            fraction=sample_kwargs.get("fraction"),  # type: ignore[arg-type]
            output_dir=get_temp_dir(),
        )
    label = sample_limit_raw.strip()
    st.info(f"📉 Pre-ingestion sampling applied: keeping **{label}** interactions.")

if raw_data is not None:
    with st.spinner("Reading Data"):
        data_id = str(hash(raw_data.explain(optimized=False)))
        da = load_decision_analyzer(
            raw_data,
            level=level,
            sample_size=sample_size,
            data_fingerprint=data_id,
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
