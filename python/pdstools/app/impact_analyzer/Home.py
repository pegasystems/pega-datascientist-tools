# python/pdstools/app/impact_analyzer/Home.py
from pathlib import Path

import polars as pl
import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import (
    handle_data_path_ia,
    load_pdc_from_uploads,
    load_sample_pdc,
    load_vbd_from_upload,
    prepare_and_save_random,
)
from pdstools.utils.streamlit_utils import (
    get_data_path,
    get_sample_limit,
    get_temp_dir,
    parse_sample_spec,
    show_sidebar_branding,
    show_version_header,
    standard_page_config,
)

standard_page_config(page_title="Impact Analyzer")
show_sidebar_branding("Impact Analyzer")

show_version_header()

"""
# Impact Analyzer

Analyze A/B test experiments from PDC exports or VBD Scenario Planner Actuals.
Two data formats are supported — format is auto-detected on upload:

| | **PDC Export** | **VBD Scenario Planner** |
|---|---|---|
| Format | JSON/NDJSON | ZIP archive |
| Source | Pega Decisioning Center | Value-Based Design |
| Metrics | CTR lift, value lift | Scenario comparison |

All charts are interactive ([Plotly](https://plotly.com/graphing-libraries/)) — pan,
zoom, and hover for details.

### Data import
"""

VBD_REQUIRED_COLS = {
    "OutcomeTime",
    "Channel",
    "Direction",
    "AggregateCount",
    "Outcome",
    "Value",
    "MktValue",
}


def _load_with_warning(loader, label: str, *, expected_input_cols=None):
    required_cols = {
        "SnapshotTime",
        "ControlGroup",
        "Impressions",
        "Accepts",
        "ValuePerImpression",
        "Channel",
    }
    try:
        ia = loader()
        if ia is None:
            return None
        try:
            schema_names = set(ia.ia_data.collect_schema().names())
        except Exception as exc:
            st.warning(f"Unable to validate {label} data schema. {exc}")
            return ia
        missing = sorted(required_cols.difference(schema_names))
        if missing:
            st.warning(
                f"{label} data is missing required columns: {', '.join(missing)}.",
            )
        return ia
    except Exception as exc:
        message = str(exc)
        if "Missing required columns" in message:
            st.warning(
                f"{label} data is missing required columns: {', '.join(sorted(required_cols))}. {message}",
            )
        elif "unable to find column" in message and expected_input_cols:
            st.warning(
                f"{label} data is missing expected input columns: {', '.join(sorted(expected_input_cols))}. {message}",
            )
        else:
            st.warning(f"Unable to load {label} data. {message}")
        return None


def _show_data_summary(ia):
    """Display a summary banner for the loaded ImpactAnalyzer."""
    try:
        schema_names = set(ia.ia_data.collect_schema().names())
        if "MktValue" in schema_names and "OutcomeTime" in schema_names:
            format_label = "**VBD Scenario Planner**"
        else:
            format_label = "**PDC Export**"
    except Exception:
        format_label = "**Unknown format**"

    try:
        rows = ia.ia_data.select(pl.len()).collect().item()
        channels = (
            ia.ia_data.select(pl.col("Channel").n_unique()).collect().item() if "Channel" in schema_names else "N/A"
        )
        st.success(
            f"Data loaded successfully. Detected format: {format_label}\n\n**{rows:,}** rows · **{channels}** channels"
        )
    except Exception:
        st.success(f"Data loaded successfully. Detected format: {format_label}")


# --- Data loading priority chain: upload > CLI path > sample ---

# File upload — always visible
uploaded_files = st.file_uploader(
    "Upload Impact Analyzer data",
    type=["json", "ndjson", "zip"],
    accept_multiple_files=True,
)

impact_analyzer = None
data_source_path = None

# 1. Handle file upload
if uploaded_files:
    suffixes = {Path(f.name).suffix.lower() for f in uploaded_files}
    if suffixes.issubset({".json", ".ndjson"}):
        with st.spinner("Loading PDC data"):
            impact_analyzer = _load_with_warning(
                lambda: load_pdc_from_uploads(uploaded_files),
                "PDC",
            )
    elif len(uploaded_files) == 1 and ".zip" in suffixes:
        with st.spinner("Loading VBD data"):
            impact_analyzer = _load_with_warning(
                lambda: load_vbd_from_upload(uploaded_files[0]),
                "VBD",
                expected_input_cols=VBD_REQUIRED_COLS,
            )
    else:
        st.error("Upload JSON/NDJSON (PDC) or a single ZIP (VBD).")

# 2. Handle --data-path CLI flag
has_existing_data = "impact_analyzer" in st.session_state
configured_path = get_data_path()

if impact_analyzer is None and configured_path and not has_existing_data:
    with st.spinner(f"Loading data from configured path: {configured_path}"):
        impact_analyzer = _load_with_warning(
            lambda: handle_data_path_ia(),
            "CLI",
        )
        data_source_path = configured_path
    if impact_analyzer is not None:
        st.info(f"Loaded data from configured path: `{configured_path}`")

# 3. Fall back to sample data
if impact_analyzer is None and not has_existing_data and not uploaded_files:
    with st.spinner("Loading sample PDC data"):
        impact_analyzer = _load_with_warning(load_sample_pdc, "Sample")
    if impact_analyzer is not None:
        st.info(
            "No file uploaded — using built-in sample data. Upload your own data above to analyze it.",
        )

# Pre-ingestion sampling (only for CLI paths)
sample_limit_raw = get_sample_limit() if data_source_path else None
if impact_analyzer is not None and sample_limit_raw:
    try:
        sample_kwargs = parse_sample_spec(sample_limit_raw)
    except ValueError as e:
        st.error(f"Invalid --sample value: {e}")
        st.stop()

    if "fraction" in sample_kwargs:
        sampling_msg = f"Sampling {sample_kwargs['fraction'] * 100:.0f}% of the rows..."
    elif "n" in sample_kwargs:
        sampling_msg = f"Sampling {sample_kwargs['n']:,} rows..."
    else:
        sampling_msg = "Sampling rows..."

    with st.spinner(sampling_msg):
        sampled_data, sample_path = prepare_and_save_random(
            impact_analyzer.ia_data,
            n=sample_kwargs.get("n"),
            fraction=sample_kwargs.get("fraction"),
            output_dir=get_temp_dir() or ".",
            source_path=data_source_path,
        )

        if sample_path is not None:
            impact_analyzer.ia_data = sampled_data

    label = sample_limit_raw.strip()
    if sample_path is not None:
        st.info(f"Pre-ingestion sampling applied: keeping **{label}** rows. Sampled data saved to `{sample_path}`.")
    else:
        st.info(f"Sampling requested (**{label}**) but data already within limit — using full dataset.")

# Store data and show summary
if impact_analyzer is not None:
    st.session_state["impact_analyzer"] = impact_analyzer
    _show_data_summary(impact_analyzer)
elif has_existing_data:
    _show_data_summary(st.session_state["impact_analyzer"])
