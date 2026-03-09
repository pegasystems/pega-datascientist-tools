# python/pdstools/app/impact_analyzer/Home.py
from pathlib import Path

import polars as pl
import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import (
    handle_data_path_ia,
    load_from_upload_auto,
    load_pdc_from_uploads,
    load_sample_pdc,
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

| | **PDC Export** | **VBD Scenario Planner** | **Interaction History** |
|---|---|---|---|
| Format | JSON/NDJSON | ZIP archive | TBD - future |
| Source | Pega Diagnostic Cloud | Pega Infinity | Pega Infinity |

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
        elif "unable to find column" in message:
            # Extract the missing column name from the error message
            import re

            match = re.search(r'unable to find column "([^"]+)"', message)
            missing_col = match.group(1) if match else "unknown"

            # Use expected_input_cols if provided, otherwise default to VBD required columns
            required_cols_list = expected_input_cols or VBD_REQUIRED_COLS
            required_cols_str = ", ".join(f"`{col}`" for col in sorted(required_cols_list))

            st.error(
                f"**Cannot load this data** — missing required column: `{missing_col}`\n\n"
                "Impact Analyzer requires VBD data from **Scenario Planner Actuals** exports, "
                "which include control group information (`MktValue`). Standard VBD Actuals exports "
                "don't contain this data.\n\n"
                f"**Required columns:** {required_cols_str}"
            )
        else:
            st.warning(f"Unable to load {label} data. {message}")
        return None


def _show_data_summary(ia, is_sample_data: bool = False):
    """Display a summary banner for the loaded ImpactAnalyzer."""
    try:
        schema_names = set(ia.ia_data.collect_schema().names())
        # Check for VBD-specific columns after from_vbd() processing
        # (MktValue becomes ControlGroup, OutcomeTime becomes SnapshotTime,
        # and VBD data has Application, Value, Outcome columns)
        has_vbd_markers = {"Application", "Value", "Outcome"}.issubset(schema_names)

        if has_vbd_markers:
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

        if is_sample_data:
            prefix = "Sample data loaded"
        else:
            prefix = "Data loaded successfully"

        st.success(f"{prefix}. Detected format: {format_label}\n\n**{rows:,}** rows · **{channels}** channels")
    except Exception:
        prefix = "Sample data loaded" if is_sample_data else "Data loaded successfully"
        st.success(f"{prefix}. Detected format: {format_label}")


# --- Data loading priority chain: upload > CLI path > sample ---

# File upload — always visible
uploaded_files = st.file_uploader(
    "Upload Impact Analyzer data",
    accept_multiple_files=True,
)

impact_analyzer = None
data_source_path = None
is_sample_data = False

# 1. Handle file upload
if uploaded_files:
    # Clear any old data when new upload is attempted
    if "impact_analyzer" in st.session_state:
        del st.session_state["impact_analyzer"]
    if "ia_is_sample_data" in st.session_state:
        del st.session_state["ia_is_sample_data"]

    # Filter out unwanted files (e.g., MANIFEST.mf from unzipped Pega exports)
    valid_suffixes = {".json", ".ndjson", ".zip"}
    filtered_files = [
        f for f in uploaded_files if Path(f.name).suffix.lower() in valid_suffixes and "META-INF" not in f.name
    ]

    if not filtered_files:
        st.error(
            "No valid data files found. Upload JSON/NDJSON (PDC) or ZIP (VBD) files. "
            "If you dragged a folder, try uploading just the `data.json` file instead."
        )
    else:
        suffixes = {Path(f.name).suffix.lower() for f in filtered_files}

        # Single file: use auto-detection
        if len(filtered_files) == 1:
            with st.spinner("Loading data (auto-detecting format)..."):
                impact_analyzer = _load_with_warning(
                    lambda: load_from_upload_auto(filtered_files[0]),
                    "uploaded",
                    expected_input_cols=VBD_REQUIRED_COLS if ".zip" in suffixes else None,
                )
        # Multiple JSON files: treat as PDC
        elif suffixes.issubset({".json", ".ndjson"}):
            with st.spinner("Loading PDC data"):
                impact_analyzer = _load_with_warning(
                    lambda: load_pdc_from_uploads(filtered_files),
                    "PDC",
                )
        else:
            st.error("Upload a single file (JSON/NDJSON/ZIP) or multiple JSON/NDJSON files (PDC).")

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
        is_sample_data = True
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
    st.session_state["ia_is_sample_data"] = is_sample_data
    _show_data_summary(impact_analyzer, is_sample_data=is_sample_data)
elif "impact_analyzer" in st.session_state:
    # Show summary for previously loaded data (only if no upload was attempted)
    was_sample = st.session_state.get("ia_is_sample_data", False)
    _show_data_summary(st.session_state["impact_analyzer"], is_sample_data=was_sample)
