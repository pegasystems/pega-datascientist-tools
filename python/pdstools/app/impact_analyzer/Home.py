# python/pdstools/app/impact_analyzer/Home.py
from pathlib import Path

import streamlit as st

from pdstools.app.impact_analyzer.ia_streamlit_utils import (
    load_pdc_from_paths,
    load_pdc_from_uploads,
    load_sample_pdc,
    load_vbd_from_path,
    load_vbd_from_upload,
)
from pdstools.utils.streamlit_utils import (
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
Upload your data below, then explore lift metrics and trends in the sidebar pages.
"""

if "impact_analyzer" in st.session_state:
    del st.session_state["impact_analyzer"]
if "impact_analyzer_source" in st.session_state:
    del st.session_state["impact_analyzer_source"]

data_sources = [
    "Sample data",
    "File upload",
    "File path",
]

source = st.selectbox("Select data source", options=data_sources, index=0)

impact_analyzer = None

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


if source == "Sample data":
    with st.spinner("Loading sample PDC data"):
        impact_analyzer = _load_with_warning(load_sample_pdc, "Sample")
elif source == "File upload":
    uploaded_files = st.file_uploader(
        "Upload Impact Analyzer data",
        type=["json", "ndjson", "zip", "parquet", "csv"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        suffixes = {Path(uploaded_file.name).suffix.lower() for uploaded_file in uploaded_files}
        if suffixes.issubset({".json", ".ndjson"}):
            with st.spinner("Loading PDC data"):
                impact_analyzer = _load_with_warning(
                    lambda: load_pdc_from_uploads(uploaded_files),
                    "PDC",
                )
        elif suffixes.intersection({".parquet", ".csv"}):
            st.info("Parquet/CSV uploads are not supported yet.")
        elif len(uploaded_files) == 1 and ".zip" in suffixes:
            with st.spinner("Loading VBD data"):
                impact_analyzer = _load_with_warning(
                    lambda: load_vbd_from_upload(uploaded_files[0]),
                    "VBD",
                    expected_input_cols=VBD_REQUIRED_COLS,
                )
        else:
            st.error("Upload JSON (PDC) or a single ZIP (VBD).")
elif source == "File path":
    file_path = st.text_input("Data file path")
    if file_path:
        path = Path(file_path)
        if path.suffix.lower() in {".json", ".ndjson"}:
            with st.spinner("Loading PDC data"):
                impact_analyzer = _load_with_warning(
                    lambda: load_pdc_from_paths((str(path),)),
                    "PDC",
                )
        elif path.suffix.lower() == ".zip":
            with st.spinner("Loading VBD data"):
                impact_analyzer = _load_with_warning(
                    lambda: load_vbd_from_path(str(path)),
                    "VBD",
                    expected_input_cols=VBD_REQUIRED_COLS,
                )
        elif path.suffix.lower() in {".parquet", ".csv"}:
            st.info("Parquet/CSV file paths are not supported yet.")
        else:
            st.error("File type not recognized. Use JSON/NDJSON (PDC) or ZIP (VBD).")

if impact_analyzer is not None:
    st.session_state["impact_analyzer"] = impact_analyzer
    st.session_state["impact_analyzer_source"] = source
    st.success("Data loaded successfully. Proceed to Overall Summary.")
elif source in {"File upload", "File path"}:
    st.info("Provide a data source to continue.")
