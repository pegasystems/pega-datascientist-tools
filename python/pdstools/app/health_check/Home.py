# python/pdstools/app/health_check/Home.py
import shutil

import streamlit as st

from pdstools.utils import streamlit_utils
from pdstools.utils.cdh_utils import setup_logger
from pdstools.utils.streamlit_utils import (
    show_sidebar_branding,
    show_version_header,
    standard_page_config,
)

standard_page_config(page_title="Adaptive Model Health Check")
show_sidebar_branding("ADM Health Check")

if "log_buffer" not in st.session_state:
    logger, st.session_state.log_buffer = setup_logger()

show_version_header()

"""
# Adaptive Model Health Check

Generate a comprehensive Health Check report from your ADM Datamart data, complete with
interactive visuals and an Excel export for further exploration.
"""

pandoc_available = shutil.which("pandoc") is not None
quarto_available = shutil.which("quarto") is not None

if not (pandoc_available and quarto_available):
    missing_deps = []
    if not pandoc_available:
        missing_deps.append("Pandoc (https://pandoc.org/installing.html)")
    if not quarto_available:
        missing_deps.append("Quarto (https://quarto.org/docs/get-started/)")

    missing_deps_list = "\n    - ".join(missing_deps)
    st.error(
        f"⚠️ The following required tools are not available on your system:\n"
        f"    - {missing_deps_list}\n"
        "The app will not function without these tools. Please install them before proceeding."
    )

# --- Data Import (previously a separate page) ---

"""
Select a data source below to get started. Advanced import options are at the bottom
of this page.
"""

# Initialize session state for import options
if "extract_pyname_keys" not in st.session_state:
    st.session_state["extract_pyname_keys"] = True
if "infer_schema_length" not in st.session_state:
    st.session_state["infer_schema_length"] = 10000

streamlit_utils.import_datamart(
    extract_pyname_keys=st.session_state["extract_pyname_keys"],
    infer_schema_length=st.session_state["infer_schema_length"],
)

# Advanced options
with st.expander("Advanced import settings"):
    st.session_state["extract_pyname_keys"] = st.checkbox(
        "Extract Treatments",
        value=st.session_state["extract_pyname_keys"],
        help=(
            'By default, ADM uses a few "Context Keys" (Issue, Group, Channel, Name) '
            "to distinguish between models. If you have custom context keys, they are "
            "embedded in the pyName column. Enable this to extract them into separate columns."
        ),
    )

    st.session_state["infer_schema_length"] = st.number_input(
        "Schema inference length",
        min_value=1000,
        value=st.session_state["infer_schema_length"],
        step=10000,
        help=(
            "Number of rows to scan when inferring the schema for CSV/JSON files. "
            "Increase for large datasets (e.g. 200 000) if columns are not detected correctly."
        ),
    )

    st.info("💡 Changes take effect when you re-import the data.")

# Clean up transient session state
essential_keys = [
    "dm",
    "params",
    "logger",
    "log_buffer",
    "data_source",
    "prediction",
    "extract_pyname_keys",
    "infer_schema_length",
]
for key in list(st.session_state.keys()):
    if key not in essential_keys and not key.startswith("_"):
        del st.session_state[key]
