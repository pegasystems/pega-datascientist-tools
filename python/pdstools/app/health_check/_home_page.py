# python/pdstools/app/health_check/_home_page.py
"""Home page for the ADM Health Check app.

Exposes a single ``home_page()`` callable that
``_navigation.build_navigation()`` registers as the default page in
``st.navigation()``. Kept as a function (not a script) so navigation
routes to it without re-executing top-level side-effects.
"""

from __future__ import annotations

import streamlit as st


def home_page() -> None:
    """Render the ADM Health Check home page."""
    import shutil

    from pdstools.app.health_check.hc_streamlit_utils import handle_data_path_hc
    from pdstools.utils import streamlit_utils
    from pdstools.utils.cdh_utils import setup_logger
    from pdstools.utils.streamlit_utils import (
        cached_sample,
        cached_sample_prediction,
        get_data_path,
        show_sidebar_branding,
        show_version_header,
        standard_page_config,
    )

    standard_page_config(page_title="Adaptive Model Health Check")
    show_sidebar_branding("ADM Health Check")

    if "log_buffer" not in st.session_state:
        _, st.session_state.log_buffer = setup_logger()

    show_version_header()

    st.markdown(
        """
# Adaptive Model Health Check

Generate a comprehensive Health Check report from your ADM Datamart data, complete with
interactive visuals and an Excel export for further exploration.
"""
    )

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
            "The app will not function without these tools. Please install them before proceeding.",
        )

    # Auto-load on first visit so the app is immediately usable —
    # matches DA / IA first-run UX. Order:
    #   1. Honour ``--data-path`` if set (HC-specific helper handles dir
    #      or zip; returns None on miss/unsupported so we fall through).
    #   2. Fall back to the bundled CDH sample.
    # The toast reflects whichever path actually loaded so users know
    # what they're looking at. ``st.rerun()`` rebuilds the script with
    # ``dm`` set so downstream pages see populated session state in the
    # same user action.
    if "dm" not in st.session_state:
        configured_path = get_data_path()
        loaded_from_path = False
        if configured_path:
            with st.spinner(f"Loading data from `{configured_path}`…"):
                dm = handle_data_path_hc()
            if dm is not None:
                loaded_from_path = True
                st.toast(f"Loaded data from `{configured_path}`.", icon="📂")
                st.rerun()

        if not loaded_from_path:
            with st.spinner("Loading sample data…"):
                st.session_state["dm"] = cached_sample()
                st.session_state["prediction"] = cached_sample_prediction()
                st.session_state["data_source"] = "CDH Sample"
            st.toast("Loaded sample data — upload your own to replace it.", icon="📊")
            st.rerun()

    # --- Data Import (previously a separate page) ---

    st.markdown(
        """
Select a data source below to get started. Advanced import options are at the bottom
of this page.
"""
    )

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
        if key not in essential_keys and not str(key).startswith("_"):
            del st.session_state[key]
