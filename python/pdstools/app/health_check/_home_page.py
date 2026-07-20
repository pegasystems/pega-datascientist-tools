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

    from pdstools.app.health_check.hc_streamlit_utils import ensure_dm_loaded, render_import_ui
    from pdstools.utils.cdh_utils import setup_logger
    from pdstools.utils.streamlit_utils import (
        get_data_path,
        set_active_app as _set_active_app,
        show_sidebar_branding,
        show_version_header,
        standard_page_config,
    )

    standard_page_config(page_title="Adaptive Model Health Check")
    show_sidebar_branding("ADM Health Check")
    _set_active_app("hc")

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
    # matches DA / IA first-run UX. Delegates the priority chain
    # (``--data-path`` → bundled sample) to the shared helper so
    # data-required sub-pages can do the same on deep-link.
    # ``st.rerun()`` rebuilds the script with ``dm`` set so the rest of
    # the home page (data summary, etc.) renders in the same user action.
    #
    # Gate on ``data_source`` being absent — i.e. *truly* first visit.
    # Once the user has interacted with the data-source dropdown (which
    # also deletes ``dm`` so the chosen branch can repopulate it), we must
    # NOT silently reload the sample, because that would leave the
    # uploader branch with ``dm`` set and the "Import Successful!" banner
    # stuck on screen, while the file_uploader widgets are skipped.
    if "dm" not in st.session_state and "data_source" not in st.session_state:
        configured_path = get_data_path()
        if ensure_dm_loaded():
            if configured_path and st.session_state.get("data_source") == "Direct file path":
                st.toast(f"Loaded data from `{configured_path}`.", icon="📂")
            else:
                st.toast("Loaded sample data — upload your own to replace it.", icon="📊")
            st.rerun()

    # --- Data Import (previously a separate page) ---

    st.markdown(
        """
Upload ADM Health Check data below, or expand **File paths** to load files from disk.
Advanced import options are at the bottom of this section.
"""
    )

    # Initialize session state for import options
    if "extract_pyname_keys" not in st.session_state:
        st.session_state["extract_pyname_keys"] = True
    if "infer_schema_length" not in st.session_state:
        st.session_state["infer_schema_length"] = 10000

    render_import_ui()

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
