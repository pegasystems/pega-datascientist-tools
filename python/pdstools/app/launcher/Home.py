# python/pdstools/app/launcher/Home.py
"""Entry script for the cross-app pdstools launcher.

Hosts Health Check, Decision Analyzer, and Impact Analyzer side by
side in a single Streamlit process using ``st.navigation``'s
sectioned dict API. Standalone tool entry points
(``app/<tool>/Home.py``) are unaffected.

URL paths are flat-namespaced (``/hc_*``, ``/da_*``, ``/ia_*``) to
avoid collisions between tools that share page names — Streamlit's
``st.Page`` rejects nested ``/`` in ``url_path`` so the launcher uses
an underscore separator instead.
"""

from pathlib import Path

import streamlit as st

from pdstools.utils.streamlit_utils import (
    show_sidebar_branding,
    standard_page_config,
)

_ABOUT_SCRIPT = Path(__file__).parent / "_about.py"


# Each tool's _navigation module is imported on first call only — the
# transitive imports (plotly, papermill, fastexcel, polars-heavy
# helpers via _home_page.py) would otherwise fire on every launcher
# rerun even when the user only ever clicks into one tool. After the
# first call sys.modules caches the module so subsequent reruns are
# free.
#
# include_about=False on every tool: the launcher exposes one shared
# top-level About entry instead of three near-identical per-tool ones.
def _hc_section() -> list[st.Page]:
    from pdstools.app.health_check._navigation import pages

    return pages(url_prefix="hc", default=True, include_about=False)


def _da_section() -> list[st.Page]:
    from pdstools.app.decision_analyzer._navigation import pages

    return pages(url_prefix="da", default=False, include_about=False)


def _ia_section() -> list[st.Page]:
    from pdstools.app.impact_analyzer._navigation import pages

    return pages(url_prefix="ia", default=False, include_about=False)


standard_page_config(page_title="pdstools")
show_sidebar_branding("pdstools")

# Health Check is the first section so its Home is the default page —
# only one page may carry default=True across the whole navigation.
# The shared About page lives in its own section so it reads as a
# top-level entry, not buried under any one tool.
sections: dict[str, list[st.Page]] = {
    "Health Check": _hc_section(),
    "Decision Analyzer": _da_section(),
    "Impact Analyzer": _ia_section(),
    "pdstools": [
        st.Page(
            str(_ABOUT_SCRIPT),
            title="About",
            icon="ℹ️",
            url_path="about",
        ),
    ],
}

st.navigation(sections, position="sidebar").run()
