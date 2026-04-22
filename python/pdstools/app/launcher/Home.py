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

import streamlit as st

from pdstools.app.decision_analyzer._navigation import pages as da_pages
from pdstools.app.health_check._navigation import pages as hc_pages
from pdstools.app.impact_analyzer._navigation import pages as ia_pages
from pdstools.utils.streamlit_utils import (
    show_sidebar_branding,
    standard_page_config,
)

standard_page_config(page_title="pdstools")
show_sidebar_branding("pdstools")

# Health Check is the first section so its Home is the default page —
# only one page may carry default=True across the whole navigation.
sections: dict[str, list[st.Page]] = {
    "Health Check": hc_pages(url_prefix="hc", default=True),
    "Decision Analyzer": da_pages(url_prefix="da", default=False),
    "Impact Analyzer": ia_pages(url_prefix="ia", default=False),
}

st.navigation(sections, position="sidebar").run()
