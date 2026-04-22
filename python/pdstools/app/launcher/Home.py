# python/pdstools/app/launcher/Home.py
"""Entry script for the cross-app pdstools launcher.

Hosts Health Check, Decision Analyzer, and Impact Analyzer side by
side in a single Streamlit process using ``st.navigation``'s
sectioned dict API. Standalone tool entry points
(``app/<tool>/Home.py``) are unaffected.

Sidebar shape adapts to the active app (tracked via
``st.session_state["_active_app"]``, set by each app's home page):

* No app active (default landing) → picker tile + 3 home links + About.
* HC / IA active → that app's sub-pages added to its section.
* DA active → DA's data pages added once ``decision_data`` is loaded
  (additional gate on top of the active-app check).

URL paths are flat-namespaced (``/hc_*``, ``/da_*``, ``/ia_*``) to
avoid collisions between tools that share page names — Streamlit's
``st.Page`` rejects nested ``/`` in ``url_path`` so the launcher uses
an underscore separator instead.
"""

from pathlib import Path

import streamlit as st

from pdstools.app.launcher._picker import picker_page, set_app_pages

_ABOUT_SCRIPT = Path(__file__).parent / "_about.py"


# Each tool's _navigation module is imported on first call only — the
# transitive imports (plotly, papermill, fastexcel, polars-heavy
# helpers via _home_page.py) would otherwise fire on every launcher
# rerun even when the user only ever clicks into one tool. After the
# first call sys.modules caches the module so subsequent reruns are
# free.
def _hc_section(active: bool) -> list[st.Page]:
    from pdstools.app.health_check._navigation import pages

    return pages(
        url_prefix="hc",
        default=False,
        include_about=False,
        include_subpages=active,
    )


def _da_section(active: bool) -> list[st.Page]:
    from pdstools.app.decision_analyzer._navigation import pages

    return pages(
        url_prefix="da",
        default=False,
        include_about=False,
        include_subpages=active,
    )


def _ia_section(active: bool) -> list[st.Page]:
    from pdstools.app.impact_analyzer._navigation import pages

    return pages(
        url_prefix="ia",
        default=False,
        include_about=False,
        include_subpages=active,
    )


# Picker is the launcher's default landing page so visiting "/" lands
# on a real "choose an app" screen instead of one tool's home.
_picker_page = st.Page(
    picker_page,
    title="Home",
    icon="🏠",
    default=True,
    url_path="home",
)

active = st.session_state.get("_active_app")
hc_pages = _hc_section(active == "hc")
da_pages = _da_section(active == "da")
ia_pages = _ia_section(active == "ia")

# Register per-app home pages with the picker so its tiles can use
# in-process ``st.page_link`` navigation (no full reload).
set_app_pages(
    {
        "hc": hc_pages[0],
        "da": da_pages[0],
        "ia": ia_pages[0],
    }
)

# Section ordering matches the picker tile order so the sidebar reads
# top-to-bottom in the same sequence the user just saw on the landing
# page. The shared About page lives in its own section so it reads as
# a top-level entry, not buried under any one tool.
sections: dict[str, list[st.Page]] = {
    "pdstools": [_picker_page],
    "ADM Health Check": hc_pages,
    "Decision Analysis": da_pages,
    "Impact Analyzer": ia_pages,
    "About": [
        st.Page(
            str(_ABOUT_SCRIPT),
            title="About",
            icon="ℹ️",
            url_path="about",
        ),
    ],
}

st.navigation(sections, position="sidebar").run()
