# python/pdstools/app/decision_analyzer/_navigation.py
"""Programmatic navigation for the Decision Analysis app.

Replaces Streamlit's auto-discovery of ``pages/`` with an explicit
``st.navigation()`` registry. Pages that need a loaded
``DecisionAnalyzer`` are hidden from the sidebar until
``st.session_state["decision_data"]`` is populated, removing the
"clickable but immediately error-out" experience.

Hidden pages stay reachable by URL, so existing per-page
``ensure_data()`` guards still apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from pdstools.app.decision_analyzer._home_page import render as render_home

_PAGES_DIR = Path(__file__).parent / "pages"


@dataclass(frozen=True)
class _DataPage:
    """Spec for a page that requires ``decision_data`` in session state."""

    filename: str
    title: str
    icon: str


# Order matches the historical numeric prefixes so the sidebar reads
# the same way users are used to. Edit here to reorder the menu.
_DATA_PAGES: tuple[_DataPage, ...] = (
    _DataPage("2_Overview.py", "Overview", "🔍"),
    _DataPage("3_Action_Distribution.py", "Action Distribution", "📊"),
    _DataPage("4_Action_Funnel.py", "Action Funnel", "🪜"),
    _DataPage("5_Global_Sensitivity.py", "Global Sensitivity", "🎯"),
    _DataPage("6_Win_Loss_Analysis.py", "Win / Loss Analysis", "🏆"),
    _DataPage("7_Optionality_Analysis.py", "Optionality Analysis", "🎨"),
    _DataPage("8_Offer_Quality_Analysis.py", "Offer Quality Analysis", "💎"),
    _DataPage("9_Thresholding_Analysis.py", "Thresholding Analysis", "🚦"),
    _DataPage("10_Arbitration_Distribution.py", "Arbitration Distribution", "⚖️"),
    _DataPage("11_Single_Decision.py", "Single Decision", "🔬"),
)

_ABOUT_PAGE_FILE = "12_About.py"


def locked_page_titles() -> list[str]:
    """Return display titles for pages currently hidden from navigation."""
    if "decision_data" in st.session_state:
        return []
    return [p.title for p in _DATA_PAGES]


def _slug(title: str) -> str:
    """Lowercase, underscore-separated slug used for stable URL paths."""
    return title.lower().replace(" / ", "_").replace(" ", "_").replace("/", "_")


def pages(
    url_prefix: str | None = None,
    default: bool = True,
    include_about: bool = True,
    include_subpages: bool = True,
) -> list[st.Page]:
    """Return the DA page list for ``st.navigation``.

    Parameters
    ----------
    url_prefix : str or None
        When set (e.g. ``"da"``), every page is registered with a
        ``url_path`` of ``f"{url_prefix}/{slug}"`` so multiple tools
        can be hosted in one Streamlit process without URL collisions.
        When ``None``, page URLs are derived from titles by Streamlit
        — preserves the standalone tool's existing URL scheme.
    default : bool
        Whether the Home page is marked as the navigation default.
        The launcher must designate exactly one default page across
        all sections, so it sets this to ``False`` for tools other
        than the one it picks as the entry point.
    include_about : bool
        Whether to append the per-tool About page. Standalone tool
        launches keep it (``True``); the cross-app launcher passes
        ``False`` so all three tools share a single top-level About
        entry instead of crowding the sidebar with three.
    include_subpages : bool
        Whether to append the data-dependent analysis pages. The
        cross-app launcher passes ``False`` until the user actively
        enters the DA app; even when ``True``, individual pages stay
        gated on ``decision_data`` being present in session state.
    """
    locked = locked_page_titles()

    def _url(name: str) -> str | None:
        # Streamlit's ``st.Page`` rejects nested ``url_path`` values
        # (no "/"). Use an underscore-prefixed flat namespace instead
        # so launcher URLs look like ``/da_overview`` while standalone
        # tool launches keep their original ``/overview`` URLs.
        return f"{url_prefix}_{name}" if url_prefix else None

    # Wrap render so st.Page can call it with no arguments while still
    # passing the locked-page list for the discoverability hint.
    home = st.Page(
        lambda: render_home(locked),
        title="Home",
        icon="🏠",
        default=default,
        url_path=_url("home"),
    )
    about = st.Page(
        str(_PAGES_DIR / _ABOUT_PAGE_FILE),
        title="About",
        icon="ℹ️",
        url_path=_url("about"),
    )

    result = [home]
    if include_subpages and "decision_data" in st.session_state:
        result.extend(
            st.Page(
                str(_PAGES_DIR / p.filename),
                title=p.title,
                icon=p.icon,
                url_path=_url(_slug(p.title)),
            )
            for p in _DATA_PAGES
        )
    if include_about:
        result.append(about)
    return result


def build_navigation():
    """Return the configured ``st.navigation`` object for the DA app.

    Always shows Home (default) and About. Data-dependent pages are
    appended to the menu only when ``decision_data`` is loaded.
    """
    return st.navigation(pages(), position="sidebar")
