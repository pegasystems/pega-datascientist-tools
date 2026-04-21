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


def build_navigation():
    """Return the configured ``st.navigation`` object for the DA app.

    Always shows Home (default) and About. Data-dependent pages are
    appended to the menu only when ``decision_data`` is loaded.
    """
    locked = locked_page_titles()
    # Wrap render so st.Page can call it with no arguments while still
    # passing the locked-page list for the discoverability hint.
    home = st.Page(lambda: render_home(locked), title="Home", icon="🏠", default=True)
    about = st.Page(str(_PAGES_DIR / _ABOUT_PAGE_FILE), title="About", icon="ℹ️")

    pages = [home]
    if "decision_data" in st.session_state:
        pages.extend(st.Page(str(_PAGES_DIR / p.filename), title=p.title, icon=p.icon) for p in _DATA_PAGES)
    pages.append(about)

    return st.navigation(pages, position="sidebar")
