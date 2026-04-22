# python/pdstools/app/health_check/_navigation.py
"""Programmatic navigation for the ADM Health Check app.

Replaces Streamlit's auto-discovery of ``pages/`` with an explicit
``st.navigation()`` registry. Mirrors the Decision Analyzer pattern
so all pdstools Streamlit apps share one navigation mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from pdstools.app.health_check._home_page import home_page

_PAGES_DIR = Path(__file__).parent / "pages"


@dataclass(frozen=True)
class _SubPage:
    """Spec for a sidebar entry backed by a script in ``pages/``."""

    filename: str
    title: str
    icon: str


# Order matches the historical numeric prefixes so the sidebar reads
# the same way users are used to. Edit here to reorder the menu.
_SUB_PAGES: tuple[_SubPage, ...] = (
    _SubPage("1_Data_Filters.py", "Data Filters", "🔍"),
    _SubPage("2_Reports.py", "Reports", "📊"),
    _SubPage("3_About.py", "About", "ℹ️"),
)


def build_navigation():
    """Return the configured ``st.navigation`` object for the HC app.

    Always shows Home (default) plus every page in ``pages/``.
    """
    home = st.Page(home_page, title="Home", icon="🏠", default=True)
    pages = [home]
    pages.extend(st.Page(str(_PAGES_DIR / p.filename), title=p.title, icon=p.icon) for p in _SUB_PAGES)
    return st.navigation(pages, position="sidebar")
