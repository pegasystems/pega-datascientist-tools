# python/pdstools/app/impact_analyzer/_navigation.py
"""Programmatic navigation for the Impact Analyzer app.

Replaces Streamlit's auto-discovery of ``pages/`` with an explicit
``st.navigation()`` registry. Mirrors the Decision Analyzer pattern
so all pdstools Streamlit apps share one navigation mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from pdstools.app.impact_analyzer._home_page import home_page

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
    _SubPage("1_Overall_Summary.py", "Overall Summary", "📈"),
    _SubPage("2_Trend.py", "Trend", "📉"),
)

_ABOUT_PAGE = _SubPage("3_About.py", "About", "ℹ️")


def _slug(title: str) -> str:
    """Lowercase, underscore-separated slug used for stable URL paths."""
    return title.lower().replace(" / ", "_").replace(" ", "_").replace("/", "_")


def pages(
    url_prefix: str | None = None,
    default: bool = True,
    include_about: bool = True,
) -> list[st.Page]:
    """Return the IA page list for ``st.navigation``.

    See ``decision_analyzer._navigation.pages`` for parameter
    semantics — same contract across all pdstools tools so the
    cross-app launcher can reuse them uniformly.
    """

    def _url(name: str) -> str | None:
        # Streamlit's ``st.Page`` rejects nested ``url_path`` values
        # (no "/"). Use an underscore-prefixed flat namespace instead.
        return f"{url_prefix}_{name}" if url_prefix else None

    home = st.Page(
        home_page,
        title="Home",
        icon="🏠",
        default=default,
        url_path=_url("home"),
    )
    result = [home]
    result.extend(
        st.Page(
            str(_PAGES_DIR / p.filename),
            title=p.title,
            icon=p.icon,
            url_path=_url(_slug(p.title)),
        )
        for p in _SUB_PAGES
    )
    if include_about:
        result.append(
            st.Page(
                str(_PAGES_DIR / _ABOUT_PAGE.filename),
                title=_ABOUT_PAGE.title,
                icon=_ABOUT_PAGE.icon,
                url_path=_url(_slug(_ABOUT_PAGE.title)),
            )
        )
    return result


def build_navigation():
    """Return the configured ``st.navigation`` object for the IA app.

    Always shows Home (default) plus every page in ``pages/``.
    """
    return st.navigation(pages(), position="sidebar")
