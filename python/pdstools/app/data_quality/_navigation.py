# python/pdstools/app/data_quality/_navigation.py
"""Programmatic navigation for the Topic Data Quality app.

Mirrors the pattern used by Impact Analyzer, Decision Analyzer, and
Health Check so all pdstools Streamlit apps share one navigation
mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from pdstools.app.data_quality._home_page import home_page

_PAGES_DIR = Path(__file__).parent / "pages"


@dataclass(frozen=True)
class _SubPage:
    """Spec for a sidebar entry backed by a script in ``pages/``."""

    filename: str
    title: str
    icon: str


_SUB_PAGES: tuple[_SubPage, ...] = (_SubPage("1_Quality_Report.py", "Quality Report", "📊"),)

_ABOUT_PAGE = _SubPage("2_About.py", "About", "ℹ️")


def _slug(title: str) -> str:
    """Lowercase, underscore-separated slug used for stable URL paths."""
    return title.lower().replace(" / ", "_").replace(" ", "_").replace("/", "_")


def pages(
    url_prefix: str | None = None,
    default: bool = True,
    include_about: bool = True,
    include_subpages: bool = True,
) -> list[Any]:
    """Return the DQ page list for ``st.navigation``.

    Parameters match the contract used by all pdstools tools so the
    cross-app launcher can reuse them uniformly.
    """

    def _url(name: str) -> str | None:
        return f"{url_prefix}_{name}" if url_prefix else None

    home = st.Page(
        home_page,
        title="Home",
        icon="🏠",
        default=default,
        url_path=_url("home"),
    )
    result = [home]
    if include_subpages:
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
    """Return the configured ``st.navigation`` object for the DQ app."""
    return st.navigation(pages(), position="sidebar")
