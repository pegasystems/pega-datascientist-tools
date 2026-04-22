# python/pdstools/app/launcher/_picker.py
"""Tile-based app picker for the cross-app pdstools launcher.

Renders three Streamlit-native cards (one per hosted tool) so the
default landing page is a real "choose an app" screen instead of the
first tool's home page. Each tile uses ``st.page_link`` so navigation
stays inside the same Streamlit process — no JS, no iframes.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class AppTile:
    """One entry in the launcher's app picker."""

    icon: str
    title: str
    description: str
    url_path: str
    active_app: str


# Order matches the launcher's sidebar section order.
TILES: tuple[AppTile, ...] = (
    AppTile(
        icon="🩺",
        title="ADM Health Check",
        description=(
            "Inspect Adaptive Decision Manager datamart exports — model "
            "performance, predictor health, and configuration drift."
        ),
        url_path="hc_home",
        active_app="hc",
    ),
    AppTile(
        icon="🎯",
        title="Decision Analysis",
        description=(
            "Explore arbitration outcomes, action funnels, and thresholding behaviour from explainability extracts."
        ),
        url_path="da_home",
        active_app="da",
    ),
    AppTile(
        icon="📈",
        title="Impact Analyzer",
        description=(
            "Compare control vs. treatment impressions and value-per-impression to quantify CDH lift over time."
        ),
        url_path="ia_home",
        active_app="ia",
    ),
)


# ``st.page_link`` resolves best when given a registered ``st.Page``
# object (so navigation runs in-process). The launcher entry script
# registers the per-app home pages here before calling ``picker_page``;
# tests that exercise the picker in isolation fall back to the raw URL.
_app_pages: dict[str, "st.Page"] = {}


def set_app_pages(pages: dict[str, "st.Page"]) -> None:
    """Register the per-app home ``st.Page`` objects for tile links."""
    _app_pages.clear()
    _app_pages.update(pages)


def _resolve_link(tile: AppTile):
    page = _app_pages.get(tile.active_app)
    return page if page is not None else f"/{tile.url_path}"


def picker_page() -> None:
    """Render the launcher landing page."""
    from pdstools.utils.streamlit_utils import (
        show_sidebar_branding,
        show_version_header,
        standard_page_config,
    )

    standard_page_config(page_title="pdstools")
    show_sidebar_branding("pdstools")

    # Visiting the picker means no app is active — clear the flag so
    # the sidebar collapses back to its picker-mode shape on rerun.
    if st.session_state.get("_active_app") is not None:
        st.session_state["_active_app"] = None

    st.markdown("# Welcome to pdstools")
    st.markdown(
        "Pick an app to get started. Each tool keeps its own data and "
        "filters in this session, so you can switch between them via "
        "the sidebar at any time.",
    )
    st.write("")

    columns = st.columns(len(TILES), gap="medium")
    for column, tile in zip(columns, TILES, strict=True):
        with column, st.container(border=True):
            st.markdown(f"### {tile.icon}&nbsp; {tile.title}")
            st.markdown(tile.description)
            st.page_link(
                _resolve_link(tile),
                label=f"Open {tile.title} →",
                use_container_width=True,
            )

    st.write("")
    show_version_header()
