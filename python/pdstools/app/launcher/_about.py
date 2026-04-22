# python/pdstools/app/launcher/_about.py
"""Shared About page for the cross-app pdstools launcher.

Single top-level About entry — replaces the three near-identical
per-tool About entries that would otherwise crowd the launcher
sidebar (one per tool section). Standalone tool launches keep
their own About page via ``pages(include_about=True)``.
"""

from pdstools.utils.streamlit_utils import show_about_page, standard_page_config

standard_page_config(page_title="About · pdstools")

show_about_page()
