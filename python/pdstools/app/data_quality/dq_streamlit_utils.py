"""Topic Data Quality app – Streamlit helpers."""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)


def ensure_dq_data_loaded() -> bool:
    """Return ``True`` when ``st.session_state["topic_dq"]`` is populated."""
    return "topic_dq" in st.session_state


def ensure_topic_dq():
    """Guard: warn and stop if no data has been analysed yet.

    Called at the top of every sub-page.
    """
    from pdstools.utils.streamlit_utils import _apply_sidebar_logo

    _apply_sidebar_logo()
    if not ensure_dq_data_loaded():
        st.warning("Please upload and analyse your data on the Home page first.")
        st.stop()
    return st.session_state["topic_dq"]
