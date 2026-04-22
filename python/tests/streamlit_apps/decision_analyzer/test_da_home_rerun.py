"""Regression tests for the Decision Analysis sample-data auto-load rerun.

When the home page auto-loads sample data on first visit,
``st.navigation()`` has already been constructed earlier in the script
run with the data-gated sub-pages hidden. Without an explicit
``st.rerun()`` the sidebar stays locked until the user clicks
something — matching the HC / IA UX requires a rerun exactly once on
the transition from "no data" to "have data".
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_sample_auto_load_triggers_rerun_then_unlocks_nav(da_app_dir: Path):
    """First-run auto-load reruns once and then exposes data-gated pages.

    AppTest's ``run()`` re-executes the script after each
    ``st.rerun()`` until the script settles, so a single ``run()``
    here covers both the initial load and the rerun pass. After the
    settled run we expect:

    - No exception bubbled out.
    - ``decision_data`` is populated in session state (sample loaded).
    - The second pass settled without triggering further reruns —
      i.e. the guard on ``has_existing_data`` prevents a loop.
    - The Overview page (one of the previously locked data-gated
      pages) is reachable via ``st.navigation`` after the rerun.
    """
    at = AppTest.from_file(str(da_app_dir / "Home.py"), default_timeout=60)
    at.run()

    assert not at.exception, f"Home raised: {at.exception}"
    assert "decision_data" in at.session_state, "Sample auto-load should populate decision_data"

    # Re-running with data already in session_state must NOT trigger
    # another rerun loop — the guard ensures the rerun fires exactly
    # once on the transition. AppTest exposes no direct rerun counter,
    # but a settled second run with the same session_state is the
    # observable signal.
    at.run()
    assert not at.exception, f"Second run raised: {at.exception}"
    assert "decision_data" in at.session_state

    # Confirm the data-gated nav would now be populated. ``_navigation``
    # gates pages on ``"decision_data" in st.session_state`` — the same
    # condition AppTest exposes via ``at.session_state``. Calling
    # ``_navigation.pages()`` directly here would need a live Streamlit
    # runtime (``st.Page`` requires it), so assert on the underlying
    # signal instead.
    from pdstools.app.decision_analyzer import _navigation

    # ``locked_page_titles()`` reads ``st.session_state`` (the bare
    # module-level one, not AppTest's). Mirror the loaded state into
    # it so we exercise the gating logic end-to-end.
    import streamlit as st

    st.session_state["decision_data"] = at.session_state["decision_data"]
    try:
        assert _navigation.locked_page_titles() == [], "Expected no locked pages once decision_data is loaded"
    finally:
        del st.session_state["decision_data"]
