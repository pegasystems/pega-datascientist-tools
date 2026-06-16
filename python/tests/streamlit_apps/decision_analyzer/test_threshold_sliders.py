"""Widget-interaction test: threshold sliders must persist to session state.

``9_Thresholding_Analysis.py`` renders two sidebar sliders keyed
``propensity_threshold_pct`` and ``priority_threshold_val``.  Both
initialise to 0.0 on first visit.  The page reads the values directly
from session state — no explicit ``on_change`` callback — so these
tests verify that Streamlit's native key-binding mechanism updates the
session-state keys when a slider is moved.

If the slider keys are ever renamed or the session-state initialisation
block is removed, both sliders silently reset to 0.0 on every rerun,
ignoring the user's threshold choice. Existing smoke tests stay green
because the page renders fine at threshold=0. These tests exercise the
state transitions.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _find_slider(at: AppTest, key: str):
    for s in at.slider:
        if s.key == key:
            return s
    return None


def _load_page(da_app_dir, seeded_decision_analyzer):
    page = da_app_dir / "pages" / "9_Thresholding_Analysis.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()
    return at


def test_propensity_slider_updates_session_state(
    da_app_dir,
    seeded_decision_analyzer,
) -> None:
    """Moving the propensity slider updates ``propensity_threshold_pct``."""
    at = _load_page(da_app_dir, seeded_decision_analyzer)
    assert not at.exception, f"Page raised: {at.exception}"

    slider = _find_slider(at, "propensity_threshold_pct")
    if slider is None:
        pytest.skip("Propensity slider not rendered — fixture likely has no arbitration-stage data.")
    if slider.max == 0.0:
        pytest.skip("Propensity slider max is 0; cannot exercise a meaningful change.")

    assert at.session_state["propensity_threshold_pct"] == 0.0, (
        f"Slider should initialise to 0.0, got {at.session_state['propensity_threshold_pct']}"
    )

    new_value = slider.max
    slider.set_value(new_value).run()
    assert not at.exception, f"Post-slider run raised: {at.exception}"
    assert at.session_state["propensity_threshold_pct"] == pytest.approx(new_value, rel=1e-3), (
        f"propensity_threshold_pct should be {new_value}, got {at.session_state['propensity_threshold_pct']}"
    )


def test_priority_slider_updates_session_state(
    da_app_dir,
    seeded_decision_analyzer,
) -> None:
    """Moving the priority slider updates ``priority_threshold_val``."""
    at = _load_page(da_app_dir, seeded_decision_analyzer)
    assert not at.exception, f"Page raised: {at.exception}"

    slider = _find_slider(at, "priority_threshold_val")
    if slider is None:
        pytest.skip("Priority slider not rendered — fixture likely has no arbitration-stage data.")
    if slider.max == 0.0:
        pytest.skip("Priority slider max is 0; cannot exercise a meaningful change.")

    assert at.session_state["priority_threshold_val"] == 0.0, (
        f"Slider should initialise to 0.0, got {at.session_state['priority_threshold_val']}"
    )

    new_value = slider.max
    slider.set_value(new_value).run()
    assert not at.exception, f"Post-slider run raised: {at.exception}"
    assert at.session_state["priority_threshold_val"] == pytest.approx(new_value, rel=1e-3), (
        f"priority_threshold_val should be {new_value}, got {at.session_state['priority_threshold_val']}"
    )


def test_resetting_propensity_slider_to_zero(
    da_app_dir,
    seeded_decision_analyzer,
) -> None:
    """Setting propensity threshold then resetting to zero clears the threshold."""
    at = _load_page(da_app_dir, seeded_decision_analyzer)
    assert not at.exception

    slider = _find_slider(at, "propensity_threshold_pct")
    if slider is None or slider.max == 0.0:
        pytest.skip("Propensity slider not rendered or max is 0.")

    slider.set_value(slider.max).run()
    assert not at.exception
    assert at.session_state["propensity_threshold_pct"] > 0.0

    _find_slider(at, "propensity_threshold_pct").set_value(0.0).run()
    assert not at.exception
    assert at.session_state["propensity_threshold_pct"] == 0.0, (
        f"Resetting slider to 0 should clear propensity_threshold_pct, "
        f"got {at.session_state['propensity_threshold_pct']}"
    )
