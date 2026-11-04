"""Widget-interaction test: baseline stage selectbox on the Optionality page.

``7_Optionality_Analysis.py`` renders an "Exclusion Rate" section (v2 data
only) with a baseline ("from") stage selectbox keyed ``exclusion_from_stage``.
The measurement ("to") stage follows the existing optionality selector
(``optionality_stage``). Changing the baseline must update session state and
re-render the exclusion-rate distribution without raising.

A plain render smoke test would stay green even if the selectbox lost its
session-state binding (every re-run would silently reset the baseline). This
test exercises the state transition explicitly.
"""

from __future__ import annotations

import polars as pl
from streamlit.testing.v1 import AppTest


def _find_selectbox(at: AppTest, key: str):
    for sb in at.selectbox:
        if sb.key == key:
            return sb
    return None


def test_baseline_selectbox_updates_session_state(
    da_app_dir,
    seeded_decision_analyzer,
) -> None:
    """Switching the baseline stage updates ``session_state['exclusion_from_stage']``."""
    page = da_app_dir / "pages" / "7_Optionality_Analysis.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"

    baseline_sb = _find_selectbox(at, "exclusion_from_stage")
    assert [widget.key for widget in at.selectbox if widget.key == "exclusion_from_stage"] == [
        "exclusion_from_stage",
    ]

    stages = list(seeded_decision_analyzer.AvailableNBADStages)
    to_stage = at.session_state["optionality_stage"] if "optionality_stage" in at.session_state else "Arbitration"
    # Pick a different baseline that is still at/before the measurement stage,
    # so the range stays valid and the distribution renders.
    valid_from = [s for s in baseline_sb.options if stages.index(s) <= stages.index(to_stage)]
    other = next((s for s in valid_from if s != baseline_sb.value), baseline_sb.value)

    baseline_sb.set_value(other).run()
    assert not at.exception, f"Post-selection run raised: {at.exception}"
    assert at.session_state["exclusion_from_stage"] == other, (
        f"After selecting {other!r}, session_state['exclusion_from_stage'] should be "
        f"{other!r}, got {at.session_state['exclusion_from_stage']!r}"
    )


def test_baseline_after_measurement_warns(
    da_app_dir,
    seeded_decision_analyzer,
) -> None:
    """A baseline later than the measurement stage shows a warning, not a broken plot."""
    page = da_app_dir / "pages" / "7_Optionality_Analysis.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"

    baseline_sb = _find_selectbox(at, "exclusion_from_stage")
    assert [widget.key for widget in at.selectbox if widget.key == "exclusion_from_stage"] == [
        "exclusion_from_stage",
    ]

    stages = list(seeded_decision_analyzer.AvailableNBADStages)
    to_stage = at.session_state["optionality_stage"] if "optionality_stage" in at.session_state else "Arbitration"
    later = [s for s in baseline_sb.options if stages.index(s) > stages.index(to_stage)]
    if not later:
        return  # No stage after the measurement point in this fixture; nothing to assert.

    baseline_sb.set_value(later[0]).run()
    assert not at.exception, f"Post-selection run raised: {at.exception}"
    assert any("comes after the measurement stage" in w.value for w in at.warning), (
        "Expected a warning when the baseline is later than the measurement stage"
    )


def test_measurement_stage_follows_optionality_selector(
    da_app_dir,
    seeded_decision_analyzer,
) -> None:
    """Changing optionality_stage changes the exclusion-rate measurement stage."""
    page = da_app_dir / "pages" / "7_Optionality_Analysis.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"

    measurement_sb = _find_selectbox(at, "optionality_stage")
    assert [widget.key for widget in at.selectbox if widget.key == "optionality_stage"] == [
        "optionality_stage",
    ]
    assert "Output" in measurement_sb.options

    measurement_sb.set_value("Output").run()
    assert not at.exception, f"Post-selection run raised: {at.exception}"
    assert at.session_state["optionality_stage"] == "Output"
    assert any("to 'Output'" in caption.value for caption in at.caption)


def test_empty_filtered_data_stops_with_warning(
    da_app_dir,
    seeded_decision_analyzer,
) -> None:
    """An empty contextual filter shows the page warning instead of rendering charts."""
    page = da_app_dir / "pages" / "7_Optionality_Analysis.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"

    at.session_state["_channel_direction_widget"] = "Mobile/Inbound"
    at.session_state["page_channel_filter"] = "Mobile/Inbound"
    at.session_state["page_channel_expr"] = pl.col("Channel") == "No data"
    at.run()

    assert not at.exception, f"Empty-filter run raised: {at.exception}"
    assert any("No data available" in warning.value for warning in at.warning)
