"""Widget-interaction test: stage-level radio must propagate to the analyzer.

``stage_level_selector()`` (in ``da_streamlit_utils.py``) renders an
``st.radio`` keyed ``_stage_level_radio`` with
``on_change=_apply_stage_level``. The callback calls
``DecisionAnalyzer.set_level(...)`` so subsequent analysis runs against
the chosen granularity.

If the callback is ever broken (key rename, wrong attribute on the
analyzer, missed wire-up) the radio still renders fine and existing
"page renders" smoke tests stay green. This test exercises the
state-transition: change the radio value, verify the analyzer's
``level`` follows.
"""

from __future__ import annotations


from streamlit.testing.v1 import AppTest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer
    from pathlib import Path


def _find_radio(at: AppTest, key: str):
    for r in at.radio:
        if r.key == key:
            return r
    return None


def test_stage_level_radio_updates_analyzer(
    da_app_dir: Path,
    seeded_decision_analyzer: DecisionAnalyzer,
) -> None:
    """Toggling the stage-level radio must update ``decision_data.level``."""
    page = da_app_dir / "pages" / "11_Single_Decision.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"

    da = at.session_state["decision_data"]
    assert da.available_levels == ["Stage Group", "Stage"], (
        f"Test depends on the minimal EEV2 fixture exposing both levels, got {da.available_levels}"
    )

    radio = _find_radio(at, "_stage_level_radio")
    assert radio is not None, "Expected stage-level radio to be rendered"
    assert da.level == "Stage Group", f"Initial level should be 'Stage Group', got {da.level!r}"

    radio.set_value("Stage").run()
    assert not at.exception, f"Post-toggle run raised: {at.exception}"

    da_after = at.session_state["decision_data"]
    assert da_after.level == "Stage", (
        f"After selecting 'Stage' on the radio, decision_data.level must be 'Stage', got {da_after.level!r}"
    )
    assert at.session_state["_stage_level_radio"] == "Stage"
