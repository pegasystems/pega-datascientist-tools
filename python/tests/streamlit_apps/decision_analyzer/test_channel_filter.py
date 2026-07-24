"""Widget-interaction test: channel-direction selectbox must update filter state.

``channel_direction_selector()`` (in ``da_streamlit_utils.py``) renders
an ``st.selectbox`` keyed ``_channel_direction_widget`` with
``on_change=_update_channel_filter``. The callback writes:

- ``page_channel_filter``: human-readable selection (``"Any"`` or ``"Channel"`` / ``"Channel/Direction"``)
- ``page_channel_expr``: a polars filter expression (``None`` for "Any")

If either side of that wiring is broken, the selectbox still renders
and existing smoke tests stay green — but every analysis page that
calls ``collect_page_filters()`` silently ignores the user's choice.
This test exercises the state transition: change the selectbox, assert
both session-state keys reflect the new selection.
"""

from __future__ import annotations


import polars as pl

from streamlit.testing.v1 import AppTest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer
    from pathlib import Path


def _find_selectbox(at: AppTest, key: str):
    for sb in at.selectbox:
        if sb.key == key:
            return sb
    return None


def test_channel_filter_selectbox_updates_session_state(
    da_app_dir: Path,
    seeded_decision_analyzer: DecisionAnalyzer,
) -> None:
    """Switching the channel selectbox to a real channel populates a polars expr."""
    page = da_app_dir / "pages" / "5_Global_Sensitivity.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"

    selectbox = _find_selectbox(at, "_channel_direction_widget")
    assert [widget.key for widget in at.selectbox if widget.key == "_channel_direction_widget"] == [
        "_channel_direction_widget",
    ]
    assert selectbox.options == ["Any", "Mobile/Inbound", "Web/Inbound", "Web/Outbound"]

    # On initial render the selectbox defaults to "Any" but doesn't
    # write the convenience keys until the user changes it. Treat
    # absence as "Any".
    initial_filter = at.session_state["page_channel_filter"] if "page_channel_filter" in at.session_state else "Any"
    initial_expr = at.session_state["page_channel_expr"] if "page_channel_expr" in at.session_state else None
    assert initial_filter == "Any"
    assert initial_expr is None
    assert selectbox.value == "Any"

    pickable = "Mobile/Inbound"

    selectbox.set_value(pickable).run()
    assert not at.exception, f"Post-selection run raised: {at.exception}"

    assert at.session_state["page_channel_filter"] == pickable, (
        f"Selecting {pickable!r} should update page_channel_filter, got {at.session_state['page_channel_filter']!r}"
    )
    expr = at.session_state["page_channel_expr"]
    assert isinstance(expr, pl.Expr), (
        f"page_channel_expr must become a polars expression after a non-'Any' selection, got {type(expr).__name__}"
    )

    # Round-trip back to "Any" — the expression must reset to None so
    # downstream filtering doesn't keep applying the stale filter.
    selectbox.set_value("Any").run()
    assert not at.exception
    assert at.session_state["page_channel_filter"] == "Any"
    assert at.session_state["page_channel_expr"] is None, "Selecting 'Any' must reset page_channel_expr to None"
