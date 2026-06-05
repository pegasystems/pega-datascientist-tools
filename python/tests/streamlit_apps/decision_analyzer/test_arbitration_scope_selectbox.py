"""Widget-interaction test: scope/granularity selectbox on Arbitration Distribution page.

``10_Arbitration_Distribution.py`` renders a "Granularity:" selectbox
keyed ``scope`` in the sidebar.  The initial value defaults to
``get_possible_scope_values()[0]`` (typically "Action"); changing it
controls whether downstream plots break results by Issue, Group, or
Action.

If the selectbox key is renamed or the session-state binding is lost,
the selectbox still renders but every re-run silently resets scope to
the default. Existing smoke tests stay green because the page renders.
This test exercises the state-transition: pick a different scope,
confirm ``session_state["scope"]`` reflects the selection.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _find_selectbox(at: AppTest, key: str):
    for sb in at.selectbox:
        if sb.key == key:
            return sb
    return None


def test_scope_selectbox_updates_session_state(
    da_app_dir,
    seeded_decision_analyzer,
) -> None:
    """Switching the granularity selectbox updates ``session_state['scope']``."""
    page = da_app_dir / "pages" / "10_Arbitration_Distribution.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"

    scope_sb = _find_selectbox(at, "scope")
    assert scope_sb is not None, "Expected 'Granularity' selectbox with key='scope' to be rendered"

    initial_scope = at.session_state["scope"]
    assert initial_scope == scope_sb.value, (
        f"session_state['scope'] ({initial_scope!r}) should match the rendered selectbox value ({scope_sb.value!r})"
    )

    options = scope_sb.options
    pickable = next((o for o in options if o != initial_scope), None)
    if pickable is None:
        pytest.skip(f"Test fixture only exposes one scope option {options!r}; cannot exercise a value change.")

    scope_sb.set_value(pickable).run()
    assert not at.exception, f"Post-selection run raised: {at.exception}"
    assert at.session_state["scope"] == pickable, (
        f"After selecting {pickable!r}, session_state['scope'] should be "
        f"{pickable!r}, got {at.session_state['scope']!r}"
    )


def test_scope_selectbox_round_trip(
    da_app_dir,
    seeded_decision_analyzer,
) -> None:
    """Scope selectbox can be changed and changed back without error."""
    page = da_app_dir / "pages" / "10_Arbitration_Distribution.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"

    scope_sb = _find_selectbox(at, "scope")
    if scope_sb is None:
        pytest.skip("Scope selectbox not rendered.")

    options = scope_sb.options
    if len(options) < 2:
        pytest.skip(f"Round-trip test requires at least two scope options, got {options!r}.")

    original = scope_sb.value
    other = next(o for o in options if o != original)

    _find_selectbox(at, "scope").set_value(other).run()
    assert not at.exception
    assert at.session_state["scope"] == other

    _find_selectbox(at, "scope").set_value(original).run()
    assert not at.exception
    assert at.session_state["scope"] == original, (
        f"After round-trip, scope should be back to {original!r}, got {at.session_state['scope']!r}"
    )
