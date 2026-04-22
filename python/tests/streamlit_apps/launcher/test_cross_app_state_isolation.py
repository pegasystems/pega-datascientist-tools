"""Cross-app state isolation regression check.

Each launcher app owns its own session-state keys
(``decision_data`` for DA, ``dm`` for HC, ``impact_analyzer`` for IA),
and each app's home page must run its own auto-load chain without
inheriting another app's loaded state. If two apps ever started
sharing a key (e.g. a refactor that renamed ``dm`` to ``datamart`` and
collided with an existing key) the launcher would leak data across
tools.

True cross-app navigation in the launcher goes through
``st.navigation`` and ``st.switch_page`` — neither of which AppTest
simulates cleanly: each ``AppTest.from_file`` call spins up a fresh
ScriptRunContext, so we can't observe "click HC tile, navigate to DA
home" inside one process. The next-best assertion this test makes is
the *contract* that each app uses a distinct, well-known key, and
that pre-seeding one app's key doesn't trip up another app's
auto-load.

If the launcher ever grows a way to multi-page-AppTest
(``st.testing.v1`` adds ``MultiPageAppTest`` or similar), this test
should be upgraded to drive the real ``switch_page`` flow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def test_da_autoload_independent_of_hc_dm(da_app_dir: Path, hc_app_dir: Path) -> None:
    """Loading HC then DA must not mix state — each app uses its own key."""
    # 1. Load HC home: autoload populates session_state["dm"].
    hc = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=60)
    hc.run()
    assert not hc.exception, f"HC Home raised: {hc.exception}"
    assert "dm" in hc.session_state, "HC autoload should populate 'dm'"
    assert "decision_data" not in hc.session_state, "HC must not write to DA's session-state key 'decision_data'"

    # 2. Pre-seed a fresh DA AppTest with HC's dm to simulate worst-case
    #    state leak (key collision via shared launcher process). DA's
    #    home page must STILL trigger its own autoload of decision_data
    #    and not mistake HC's dm for its own data object.
    da = AppTest.from_file(str(da_app_dir / "Home.py"), default_timeout=60)
    da.session_state["dm"] = hc.session_state["dm"]
    da.run()
    assert not da.exception, f"DA Home raised when 'dm' was pre-seeded: {da.exception}"

    # DA's autoload populates decision_data. The leaked dm is left
    # alone (DA never reads it) — proving the two apps use disjoint
    # keys.
    assert "decision_data" in da.session_state, (
        "DA autoload should populate 'decision_data' even when an unrelated 'dm' key exists"
    )
    da_obj = da.session_state["decision_data"]
    hc_obj = hc.session_state["dm"]
    assert type(da_obj).__name__ != type(hc_obj).__name__, (
        f"DA and HC must produce different analyzer types — got {type(da_obj).__name__} for both"
    )


def test_cross_app_switch_via_navigation_unsupported() -> None:
    """Document why we can't drive real ``st.switch_page`` in AppTest.

    AppTest spins up a fresh ScriptRunContext per ``from_file()`` and
    has no ``switch_page`` simulation as of streamlit 1.x. The real
    cross-app navigation from the launcher (``picker.click → DA Home``)
    can only be exercised end-to-end via Playwright. This stub keeps
    the gap discoverable.
    """
    pytest.skip(
        "AppTest does not simulate st.navigation / st.switch_page. "
        "Real cross-app launcher navigation needs an end-to-end test "
        "(Playwright). The session-state isolation contract is covered "
        "by test_da_autoload_independent_of_hc_dm."
    )
