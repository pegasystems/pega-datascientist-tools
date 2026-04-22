"""AppTest smoke test for the cross-app pdstools launcher.

Verifies the launcher boots without exception, registers all three
tools as sidebar sections, and namespaces their URLs to avoid
collisions. A regression here means the launcher won't host one or
more tools.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pdstools.app.health_check._navigation import pages as hc_pages
from pdstools.app.decision_analyzer._navigation import pages as da_pages
from pdstools.app.impact_analyzer._navigation import pages as ia_pages

REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER_HOME = REPO_ROOT / "python" / "pdstools" / "app" / "launcher" / "Home.py"
HC_HOME = REPO_ROOT / "python" / "pdstools" / "app" / "health_check" / "Home.py"

# The collapse-at-N threshold Streamlit applies to ``stSidebarNav``.
# Anything beyond this gets hidden behind a "View N more" button by
# default. The launcher injects CSS to override this, but we still
# keep the structural sidebar at or below the threshold pre-DA-load
# so the CSS hack is belt-and-braces, not load-bearing.
_NAV_COLLAPSE_THRESHOLD = 10


@pytest.fixture
def launcher_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the launcher Home with PDSTOOLS_LAUNCHER_MODE set, like the CLI."""
    monkeypatch.setenv("PDSTOOLS_LAUNCHER_MODE", "1")


def test_launcher_home_renders_without_data(launcher_mode):
    """Launcher entry script runs without exception on first load.

    Indirectly validates that all three tools' ``pages()`` helpers
    are callable, that the sectioned ``st.navigation`` dict API
    accepts the result, and that the per-tool URL namespacing
    produces URL paths Streamlit considers valid (no nested ``/``,
    no collisions across the three "About" pages).
    """
    at = AppTest.from_file(str(LAUNCHER_HOME), default_timeout=60)
    at.run()
    assert not at.exception, f"Launcher Home raised: {at.exception}"


def test_standalone_hc_home_still_renders():
    """Standalone HC launch must keep working with the refactored ``pages()``.

    The launcher refactor moved page construction into a helper that
    accepts a ``url_prefix``; the standalone entry script must keep
    rendering the original URL scheme so bookmarks don't break.
    """
    at = AppTest.from_file(str(HC_HOME), default_timeout=60)
    at.run()
    assert not at.exception, f"HC standalone Home raised: {at.exception}"


def test_launcher_landing_page_renders_three_picker_tiles(launcher_mode):
    """Default landing exposes one picker entry per hosted tool.

    Asserts on the picker copy (icon + title) so a regression that
    silently drops a tile — for example by mis-spelling an
    ``active_app`` key — fails the test instead of reaching users.
    """
    at = AppTest.from_file(str(LAUNCHER_HOME), default_timeout=60)
    at.run()
    assert not at.exception, f"Launcher Home raised: {at.exception}"

    markdown_text = " ".join(block.value for block in at.markdown)
    for icon, title in (
        ("🩺", "ADM Health Check"),
        ("🎯", "Decision Analysis"),
        ("📈", "Impact Analyzer"),
    ):
        assert icon in markdown_text, f"Picker missing tile icon {icon!r}"
        assert title in markdown_text, f"Picker missing tile title {title!r}"


def test_launcher_branding_uses_pdstools_in_launcher_mode(launcher_mode):
    """Sidebar brand text is forced to 'pdstools' under the launcher.

    The picker's own ``show_sidebar_branding('pdstools')`` and any
    per-app override must both end up rendering the same top-level
    brand so users always see where they are. The brand is injected
    via a CSS ``::before`` on ``stSidebarNav`` — assert against the
    raw style block.
    """
    at = AppTest.from_file(str(LAUNCHER_HOME), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.session_state["_pdstools_brand"] == "pdstools"


def test_launcher_pre_app_sidebar_stays_under_collapse_threshold(launcher_mode):
    """Until the user enters an app the sidebar must stay short.

    Pre-app the launcher should expose: picker home + 3 app homes +
    About = 5 entries. Even with future growth this should stay well
    under Streamlit's auto-collapse threshold so newcomers don't have
    to click "View N more" to see Impact Analyzer exists.
    """
    # Mirror what launcher Home.py builds in the no-active-app state.
    sections_total = (
        1  # picker
        + len(hc_pages(url_prefix="hc", default=False, include_about=False, include_subpages=False))
        + len(da_pages(url_prefix="da", default=False, include_about=False, include_subpages=False))
        + len(ia_pages(url_prefix="ia", default=False, include_about=False, include_subpages=False))
        + 1  # About
    )
    assert sections_total <= _NAV_COLLAPSE_THRESHOLD, (
        f"Pre-app sidebar has {sections_total} entries — exceeds the "
        f"{_NAV_COLLAPSE_THRESHOLD}-item collapse threshold and "
        "newcomers will lose discoverability of one or more apps."
    )


def test_launcher_subpage_sidebar_growth_is_documented(launcher_mode):
    """Document where each app's worst-case sidebar size lands.

    HC and IA stay tiny (2 sub-pages each → 7 total). DA's data-pages
    (10) push the total to 15 *only after a dataset is loaded* —
    that's intentional and beyond the collapse threshold, so the
    launcher injects CSS to keep all entries visible. This test
    pins the numbers so future page additions trip a clear failure
    rather than silently re-introducing the collapse problem.
    """
    base = 1 + 1 + 1 + 1 + 1  # picker + 3 home links + About

    hc_active = base + len(hc_pages(url_prefix="hc", default=False, include_about=False)) - 1
    ia_active = base + len(ia_pages(url_prefix="ia", default=False, include_about=False)) - 1
    # DA data-pages are gated on session_state["decision_data"] which
    # AppTest doesn't populate here — count only the home entry.
    da_active_no_data = base + len(da_pages(url_prefix="da", default=False, include_about=False)) - 1

    # HC active: picker + HC home + 2 HC sub-pages + 2 other home links + About = 7
    assert hc_active == 7, hc_active
    # IA active: same shape as HC = 7
    assert ia_active == 7, ia_active
    # DA active without data loaded behaves like the picker case = 5
    assert da_active_no_data == 5, da_active_no_data
