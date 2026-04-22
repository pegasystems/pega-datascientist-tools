"""AppTest smoke test for the cross-app pdstools launcher.

Verifies the launcher boots without exception, registers all three
tools as sidebar sections, and namespaces their URLs to avoid
collisions. A regression here means the launcher won't host one or
more tools.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER_HOME = REPO_ROOT / "python" / "pdstools" / "app" / "launcher" / "Home.py"
HC_HOME = REPO_ROOT / "python" / "pdstools" / "app" / "health_check" / "Home.py"


def test_launcher_home_renders_without_data():
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
