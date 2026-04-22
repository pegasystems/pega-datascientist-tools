"""Auto-load failure path: missing sample must not crash DA Home.

If the bundled DA sample disappears (e.g. wheel packaging mistake or
offline GitHub fallback failure), DA Home should render a graceful
state — locked-pages hint or "please upload" — rather than crashing.

Mocks ``handle_sample_data`` to return ``None`` so the auto-load
branch falls through with no data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def test_da_home_renders_when_sample_missing(
    da_app_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DA Home must not crash when the bundled sample auto-load returns None."""
    monkeypatch.setattr(
        "pdstools.app.decision_analyzer.da_streamlit_utils.handle_sample_data",
        lambda: None,
    )

    at = AppTest.from_file(str(da_app_dir / "Home.py"), default_timeout=60)
    at.run()

    assert not at.exception, f"DA Home must not crash when sample autoload returns None: {at.exception}"
    assert "decision_data" not in at.session_state, (
        "When the sample auto-load fails, decision_data must NOT be populated"
    )

    # The file uploader is the recovery affordance — must still render.
    file_uploaders = at.get("file_uploader") if hasattr(at, "get") else []
    assert list(file_uploaders), "Expected file_uploader to remain available so the user can recover by uploading"
