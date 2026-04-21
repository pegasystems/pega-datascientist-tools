"""AppTest smoke tests for the Decision Analysis Home page.

Home is the data-entry point — it doesn't depend on prior
``session_state``, so these tests exercise the "no data loaded yet"
rendering path. Pages 2+ are tested with ``session_state.decision_data``
seeded via the ``seeded_decision_analyzer`` fixture.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_home_renders_without_data(da_app_dir: Path):
    """Home page renders cleanly before any data is uploaded.

    This is the first thing a user sees — if it raises we have a
    boot-time regression. Checks:
    - No exception bubbles out of the script.
    - The page branding markdown appears (proves
      ``show_sidebar_branding`` and the first ``st.write`` ran).
    - The sample-size ``number_input`` rendered.
    - A ``file_uploader`` widget rendered so the user actually has
      a way to load data.
    """
    at = AppTest.from_file(str(da_app_dir / "Home.py"), default_timeout=30)
    at.run()

    assert not at.exception, f"Home raised: {at.exception}"
    headings = [str(h.value) for h in at.title] + [str(h.value) for h in at.header]
    headings += [str(m.value) for m in at.markdown]
    assert any("Decision Analysis" in s for s in headings), "Expected 'Decision Analysis' header/title not found"
    assert list(at.number_input), "Expected at least one number_input (sample size)"
    file_uploaders = at.get("file_uploader") if hasattr(at, "get") else []
    assert list(file_uploaders), "Expected a file_uploader widget for data entry"
