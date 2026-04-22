"""Auto-load failure path: missing sample must surface a graceful message.

If the bundled sample disappears (e.g. a packaging mistake leaves
``data/`` out of the wheel, or the network is offline and the GitHub
fallback fails), HC's auto-load should surface a user-facing
"please configure your files" warning rather than crashing or
silently rendering a blank page.

Mocks ``cached_sample`` to raise so the auto-load goes through its
exception branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def _boom() -> object:
    raise FileNotFoundError("simulated missing sample bundle")


def test_hc_home_renders_when_sample_missing(
    hc_app_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HC Home must not crash when the bundled sample fails to load.

    ``ensure_dm_loaded`` catches the exception and returns False; the
    home page should therefore not have ``dm`` in session_state, and
    must not raise. The data-source dropdown must still render so
    the user can recover by uploading manually.
    """
    monkeypatch.setattr(
        "pdstools.app.health_check.hc_streamlit_utils.cached_sample",
        _boom,
    )
    # ``cached_sample`` is also imported into the shared
    # ``streamlit_utils`` module under the same name; HC's
    # ``import_datamart`` reads ``data_source`` then re-resolves it
    # there, so patch both call sites.
    monkeypatch.setattr(
        "pdstools.utils.streamlit_utils.cached_sample",
        _boom,
    )

    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=60)
    at.run()

    assert not at.exception, f"HC Home must not crash when sample autoload fails: {at.exception}"
    assert "dm" not in at.session_state, "When the sample fails to load, dm must NOT end up in session_state"

    # The data-source dropdown is the recovery affordance — must still render.
    selectbox_keys = [sb.key for sb in at.selectbox]
    assert "_data_source" in selectbox_keys, (
        f"Expected '_data_source' selectbox to remain available as a recovery "
        f"affordance, got selectboxes: {selectbox_keys}"
    )
