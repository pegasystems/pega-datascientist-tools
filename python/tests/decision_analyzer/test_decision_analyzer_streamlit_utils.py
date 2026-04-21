"""Tests for da_streamlit_utils helper functions.

Only pure-Python helpers are tested here; Streamlit UI rendering is not
exercised. Session-state clearing is tested by injecting a mock session
state dict rather than launching a Streamlit server.
"""

from __future__ import annotations

import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from pdstools.app.decision_analyzer.da_streamlit_utils import (
    _DECISION_STATE_KEYS,
    _is_upload_noise,
    clear_decision_state,
)


# ---------------------------------------------------------------------------
# _is_upload_noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        # META-INF entries
        ("META-INF/MANIFEST.MF", True),
        ("META-INF/manifest.mf", True),
        ("META-INF/", True),
        # macOS resource fork directory
        ("__MACOSX/._data.csv", True),
        ("__MACOSX/", True),
        # OS artefacts
        (".DS_Store", True),
        ("Thumbs.db", True),
        ("desktop.ini", True),
        # Nested noise path
        ("some/nested/__MACOSX/file.txt", True),
        ("some/META-INF/entry", True),
        # Valid data files — must NOT be treated as noise
        ("data.csv", False),
        ("data.parquet", False),
        ("data.json", False),
        ("export.zip", False),
        ("subdir/data.arrow", False),
        # Filename that merely *contains* meta-inf as substring (not a component)
        ("my_meta_inf_data.csv", False),
    ],
)
def test_is_upload_noise(name: str, expected: bool) -> None:
    assert _is_upload_noise(name) is expected


# ---------------------------------------------------------------------------
# clear_decision_state
# ---------------------------------------------------------------------------


def _make_session_state(extra: dict | None = None) -> dict:
    """Build a mock session-state dict with known DA keys."""
    state: dict = {
        "decision_data": object(),
        "filters": ["some_filter"],
        "sample_metadata": {"foo": "bar"},
        "page_channel_filter": "Web/Inbound",
        "page_channel_expr": MagicMock(),
        # Filter widget keys that should also be cleared
        "globalselected_Action": ["Offer A"],
        "local_selected_Channel": ["Web"],
        "globalmultiselect": ["Action"],
        "categories_Stage": ["Arbitration"],
        # Keys that should survive
        "unrelated_key": "should_survive",
    }
    if extra:
        state.update(extra)
    return state


def test_clear_decision_state_removes_primary_keys() -> None:
    """All _DECISION_STATE_KEYS are removed."""
    state = _make_session_state()

    with patch("streamlit.session_state", state):
        clear_decision_state()

    for key in _DECISION_STATE_KEYS:
        assert key not in state, f"Expected '{key}' to be removed"


def test_clear_decision_state_removes_filter_widget_keys() -> None:
    """Filter widget state (selected_*, *multiselect*, categories_*) is removed."""
    state = _make_session_state()

    with patch("streamlit.session_state", state):
        clear_decision_state()

    filter_keys = [k for k in state if "selected_" in k or "multiselect" in k or "categories_" in k]
    assert filter_keys == [], f"Filter widget keys not cleared: {filter_keys}"


def test_clear_decision_state_preserves_unrelated_keys() -> None:
    """Keys unrelated to Decision Analyzer are not touched."""
    state = _make_session_state()

    with patch("streamlit.session_state", state):
        clear_decision_state()

    assert state.get("unrelated_key") == "should_survive"


def test_clear_decision_state_is_idempotent() -> None:
    """Calling clear_decision_state twice does not raise."""
    state = _make_session_state()
    with patch("streamlit.session_state", state):
        clear_decision_state()
        clear_decision_state()  # second call on already-cleared state


def test_clear_decision_state_handles_empty_session() -> None:
    """clear_decision_state is safe when no keys are present."""
    state: dict = {}
    with patch("streamlit.session_state", state):
        clear_decision_state()
    assert state == {}


# ---------------------------------------------------------------------------
# _read_uploaded_zip — noise filtering inside archives
# ---------------------------------------------------------------------------


def _make_zip(file_entries: dict[str, bytes]) -> io.BytesIO:
    """Return an in-memory ZIP containing the given entries."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in file_entries.items():
            zf.writestr(name, content)
    buf.seek(0)
    return buf


def test_read_uploaded_zip_filters_manifest() -> None:
    """A ZIP with only a META-INF/MANIFEST.MF plus a data file reads successfully."""
    # We can't easily test the full read path without a running Streamlit context,
    # but we can verify that the noise-filtering logic inside _read_uploaded_zip
    # would NOT raise on a ZIP that contains both a manifest and a CSV.
    # The inner_names calculation is the key part — confirm it excludes META-INF.
    buf = _make_zip(
        {
            "META-INF/MANIFEST.MF": b"Manifest-Version: 1.0\n",
            "__MACOSX/._data.csv": b"",
            "data.csv": b"col1,col2\n1,2\n",
        }
    )
    with zipfile.ZipFile(buf, "r") as zf:
        inner_names = [n for n in zf.namelist() if not _is_upload_noise(n)]

    assert inner_names == ["data.csv"]


def test_read_uploaded_zip_all_noise_gives_empty_inner_names() -> None:
    """A ZIP containing only noise files results in an empty inner_names list."""
    buf = _make_zip(
        {
            "META-INF/MANIFEST.MF": b"Manifest-Version: 1.0\n",
            "__MACOSX/._data.csv": b"",
            ".DS_Store": b"",
        }
    )
    with zipfile.ZipFile(buf, "r") as zf:
        inner_names = [n for n in zf.namelist() if not _is_upload_noise(n)]

    assert inner_names == []
