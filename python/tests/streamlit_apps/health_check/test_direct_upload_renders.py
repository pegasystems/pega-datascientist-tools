"""Regression test: Health Check uploaders are available immediately."""

from __future__ import annotations


from streamlit.testing.v1 import AppTest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_direct_file_upload_renders_uploaders(hc_app_dir: Path):
    """First render shows uploaders without requiring a source selector."""
    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"HC Home raised on first run: {at.exception}"
    assert "dm" in at.session_state, "auto-load should have populated dm on first run"
    assert at.session_state["data_source"] == "CDH Sample"
    assert "_data_source" not in [selectbox.key for selectbox in at.selectbox]

    uploader_labels = [u.label for u in at.get("file_uploader")]
    assert any("Model Snapshot" in lbl for lbl in uploader_labels), (
        f"expected a 'Model Snapshot' uploader, got: {uploader_labels}"
    )
    assert any("Predictor" in lbl for lbl in uploader_labels), (
        f"expected a 'Predictor' uploader, got: {uploader_labels}"
    )

    assert [expander.label for expander in at.expander] == [
        "File paths (optional)",
        "Predictor categorization (optional)",
        "Processed parquet cache (optional)",
        "Advanced import settings (optional)",
    ]
    predictor_tip = " ".join(info.value for info in at.info)
    assert "literal substring matches" in predictor_tip
    assert "ADMDatamart.apply_predictor_categorization" in predictor_tip

    success_messages = [s.value for s in at.success]
    assert not any("Import Successful" in m for m in success_messages), (
        f"stale 'Import Successful!' banner should be gone, got: {success_messages}"
    )
