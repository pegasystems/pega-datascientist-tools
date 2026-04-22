"""Regression test: switching to "Direct file upload" must show uploaders.

Reproduces the bug where the auto-load on first visit (CDH Sample) left
the file_uploader branch unrendered when the user later flipped the
data-source dropdown to "Direct file upload" — because the home page's
"if dm not in session_state, auto-load" guard re-fired on every
selection change and silently put dm + data_source back to CDH Sample,
while the selectbox widget held the user's chosen "Direct file upload"
value.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_direct_file_upload_renders_uploaders(hc_app_dir: Path):
    """Selecting "Direct file upload" renders the model + predictor uploaders.

    First run auto-loads the bundled CDH sample. The second run mimics
    the user picking "Direct file upload": the on_change callback in
    ``import_datamart`` deletes ``dm`` and sets ``data_source`` to the
    new selection. The home page must NOT re-trigger auto-load — it
    must let the upload branch render its file_uploader widgets.
    """
    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"HC Home raised on first run: {at.exception}"
    assert "dm" in at.session_state, "auto-load should have populated dm on first run"
    assert at.session_state["data_source"] == "CDH Sample"

    # Mimic the dropdown's on_change callback: delete dm, switch source.
    del at.session_state["dm"]
    at.session_state["data_source"] = "Direct file upload"
    at.session_state["_data_source"] = "Direct file upload"
    at.run()
    assert not at.exception, f"HC Home raised after switching source: {at.exception}"

    assert "dm" not in at.session_state, "Switching to 'Direct file upload' should not silently re-load CDH Sample"
    assert at.session_state["data_source"] == "Direct file upload"

    uploader_labels = [u.label for u in at.get("file_uploader")]
    assert any("Model Snapshot" in lbl for lbl in uploader_labels), (
        f"expected a 'Model Snapshot' uploader, got: {uploader_labels}"
    )
    assert any("Predictor" in lbl for lbl in uploader_labels), (
        f"expected a 'Predictor' uploader, got: {uploader_labels}"
    )

    success_messages = [s.value for s in at.success]
    assert not any("Import Successful" in m for m in success_messages), (
        f"stale 'Import Successful!' banner should be gone, got: {success_messages}"
    )
