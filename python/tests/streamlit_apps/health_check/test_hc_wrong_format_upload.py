"""Wrong-format upload: HC must reject non-zip bytes gracefully."""

from __future__ import annotations


from streamlit.testing.v1 import AppTest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_hc_garbage_zip_upload_handled_gracefully(hc_app_dir: Path) -> None:
    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"Initial HC run raised: {at.exception}"

    uploaders = {u.label: u for u in at.get("file_uploader")}
    assert list(uploaders) == [
        "Upload Model Snapshot",
        "Upload Predictor Binning snapshot (optional)",
        "Upload Prediction Table (optional)",
    ]
    model_uploader = uploaders["Upload Model Snapshot"]
    predictor_uploader = uploaders["Upload Predictor Binning snapshot (optional)"]

    # ZIPs whose bytes aren't a real archive — the parser catches this
    # and ``cached_datamart`` returns None.
    junk = b"not a zip\n"
    model_uploader.upload("nope.zip", junk, "application/zip")
    predictor_uploader.upload("nope.zip", junk, "application/zip")
    at.run()

    import_button = next(button for button in at.button if button.label == "Import")
    import_button.click().run()

    assert not at.exception, f"HC Home must not crash on garbage zip uploads, got: {at.exception}"
    assert "dm" not in at.session_state
    assert any(error.value.startswith("Import failed:") for error in at.error)
