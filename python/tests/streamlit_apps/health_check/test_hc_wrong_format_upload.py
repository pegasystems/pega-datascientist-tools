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
    model_uploader = next((u for lbl, u in uploaders.items() if "Model Snapshot" in lbl), None)
    predictor_uploader = next((u for lbl, u in uploaders.items() if "Predictor" in lbl), None)
    assert model_uploader is not None and predictor_uploader is not None, (
        f"Expected Model + Predictor uploaders, got: {list(uploaders)}"
    )

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
