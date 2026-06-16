"""Wrong-format upload: HC must reject non-zip bytes gracefully.

Mirrors ``decision_analyzer/test_da_wrong_format_upload.py``. After
switching to "Direct file upload", dropping garbage bytes with a
whitelisted extension into the Model + Predictor uploaders must not
crash the page — the parser in ``cached_datamart`` catches the
exception and surfaces an ``st.error``, leaving ``dm`` unset so the
user can retry.
"""

from __future__ import annotations


from streamlit.testing.v1 import AppTest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_hc_garbage_zip_upload_handled_gracefully(hc_app_dir: Path) -> None:
    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"Initial HC run raised: {at.exception}"

    # Switch to "Direct file upload" so the model+predictor uploaders render.
    del at.session_state["dm"]
    at.session_state["data_source"] = "Direct file upload"
    at.session_state["_data_source"] = "Direct file upload"
    at.run()
    assert not at.exception, f"HC Home raised after switching source: {at.exception}"

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

    assert not at.exception, f"HC Home must not crash on garbage zip uploads, got: {at.exception}"
    # Either dm stays unset (cached_datamart returned None) OR the
    # parser surfaced an error/warning. A silent success would mean
    # we accidentally produced an empty datamart.
    no_dm = "dm" not in at.session_state
    error_surfaced = bool(at.error) or bool(at.warning)
    assert no_dm or error_surfaced, (
        "Garbage zip upload must either leave dm unset or surface an error/warning. "
        f"dm in state: {'dm' in at.session_state}, errors: {[e.value for e in at.error]}, "
        f"warnings: {[w.value for w in at.warning]}"
    )
