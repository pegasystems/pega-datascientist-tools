"""Regression test: the "Import Successful!" banner must be transient.

The banner used to be driven by the persistent ``_hc_import_result``
session key, so it re-appeared on every rerun and page navigation after
the first import — making it look like an import had just happened even
when the user had only navigated back to the page. It must now show only
on the exact run that performed the import; subsequent runs fall back to
a status banner that names the currently-loaded data source.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data"
MODEL_ZIP = DATA_DIR / "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip"
PREDICTOR_ZIP = DATA_DIR / "Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip"


def test_import_success_banner_is_transient(hc_app_dir: Path) -> None:
    """ "Import Successful!" shows once, then yields to an accurate status banner."""
    if not MODEL_ZIP.exists() or not PREDICTOR_ZIP.exists():
        pytest.skip(f"Required HC sample zips missing in {DATA_DIR}")

    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=120)
    at.run()
    assert not at.exception, f"HC Home raised on first run: {at.exception}"

    uploaders = {u.label: u for u in at.get("file_uploader")}
    model_uploader = next(u for lbl, u in uploaders.items() if "Model Snapshot" in lbl)
    predictor_uploader = next(u for lbl, u in uploaders.items() if "Predictor" in lbl)
    model_uploader.upload(MODEL_ZIP.name, MODEL_ZIP.read_bytes(), "application/zip")
    predictor_uploader.upload(PREDICTOR_ZIP.name, PREDICTOR_ZIP.read_bytes(), "application/zip")
    at.run()

    import_button = next(button for button in at.button if button.label == "Import")
    import_button.click().run()
    assert not at.exception, f"HC Home raised after import: {at.exception}"

    # On the import run itself, the success banner is shown.
    assert any("Import Successful" in s.value for s in at.success), (
        f"expected 'Import Successful!' on the import run, got: {[s.value for s in at.success]}"
    )
    assert at.session_state["_hc_import_result"] is not None

    # A subsequent rerun (e.g. navigating back to the page) must NOT keep the
    # stale success banner; instead it shows the imported-data status.
    at.run()
    assert not at.exception, f"HC Home raised on rerun: {at.exception}"
    assert not any("Import Successful" in s.value for s in at.success), (
        f"'Import Successful!' must not persist across reruns, got: {[s.value for s in at.success]}"
    )
    info_messages = [i.value for i in at.info]
    assert any("Your imported data is currently loaded" in m for m in info_messages), (
        f"expected an imported-data status banner on rerun, got: {info_messages}"
    )
