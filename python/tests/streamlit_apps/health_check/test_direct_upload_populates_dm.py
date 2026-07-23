"""Regression test: HC upload must populate ``dm`` end-to-end.

The sibling ``test_direct_upload_renders.py`` only asserts that the
uploader widgets appear immediately. This test goes one step further:
simulate uploading bundled model + predictor zips and assert that
``st.session_state["dm"]`` becomes a real :class:`ADMDatamart` with
non-zero model count.

Together with the renders-test, this closes the gap between "uploader
exists" and "uploader works".
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from pdstools.adm.ADMDatamart import ADMDatamart


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data"
MODEL_ZIP = DATA_DIR / "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip"
PREDICTOR_ZIP = DATA_DIR / "Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip"


def test_direct_upload_populates_dm(hc_app_dir: Path) -> None:
    """Uploading model + predictor zips through the HC home page must populate ``dm``.

    Steps:
    1. Run Home — auto-load lands a CDH-Sample ``dm``.
        2. Drop the bundled model + predictor zips into the
       two file_uploader widgets.
        3. Verify upload alone does not import, then click ``Import``.
        4. Assert ``dm`` is a real ``ADMDatamart`` with non-zero models.
    """
    if not MODEL_ZIP.exists() or not PREDICTOR_ZIP.exists():
        pytest.skip(f"Required HC sample zips missing in {DATA_DIR}")

    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=120)
    at.run()
    assert not at.exception, f"HC Home raised on first run: {at.exception}"
    assert "dm" in at.session_state

    autoloaded_dm = at.session_state["dm"]

    # Find the model + predictor uploaders by label and drop the bundled zips in.
    uploaders = {u.label: u for u in at.get("file_uploader")}
    model_uploader = next((u for lbl, u in uploaders.items() if "Model Snapshot" in lbl), None)
    predictor_uploader = next((u for lbl, u in uploaders.items() if "Predictor" in lbl), None)
    assert model_uploader is not None and predictor_uploader is not None, (
        f"Expected Model + Predictor uploaders, got: {list(uploaders)}"
    )

    model_uploader.upload(MODEL_ZIP.name, MODEL_ZIP.read_bytes(), "application/zip")
    predictor_uploader.upload(PREDICTOR_ZIP.name, PREDICTOR_ZIP.read_bytes(), "application/zip")
    at.run()

    assert not at.exception, f"HC Home raised after upload: {at.exception}"
    assert "dm" not in at.session_state, "Uploading files must wait for the explicit Import action"

    category_mapping = next(text_area for text_area in at.text_area if text_area.label == "Predictor category mappings")
    category_mapping.set_value("Uploaded=.*")
    regex_checkbox = next(
        checkbox for checkbox in at.checkbox if checkbox.label == "Use regular expressions for predictor mappings"
    )
    regex_checkbox.set_value(True)
    at.run()

    import_button = next(button for button in at.button if button.label == "Import")
    import_button.click().run()

    assert not at.exception, f"HC Home raised after import: {at.exception}"
    assert "dm" in at.session_state, (
        "After uploading model + predictor zips, st.session_state['dm'] must be populated; "
        f"errors: {[error.value for error in at.error]}"
    )
    dm = at.session_state["dm"]
    assert isinstance(dm, ADMDatamart), f"Expected ADMDatamart, got {type(dm).__name__}"
    assert dm is not autoloaded_dm, (
        "The post-upload datamart must be a freshly built instance, not the same object as the autoloaded CDH Sample"
    )

    model_count = dm.aggregates.last().select("ModelID").unique().collect().height
    assert model_count > 0, f"Uploaded datamart must have at least one model, got {model_count}"
    categories = (
        dm.predictor_data.select("PredictorCategory").filter(pl.col("PredictorCategory") == "Uploaded").collect()
    )
    assert categories.height > 0
