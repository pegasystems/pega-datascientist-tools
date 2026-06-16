"""Regression test: HC "Direct file upload" must populate ``dm`` end-to-end.

The sibling ``test_direct_upload_renders.py`` only asserts that the
uploader widgets *appear* when the data-source dropdown is flipped.
This test goes one step further: simulate uploading bundled model +
predictor zips and assert that ``st.session_state["dm"]`` becomes a
real :class:`ADMDatamart` with non-zero model count.

Together with the renders-test, this closes the gap between "uploader
exists" and "uploader works".
"""

from __future__ import annotations

from pathlib import Path

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
    2. Switch to "Direct file upload" (delete dm, set data_source).
    3. Re-run, then drop the bundled model + predictor zips into the
       two file_uploader widgets.
    4. Re-run again — assert ``dm`` is a real ``ADMDatamart`` with
       non-zero models.

    .. note::

       Currently ``xfail``ed because ``ADMDatamart.from_ds_export``
       fails schema inference on in-memory ZIP readers (the path the
       Streamlit uploader takes). See
       ``docs/plans/health-check/in-memory-zip-upload-schema-error.md``
       — once that lands, drop the xfail marker.
    """
    if not MODEL_ZIP.exists() or not PREDICTOR_ZIP.exists():
        pytest.skip(f"Required HC sample zips missing in {DATA_DIR}")

    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=120)
    at.run()
    assert not at.exception, f"HC Home raised on first run: {at.exception}"
    assert "dm" in at.session_state

    autoloaded_dm = at.session_state["dm"]

    # Switch the dropdown via its on_change semantics (mirrors
    # test_direct_upload_renders.py).
    del at.session_state["dm"]
    at.session_state["data_source"] = "Direct file upload"
    at.session_state["_data_source"] = "Direct file upload"
    at.run()
    assert not at.exception, f"HC Home raised after switching source: {at.exception}"
    assert "dm" not in at.session_state, "Switching to 'Direct file upload' must not silently re-load CDH Sample"

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
    assert "dm" in at.session_state, "After uploading model + predictor zips, st.session_state['dm'] must be populated"
    dm = at.session_state["dm"]
    assert isinstance(dm, ADMDatamart), f"Expected ADMDatamart, got {type(dm).__name__}"
    assert dm is not autoloaded_dm, (
        "The post-upload datamart must be a freshly built instance, not the same object as the autoloaded CDH Sample"
    )

    model_count = dm.aggregates.last().select("ModelID").unique().collect().height
    assert model_count > 0, f"Uploaded datamart must have at least one model, got {model_count}"


# The "uploader populates dm end-to-end" assertion fails today because
# in-memory ZIP ingest hits a polars schema-inference bug (see plan
# file). Mark it xfail so the test runs in CI and starts passing
# automatically once the parser is fixed — at which point the marker
# (and the plan file) should be removed.
test_direct_upload_populates_dm = pytest.mark.xfail(
    reason=(
        "ADMDatamart.from_ds_export fails schema inference on in-memory "
        "ZIP readers — see docs/plans/health-check/"
        "in-memory-zip-upload-schema-error.md"
    ),
    strict=False,
)(test_direct_upload_populates_dm)
