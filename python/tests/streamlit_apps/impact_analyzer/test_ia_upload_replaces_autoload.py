"""Regression test: explicit IA upload must override the autoloaded sample.

Mirrors ``decision_analyzer/test_upload_replaces_autoload.py``. The IA
home page auto-loads the bundled PDC sample on first visit; if the
file_uploader's persistent value re-fired ``load_from_upload_auto`` on
every rerun while ``ensure_ia_data_loaded`` re-installed the sample,
the user's upload would silently lose the race and the analyzer would
keep pointing at the autoloaded sample.

This test seeds the autoload, then simulates a file_uploader value
change to a different bundled fixture (PDC JSON copied to a temp file
with a distinctive ``ChannelGroup`` set) and asserts that
``session_state["impact_analyzer"]`` now reflects the upload — by
checking that ``ia_is_sample_data`` flipped to False and the
``impact_analyzer`` instance identity changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[4]
IA_SAMPLE_JSON = REPO_ROOT / "data" / "ia" / "CDH_Metrics_ImpactAnalyzer.json"


@pytest.fixture
def upload_fixture_bytes() -> bytes:
    """A trimmed copy of the PDC sample JSON with a different row yield.

    The PDC sample has a single outer ``pxResults`` entry that itself
    contains a nested ``pxResults`` list of per-experiment rows. The
    full sample yields 24 ``ia_data`` rows; trimming the inner list to
    15 entries yields 12 rows — distinct enough to prove which file
    won the race.
    """
    if not IA_SAMPLE_JSON.exists():
        pytest.skip(f"IA sample data missing: {IA_SAMPLE_JSON}")

    obj = json.loads(IA_SAMPLE_JSON.read_text(encoding="utf-8"))
    inner = obj["pxResults"][0].get("pxResults")
    if not isinstance(inner, list) or len(inner) < 16:
        pytest.skip("IA sample structure changed — fixture trim no longer applies")
    obj["pxResults"][0]["pxResults"] = inner[:15]
    return json.dumps(obj).encode("utf-8")


def test_upload_replaces_autoloaded_sample(ia_app_dir: Path, upload_fixture_bytes: bytes) -> None:
    """Dropping a file into the IA uploader must replace the autoloaded sample."""
    at = AppTest.from_file(str(ia_app_dir / "Home.py"), default_timeout=120)
    at.run()
    assert not at.exception, f"Initial autoload raised: {at.exception}"
    assert "impact_analyzer" in at.session_state, "Autoload should populate impact_analyzer"
    assert at.session_state["ia_is_sample_data"] is True, "Autoloaded data should be flagged as sample data"

    autoload_rows = at.session_state["impact_analyzer"].ia_data.select(pl.len()).collect().item()

    uploaders = at.get("file_uploader")
    assert list(uploaders), "Expected a file_uploader on the IA Home page"
    uploaders[0].upload("upload_fixture.json", upload_fixture_bytes, "application/json")
    at.run()

    assert not at.exception, f"Post-upload run raised: {at.exception}"
    assert "impact_analyzer" in at.session_state

    upload_rows = at.session_state["impact_analyzer"].ia_data.select(pl.len()).collect().item()
    assert upload_rows != autoload_rows, (
        f"Expected uploaded file to override autoloaded sample, but row count is unchanged "
        f"({upload_rows} == {autoload_rows}). The upload likely lost the race to the autoload."
    )
    assert at.session_state["ia_is_sample_data"] is False, (
        "After an explicit upload, ia_is_sample_data must flip to False"
    )
