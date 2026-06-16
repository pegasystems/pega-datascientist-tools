"""Regression test: explicit upload must override the autoloaded sample.

PR #762 added an autoload of the bundled sample on first visit so the
home page is no longer blank. A side-effect: the file uploader's
persistent value re-fired ``handle_file_upload`` on every rerun, which
called ``clear_decision_state()`` and reloaded the analyzer. Combined
with the post-load ``st.rerun()``, this looped indefinitely (rendering
multiple "Data loaded" banners) and left ``decision_data`` reflecting
whichever load won the race — typically the autoloaded sample, not the
user's upload.

This test seeds the autoload sample, then simulates a file_uploader
value change to a small synthetic parquet and asserts that
``session_state.decision_data`` reflects the uploaded file's row count,
not the sample's.
"""

from __future__ import annotations


import polars as pl
import pytest
from streamlit.testing.v1 import AppTest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


# Distinctive row count — must not collide with the bundled sample's
# size, the minimal CSV fixture's size, or any other dataset that
# appears in DA tests, so the assertion proves which file actually
# made it into ``decision_data``.
_FIXTURE_ROW_COUNT = 17


@pytest.fixture
def upload_fixture_bytes(tmp_path: Path, da_sample_path: Path) -> bytes:
    """Tiny EEV2-shaped parquet for upload simulation.

    Reuses the minimal CSV fixture's schema and trims to a unique row
    count so the test can distinguish "loaded the upload" from "still
    showing the autoload sample".
    """
    df = pl.read_csv(da_sample_path).head(_FIXTURE_ROW_COUNT)
    if df.height < _FIXTURE_ROW_COUNT:
        # Repeat rows if the fixture is smaller than our target — we
        # only care that the row count is distinctive, not the values.
        repeats = (_FIXTURE_ROW_COUNT // max(df.height, 1)) + 1
        df = pl.concat([df] * repeats).head(_FIXTURE_ROW_COUNT)
    out = tmp_path / "upload_fixture.parquet"
    df.write_parquet(out)
    return out.read_bytes()


def test_upload_replaces_autoloaded_sample(da_app_dir: Path, upload_fixture_bytes: bytes) -> None:
    """Dropping a file into the uploader must replace the autoloaded sample.

    Steps:
    1. Run Home — sample autoloads, ``decision_data`` populated.
    2. Simulate the user dropping a small parquet into the uploader.
    3. Re-run — the analyzer must rebuild from the upload, not the sample.

    Asserts:
    - No exception bubbled out.
    - ``decision_data`` row count equals the upload fixture's row
      count (proves the upload won, not the autoloaded sample).
    - Exactly one ``st.success`` "Data loaded" banner is rendered on
      the post-upload run (no rerun-loop duplicate alerts).
    """
    at = AppTest.from_file(str(da_app_dir / "Home.py"), default_timeout=120)
    at.run()
    assert not at.exception, f"Initial autoload raised: {at.exception}"
    assert "decision_data" in at.session_state, "Autoload should populate decision_data"

    autoload_rows = at.session_state["decision_data"].decision_data.select(pl.len()).collect().item()
    assert autoload_rows != _FIXTURE_ROW_COUNT, (
        "Test fixture row count must differ from the autoload sample's row count to be a meaningful assertion"
    )

    uploaders = at.get("file_uploader")
    assert list(uploaders), "Expected a file_uploader on the Home page"
    uploaders[0].upload("upload_fixture.parquet", upload_fixture_bytes, "application/octet-stream")
    at.run()

    assert not at.exception, f"Post-upload run raised: {at.exception}"
    assert "decision_data" in at.session_state

    upload_rows = at.session_state["decision_data"].decision_data.select(pl.len()).collect().item()
    assert upload_rows == _FIXTURE_ROW_COUNT, (
        f"Expected uploaded file ({_FIXTURE_ROW_COUNT} rows) to override autoloaded "
        f"sample ({autoload_rows} rows), but decision_data has {upload_rows} rows"
    )

    success_banners = [s for s in at.success if "data loaded successfully" in str(s.value).lower()]
    assert len(success_banners) == 1, (
        f"Expected exactly one 'Data loaded successfully' banner, got {len(success_banners)}: "
        f"{[s.value for s in success_banners]}"
    )

    # A subsequent rerun without changing the uploader must be a no-op
    # for ingest — the dedup guard should short-circuit. If it didn't,
    # decision_data would either be cleared or rebuilt with a fresh
    # cache key (still 17 rows, but a different object identity).
    da_id_before = id(at.session_state["decision_data"])
    at.run()
    assert not at.exception, f"Idle rerun raised: {at.exception}"
    assert id(at.session_state["decision_data"]) == da_id_before, (
        "Idle rerun must not rebuild the analyzer when the upload signature is unchanged"
    )
