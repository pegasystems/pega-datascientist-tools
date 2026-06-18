"""Wrong-format upload: DA must surface a clear error and preserve autoload.

The DA file_uploader has ``type=["zip", "parquet", "json", "csv",
"arrow", "gz", "tgz", "tar"]``. A user can still bypass the
extension check by renaming a file (or just dropping a corrupt file
with a whitelisted extension). The page must surface a graceful
error rather than crashing with a Python stack trace.
"""

from __future__ import annotations


import pytest
from streamlit.testing.v1 import AppTest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_da_garbage_parquet_upload_handled_gracefully(da_app_dir: Path) -> None:
    """A ``.parquet`` file with garbage bytes must surface an error, not crash.

    The extension passes the uploader's allow-list but the bytes
    aren't a valid parquet — the parser must reject and the page must
    render an ``st.error`` (not raise an unhandled ComputeError).

    Currently ``xfail``ed because ``raw_data.explain(...)`` is called
    outside the try/except in ``_home_page.py``, so polars
    ``ComputeError`` escapes as an unhandled exception. Tracked in
    ``docs/plans/decision-analyzer/garbage-upload-crashes-home.md``.
    """
    at = AppTest.from_file(str(da_app_dir / "Home.py"), default_timeout=120)
    at.run()
    assert not at.exception, f"Initial run raised: {at.exception}"
    assert "decision_data" in at.session_state

    uploaders = at.get("file_uploader")
    assert list(uploaders), "Expected a file_uploader on Home"

    junk_bytes = b"PAR-not-actually-parquet\x00\x00\x00\x00"
    uploaders[0].upload("nonsense.parquet", junk_bytes, "application/octet-stream")
    at.run()

    assert not at.exception, f"DA Home must not crash on a garbage .parquet upload, got: {at.exception}"
    error_surfaced = bool(at.error) or bool(at.warning)
    assert error_surfaced, (
        "A garbage .parquet upload must surface an st.error/st.warning so the user "
        "knows the upload was rejected by the parser"
    )


test_da_garbage_parquet_upload_handled_gracefully = pytest.mark.xfail(
    reason=(
        "DA Home does not catch polars ComputeError from a corrupt parquet upload — "
        "see docs/plans/decision-analyzer/garbage-upload-crashes-home.md"
    ),
    strict=False,
)(test_da_garbage_parquet_upload_handled_gracefully)
