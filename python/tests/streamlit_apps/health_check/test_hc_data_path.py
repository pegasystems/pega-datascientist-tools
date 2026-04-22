"""Tests for ``handle_data_path_hc``.

Exercises the per-app ``--data-path`` wiring: nothing configured,
nonexistent path, directory autodetection (full / model-only), zip
extraction, and unsupported single-file rejection.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from pdstools.app.health_check import hc_streamlit_utils as hc_utils

DATA_DIR = Path(__file__).resolve().parents[4] / "data"
ADM_MODEL_FILE = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip"
ADM_PREDICTOR_FILE = "Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip"


def _require_sample_files() -> None:
    if not (DATA_DIR / ADM_MODEL_FILE).exists():
        pytest.skip(f"ADM sample data missing: {DATA_DIR / ADM_MODEL_FILE}")


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """Directory containing all three Pega export files (model+predictor)."""
    _require_sample_files()
    shutil.copy(DATA_DIR / ADM_MODEL_FILE, tmp_path / ADM_MODEL_FILE)
    shutil.copy(DATA_DIR / ADM_PREDICTOR_FILE, tmp_path / ADM_PREDICTOR_FILE)
    return tmp_path


@pytest.fixture
def model_only_dir(tmp_path: Path) -> Path:
    _require_sample_files()
    shutil.copy(DATA_DIR / ADM_MODEL_FILE, tmp_path / ADM_MODEL_FILE)
    return tmp_path


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture(autouse=True)
def _clear_streamlit_state(monkeypatch):
    """Each test gets a clean session_state and no PDSTOOLS_DATA_PATH leak."""
    import streamlit as st

    monkeypatch.delenv("PDSTOOLS_DATA_PATH", raising=False)
    st.session_state.clear()
    st.cache_resource.clear()
    yield
    st.session_state.clear()
    st.cache_resource.clear()


def test_returns_none_when_no_data_path_set():
    assert hc_utils.handle_data_path_hc() is None


def test_returns_none_when_path_does_not_exist(monkeypatch):
    monkeypatch.setenv("PDSTOOLS_DATA_PATH", "/nonexistent/path/does/not/exist")
    assert hc_utils.handle_data_path_hc() is None


def test_directory_with_all_files_loads_dm(monkeypatch, sample_dir: Path):
    import streamlit as st

    monkeypatch.setenv("PDSTOOLS_DATA_PATH", str(sample_dir))
    dm = hc_utils.handle_data_path_hc()
    assert dm is not None
    assert st.session_state["dm"] is dm
    assert st.session_state["data_source"] == "Direct file path"
    # Predictor was found, so the dm has a non-empty predictor frame.
    assert dm.predictor_data is not None


def test_directory_model_only_loads_dm_without_predictor(monkeypatch, model_only_dir: Path):
    import streamlit as st

    monkeypatch.setenv("PDSTOOLS_DATA_PATH", str(model_only_dir))
    dm = hc_utils.handle_data_path_hc()
    assert dm is not None
    assert st.session_state["dm"] is dm
    assert st.session_state["data_source"] == "Direct file path"


def test_empty_directory_returns_none_with_error(monkeypatch, empty_dir: Path):
    monkeypatch.setenv("PDSTOOLS_DATA_PATH", str(empty_dir))
    with mock.patch("streamlit.error") as mock_err:
        dm = hc_utils.handle_data_path_hc()
    assert dm is None
    assert mock_err.called
    assert "model snapshot" in mock_err.call_args[0][0].lower()


def test_zip_file_extracts_and_loads(monkeypatch, sample_dir: Path, tmp_path: Path):
    import streamlit as st

    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in (ADM_MODEL_FILE, ADM_PREDICTOR_FILE):
            zf.write(sample_dir / name, arcname=name)

    monkeypatch.setenv("PDSTOOLS_DATA_PATH", str(zip_path))
    dm = hc_utils.handle_data_path_hc()
    assert dm is not None
    assert st.session_state["dm"] is dm
    assert st.session_state["data_source"] == "Direct file path"


def test_unsupported_single_file_returns_none_with_error(monkeypatch, tmp_path: Path):
    bogus = tmp_path / "data.parquet"
    bogus.write_bytes(b"not a real parquet")
    monkeypatch.setenv("PDSTOOLS_DATA_PATH", str(bogus))
    with mock.patch("streamlit.error") as mock_err:
        dm = hc_utils.handle_data_path_hc()
    assert dm is None
    assert mock_err.called
    msg = mock_err.call_args[0][0]
    assert "directory or zip" in msg.lower()
