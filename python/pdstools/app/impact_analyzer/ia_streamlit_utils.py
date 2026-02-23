# python/pdstools/app/impact_analyzer/ia_streamlit_utils.py
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

import streamlit as st

from pdstools import ImpactAnalyzer
from pdstools.utils.streamlit_utils import ensure_session_data


SAMPLE_PDC_URL = "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data/ia/CDH_Metrics_ImpactAnalyzer.json"


def ensure_impact_analyzer() -> None:
    """Guard: stop if Impact Analyzer data is not loaded."""
    ensure_session_data("impact_analyzer")


def _write_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".json"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _write_uploaded_files(uploaded_files: Iterable) -> List[str]:
    return [_write_uploaded_file(uploaded_file) for uploaded_file in uploaded_files]


def _resolve_sample_pdc_path() -> Path:
    local_path = (
        Path(__file__).resolve().parents[4]
        / "data"
        / "ia"
        / "CDH_Metrics_ImpactAnalyzer.json"
    )
    if local_path.exists():
        return local_path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        with urllib.request.urlopen(SAMPLE_PDC_URL) as response:
            tmp.write(response.read())
        return Path(tmp.name)


@st.cache_resource
def load_sample_pdc() -> ImpactAnalyzer:
    sample_path = _resolve_sample_pdc_path()
    return ImpactAnalyzer.from_pdc(str(sample_path))


@st.cache_resource
def load_pdc_from_paths(paths: tuple[str, ...]) -> ImpactAnalyzer:
    cleaned_paths = [path for path in paths if path]
    if len(cleaned_paths) == 1:
        return ImpactAnalyzer.from_pdc(cleaned_paths[0])
    return ImpactAnalyzer.from_pdc(cleaned_paths)


def load_pdc_from_uploads(uploaded_files: Iterable) -> ImpactAnalyzer:
    paths = _write_uploaded_files(uploaded_files)
    return load_pdc_from_paths(tuple(paths))


@st.cache_resource
def load_vbd_from_path(path: str) -> Optional[ImpactAnalyzer]:
    return ImpactAnalyzer.from_vbd(path)


def load_vbd_from_upload(uploaded_file) -> Optional[ImpactAnalyzer]:
    path = _write_uploaded_file(uploaded_file)
    return load_vbd_from_path(path)
