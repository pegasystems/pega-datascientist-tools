"""Shared fixtures for Streamlit AppTest-based page tests.

These tests exercise the Streamlit layer in-process via
``streamlit.testing.v1.AppTest`` — no browser, no server. They cover
the glue that unit tests can't: session_state wiring, widget
persistence, ``ensure_data()`` guards, filter namespacing, and the
assumptions made by ``standard_page_config`` / ``show_sidebar_branding``.

Library-level logic (plots, aggregates, compute) is covered by the
main test suite; these tests intentionally stay thin at the page
level — assert the page renders without exception and the expected
widgets / session_state are present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pdstools.adm.ADMDatamart import ADMDatamart
from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer
from pdstools.impactanalyzer.ImpactAnalyzer import ImpactAnalyzer


@pytest.fixture(autouse=True)
def _stub_pypi_version_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent AppTest pages from calling PyPI for the latest-version check.

    ``show_about_page()`` (and the sidebar branding helper) call
    ``cdh_utils.get_latest_pdstools_version()`` which hits PyPI with a
    5 s timeout. Under AppTest the call slows tests down and adds
    network flake — the version-check helper has its own dedicated
    unit test, so mocking it here is the right boundary.
    """
    monkeypatch.setattr(
        "pdstools.utils.cdh_utils._io.get_latest_pdstools_version",
        lambda: None,
    )
    # Streamlit's ``@st.cache_data`` persists across AppTest script
    # runs in the same process — clear it so the patched function is
    # actually called instead of a stale cached return value.
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

DA_APP_DIR = REPO_ROOT / "python" / "pdstools" / "app" / "decision_analyzer"
HC_APP_DIR = REPO_ROOT / "python" / "pdstools" / "app" / "health_check"
IA_APP_DIR = REPO_ROOT / "python" / "pdstools" / "app" / "impact_analyzer"

DA_SAMPLE_CSV = DATA_DIR / "da" / "sample_eev2_minimal.csv"
ADM_MODEL_FILE = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip"
ADM_PREDICTOR_FILE = "Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip"


@pytest.fixture(scope="session")
def da_app_dir() -> Path:
    """Absolute path to the DA Streamlit app directory."""
    return DA_APP_DIR


@pytest.fixture(scope="session")
def hc_app_dir() -> Path:
    """Absolute path to the Health Check Streamlit app directory."""
    return HC_APP_DIR


@pytest.fixture(scope="session")
def ia_app_dir() -> Path:
    """Absolute path to the Impact Analyzer Streamlit app directory."""
    return IA_APP_DIR


@pytest.fixture(scope="session")
def da_sample_path() -> Path:
    """Path to the minimal EEV2 sample used for DA smoke tests.

    Using the minimal CSV (not the full parquet) keeps test runtime
    down — every value in it is traceable back to the fixture, which
    also helps when we promote smoke tests to exact-value tests.
    """
    if not DA_SAMPLE_CSV.exists():
        pytest.skip(f"DA sample data missing: {DA_SAMPLE_CSV}")
    return DA_SAMPLE_CSV


@pytest.fixture
def seeded_decision_analyzer(da_sample_path: Path) -> DecisionAnalyzer:
    """A DecisionAnalyzer pre-loaded with the minimal EEV2 fixture.

    Tests inject this into ``session_state["decision_data"]`` before
    running the page so ``ensure_data()`` guards pass.
    """
    return DecisionAnalyzer.from_decision_analyzer(da_sample_path)


@pytest.fixture
def seeded_admdatamart() -> ADMDatamart:
    """An ADMDatamart loaded from the canonical CDH sample.

    Health Check pages read this from ``st.session_state["dm"]``. We
    re-use the same fixture data as ``test_ADMDatamart.py`` so any
    schema drift surfaces in both layers.
    """
    if not (DATA_DIR / ADM_MODEL_FILE).exists():
        pytest.skip(f"ADM sample data missing: {DATA_DIR / ADM_MODEL_FILE}")
    return ADMDatamart.from_ds_export(
        base_path=str(DATA_DIR),
        model_filename=ADM_MODEL_FILE,
        predictor_filename=ADM_PREDICTOR_FILE,
    )


IA_SAMPLE_JSON = DATA_DIR / "ia" / "CDH_Metrics_ImpactAnalyzer.json"


@pytest.fixture
def seeded_impact_analyzer() -> ImpactAnalyzer:
    """An ImpactAnalyzer loaded from the bundled PDC sample."""
    if not IA_SAMPLE_JSON.exists():
        pytest.skip(f"IA sample data missing: {IA_SAMPLE_JSON}")
    return ImpactAnalyzer.from_pdc(str(IA_SAMPLE_JSON))
