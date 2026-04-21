"""AppTest smoke tests for the Impact Analyzer Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pdstools.impactanalyzer.ImpactAnalyzer import ImpactAnalyzer


def test_home_renders_without_data(ia_app_dir: Path):
    """Home renders cleanly before any data is uploaded."""
    at = AppTest.from_file(str(ia_app_dir / "Home.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"IA Home raised: {at.exception}"


# Page 3 (About) renders the shared about page and is covered by the
# no-data path; sub-pages 1 and 2 require ``impact_analyzer`` in
# session_state via ``ensure_impact_analyzer()``.
IA_PAGES = ["1_Overall_Summary.py", "2_Trend.py", "3_About.py"]


@pytest.mark.parametrize("page", IA_PAGES, ids=lambda p: p.removesuffix(".py"))
def test_ia_page_renders(
    page: str,
    seeded_impact_analyzer: ImpactAnalyzer,
    ia_app_dir: Path,
):
    """Each sub-page renders with a seeded ImpactAnalyzer and no exception."""
    at = AppTest.from_file(str(ia_app_dir / "pages" / page), default_timeout=30)
    at.session_state["impact_analyzer"] = seeded_impact_analyzer
    at.session_state["ia_is_sample_data"] = True
    at.run()
    assert not at.exception, f"{page} raised: {at.exception}"
