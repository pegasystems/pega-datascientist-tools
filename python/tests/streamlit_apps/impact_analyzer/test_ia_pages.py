"""AppTest smoke tests for the Impact Analyzer Streamlit app.

Every page asserts two things: the script ran without exception, and
one diagnostic content signal (heading text or a minimum widget
count) so an empty body or missing widget surface fails the test
instead of passing silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pdstools.impactanalyzer.ImpactAnalyzer import ImpactAnalyzer


def test_home_renders_without_data(ia_app_dir: Path):
    """Home renders cleanly before any data is uploaded.

    The branding markdown and the file uploader together prove the
    page reached its data-entry surface (the upload widget is the
    primary CTA — if it goes missing the user has no way to load
    data).
    """
    at = AppTest.from_file(str(ia_app_dir / "Home.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"IA Home raised: {at.exception}"
    assert any("Impact Analyzer" in m.value for m in at.markdown), "IA Home missing branding markdown"
    # AppTest doesn't expose ``file_uploader`` as a typed accessor on
    # every version, so go through the generic ``get()`` bag.
    file_uploaders = at.get("file_uploader") if hasattr(at, "get") else []
    assert list(file_uploaders), "IA Home expected a file_uploader widget for data entry"


# Page 3 (About) renders the shared about page and is covered by the
# no-data path; sub-pages 1 and 2 require ``impact_analyzer`` in
# session_state via ``ensure_impact_analyzer()``.
#
# Each entry: (filename, heading substring, {widget attribute: min count}).
IA_PAGES: list[tuple[str, str, dict[str, int]]] = [
    ("1_Overall_Summary.py", "# Overall Summary", {"dataframe": 1, "selectbox": 1}),
    ("2_Trend.py", "# Trend Analysis", {"dataframe": 1, "selectbox": 1}),
    ("3_About.py", "# About", {"expander": 1}),
]


@pytest.mark.parametrize(
    ("page", "heading", "widget_checks"),
    IA_PAGES,
    ids=[p[0].removesuffix(".py") for p in IA_PAGES],
)
def test_ia_page_renders(
    page: str,
    heading: str,
    widget_checks: dict[str, int],
    seeded_impact_analyzer: ImpactAnalyzer,
    ia_app_dir: Path,
):
    """Each sub-page renders with a seeded ImpactAnalyzer and produces
    its expected heading + widget surface."""
    at = AppTest.from_file(str(ia_app_dir / "pages" / page), default_timeout=30)
    at.session_state["impact_analyzer"] = seeded_impact_analyzer
    at.session_state["ia_is_sample_data"] = True
    at.run()
    assert not at.exception, f"{page} raised: {at.exception}"

    assert any(heading in m.value for m in at.markdown), (
        f"{page} missing heading {heading!r} — page body did not render"
    )
    for attr, minimum in widget_checks.items():
        actual = len(getattr(at, attr))
        assert actual >= minimum, f"{page} expected at least {minimum} {attr} widget(s), got {actual}"
