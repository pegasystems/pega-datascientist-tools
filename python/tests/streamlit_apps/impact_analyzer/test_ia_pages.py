"""AppTest smoke tests for the Impact Analyzer Streamlit app.

Every page asserts two things: the script ran without exception, and
one diagnostic content signal (heading text or a minimum widget
count) so an empty body or missing widget surface fails the test
instead of passing silently.
"""

from __future__ import annotations

import copy
from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from pdstools.impactanalyzer.ImpactAnalyzer import ImpactAnalyzer


def test_home_pre_load_text_is_product_neutral(ia_app_dir: Path):
    """Pre-load home text must not mention internal product names.

    Before the user uploads data, the UI describes Impact Analyzer
    generically. Once data is loaded and a format is detected, the
    detected-format label (``**PDC Export**``) is allowed.
    """
    at = AppTest.from_file(str(ia_app_dir / "Home.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"IA Home raised: {at.exception}"
    pre_load_text = "\n".join(m.value for m in at.markdown)
    for forbidden in ("PDC", "Pega Diagnostic Cloud"):
        assert forbidden not in pre_load_text, f"Pre-load IA Home text leaks {forbidden!r}: {pre_load_text!r}"


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


# Each entry: (filename, expected page_title). Mirrors the
# ``"<Page> · <App>"`` convention used by HC and DA so the browser tab
# reads the page name first.
IA_PAGE_TITLES: list[tuple[str, str]] = [
    ("1_Overall_Summary.py", "Overall Summary · Impact Analyzer"),
    ("2_Trend.py", "Trend · Impact Analyzer"),
    ("3_About.py", "About · Impact Analyzer"),
]


@pytest.mark.parametrize(
    ("page", "expected_title"),
    IA_PAGE_TITLES,
    ids=[p[0].removesuffix(".py") for p in IA_PAGE_TITLES],
)
def test_ia_page_title_order(
    page: str,
    expected_title: str,
    ia_app_dir: Path,
):
    """Browser-tab title is ``"<Page> · Impact Analyzer"``.

    AppTest doesn't expose ``st.set_page_config(page_title=...)`` as a
    queryable attribute, so assert against the source to guarantee the
    convention is followed.
    """
    source = (ia_app_dir / "pages" / page).read_text()
    assert f'page_title="{expected_title}"' in source, f"{page} expected page_title={expected_title!r} in source"


def test_trend_guards_single_timestamp_data(
    seeded_impact_analyzer: ImpactAnalyzer,
    ia_app_dir: Path,
):
    """Trend page shows an info message and stops when data spans a
    single point in time, instead of rendering a degenerate axis with
    microsecond ticks."""
    ia = copy.copy(seeded_impact_analyzer)
    df = ia.ia_data.collect()
    fixed = df.with_columns(pl.col("SnapshotTime").min().alias("SnapshotTime"))
    ia.ia_data = fixed.lazy()

    at = AppTest.from_file(str(ia_app_dir / "pages" / "2_Trend.py"), default_timeout=30)
    at.session_state["impact_analyzer"] = ia
    at.session_state["ia_is_sample_data"] = True
    at.run()
    assert not at.exception, f"2_Trend.py raised: {at.exception}"

    info_messages = [i.value for i in at.info]
    assert any("single point in time" in msg for msg in info_messages), (
        f"Expected single-timestamp guard message; got info={info_messages!r}"
    )
    # Chart must not render when the page has stopped early.
    assert not at.get("plotly_chart"), "Trend chart rendered despite single-timestamp guard"


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
    ia = seeded_impact_analyzer
    if page == "2_Trend.py":
        # The bundled IA sample only has one SnapshotTime, which would
        # trigger the single-timestamp guard. Expand it to two so the
        # Trend page renders its chart + table for the smoke check.
        ia = copy.copy(ia)
        df = ia.ia_data.collect()
        df2 = df.with_columns((pl.col("SnapshotTime") + pl.duration(days=1)).alias("SnapshotTime"))
        ia.ia_data = pl.concat([df, df2]).lazy()

    at = AppTest.from_file(str(ia_app_dir / "pages" / page), default_timeout=30)
    at.session_state["impact_analyzer"] = ia
    at.session_state["ia_is_sample_data"] = True
    at.run()
    assert not at.exception, f"{page} raised: {at.exception}"

    assert any(heading in m.value for m in at.markdown), (
        f"{page} missing heading {heading!r} — page body did not render"
    )
    for attr, minimum in widget_checks.items():
        actual = len(getattr(at, attr))
        assert actual >= minimum, f"{page} expected at least {minimum} {attr} widget(s), got {actual}"


IA_DATA_REQUIRED_PAGES = [
    ("1_Overall_Summary.py", "# Overall Summary"),
    ("2_Trend.py", "# Trend Analysis"),
]


@pytest.mark.parametrize(
    ("page", "heading"),
    IA_DATA_REQUIRED_PAGES,
    ids=[p[0].removesuffix(".py") for p in IA_DATA_REQUIRED_PAGES],
)
def test_ia_subpage_autoloads_sample_on_deep_link(page: str, heading: str, ia_app_dir: Path):
    """Visiting a data-required sub-page directly auto-loads the bundled
    PDC sample instead of dead-ending on a "please upload" warning.
    """
    at = AppTest.from_file(str(ia_app_dir / "pages" / page), default_timeout=60)
    at.run()
    assert not at.exception, f"{page} raised: {at.exception}"

    warnings = [w.value for w in at.warning]
    assert not any("please upload" in w.lower() or "please load" in w.lower() for w in warnings), (
        f"{page} showed a please-upload warning instead of auto-loading: {warnings}"
    )
    assert "impact_analyzer" in at.session_state, f"{page} did not populate session_state['impact_analyzer']"

    rendered_text = [m.value for m in at.markdown] + [t.value for t in at.title]
    assert any(heading in s for s in rendered_text), f"{page} missing heading {heading!r} — body did not render"


def test_ia_subpage_warns_when_autoload_fails(ia_app_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """When the sample loader raises, the sub-page falls back to the
    original "please upload" warning instead of crashing.
    """
    import streamlit as st

    st.cache_resource.clear()

    def _broken_sample():
        raise RuntimeError("sample bundle unavailable")

    monkeypatch.setattr(
        "pdstools.app.impact_analyzer.ia_streamlit_utils.load_sample_pdc",
        _broken_sample,
    )

    at = AppTest.from_file(str(ia_app_dir / "pages" / "1_Overall_Summary.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"page raised: {at.exception}"
    warnings = [w.value for w in at.warning]
    assert any("please upload" in w.lower() for w in warnings), (
        f"expected please-upload warning when auto-load fails, got: {warnings}"
    )
    assert "impact_analyzer" not in at.session_state
