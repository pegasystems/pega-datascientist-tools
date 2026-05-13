"""AppTest smoke tests for the Impact Analyzer Streamlit app.

Every page asserts two things: the script ran without exception, and
one diagnostic content signal (heading text or a minimum widget
count) so an empty body or missing widget surface fails the test
instead of passing silently.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdstools.impactanalyzer.ImpactAnalyzer import ImpactAnalyzer
    from pathlib import Path


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
    ("1_Overall_Summary.py", "# Overall Summary", {"selectbox": 1}),
    ("2_Channels.py", "# Channels", {"dataframe": 1, "selectbox": 1}),
    ("3_About.py", "# About", {"expander": 1}),
]


# Each entry: (filename, expected page_title). Mirrors the
# ``"<Page> · <App>"`` convention used by HC and DA so the browser tab
# reads the page name first.
IA_PAGE_TITLES: list[tuple[str, str]] = [
    ("1_Overall_Summary.py", "Impact Analyzer · Overall Summary"),
    ("2_Channels.py", "Impact Analyzer · Channels"),
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
    ("2_Channels.py", "# Channels"),
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
        "pdstools.app.impact_analyzer.ia_streamlit_utils.load_sample",
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
