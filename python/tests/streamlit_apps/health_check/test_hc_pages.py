"""AppTest smoke tests for the Health Check Streamlit app.

Every page asserts two things: the script ran without exception, and
one diagnostic content signal (heading text or a minimum widget
count) so an empty body or missing widget surface fails the test
instead of passing silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pdstools.adm.ADMDatamart import ADMDatamart


def test_home_renders_without_data(hc_app_dir: Path):
    """Home renders cleanly before any data is uploaded.

    The data-source picker (selectbox) and the "Adaptive Model Health
    Check" branding markdown both have to be present — together they
    prove ``standard_page_config`` ran *and* the data-entry surface
    is wired up.
    """
    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"HC Home raised: {at.exception}"
    assert any("Adaptive Model Health Check" in m.value for m in at.markdown), "HC Home missing branding markdown"
    assert len(at.selectbox) >= 1, "HC Home expected at least one selectbox (data source picker)"


# Page 3 (About) renders the shared about page with no session_state
# requirement, so it's covered by the no-data path.
#
# Each entry: (filename, heading substring, {widget attribute: min count}).
HC_PAGES: list[tuple[str, str, dict[str, int]]] = [
    ("1_Data_Filters.py", "# Add custom filters", {"multiselect": 1}),
    ("2_Reports.py", "Generate Health Check", {"checkbox": 5, "tabs": 2, "button": 2}),
    ("3_About.py", "# About", {"expander": 1}),
]


@pytest.mark.parametrize(
    ("page", "heading", "widget_checks"),
    HC_PAGES,
    ids=[p[0].removesuffix(".py") for p in HC_PAGES],
)
def test_hc_page_renders(
    page: str,
    heading: str,
    widget_checks: dict[str, int],
    seeded_admdatamart: ADMDatamart,
    hc_app_dir: Path,
):
    """Each sub-page renders with a seeded ADMDatamart and produces
    its expected heading + widget surface.

    HC pages guard with ``if "dm" in st.session_state`` rather than a
    helper, so seeding the key reaches the populated branch.
    """
    at = AppTest.from_file(str(hc_app_dir / "pages" / page), default_timeout=30)
    at.session_state["dm"] = seeded_admdatamart
    at.run()
    assert not at.exception, f"{page} raised: {at.exception}"

    rendered_text = [m.value for m in at.markdown] + [t.value for t in at.title]
    assert any(heading in s for s in rendered_text), f"{page} missing heading {heading!r} — page body did not render"
    for attr, minimum in widget_checks.items():
        actual = len(getattr(at, attr))
        assert actual >= minimum, f"{page} expected at least {minimum} {attr} widget(s), got {actual}"


# Sub-pages that auto-load on deep-link — exclude the About page since
# it's not data-gated.
HC_DATA_REQUIRED_PAGES = [
    ("1_Data_Filters.py", "# Add custom filters"),
    ("2_Reports.py", "Generate Health Check"),
]


@pytest.mark.parametrize(
    ("page", "heading"),
    HC_DATA_REQUIRED_PAGES,
    ids=[p[0].removesuffix(".py") for p in HC_DATA_REQUIRED_PAGES],
)
def test_hc_subpage_autoloads_sample_on_deep_link(page: str, heading: str, hc_app_dir: Path):
    """Visiting a data-required sub-page directly auto-loads the bundled
    CDH sample instead of dead-ending on a "please configure" warning.

    Mirrors the launcher / deep-link flow: the user clicks a sub-page
    without visiting Home first. The page should render and ``dm``
    should be populated.
    """
    at = AppTest.from_file(str(hc_app_dir / "pages" / page), default_timeout=60)
    at.run()
    assert not at.exception, f"{page} raised: {at.exception}"

    warnings = [w.value for w in at.warning]
    assert not any("please configure" in w.lower() or "please load" in w.lower() for w in warnings), (
        f"{page} showed a please-upload warning instead of auto-loading: {warnings}"
    )
    assert "dm" in at.session_state, f"{page} did not populate session_state['dm']"

    rendered_text = [m.value for m in at.markdown] + [t.value for t in at.title]
    assert any(heading in s for s in rendered_text), f"{page} missing heading {heading!r} — body did not render"


def test_hc_subpage_warns_when_autoload_fails(hc_app_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """When the sample loader raises, the sub-page falls back to the
    original "please configure" warning instead of crashing.

    Patches both the CLI-path helper (returns None — no path configured)
    and ``cached_sample`` (raises) so the helper exhausts its priority
    chain and returns False.
    """
    import streamlit as st

    st.cache_resource.clear()

    def _broken_sample():
        raise RuntimeError("sample bundle unavailable")

    monkeypatch.setattr(
        "pdstools.app.health_check.hc_streamlit_utils.cached_sample",
        _broken_sample,
    )

    at = AppTest.from_file(str(hc_app_dir / "pages" / "2_Reports.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"page raised: {at.exception}"
    warnings = [w.value for w in at.warning]
    assert any("please configure" in w.lower() for w in warnings), (
        f"expected please-configure warning when auto-load fails, got: {warnings}"
    )
    assert "dm" not in at.session_state
