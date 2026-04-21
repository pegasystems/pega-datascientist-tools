"""AppTest smoke tests for the Health Check Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pdstools.adm.ADMDatamart import ADMDatamart


def test_home_renders_without_data(hc_app_dir: Path):
    """Home renders cleanly before any data is uploaded."""
    at = AppTest.from_file(str(hc_app_dir / "Home.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"HC Home raised: {at.exception}"


# Page 3 (About) renders the shared about page with no session_state
# requirement, so it's covered by the no-data path.
HC_PAGES = ["1_Data_Filters.py", "2_Reports.py", "3_About.py"]


@pytest.mark.parametrize("page", HC_PAGES, ids=lambda p: p.removesuffix(".py"))
def test_hc_page_renders(
    page: str,
    seeded_admdatamart: ADMDatamart,
    hc_app_dir: Path,
):
    """Each sub-page renders with a seeded ADMDatamart and no exception.

    HC pages guard with ``if "dm" in st.session_state`` rather than a
    helper, so seeding the key reaches the populated branch.
    """
    at = AppTest.from_file(str(hc_app_dir / "pages" / page), default_timeout=30)
    at.session_state["dm"] = seeded_admdatamart
    at.run()
    assert not at.exception, f"{page} raised: {at.exception}"
