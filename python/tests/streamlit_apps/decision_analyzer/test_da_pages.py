"""AppTest smoke tests for Decision Analyzer sub-pages.

Each page is exercised once with a seeded ``DecisionAnalyzer`` in
``session_state["decision_data"]`` — mirroring how a real user
arrives at the page after loading data on Home. For every page we
assert two things:

1. No exception bubbled out of the script.
2. One diagnostic content signal — either the leading page heading
   appearing in the rendered markdown (proves the script reached its
   first ``st.write``) or a minimum widget count (proves the page
   actually built its filter / metric / table surface, not just an
   empty body).

Library-layer behaviour stays covered by ``test_DecisionAnalyzer.py``
and friends.

Adding a new page test
----------------------
Append a ``(filename, heading_substring, widget_checks)`` tuple to
``DA_PAGES``. ``widget_checks`` is a dict of attribute → minimum
count. Use the smallest signal that reliably distinguishes "rendered"
from "didn't render" so the assertion doesn't churn on cosmetic
edits.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer

# Page 12 (About) is intentionally excluded — it uses the shared
# `show_about_page()` helper, which is exercised by the HC/IA About
# tests and doesn't depend on `decision_data`.
#
# Each entry: (page filename, heading substring expected in markdown,
# {widget attribute: minimum count}). The widget counts are derived
# from a discovery run against the seeded EEV2 fixture and are set
# to the actual count so a regression that drops a filter / metric
# / table surfaces immediately.
DA_PAGES: list[tuple[str, str, dict[str, int]]] = [
    ("2_Overview.py", "# Overview", {}),
    ("3_Action_Distribution.py", "# Action Distribution", {"selectbox": 3, "radio": 1}),
    ("4_Action_Funnel.py", "# Action Funnel", {"dataframe": 2, "tabs": 3}),
    ("5_Global_Sensitivity.py", "# Global Sensitivity Analysis", {"selectbox": 2}),
    ("6_Win_Loss_Analysis.py", "# Win/Loss Analysis", {"multiselect": 1, "selectbox": 2}),
    ("7_Optionality_Analysis.py", "# Optionality Analysis", {"selectbox": 2}),
    ("8_Offer_Quality_Analysis.py", "# Offer Quality Analysis", {"slider": 2}),
    ("9_Thresholding_Analysis.py", "# Thresholding Analysis", {"metric": 4, "slider": 2}),
    ("10_Arbitration_Distribution.py", "# Arbitration Distribution", {"selectbox": 4, "tabs": 3}),
    ("11_Single_Decision.py", "# Single Decision", {"selectbox": 1, "text_input": 1}),
]


@pytest.mark.parametrize(
    ("page", "heading", "widget_checks"),
    DA_PAGES,
    ids=[p[0].removesuffix(".py") for p in DA_PAGES],
)
def test_da_page_renders(
    page: str,
    heading: str,
    widget_checks: dict[str, int],
    seeded_decision_analyzer: DecisionAnalyzer,
    da_app_dir: Path,
):
    """Page renders with seeded ``decision_data``, no exception, and
    produces the expected heading + widget surface."""
    at = AppTest.from_file(str(da_app_dir / "pages" / page), default_timeout=30)
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()

    assert not at.exception, f"{page} raised: {at.exception}"
    assert any(heading in m.value for m in at.markdown), (
        f"{page} missing heading {heading!r} — page body did not render"
    )
    for attr, minimum in widget_checks.items():
        actual = len(getattr(at, attr))
        assert actual >= minimum, f"{page} expected at least {minimum} {attr} widget(s), got {actual}"
