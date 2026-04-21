"""AppTest smoke tests for Decision Analyzer sub-pages.

Each page is exercised once with a seeded ``DecisionAnalyzer`` in
``session_state["decision_data"]`` — mirroring how a real user
arrives at the page after loading data on Home. We assert only that
the page renders without exception; library-layer behaviour is
covered by ``test_DecisionAnalyzer.py`` and friends.

Adding a new page test
----------------------
Append the page filename to ``DA_PAGES`` below. No other change
needed — the parametrize id keeps reports readable.

Phase 2 (filter persistence, ensure_data guards, #649-style
regression locks) will get its own test files; this file stays
narrowly scoped to "boots without crashing".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer

# Page 12 (About) is intentionally excluded — it uses the shared
# `show_about_page()` helper, which is exercised by the HC/IA About
# tests and doesn't depend on `decision_data`.
DA_PAGES = [
    "2_Overview.py",
    "3_Action_Distribution.py",
    "4_Action_Funnel.py",
    "5_Global_Sensitivity.py",
    "6_Win_Loss_Analysis.py",
    "7_Optionality_Analysis.py",
    "8_Offer_Quality_Analysis.py",
    "9_Thresholding_Analysis.py",
    "10_Arbitration_Distribution.py",
    "11_Single_Decision.py",
]


@pytest.mark.parametrize("page", DA_PAGES, ids=lambda p: p.removesuffix(".py"))
def test_da_page_renders(
    page: str,
    seeded_decision_analyzer: DecisionAnalyzer,
    da_app_dir: Path,
):
    """Page renders with seeded ``decision_data`` and no exception."""
    at = AppTest.from_file(str(da_app_dir / "pages" / page), default_timeout=30)
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()

    assert not at.exception, f"{page} raised: {at.exception}"
