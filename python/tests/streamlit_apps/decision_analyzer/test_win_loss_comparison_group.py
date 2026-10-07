"""Widget-interaction test: defining a comparison group on the Win/Loss page.

Picking a comparison group renders the prioritization-factor boxplots.
``prio_factor_boxplots`` downsamples the plotted values and returns a
sampling notice alongside the figure. That notice is informational and
must not surface as an ``st.warning`` on every render.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pdstools.decision_analyzer.plots import Plot
from pdstools.decision_analyzer.plots._sensitivity import prio_factor_boxplots
from streamlit.testing.v1 import AppTest

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer


def test_comparison_group_renders_boxplots_without_sampling_warning(
    da_app_dir: Path,
    seeded_decision_analyzer: DecisionAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampling_notices: list[str | None] = []

    # The minimal fixture is below the default cap, so force sampling to occur.
    def tiny_cap_boxplots(self, *args, **kwargs):
        result = prio_factor_boxplots(self, *args, max_points_per_group=2, **kwargs)
        sampling_notices.append(result[1])
        return result

    monkeypatch.setattr(Plot, "prio_factor_boxplots", tiny_cap_boxplots)

    page = da_app_dir / "pages" / "6_Win_Loss_Analysis.py"
    at = AppTest.from_file(str(page), default_timeout=60)
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"

    at.multiselect(key="local_multiselect").set_value(["Issue"]).run()
    assert not at.exception, f"Column selection raised: {at.exception}"

    issue_values = at.multiselect(key="local_selected_Issue")
    issue_values.set_value(issue_values.options[:1]).run()
    assert not at.exception, f"Value selection raised: {at.exception}"

    sampling_notice = "Showing a quantile-stratified sample of up to 2 values per segment to keep the chart compact."
    assert sampling_notices[-1] == sampling_notice

    warnings = [w.value for w in at.warning]
    assert sampling_notice not in warnings, f"Sampling notice leaked to UI: {warnings}"
    assert "No comparison group defined" not in warnings
    assert any("How Often Do These Offers Rank First?" in m.value for m in at.markdown)
