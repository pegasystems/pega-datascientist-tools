"""State transitions for hierarchical lever replacement and Reset."""

from datetime import datetime

import polars as pl
import pytest
from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.filterwarnings("ignore:The following default columns are missing:UserWarning")


@pytest.fixture
def lever_page(da_app_dir):
    raw = pl.LazyFrame(
        {
            "Interaction ID": ["I1", "I1", "I2", "I2"],
            "Subject ID": ["S1", "S1", "S2", "S2"],
            "Decision Time": [datetime(2024, 1, 1)] * 4,
            "Issue": ["Sales", "Service", "Sales", "Service"],
            "Group": ["Cards", "Cards", "Loans", "Loans"],
            "Action": ["A", "A", "C", "D"],
            "Channel": ["Web", "Web", "Mobile", "Mobile"],
            "Direction": ["Inbound"] * 4,
            "Propensity": [1.0] * 4,
            "Value": [1.0, 1.4, 1.0, 1.05],
            "Context Weight": [1.0] * 4,
            "Levers": [2.0, 1.0, 1.0, 1.0],
            "Priority": [2.0, 1.4, 1.0, 1.05],
            "Stage_pyStageGroup": ["Arbitration"] * 4,
            "Stage_pyName": ["Arbitration"] * 4,
            "Stage_pyOrder": [3800] * 4,
        }
    )
    at = AppTest.from_file(str(da_app_dir / "pages" / "13_Business_Levers.py"), default_timeout=60)
    at.session_state["decision_data"] = DecisionAnalyzer(raw, sample_size=10)
    at.run()
    assert not at.exception
    return at


def _selected_counts(at):
    data = pl.from_pandas(at.dataframe[0].value)
    return data.filter(pl.col("Selected")).select("Before", "After").rows()


@pytest.mark.parametrize(("percentage", "expected_after"), [(-50, 0), (10, 0), (50, 1)])
def test_slider_replaces_action_lever(lever_page, percentage, expected_after):
    at = lever_page
    at.selectbox(key="_lever_level").set_value("Action").run()
    assert at.main.subheader[0].value == "Winning share for selected action: Sales / Cards / A"
    assert _selected_counts(at) == [(1, 1)]
    at.slider(key="_lever_percentage").set_value(percentage).run()
    assert not at.exception
    assert _selected_counts(at) == [(1, expected_after)]
    assert at.metric[0].value == "50.0%"
    assert at.metric[1].value == f"{50 * expected_after:.1f}%"
    assert at.session_state["lever_percentage"] == percentage
    assert at.session_state["lever_override_active"] is True
    assert len(at.get("plotly_chart")) == 2


def test_zero_is_replacement_and_reset_removes_override(lever_page):
    at = lever_page
    at.selectbox(key="_lever_level").set_value("Action").run()
    at.slider(key="_lever_percentage").set_value(10).run()
    at.slider(key="_lever_percentage").set_value(0).run()
    assert not at.exception
    assert _selected_counts(at) == [(1, 0)]
    at.button[0].click().run()
    assert not at.exception
    assert _selected_counts(at) == [(1, 1)]
    assert at.session_state["lever_override_active"] is False
    assert at.slider(key="_lever_percentage").value == 0


def test_cascading_selection_and_issue_group_scopes(lever_page):
    at = lever_page
    at.slider(key="_lever_percentage").set_value(10).run()
    assert at.main.subheader[0].value == "Winning share for selected issue: Sales"
    assert _selected_counts(at) == [(0, 1), (1, 0)]
    at.selectbox(key="_lever_level").set_value("Group").run()
    at.selectbox(key="_lever_group").set_value("Loans").run()
    assert not at.exception
    assert at.main.subheader[0].value == "Winning share for selected group: Sales / Loans"
    assert _selected_counts(at) == [(0, 1)]
    at.selectbox(key="_lever_level").set_value("Action").run()
    assert at.selectbox(key="_lever_action").value == "C"
    at.selectbox(key="_lever_issue").set_value("Service").run()
    assert not at.exception
    assert at.main.subheader[0].value == "Winning share for selected action: Service / Cards / A"
    assert at.selectbox(key="_lever_group").value == "Cards"
    assert at.selectbox(key="_lever_action").value == "A"
    data = pl.from_pandas(at.dataframe[0].value)
    assert data.filter(pl.col("Selected")).select("Issue", "Group", "Action").row(0) == ("Service", "Cards", "A")


def test_filter_change_invalidates_target_and_restores_baseline(lever_page):
    at = lever_page
    at.selectbox(key="_lever_level").set_value("Action").run()
    at.slider(key="_lever_percentage").set_value(10).run()
    at.selectbox(key="_channel_direction_widget").set_value("Mobile/Inbound").run()
    assert not at.exception
    assert at.selectbox(key="_lever_group").value == "Loans"
    assert at.selectbox(key="_lever_action").value == "C"
    assert at.session_state["lever_override_active"] is False
    assert _selected_counts(at) == [(0, 0)]
    assert "Re-evaluated **1 decision**" in at.main.caption[-1].value


def test_new_data_clears_override(lever_page):
    at = lever_page
    at.slider(key="_lever_percentage").set_value(10).run()
    da = at.session_state["decision_data"]
    at.session_state["decision_data"] = DecisionAnalyzer(da.decision_data, sample_size=10)
    at.run()
    assert not at.exception
    assert at.session_state["lever_override_active"] is False
    assert at.slider(key="_lever_percentage").value == 0
    assert at.metric[0].value == at.metric[1].value


def test_evaluation_summary_below_table(lever_page, monkeypatch):
    clock = iter([20.0, 20.125])
    monkeypatch.setattr("pdstools.decision_analyzer._scoring.perf_counter", lambda: next(clock))
    at = lever_page
    at.slider(key="_lever_percentage").set_value(10).run()
    assert not at.exception
    assert at.main.caption[-1].value == (
        "Re-evaluated **2 decisions** at arbitration in **0.125 s** "
        "(sampled, filtered population; scoring and aggregation only)."
    )


def test_top_n_control_keeps_highest_offers_per_interaction(lever_page):
    at = lever_page
    at.selectbox(key="_lever_level").set_value("Action").run()
    at.slider(key="_lever_percentage").set_value(10).run()
    assert _selected_counts(at) == [(1, 0)]
    at.number_input(key="_win_rank").set_value(2).run()
    assert not at.exception
    assert _selected_counts(at) == [(1, 1)]
    assert at.metric[0].value == at.metric[1].value == "50.0%"
    assert "at least one selected offer in the Top-2" in " ".join(c.value for c in at.main.caption)
    assert at.session_state["win_rank"] == 2
    at.number_input(key="_win_rank").set_value(1).run()
    assert not at.exception
    assert _selected_counts(at) == [(1, 0)]


def test_top_n_persists_between_pages(da_app_dir, seeded_decision_analyzer):
    at = AppTest.from_file(str(da_app_dir / "Home.py"), default_timeout=60)
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()
    at.switch_page("pages/13_Business_Levers.py").run()
    assert not at.exception
    at.number_input(key="_win_rank").set_value(3).run()
    at.switch_page("pages/5_Global_Sensitivity.py").run()
    assert not at.exception
    assert at.number_input(key="_win_rank").value == 3
    at.number_input(key="_win_rank").set_value(2).run()
    at.switch_page("pages/3_Action_Distribution.py").run()
    assert not at.exception
    assert at.session_state["win_rank"] == 2
    at.switch_page("pages/6_Win_Loss_Analysis.py").run()
    assert not at.exception
    assert at.number_input(key="_win_rank").value == 2
    at.number_input(key="_win_rank").set_value(1).run()
    at.switch_page("pages/13_Business_Levers.py").run()
    assert not at.exception
    assert at.number_input(key="_win_rank").value == 1
