"""Color controls preserve their selection and do not change the analysis."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl
from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer
from streamlit.testing.v1 import AppTest

if TYPE_CHECKING:
    from pathlib import Path


def test_optionality_color_selection_survives_navigation(
    da_app_dir: Path,
    seeded_decision_analyzer: DecisionAnalyzer,
) -> None:
    page = da_app_dir / "pages" / "7_Optionality_Analysis.py"
    at = AppTest.from_string(
        f"""
import runpy
import streamlit as st
if st.session_state.get("away", False):
    st.write("Another page")
else:
    runpy.run_path({str(page)!r})
""",
        default_timeout=60,
    )
    at.session_state["decision_data"] = seeded_decision_analyzer
    at.run()
    assert not at.exception
    initial = json.loads(at.get("plotly_chart")[0].proto.spec)
    initial_propensity = next(trace for trace in initial["data"] if trace["type"] == "scatter")

    for dimension in ("Issue", "Group"):
        at.selectbox(key="_optionality_color_by").set_value(dimension).run()
        assert not at.exception
        assert at.session_state["optionality_color_by"] == dimension
        figure = json.loads(at.get("plotly_chart")[0].proto.spec)
        assert next(trace for trace in figure["data"] if trace["type"] == "scatter")["y"] == initial_propensity["y"]
        assert all(
            trace["marker"]["color"] == seeded_decision_analyzer.color_mappings[dimension][trace["name"]]
            for trace in figure["data"]
            if trace["type"] == "bar" and trace["name"] != "No actions"
        )
        assert any("proportionally" in caption.value for caption in at.caption)

        at.session_state["away"] = True
        at.run()
        assert not at.exception
        assert len(at.selectbox) == 0
        at.session_state["away"] = False
        at.run()
        assert not at.exception
        assert at.selectbox(key="_optionality_color_by").value == dimension

    at.selectbox(key="_optionality_color_by").set_value("None").run()
    assert not at.exception
    final = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert final["data"] == initial["data"]


def test_action_color_cap_notice_appears_only_for_action_granularity(
    da_app_dir: Path,
    seeded_decision_analyzer: DecisionAnalyzer,
) -> None:
    original = seeded_decision_analyzer.decision_data
    extra = [original.head(1).with_columns(pl.lit(f"Extra{i}").alias("Action")) for i in range(12)]
    da = DecisionAnalyzer(pl.concat([original, *extra]))
    at = AppTest.from_file(str(da_app_dir / "pages" / "5_Global_Sensitivity.py"), default_timeout=60)
    at.session_state["decision_data"] = da
    at.run()
    assert not at.exception
    assert not any("other actions are gray" in caption.value for caption in at.caption)
    at.selectbox(key="glob_sensitivity_scope").set_value("Action").run()
    assert not at.exception
    assert any("10 actions" in caption.value and "other actions are gray" in caption.value for caption in at.caption)
    at.selectbox(key="glob_sensitivity_scope").set_value("Issue").run()
    assert not at.exception
    assert not any("other actions are gray" in caption.value for caption in at.caption)
