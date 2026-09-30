"""Regression tests for the Health Check's insufficient-feedback fallback."""

import re
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import polars.selectors as cs
import pytest
from pdstools import ADMDatamart


@pytest.fixture
def report_cells():
    """Load the headline and exclusion cells without requiring Quarto."""
    template = Path(__file__).parents[2] / "pdstools" / "reports" / "HealthCheck.qmd"
    cells = re.findall(r"```\{python\}\n(.*?)\n```", template.read_text(encoding="utf-8"), re.DOTALL)
    result = []
    for marker in ("Insufficient feedback across all channels", "unused_channels = "):
        matches = [cell for cell in cells if marker in cell]
        assert len(matches) == 1
        result.append(compile(matches[0], str(template), "exec"))
    return result


@pytest.mark.parametrize(
    ("outcomes", "expected_ids", "shows_finding"),
    [
        ([(0, 2000)], ["model-0"], True),
        ([(0, 2000), (200, 999)], ["model-0", "model-1"], True),
        ([(0, 2000), (200, 1000)], ["model-1"], False),
        ([(200, 1000), (300, 2000)], ["model-0", "model-1"], False),
    ],
    ids=["single-invalid", "all-invalid", "mixed", "all-valid"],
)
def test_channel_feedback_fallback(report_cells, outcomes, expected_ids, shows_finding):
    """Keep and warn on all-invalid data; otherwise preserve channel exclusion."""
    rows = [
        {
            "ModelID": f"model-{index}",
            "Name": f"action-{index}",
            "Configuration": "TestModel",
            "Channel": "Email",
            "Direction": "Outbound" if index == 0 else "Inbound",
            "SnapshotTime": "20260101",
            "Positives": positives,
            "ResponseCount": responses,
            "Performance": 0.5,
        }
        for index, (positives, responses) in enumerate(outcomes)
    ]
    datamart = ADMDatamart(model_df=pl.DataFrame(rows).lazy())
    report_utils = Mock()
    namespace = {
        "pl": pl,
        "cs": cs,
        "ADMDatamart": ADMDatamart,
        "datamart": datamart,
        "last_data": datamart.get_last_data_for_report(),
        "active_models_filter_expr": pl.lit(True),
        "report_utils": report_utils,
        "GT": Mock(),
        "display": Mock(),
    }

    for cell in report_cells:
        exec(cell, namespace)

    result = namespace["datamart"]
    assert result.model_data.select("ModelID").collect()["ModelID"].sort().to_list() == expected_ids
    assert namespace["last_data"]["ModelID"].sort().to_list() == expected_ids
    if shows_finding:
        assert result is datamart
        report_utils.quarto_callout_important.assert_called_once()
        assert "Insufficient feedback" in report_utils.quarto_callout_important.call_args.args[0]
    else:
        report_utils.quarto_callout_important.assert_not_called()
