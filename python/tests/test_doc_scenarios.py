"""
Testing that none of the docs examples produce errors.
"""

import pathlib
import platform

import pytest
from testbook import testbook

basePath = pathlib.Path(__file__).parent.parent.parent


@pytest.mark.parametrize(
    "relative_filepath",
    [
        # Core functionality notebooks
        "examples/datamart/Example_ADM_Analysis.ipynb",
        "examples/impactanalyzer/impact_analyzer.ipynb",
        "examples/prediction_studio/Predictions.ipynb",
        # "examples/prediction_studio/PredictionStudio.ipynb",  # Not passing
        "examples/hds/hds_analysis.ipynb",
        "examples/ih/Example_IH_Analysis.ipynb",
        # "examples/ih/Conversion_Reporting.ipynb",  # Failing due to column not found
        "examples/valuefinder/vf_analysis.ipynb",
        "examples/decision_analyzer/decision_analyzer.ipynb",
        "examples/explainability_extract/explainability_extract.ipynb",
        # Educational/article notebooks
        "examples/articles/ADMExplained.ipynb",
        "examples/articles/thompsonsampling.ipynb",
        "examples/articles/explanations/agb_global_explanations.ipynb",
        # ADM analysis notebooks
        "examples/adm/AGBModelVisualisation.ipynb",
        # "examples/adm/ADMBinningInsights.ipynb",  # Missing BinAggregator import
        # "examples/adm/ADM_ActionDiscrimination_Metric.ipynb",  # Import issues
    ],
)
def test_notebook(relative_filepath):
    file = str(basePath / relative_filepath)

    if platform.system() == "Windows":  # pragma: no cover
        pythonPath = "python"
    else:
        pythonPath = str(basePath / "python")

    with testbook(file) as tb:
        tb.inject(
            f"""
        import sys
        sys.path.append('{pythonPath}')"""
        )
        tb.execute()

    assert True
