"""
Testing that none of the docs examples produce errors.
"""

import platform

import pytest
from testbook import testbook


@pytest.mark.parametrize(
    "relative_filepath",
    [
        # TODO shouldn't we have all the notebooks here? like *.ipynb?
        "examples/datamart/Example_ADM_Analysis.ipynb",
        # "examples/adm/AGBModelVisualisation.ipynb",
        # "examples/adm/ADMBinningInsights.ipynb",
        # test_ExampleDataAnonymization
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
