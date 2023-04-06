"""
Testing that none of the docs examples produce errors.
"""


import pytest
import sys

import pathlib

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import datasets


@pytest.fixture
def data():
    return datasets.CDHSample()


def test_all_notebooks():
    from testbook import testbook

    files = [
        str(basePath / f)
        for f in [
            "examples/datamart/Example_ADM_Analysis.ipynb",
            "examples/graph_gallery/graph_gallery.ipynb",
            "examples/helloworld/hello_cdhtools.ipynb",
            "examples/adm/AGBModelVisualisation.ipynb",
        ]
    ]

    def test_get_details(file):
        pythonPath = str(basePath / "python")
        with testbook(file) as tb:
            tb.inject(
                f"""
            import sys
            sys.path.append({pythonPath})"""
            )
            tb.execute()
        return True

    assert all(map(test_get_details, files))


def test_ExampleDataAnonymization():
    pass
