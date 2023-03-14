"""
Testing that none of the docs examples produce errors.
"""


import pytest
import sys

sys.path.append("python")
from pdstools import datasets


@pytest.fixture
def data():
    return datasets.CDHSample()


def test_all_notebooks():
    from testbook import testbook
    import glob

    root_dir = "./"

    files = [
        root_dir + f
        for f in [
            "examples/datamart/Example_ADM_Analysis.ipynb",
            "examples/graph_gallery/graph_gallery.ipynb",
            "examples/helloworld/hello_cdhtools.ipynb",
            "examples/adm/AGBModelVisualisation.ipynb",
        ]
    ]

    # files += glob.glob("examples/valuefinder/*.ipynb", root_dir=root_dir)

    def test_get_details(file):
        with testbook(file) as tb:
            tb.inject(
                """
            import sys
            sys.path.append('./python')"""
            )
            tb.execute()
        return True

    assert all(map(test_get_details, files))


def test_ExampleDataAnonymization():
    pass
