"""
Testing the functionality of utils/show_versions functions
"""

import pathlib
import sys

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
import pdstools


def test_show_versions(capsys):
    pdstools.show_versions()
    captured = capsys.readouterr()
    assert "pdstools" in captured.out
    assert "---Version info---" in captured.out
    assert "---Dependencies---" in captured.out
    assert "---Streamlit app dependencies---" in captured.out
    
