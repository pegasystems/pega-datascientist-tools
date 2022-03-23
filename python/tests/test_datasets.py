import sys
sys.path.append("python")
from cdhtools import datasets

def test_import_CDHSample():
    Sample = datasets.CDHSample()
    assert Sample.modelData.shape == (1047,12)