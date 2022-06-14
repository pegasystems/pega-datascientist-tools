import sys
sys.path.append("python")
from cdhtools import datasets

def test_import_CDHSample():
    Sample = datasets.CDHSample()
    assert Sample.modelData.shape == (1047,12)

def test_import_SampleTrees():
    datasets.SampleTrees()

def test_import_SampleValueFinder():
    vf = datasets.SampleValueFinder()
    assert vf.df.shape == (27133, 11)