"""
Testing the functionality of the ADMDatamart functions
"""

import pytest
import sys

import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from pdstools import ADMTrees


@pytest.fixture
def treeSample():
    """Fixture to serve as class to call functions from."""
    return ADMTrees("data/agb/_974a7f9c-66a6-4f00-bf3e-3acf5f188b1d.txt")


def test_has_models(treeSample):
    assert isinstance(treeSample.model, list)
    assert len(treeSample.model) > 1


def test_has_properties(treeSample):
    required_properties = {
        "trees",
        "properties",
        "learning_rate",
        "model",
        "treeStats",
        "splitsPerTree",
        "gainsPerTree",
        "gainsPerSplit",
        "groupedGainsPerSplit",
        "predictors",
        "allValuesPerSplit",
    }
    assert all(hasattr(treeSample, attr) for attr in required_properties)


def test_plotSplitsPerVariable(treeSample):
    treeSample.plotSplitsPerVariable(show=False)


def sampleX(trees):
    from random import sample
    x = {}
    for variable, values in trees.allValuesPerSplit.items():
        if len(values) == 1:
            if 'true' in values or 'false' in values:
                values = {'true', 'false'}
            if isinstance(list(values)[0], str):
                try: 
                    float(list(values)[0])
                except:
                    print('FAILED ON ', values)
                    values = values.union({'Other'})
        x[variable]= sample(list(values), 1)[0]
    return x


@pytest.fixture
def sampledX(treeSample):
    return sampleX(treeSample)


def test_plotTreeZero(treeSample, sampledX):
    treeSample.plotTree(42, highlighted=sampledX, show=False)


def test_score(treeSample, sampledX):
    assert 0 <= treeSample.score(sampledX) <= 1


def test_plotContributionPerTree(treeSample, sampledX):
    treeSample.plotContributionPerTree(sampledX, show=False)


def test_plotSplitsPerVariableType(treeSample):
    treeSample.plotSplitsPerVariableType()
