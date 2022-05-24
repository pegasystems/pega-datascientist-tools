import pytest
import sys

sys.path.append("python")
from cdhtools import ADMTrees


def test_import_txt_file():
    ADMTrees(
        "data/agb/_87288230-d2fd-408f-822c-3d883f45701f.txt"
    )


@pytest.fixture
def treeSample():
    """Fixture to serve as class to call functions from."""
    return ADMTrees(
        "data/agb/IHmodelbeforeRetrain.txt"
    )


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

@pytest.fixture
def sampledX():
    return {'Age': 49.0,
 'CustomerName': 'FrancinaKunze',
 'EyeColor': 'LightYellow',
 'IH.PegaBatch.E2E Test.Accept.pxLastOutcomeTime.DaysSince': 0.0003451851851851852,
 'IH.PegaBatch.E2E Test.Accept.pyHistoricalOutcomeCount': 69.0,
 'IH.PegaBatch.E2E Test.Decline.pxLastOutcomeTime.DaysSince': 0.001279548611111111,
 'IH.PegaBatch.E2E Test.Decline.pyHistoricalOutcomeCount': 352.0,
 'Income': 38778.1227000507,
 'NumX': 63.0,
 'Occupation': 'Communityeducationofficer',
 'pyName': 'P1'}

def test_plotTreeZero(treeSample, sampledX):
    treeSample.plotTree(42, highlighted=sampledX, show=False)

def test_score(treeSample, sampledX):
    assert treeSample.score(sampledX) == 0.020463661594252957

def test_plotContributionPerTree(treeSample, sampledX):
    treeSample.plotContributionPerTree(sampledX, show=False)