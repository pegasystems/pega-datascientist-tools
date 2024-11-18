"""
Testing the functionality of the ADMDatamart functions
"""

import pathlib

import pytest
from pdstools.adm.ADMTrees import ADMTrees, ADMTreesModel

basePath = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture
def tree_sample() -> ADMTreesModel:
    """Fixture to serve as class to call functions from."""
    return ADMTrees(f"{basePath}/data/agb/_974a7f9c-66a6-4f00-bf3e-3acf5f188b1d.txt")


def test_has_models(tree_sample: ADMTreesModel):
    assert isinstance(tree_sample.model, list)
    assert len(tree_sample.model) > 1


def test_plot_splits_per_variable(tree_sample: ADMTreesModel):
    tree_sample.plot_splits_per_variable(show=False)


def sample_x(trees):
    from random import sample

    x = {}
    for variable, values in trees.all_values_per_split.items():
        if len(values) == 1:
            if "true" in values or "false" in values:
                values = {"true", "false"}
            if isinstance(list(values)[0], str):
                try:
                    float(list(values)[0])
                except Exception:
                    print("FAILED ON ", values)
                    values = values.union({"Other"})
        x[variable] = sample(list(values), 1)[0]
    return x


@pytest.fixture
def sampledX(tree_sample: ADMTreesModel):
    return sample_x(tree_sample)


# def test_plot_first_tree(tree_sample, sampledX):
#     tree_sample.plot_tree(42, highlighted=sampledX, show=False)


# def test_score(tree_sample, sampledX):
#     assert 0 <= tree_sample.score(sampledX) <= 1


# def test_plotContributionPerTree(tree_sample, sampledX):
#     tree_sample.plot_contribution_per_tree(sampledX, show=False)


# def test_plotSplitsPerVariableType(tree_sample):
#     tree_sample.plot_splits_per_variable_type()
