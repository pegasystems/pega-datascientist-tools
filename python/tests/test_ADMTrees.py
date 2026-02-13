"""
Testing the functionality of the ADMDatamart functions
"""

import pathlib

import pytest
from pdstools.adm.ADMTrees import ADMTrees, ADMTreesModel

basePath = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture
def tree_sample() -> ADMTrees:
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
def sampledX(tree_sample: ADMTrees):
    return sample_x(tree_sample)


@pytest.mark.skip(reason="Test disabled - needs investigation")
def test_plot_first_tree(tree_sample, sampledX):
    tree_sample.plot_tree(42, highlighted=sampledX, show=False)


@pytest.mark.skip(reason="Test disabled - needs investigation")
def test_score(tree_sample, sampledX):
    assert 0 <= tree_sample.score(sampledX) <= 1


@pytest.mark.skip(reason="Test disabled - needs investigation")
def test_plotContributionPerTree(tree_sample, sampledX):
    tree_sample.plot_contribution_per_tree(sampledX, show=False)


@pytest.mark.skip(reason="Test disabled - needs investigation")
def test_plotSplitsPerVariableType(tree_sample):
    tree_sample.plot_splits_per_variable_type()


# --- metrics tests ---------------------------------------------------------


@pytest.fixture
def exported_model() -> ADMTreesModel:
    """Fixture for a decoded/exported model (no inputsEncoder)."""
    return ADMTrees(f"{basePath}/data/agb/ModelExportExample.json")


def test_metrics_returns_dict(tree_sample: ADMTreesModel):
    m = tree_sample.metrics
    assert isinstance(m, dict)
    assert len(m) > 0


def test_metrics_required_keys(tree_sample: ADMTreesModel):
    m = tree_sample.metrics
    required = {
        "auc",
        "number_of_tree_nodes",
        "tree_depth_max",
        "tree_depth_avg",
        "number_of_trees",
        "total_number_of_active_predictors",
        "total_number_of_predictors",
        "number_of_active_ih_predictors",
        "number_of_active_context_key_predictors",
        "number_of_active_symbolic_predictors",
        "number_of_active_numeric_predictors",
        "number_of_splits_on_ih_predictors",
        "number_of_splits_on_context_key_predictors",
        "number_of_splits_on_other_predictors",
    }
    assert required.issubset(m.keys()), f"Missing keys: {required - m.keys()}"


def test_metrics_tree_counts(tree_sample: ADMTreesModel):
    m = tree_sample.metrics
    assert m["number_of_trees"] == len(tree_sample.model)
    assert m["number_of_trees"] > 0
    assert m["number_of_tree_nodes"] > m["number_of_trees"]
    assert m["tree_depth_max"] > 0
    assert m["tree_depth_avg"] > 0


def test_metrics_splits_add_up(tree_sample: ADMTreesModel):
    m = tree_sample.metrics
    total_splits = (
        m["number_of_splits_on_ih_predictors"]
        + m["number_of_splits_on_context_key_predictors"]
        + m["number_of_splits_on_other_predictors"]
    )
    assert total_splits > 0


def test_metrics_predictor_counts(tree_sample: ADMTreesModel):
    m = tree_sample.metrics
    assert m["total_number_of_active_predictors"] > 0
    assert m["total_number_of_predictors"] >= m["total_number_of_active_predictors"]


def test_metrics_exported_model(exported_model: ADMTreesModel):
    m = exported_model.metrics
    assert m["number_of_trees"] == 50
    assert m["number_of_tree_nodes"] > 0
    assert m["auc"] is not None
    assert m["response_positive_count"] == 803
    assert m["response_negative_count"] == 14197


def test_metrics_no_encoder(exported_model: ADMTreesModel):
    """Exported models have no inputsEncoder, so no saturation metrics."""
    m = exported_model.metrics
    assert "number_of_saturated_context_key_predictors" not in m
    assert exported_model._get_encoder_info() is None
