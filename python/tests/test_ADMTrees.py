"""Tests for ADM Gradient Boosting (ADMTrees) functionality."""

import pathlib

import pytest
from pdstools.adm.ADMTrees import ADMTreesModel, parse_split

basePath = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture
def tree_sample() -> ADMTreesModel:
    """Fixture: legacy AGB export (no encoder, no AUC/response counts)."""
    return ADMTreesModel.from_file(f"{basePath}/data/agb/_974a7f9c-66a6-4f00-bf3e-3acf5f188b1d.txt")


def test_has_models(tree_sample: ADMTreesModel):
    assert isinstance(tree_sample.model, list)
    assert len(tree_sample.model) == 50


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


def test_plot_first_tree(tree_sample, sampledX):
    tree_sample.plot_tree(42, highlighted=sampledX, show=False)


def test_score(tree_sample, sampledX):
    assert 0 <= tree_sample.score(sampledX) <= 1


def test_plotContributionPerTree(tree_sample, sampledX):
    tree_sample.plot_contribution_per_tree(sampledX, show=False)


def test_plotSplitsPerVariableType(tree_sample):
    tree_sample.plot_splits_per_variable_type()


# --- metrics tests ---------------------------------------------------------


@pytest.fixture
def exported_model() -> ADMTreesModel:
    """Fixture for a decoded/exported model (no inputsEncoder)."""
    return ADMTreesModel.from_file(f"{basePath}/data/agb/ModelExportExample.json")


def test_metrics_returns_dict(tree_sample: ADMTreesModel):
    m = tree_sample.metrics
    assert isinstance(m, dict)
    # Sanity floor on metric coverage — refactor must not silently drop metrics.
    assert len(m) >= 40


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
    assert m["number_of_trees"] == 50
    assert m["number_of_tree_nodes"] == 2232
    assert m["tree_depth_max"] == 10
    assert m["tree_depth_avg"] == 7.8
    assert m["tree_depth_std"] == 1.92


def test_metrics_splits_add_up(tree_sample: ADMTreesModel):
    m = tree_sample.metrics
    by_role = (
        m["number_of_splits_on_ih_predictors"]
        + m["number_of_splits_on_context_key_predictors"]
        + m["number_of_splits_on_other_predictors"]
    )
    by_type = m["number_of_numeric_splits"] + m["number_of_symbolic_splits"]
    assert by_role == 1091
    # by_type only counts <, in, ==; "is" splits are excluded — so always <= role count.
    assert by_type == 1033
    assert m["number_of_splits_on_ih_predictors"] == 292
    assert m["number_of_splits_on_context_key_predictors"] == 236
    assert m["number_of_splits_on_other_predictors"] == 563


def test_metrics_predictor_counts(tree_sample: ADMTreesModel):
    m = tree_sample.metrics
    assert m["total_number_of_active_predictors"] == 41
    assert m["total_number_of_predictors"] == 83


def test_metrics_exported_model(exported_model: ADMTreesModel):
    m = exported_model.metrics
    assert m["number_of_trees"] == 50
    assert m["number_of_tree_nodes"] == 866
    assert m["auc"] == pytest.approx(0.0535, abs=1e-3)
    assert m["response_positive_count"] == 803
    assert m["response_negative_count"] == 14197
    assert m["factory_update_time"] == "2022-03-24T14:36:19.902Z"


def test_metrics_no_encoder(exported_model: ADMTreesModel):
    """Exported models have no inputsEncoder, so no saturation metrics."""
    m = exported_model.metrics
    assert "number_of_saturated_context_key_predictors" not in m
    assert exported_model._get_encoder_info() is None


# --- gain distribution tests -----------------------------------------------


def test_metrics_gain_distribution(tree_sample: ADMTreesModel):
    """Gain distribution metrics for the legacy export."""
    m = tree_sample.metrics
    assert m["total_gain"] == pytest.approx(10306.476, rel=1e-4)
    assert m["max_gain_per_split"] >= m["mean_gain_per_split"] > 0
    assert m["median_gain_per_split"] > 0
    assert m["gain_std"] >= 0


def test_metrics_gain_exported_model(exported_model: ADMTreesModel):
    """Exported model gain metrics — pinned to captured values."""
    m = exported_model.metrics
    assert m["total_gain"] == pytest.approx(1134.7307, rel=1e-4)
    assert m["mean_gain_per_split"] == pytest.approx(2.7812, abs=1e-3)
    assert m["median_gain_per_split"] == pytest.approx(2.0055, abs=1e-3)
    assert m["max_gain_per_split"] == pytest.approx(27.6557, abs=1e-3)


# --- leaf score tests -------------------------------------------------------


def test_metrics_leaf_scores(tree_sample: ADMTreesModel):
    """Leaf score metrics."""
    m = tree_sample.metrics
    # 50 trees * 22.82 avg leaves ≈ 1141.
    assert m["number_of_leaves"] == 1141
    assert m["leaf_score_min"] <= m["leaf_score_mean"] <= m["leaf_score_max"]
    assert m["leaf_score_std"] > 0


def test_metrics_leaf_scores_exported(exported_model: ADMTreesModel):
    """Exported model leaf scores — pinned values."""
    m = exported_model.metrics
    assert m["number_of_leaves"] == 458
    assert m["leaf_score_min"] == pytest.approx(-0.535617, abs=1e-4)
    assert m["leaf_score_max"] == pytest.approx(0.481624, abs=1e-4)
    assert m["leaf_score_mean"] == pytest.approx(0.001646, abs=1e-4)


# --- tree structure tests ---------------------------------------------------


def test_metrics_tree_structure(tree_sample: ADMTreesModel):
    """Structure metrics: stumps, depth std, avg leaves."""
    m = tree_sample.metrics
    assert m["number_of_stump_trees"] == 1
    assert m["avg_leaves_per_tree"] == 22.82


def test_metrics_stump_count_exported(exported_model: ADMTreesModel):
    """Exported model: 4 of 50 trees are stumps."""
    m = exported_model.metrics
    assert m["number_of_stump_trees"] == 4
    assert m["avg_leaves_per_tree"] == 9.16
    assert m["tree_depth_std"] == 2.11


# --- split type tests -------------------------------------------------------


def test_metrics_split_types(tree_sample: ADMTreesModel):
    """Split-type metrics for the legacy export — pinned values."""
    m = tree_sample.metrics
    assert m["number_of_numeric_splits"] == 750
    assert m["number_of_symbolic_splits"] == 283
    assert m["symbolic_split_fraction"] == pytest.approx(283 / 1033, abs=1e-3)
    assert m["split_reuse_ratio"] >= 1.0
    assert m["avg_symbolic_set_size"] == pytest.approx(9.77, abs=1e-2)


def test_metrics_split_types_exported(exported_model: ADMTreesModel):
    """Exported model split-type metrics — pinned values."""
    m = exported_model.metrics
    assert m["number_of_numeric_splits"] == 180
    assert m["number_of_symbolic_splits"] == 225
    assert m["symbolic_split_fraction"] == pytest.approx(0.5556, abs=1e-3)
    assert m["number_of_unique_splits"] == 215
    assert m["number_of_unique_predictors_split_on"] == 7
    assert m["avg_symbolic_set_size"] == pytest.approx(31.31, abs=1e-2)


# --- convergence tests ------------------------------------------------------


def test_metrics_convergence(tree_sample: ADMTreesModel):
    """Convergence metrics — pinned values."""
    m = tree_sample.metrics
    assert m["mean_abs_score_first_10"] == pytest.approx(0.308436, abs=1e-4)
    assert m["score_decay_ratio"] == pytest.approx(0.0684, abs=1e-3)
    # First half should boost more than the last half (convergence).
    assert m["mean_gain_first_half"] > m["mean_gain_last_half"]


def test_metrics_convergence_exported(exported_model: ADMTreesModel):
    """Exported model convergence metrics."""
    m = exported_model.metrics
    assert m["mean_abs_score_first_10"] == pytest.approx(0.267244, abs=1e-4)
    assert m["mean_abs_score_last_10"] == pytest.approx(0.031273, abs=1e-4)
    assert m["score_decay_ratio"] == pytest.approx(0.117, abs=1e-3)


# --- feature importance concentration tests ---------------------------------


def test_metrics_feature_importance(tree_sample: ADMTreesModel):
    """Feature importance concentration metrics — pinned values."""
    m = tree_sample.metrics
    assert m["top_predictor_by_gain"] == "pyName"
    assert m["top_predictor_gain_share"] == pytest.approx(0.5481, abs=1e-3)
    assert m["predictor_gain_entropy"] == pytest.approx(0.4206, abs=1e-3)


def test_metrics_feature_importance_exported(exported_model: ADMTreesModel):
    """Exported model feature importance metrics — pinned values."""
    m = exported_model.metrics
    assert m["top_predictor_by_gain"] == "pyName"
    assert m["top_predictor_gain_share"] == pytest.approx(0.2197, abs=1e-3)
    assert m["predictor_gain_entropy"] == pytest.approx(0.9619, abs=1e-3)


# --- safe evaluation helpers -----------------------------------------------


@pytest.mark.parametrize(
    ("left", "op", "right", "expected"),
    [
        (1.0, "<", 2.0, True),
        (3.0, "<", 2.0, False),
        (2.0, ">", 1.0, True),
        (1.0, ">", 2.0, False),
        (1.0, "==", 1.0, True),
        (1.0, "==", 2.0, False),
        (1.0, "<=", 1.0, True),
        (2.0, "<=", 1.0, False),
        (1.0, ">=", 1.0, True),
        (1.0, ">=", 2.0, False),
        (1.0, "!=", 2.0, True),
        (1.0, "!=", 1.0, False),
    ],
)
def test_safe_numeric_compare(tree_sample, left, op, right, expected):
    assert tree_sample._safe_numeric_compare(left, op, right) is expected


def test_safe_numeric_compare_unsupported_operator(tree_sample):
    with pytest.raises(ValueError, match="Unsupported operator"):
        tree_sample._safe_numeric_compare(1.0, "??", 2.0)


@pytest.mark.parametrize(
    ("value", "op", "comparison", "expected"),
    [
        ("a", "in", {"a", "b"}, True),
        ("c", "in", {"a", "b"}, False),
        ("1.5", "<", 2.0, True),
        ("3.0", "<", 2.0, False),
        ("3.0", ">", 2.0, True),
        ("1.0", ">", 2.0, False),
        ("foo", "==", "foo", True),
        ("foo", "==", "bar", False),
    ],
)
def test_safe_condition_evaluate(tree_sample, value, op, comparison, expected):
    assert tree_sample._safe_condition_evaluate(value, op, comparison) is expected


def test_safe_condition_evaluate_unsupported_operator(tree_sample):
    # Unsupported operator raises inside the try, gets caught, returns False
    assert tree_sample._safe_condition_evaluate("x", "??", "y") is False


def test_safe_condition_evaluate_handles_bad_numeric(tree_sample, caplog):
    # Non-numeric string with "<" triggers ValueError -> debug log -> False
    import logging

    with caplog.at_level(logging.DEBUG, logger="pdstools.adm.ADMTrees"):
        assert tree_sample._safe_condition_evaluate("not_a_number", "<", 2.0) is False
    assert any("Safe evaluation failed" in r.message for r in caplog.records)
    # Verify we log at DEBUG, not WARNING — the function is hot in scoring loops.
    assert all(r.levelno <= logging.DEBUG for r in caplog.records if "Safe evaluation failed" in r.message)


# --- top-level Split / parse_split helpers ---------------------------------


def test_parse_split_numeric():
    s = parse_split("Age < 42.5")
    assert s.variable == "Age"
    assert s.operator == "<"
    assert s.value == 42.5
    assert s.is_numeric and not s.is_symbolic


def test_parse_split_set_membership():
    s = parse_split("Color in { red, blue, green }")
    assert s.variable == "Color"
    assert s.operator == "in"
    assert s.value == ("red", "blue", "green")
    assert s.is_symbolic and not s.is_numeric


def test_parse_split_set_membership_preserves_duplicates():
    # Legacy semantics counted duplicates — we model that with tuple, not set.
    s = parse_split("Color in { red, red, blue }")
    assert s.value == ("red", "red", "blue")
    assert len(s.value) == 3


def test_parse_split_is_missing():
    s = parse_split("Status is Missing")
    assert s.operator == "is"
    assert s.value == "Missing"


def test_parse_split_invalid_raises():
    with pytest.raises(ValueError, match="Cannot parse split"):
        parse_split("totally not a split")


# --- deprecated entry point ------------------------------------------------


def test_legacy_admtrees_factory_warns():
    """The legacy ADMTrees(...) entry point still works but warns."""
    from pdstools.adm.ADMTrees import ADMTrees

    with pytest.warns(DeprecationWarning, match="ADMTrees"):
        m = ADMTrees(f"{basePath}/data/agb/ModelExportExample.json")
    assert isinstance(m, ADMTreesModel)
    assert m.metrics["number_of_trees"] == 50


# --- read pipeline dispatch ------------------------------------------------


def test_from_dict_round_trips_the_model():
    """Loading from a dict skips IO and produces the same model."""
    import json

    raw = json.loads(pathlib.Path(f"{basePath}/data/agb/ModelExportExample.json").read_text())
    m = ADMTreesModel.from_dict(raw)
    assert m.metrics["number_of_trees"] == 50


def test_from_anything_dispatches_dict():
    """dict input goes through from_dict (no IO required)."""
    import json

    raw = json.loads(pathlib.Path(f"{basePath}/data/agb/ModelExportExample.json").read_text())
    m = ADMTreesModel._from_anything(raw)
    assert m.metrics["number_of_trees"] == 50


def test_from_anything_dispatches_existing_path():
    """A path string that exists routes to from_file, not base64."""
    path = f"{basePath}/data/agb/ModelExportExample.json"
    m = ADMTreesModel._from_anything(path)
    assert m.raw_input == path


def test_from_anything_url_string_routes_to_url(monkeypatch):
    """A string starting with http(s):// routes to from_url, not from_file."""
    sentinel = object()

    def fake_from_url(cls, url, **kwargs):
        assert url == "https://example.com/model.json"
        return sentinel

    monkeypatch.setattr(ADMTreesModel, "from_url", classmethod(fake_from_url), raising=True)
    assert ADMTreesModel._from_anything("https://example.com/model.json") is sentinel


def test_from_anything_unknown_string_treated_as_blob():
    """A string that is neither URL nor existing path falls through to
    from_datamart_blob and surfaces its decode error directly."""
    with pytest.raises(Exception):  # noqa: B017 — base64/zlib chain raises various
        ADMTreesModel._from_anything("definitely-not-a-real-path-or-blob")


def test_from_anything_rejects_unsupported_type():
    with pytest.raises(TypeError, match="Unsupported input type"):
        ADMTreesModel._from_anything(12345)


# --- MultiTrees -----------------------------------------------------------


@pytest.fixture
def two_models(exported_model: ADMTreesModel) -> tuple[ADMTreesModel, ADMTreesModel]:
    """Two distinct ADMTreesModel instances loaded from the same export."""
    second = ADMTreesModel.from_file(f"{basePath}/data/agb/ModelExportExample.json")
    return exported_model, second


def test_multitrees_indexing_by_int(two_models):
    from pdstools.adm.ADMTrees import MultiTrees

    a, b = two_models
    mt = MultiTrees(trees={"2024-01-01": a, "2024-02-01": b}, model_name="cfg")
    # Integer index returns the (key, value) pair from insertion order.
    assert mt[0] == ("2024-01-01", a)
    assert mt[-1] == ("2024-02-01", b)
    assert mt.first == ("2024-01-01", a)
    assert mt.last == ("2024-02-01", b)


def test_multitrees_indexing_by_str(two_models):
    """String indexing returns the model directly — fixes a latent bug
    where the old code checked ``isinstance(index, pl.datetime)`` which
    is always False (``pl.datetime`` is a function, not a class)."""
    from pdstools.adm.ADMTrees import MultiTrees

    a, b = two_models
    mt = MultiTrees(trees={"2024-01-01": a, "2024-02-01": b})
    assert mt["2024-01-01"] is a
    assert mt["2024-02-01"] is b


def test_multitrees_add_preserves_metadata(two_models):
    """``__add__`` must preserve ``model_name`` and ``context_keys`` —
    legacy code dropped them silently."""
    from pdstools.adm.ADMTrees import MultiTrees

    a, b = two_models
    left = MultiTrees(trees={"2024-01-01": a}, model_name="cfg", context_keys=["Issue", "Group"])
    right = MultiTrees(trees={"2024-02-01": b})
    combined = left + right
    assert combined.model_name == "cfg"
    assert combined.context_keys == ["Issue", "Group"]
    assert set(combined.trees) == {"2024-01-01", "2024-02-01"}


def test_multitrees_add_with_admtreesmodel(two_models):
    """Adding an ADMTreesModel keys it by its factory_update_time."""
    from pdstools.adm.ADMTrees import MultiTrees

    a, b = two_models
    mt = MultiTrees(trees={"2024-01-01": a}, model_name="cfg")
    combined = mt + b
    # Exported model has factory_update_time = '2022-03-24T14:36:19.902Z'
    assert "2022-03-24T14:36:19.902Z" in combined.trees
    assert combined.model_name == "cfg"


def test_multitrees_len_and_repr(two_models):
    from pdstools.adm.ADMTrees import MultiTrees

    a, b = two_models
    mt = MultiTrees(trees={"2024-01-01": a, "2024-02-01": b}, model_name="cfg")
    assert len(mt) == 2
    r = repr(mt)
    assert "cfg" in r and "2" in r
