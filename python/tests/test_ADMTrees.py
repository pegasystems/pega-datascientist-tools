"""Tests for ADM Gradient Boosting (ADMTrees) functionality."""

import pathlib
import re

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
    """First failure logs at INFO; subsequent identical failures log at DEBUG."""
    import logging
    from pdstools.adm.ADMTrees import ADMTreesModel

    # Reset the dedupe set so we get a deterministic INFO on first call.
    ADMTreesModel._safe_eval_seen_errors.clear()
    with caplog.at_level(logging.DEBUG, logger="pdstools.adm.ADMTrees"):
        assert tree_sample._safe_condition_evaluate("not_a_number", "<", 2.0) is False
        assert tree_sample._safe_condition_evaluate("also_bad", "<", 3.0) is False
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("Safe scoring evaluation failed" in r.message for r in info_records), (
        "First failure should log at INFO so users notice scoring is degraded."
    )
    assert any("Safe evaluation failed" in r.message for r in debug_records), (
        "Subsequent failures should log at DEBUG to avoid spamming logs."
    )


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
    # Integer index returns the model directly (was a (key, value) tuple
    # in the legacy code — confusingly inconsistent with string indexing).
    assert mt[0] is a
    assert mt[-1] is b
    assert mt.first is a
    assert mt.last is b
    # items() / values() / keys() expose the dict view explicitly.
    assert list(mt.values()) == [a, b]
    assert list(mt.keys()) == ["2024-01-01", "2024-02-01"]
    assert list(mt.items()) == [("2024-01-01", a), ("2024-02-01", b)]


def test_multitrees_indexing_by_str(two_models):
    """String indexing returns the model directly — fixes a latent bug
    where the old code checked ``isinstance(index, pl.datetime)`` which
    is always False (``pl.datetime`` is a function, not a class)."""
    from pdstools.adm.ADMTrees import MultiTrees

    a, b = two_models
    mt = MultiTrees(trees={"2024-01-01": a, "2024-02-01": b})
    assert mt["2024-01-01"] is a
    assert mt["2024-02-01"] is b


def test_multitrees_first_last_callable(two_models):
    """``mt.first.score(x)`` must work — guards against the asymmetric
    indexing footgun."""
    from pdstools.adm.ADMTrees import MultiTrees

    a, b = two_models
    mt = MultiTrees(trees={"2024-01-01": a, "2024-02-01": b})
    # If first returned a tuple this would AttributeError.
    assert isinstance(mt.first.metrics, dict)
    assert isinstance(mt.last.metrics, dict)


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


def test_multitrees_from_datamart_timestamp_formatting():
    """Regression: ``str.strip_chars_end(\".000000000\")`` strips a *set*
    of characters, not a literal suffix, so timestamps ending in 0 (e.g.
    12:30:20) used to get mangled to ``12:30:2``.  The fix uses
    ``dt.strftime`` which formats explicitly.
    """
    import polars as pl
    from datetime import datetime
    from pdstools.adm.ADMTrees import MultiTrees

    df = pl.DataFrame(
        {
            "SnapshotTime": [
                datetime(2024, 1, 1, 12, 30, 20),  # ends in 0 — would be mangled
                datetime(2024, 1, 1, 12, 30, 5),
            ],
            "Modeldata": [None, None],  # filtered out, we only want the strftime
            "Configuration": ["cfg", "cfg"],
        }
    )
    # Reproduce the projection that from_datamart applies to confirm the
    # formatting; the actual decode pipeline needs real blobs.
    formatted = df.select(
        pl.col("SnapshotTime").dt.round("1s").dt.strftime("%Y-%m-%d %H:%M:%S"),
    )["SnapshotTime"].to_list()
    assert formatted == ["2024-01-01 12:30:20", "2024-01-01 12:30:05"]
    # Sanity: the buggy approach mangles the trailing 0.
    legacy_buggy = df.select(pl.col("SnapshotTime").dt.round("1s").cast(pl.Utf8).str.strip_chars_end(".000000000"))[
        "SnapshotTime"
    ].to_list()
    assert "12:30:20" not in legacy_buggy[0]  # confirms the bug exists
    # Reference MultiTrees so it's exercised even though we don't construct it.
    assert callable(MultiTrees.from_datamart)


def test_multitrees_add_admtreesmodel_without_timestamp_raises(exported_model):
    """A model without ``factory_update_time`` would silently use 'None'
    as a key in legacy code; now we raise instead."""
    from pdstools.adm.ADMTrees import MultiTrees

    mt = MultiTrees(trees={})
    # Wipe the factory timestamp on the underlying _properties and clear
    # the metrics cached_property so __add__ sees a falsy timestamp.
    exported_model._properties = {**exported_model._properties, "factoryUpdateTime": None}
    exported_model.__dict__.pop("metrics", None)
    with pytest.raises(ValueError, match="factory_update_time"):
        mt + exported_model


def test_multitrees_from_datamart_rejects_multi_config(exported_model):
    """Passing a multi-config DataFrame to from_datamart should raise —
    callers must use from_datamart_grouped or pass `configuration=`."""
    import polars as pl
    from datetime import datetime
    from pdstools.adm.ADMTrees import MultiTrees, ADMTreesModel
    import pdstools.adm.ADMTrees as mod

    df = pl.DataFrame(
        {
            "SnapshotTime": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            # Valid base64 (any string works since we stub the decoder).
            "Modeldata": ["YQ==", "Yg=="],
            "Configuration": ["cfg_a", "cfg_b"],
        }
    )
    real = ADMTreesModel.from_datamart_blob

    def fake(_):
        return exported_model

    mod.ADMTreesModel.from_datamart_blob = staticmethod(fake)  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="2 configurations"):
            MultiTrees.from_datamart(df)
        # ...but explicit configuration= picks one cleanly.
        picked = MultiTrees.from_datamart(df, configuration="cfg_a")
        assert picked.model_name == "cfg_a"
        assert len(picked.trees) == 1
    finally:
        mod.ADMTreesModel.from_datamart_blob = real  # type: ignore[assignment]


def test_multitrees_from_datamart_grouped(exported_model):
    """from_datamart_grouped returns a {config: MultiTrees} dict."""
    import polars as pl
    from datetime import datetime
    from pdstools.adm.ADMTrees import MultiTrees, ADMTreesModel
    import pdstools.adm.ADMTrees as mod

    df = pl.DataFrame(
        {
            "SnapshotTime": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "Modeldata": ["YQ==", "Yg==", "Yw=="],
            "Configuration": ["cfg_a", "cfg_a", "cfg_b"],
        }
    )
    real = ADMTreesModel.from_datamart_blob
    mod.ADMTreesModel.from_datamart_blob = staticmethod(lambda _: exported_model)  # type: ignore[assignment]
    try:
        out = MultiTrees.from_datamart_grouped(df)
    finally:
        mod.ADMTreesModel.from_datamart_blob = real  # type: ignore[assignment]
    assert set(out) == {"cfg_a", "cfg_b"}
    assert len(out["cfg_a"]) == 2
    assert len(out["cfg_b"]) == 1
    assert out["cfg_a"].model_name == "cfg_a"


def test_parse_split_values_emits_deprecation_warning(tree_sample):
    with pytest.warns(DeprecationWarning, match="parse_split_values"):
        tree_sample.parse_split_values("Age < 42")


def test_admtreesmodel_string_constructor_emits_deprecation_warning():
    """The legacy ``ADMTreesModel(file)`` constructor must warn."""
    with pytest.warns(DeprecationWarning, match="from_file"):
        ADMTreesModel(f"{basePath}/data/agb/ModelExportExample.json")


def test_score_matches_per_tree_sum(tree_sample, sampledX):
    """``score(x)`` is sigmoid of the sum of per-tree leaf scores; the
    refactored fast path must match the equivalent
    ``get_all_visited_nodes`` aggregation exactly."""
    from math import exp

    fast = tree_sample.score(sampledX)
    df = tree_sample.get_all_visited_nodes(sampledX)
    slow = 1 / (1 + exp(-df["score"].sum()))
    assert fast == pytest.approx(slow, rel=1e-12)


def test_splits_and_gains_lengths_aligned(tree_sample):
    """Regression: ``gains_per_split`` rows must match ``splits_per_tree``
    in count.  Legacy code dropped zero-gain splits from the gains list,
    silently misaligning split→gain pairs in the fallback path."""
    total_splits = sum(len(s) for s in tree_sample.splits_per_tree.values())
    assert tree_sample.gains_per_split.height == total_splits


def test_score_missing_predictor_raises_helpful_keyerror(tree_sample, sampledX):
    """Calling score with an incomplete feature dict should raise a KeyError
    that names the missing predictor (not a bare KeyError on the variable)."""
    # Capture the predictors actually accessed when scoring the full dict —
    # picking from the model's full predictor list isn't enough because some
    # predictors may never be hit on this particular traversal path.
    accessed: set[str] = set()

    class _RecordingDict(dict):
        def __getitem__(self, key):
            accessed.add(key)
            return super().__getitem__(key)

    tree_sample.score(_RecordingDict(sampledX))
    assert accessed, "fixture sanity: scoring sampledX should access at least one predictor"
    a_predictor = next(iter(accessed))
    incomplete = dict(sampledX)
    incomplete.pop(a_predictor)
    with pytest.raises(KeyError, match=re.escape(repr(a_predictor))):
        tree_sample.score(incomplete)


# ---------------------------------------------------------------------------
# AGB datamart-helper tests (synthetic fixture)
# ---------------------------------------------------------------------------


@pytest.fixture
def agb_datamart_stub():
    """Build a stub object that quacks like ADMDatamart for AGB's purposes.

    AGB only touches ``datamart.model_data`` (a LazyFrame) and
    ``datamart.aggregates.last(table=...)`` — so a SimpleNamespace with
    those attributes is enough to exercise discover_model_types and
    get_agb_models without standing up a real ADMDatamart.

    The frame contains two synthetic configurations:
    - ``WebClickthroughAGB`` — a real AGB blob (re-encoded from the
      shipped sample model JSON), so its _serialClass ends with
      ``GbModel``.
    - ``CaseModelNB`` — a hand-written NaiveBayes blob, which AGB
      should *exclude* from get_agb_models.
    """
    import base64
    import json
    import zlib
    from datetime import datetime
    from types import SimpleNamespace

    import polars as pl

    sample_path = basePath / "data" / "agb" / "_974a7f9c-66a6-4f00-bf3e-3acf5f188b1d.txt"
    raw_agb_save_model = json.loads(sample_path.read_text())
    # The "save model" export omits _serialClass and isn't wrapped the way a
    # datamart Modeldata blob is. Wrap it so that trees["model"]["model"]
    # ["boosters"][0]["trees"] resolves to the actual tree list.
    datamart_payload = {
        "_serialClass": "com.pega.decision.adm.client.impl.GbModel",
        "model": raw_agb_save_model["model"],
    }
    agb_blob = base64.b64encode(zlib.compress(json.dumps(datamart_payload).encode())).decode()

    nb_obj = {"_serialClass": "com.pega.decision.adm.client.impl.NaiveBayesModel"}
    nb_blob = base64.b64encode(zlib.compress(json.dumps(nb_obj).encode())).decode()

    df = pl.LazyFrame(
        {
            "Configuration": ["WebClickthroughAGB", "WebClickthroughAGB", "CaseModelNB"],
            "Modeldata": [agb_blob, agb_blob, nb_blob],
            "SnapshotTime": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 1),
            ],
        }
    )
    aggregates = SimpleNamespace(last=lambda table="model_data": df)
    return SimpleNamespace(model_data=df, aggregates=aggregates)


def test_agb_discover_model_types_returns_serial_classes(agb_datamart_stub):
    from pdstools.adm.ADMTrees import AGB

    agb = AGB(agb_datamart_stub)
    types = agb.discover_model_types(agb_datamart_stub.model_data, by="Configuration")
    assert types["WebClickthroughAGB"].endswith("GbModel")
    assert types["CaseModelNB"].endswith("NaiveBayesModel")


def test_agb_discover_model_types_rejects_missing_modeldata():
    import polars as pl
    from pdstools.adm.ADMTrees import AGB

    df = pl.LazyFrame({"Configuration": ["x"]})
    agb = AGB(datamart=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Modeldata column"):
        agb.discover_model_types(df)


def test_agb_get_agb_models_filters_to_gb_only(agb_datamart_stub, monkeypatch):
    """``get_agb_models`` should filter to ``GbModel`` configs and call
    ``MultiTrees.from_datamart`` on each. We monkeypatch the blob decoder
    because the save-model JSON we use as a fixture lacks the
    ``inputsEncoder`` block that real datamart blobs carry; the actual
    decoding path is covered by ``tree_sample``.
    """
    from pdstools.adm import ADMTrees as mod
    from pdstools.adm.ADMTrees import AGB, MultiTrees

    monkeypatch.setattr(
        mod.ADMTreesModel,
        "from_datamart_blob",
        staticmethod(
            lambda blob, **_: mod.ADMTreesModel.from_file(
                str(basePath / "data" / "agb" / "_974a7f9c-66a6-4f00-bf3e-3acf5f188b1d.txt"),
            )
        ),
    )

    agb = AGB(agb_datamart_stub)
    result = agb.get_agb_models(n_threads=1)
    assert set(result) == {"WebClickthroughAGB"}, "non-AGB configs must be excluded"
    assert isinstance(result["WebClickthroughAGB"], MultiTrees)
    # Two snapshots in the fixture -> two trees decoded into the MultiTrees
    assert len(result["WebClickthroughAGB"].trees) == 2


def test_agb_get_agb_models_with_last_uses_aggregates(agb_datamart_stub, monkeypatch):
    from pdstools.adm import ADMTrees as mod
    from pdstools.adm.ADMTrees import AGB

    monkeypatch.setattr(
        mod.ADMTreesModel,
        "from_datamart_blob",
        staticmethod(
            lambda blob, **_: mod.ADMTreesModel.from_file(
                str(basePath / "data" / "agb" / "_974a7f9c-66a6-4f00-bf3e-3acf5f188b1d.txt"),
            )
        ),
    )

    calls: list[str] = []

    def spy_last(table="model_data"):
        calls.append(table)
        return agb_datamart_stub.model_data

    agb_datamart_stub.aggregates.last = spy_last
    agb = AGB(agb_datamart_stub)
    agb.get_agb_models(last=True, n_threads=1)
    assert calls == ["model_data"]
