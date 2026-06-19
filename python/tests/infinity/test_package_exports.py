"""Tests for the lazy re-exports on the ``pdstools.infinity`` package."""

from __future__ import annotations

import pytest

import pdstools.infinity as inf
from pdstools.infinity import (
    ModelData,
    ModelInstanceData,
    NotificationData,
    PredictionData,
)
from pdstools.infinity.resources.prediction_studio import schemas


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ModelData", schemas.ModelData),
        ("ModelInstanceData", schemas.ModelInstanceData),
        ("NotificationData", schemas.NotificationData),
        ("PredictionData", schemas.PredictionData),
    ],
)
def test_data_model_reexport_is_canonical_class(name, expected):
    """The package-level alias is the same object as the deep-path class."""
    assert getattr(inf, name) is expected


def test_data_models_in_all():
    for name in (
        "ModelData",
        "ModelInstanceData",
        "NotificationData",
        "PredictionData",
    ):
        assert name in inf.__all__


def test_client_entrypoints_in_all():
    assert "Infinity" in inf.__all__
    assert "AsyncInfinity" in inf.__all__


def test_reexported_model_exposes_locked_schema():
    """A re-exported model still drives the locked DataFrame schema."""
    assert list(ModelData.polars_schema().keys()) == [
        "model_id",
        "label",
        "model_type",
        "modeling_technique",
        "source",
        "status",
        "component_name",
        "last_update_time",
        "updated_by",
    ]


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="has no attribute 'Nonexistent'"):
        inf.__getattr__("Nonexistent")


def test_data_model_missing_pydantic_returns_sentinel(monkeypatch):
    """When pydantic is absent the lazy branch yields a deferred-error sentinel."""

    def fake_find_spec(name: str):
        if name == "pydantic":
            return None
        return object()

    monkeypatch.setattr(inf, "find_spec", fake_find_spec)
    sentinel = inf.__getattr__("PredictionData")
    assert isinstance(sentinel, inf.DependencyNotFound)
    assert sentinel.dependencies == ["pydantic"]


def test_reexported_classes_are_pydantic_models():
    for cls in (ModelData, ModelInstanceData, NotificationData, PredictionData):
        assert hasattr(cls, "model_validate")
        assert hasattr(cls, "polars_schema")
