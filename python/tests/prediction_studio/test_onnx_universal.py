"""
Tests for PyTorch ONNX conversion, enterprise metadata, and security helpers.

Run with:
    uv run pytest python/tests/prediction_studio/test_onnx_universal.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _pytorch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def simple_pytorch_model():
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(self.linear(x))

    return TinyModel()


@pytest.fixture()
def dummy_input():
    torch = pytest.importorskip("torch")
    return torch.randn(1, 4)


@pytest.fixture()
def tmp_onnx_path(tmp_path: Path) -> Path:
    return tmp_path / "test_model.onnx"


# ============================================================================
# Test: Predictor enhancements
# ============================================================================


class TestPredictor:
    def test_predictor_basic(self):
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Predictor

        p = Predictor(name="Age", index=1, input_name="features")
        assert p.name == "Age"
        assert p.index == 1
        assert p.data_type == "Numeric"
        assert p.pega_property is None

    def test_predictor_symbolic(self):
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Predictor

        p = Predictor(
            name="ContractType",
            index=6,
            input_name="features",
            data_type="Symbolic",
            pega_property=".Customer.ContractType",
        )
        assert p.data_type == "Symbolic"
        assert p.pega_property == ".Customer.ContractType"

    def test_predictor_invalid_data_type(self):
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Predictor

        with pytest.raises(ValueError, match="data_type"):
            Predictor(name="X", index=1, input_name="inp", data_type="Invalid")

    def test_predictor_invalid_index(self):
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Predictor

        with pytest.raises(ValueError, match="index"):
            Predictor(name="X", index=0, input_name="inp")

    def test_auto_name_from_pega_property(self):
        """When name is omitted, it is derived from pega_property leaf."""
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Predictor

        p = Predictor(pega_property=".Customer.Age", index=1, input_name="features")
        assert p.name == "Age"
        assert p.pega_property == ".Customer.Age"

    def test_auto_name_deep_path(self):
        """Leaf extraction works for deeply nested Pega property paths."""
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Predictor

        p = Predictor(
            pega_property=".Customer.Account.MonthlyCharges",
            index=3,
            input_name="features",
        )
        assert p.name == "MonthlyCharges"
        assert p.pega_property == ".Customer.Account.MonthlyCharges"

    def test_explicit_name_overrides_pega_property(self):
        """When both name and pega_property are given, name wins."""
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Predictor

        p = Predictor(
            name="custom_age",
            pega_property=".Customer.Age",
            index=1,
            input_name="features",
        )
        assert p.name == "custom_age"
        assert p.pega_property == ".Customer.Age"

    def test_no_name_no_pega_property_raises(self):
        """Omitting both name and pega_property must raise."""
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Predictor

        with pytest.raises(ValueError, match="name"):
            Predictor(index=1, input_name="inp")


# ============================================================================
# Test: Metadata with enterprise fields
# ============================================================================


class TestMetadata:
    def test_metadata_roundtrip_json(self):
        from pdstools.infinity.resources.prediction_studio.local_model_utils import (
            Metadata,
            OutcomeType,
            Output,
            Predictor,
        )

        meta = Metadata(
            type=OutcomeType.BINARY,
            output=Output(label_name="label", score_name="prob"),
            predictor_list=[Predictor(name="Age", index=1, input_name="features")],
            model_version="1.0.0",
            created_by="Falcons",
            baseline_auc=0.87,
        )
        json_str = meta.to_json()
        parsed = json.loads(json_str)

        # Enterprise fields are camelCased in JSON
        assert parsed["type"] == "binary"
        assert parsed["pyModelVersion"] == "1.0.0"
        assert parsed["pyCreatedBy"] == "Falcons"
        assert parsed["pyBaselineAUC"] == 0.87

    def test_metadata_from_json(self):
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Metadata

        raw = json.dumps(
            {
                "type": "binary",
                "output": {"labelName": "pred"},
                "predictorList": [{"name": "A", "index": 1, "inputName": "inp"}],
                "pyModelVersion": "2.0",
            }
        )
        meta = Metadata.from_json(raw)
        assert meta.model_version == "2.0"

    def test_metadata_predictor_data_type_roundtrip(self):
        from pdstools.infinity.resources.prediction_studio.local_model_utils import (
            Metadata,
            OutcomeType,
            Output,
            Predictor,
        )

        meta = Metadata(
            type=OutcomeType.BINARY,
            output=Output(label_name="label"),
            predictor_list=[
                Predictor(name="Age", index=1, input_name="inp", data_type="Numeric"),
                Predictor(
                    name="Region",
                    index=2,
                    input_name="inp",
                    data_type="Symbolic",
                    pega_property=".Customer.Region",
                ),
            ],
        )
        json_str = meta.to_json()
        parsed = json.loads(json_str)
        predictors = parsed["predictorList"]
        # dataType appears only when explicitly set
        assert predictors[0]["dataType"] == "Numeric"
        assert predictors[1]["dataType"] == "Symbolic"
        assert predictors[1]["pegaProperty"] == ".Customer.Region"

    def test_build_predictor_list_pega_paths(self):
        """Pega property paths: name auto-derived, pegaProperty stored."""
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Metadata

        predictors = Metadata.build_predictor_list(
            [".Customer.Age", ".Customer.Tenure", ".Customer.MonthlyCharges"],
            input_name="features",
        )
        assert len(predictors) == 3
        assert predictors[0].name == "Age"
        assert predictors[0].pega_property == ".Customer.Age"
        assert predictors[0].index == 1
        assert predictors[1].name == "Tenure"
        assert predictors[1].index == 2
        assert predictors[2].name == "MonthlyCharges"
        assert predictors[2].index == 3

    def test_build_predictor_list_plain_names(self):
        """Plain feature names: used as-is, no pegaProperty."""
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Metadata

        predictors = Metadata.build_predictor_list(
            ["Age", "Tenure", "MonthlyCharges"],
        )
        assert predictors[0].name == "Age"
        assert predictors[0].pega_property is None
        assert predictors[1].name == "Tenure"

    def test_build_predictor_list_mixed(self):
        """Mix of Pega paths and plain names in a single call."""
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Metadata

        predictors = Metadata.build_predictor_list(
            [".Customer.Age", "Tenure", ".Customer.MonthlyCharges"],
        )
        assert predictors[0].name == "Age"
        assert predictors[0].pega_property == ".Customer.Age"
        assert predictors[1].name == "Tenure"
        assert predictors[1].pega_property is None
        assert predictors[2].name == "MonthlyCharges"
        assert predictors[2].pega_property == ".Customer.MonthlyCharges"

    def test_build_predictor_list_with_data_types(self):
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Metadata

        predictors = Metadata.build_predictor_list(
            [".Customer.Age", ".Customer.Region"],
            input_name="inp",
            data_types=["Numeric", "Symbolic"],
        )
        assert predictors[0].data_type == "Numeric"
        assert predictors[1].data_type == "Symbolic"
        assert predictors[1].name == "Region"

    def test_build_predictor_list_with_data_types_dict(self):
        """Dict-based data_types applies sparse overrides; rest default Numeric."""
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Metadata

        predictors = Metadata.build_predictor_list(
            [".Customer.Age", ".Customer.Region", ".Customer.Tenure"],
            input_name="inp",
            data_types={"Region": "Symbolic"},
        )
        assert predictors[0].data_type == "Numeric"
        assert predictors[1].data_type == "Symbolic"
        assert predictors[2].data_type == "Numeric"

    def test_build_predictor_list_json_auto_mapping(self):
        """Verify JSON output has correct name/pegaProperty for auto-mapping."""
        from pdstools.infinity.resources.prediction_studio.local_model_utils import (
            Metadata,
            OutcomeType,
            Output,
        )

        meta = Metadata(
            type=OutcomeType.BINARY,
            output=Output(label_name="pred"),
            predictor_list=Metadata.build_predictor_list(
                [".Customer.Age", ".Customer.NumSupportTickets"],
                input_name="features",
            ),
        )
        parsed = json.loads(meta.to_json())
        names = [p["name"] for p in parsed["predictorList"]]
        assert names == ["Age", "NumSupportTickets"]
        pega_props = [p["pegaProperty"] for p in parsed["predictorList"]]
        assert pega_props == [".Customer.Age", ".Customer.NumSupportTickets"]

    def test_build_predictor_list_plain_names_no_pega_property_in_json(self):
        """Plain names should NOT produce pegaProperty in JSON output."""
        from pdstools.infinity.resources.prediction_studio.local_model_utils import (
            Metadata,
            OutcomeType,
            Output,
        )

        meta = Metadata(
            type=OutcomeType.BINARY,
            output=Output(label_name="pred"),
            predictor_list=Metadata.build_predictor_list(
                ["Age", "Tenure"],
                input_name="features",
            ),
        )
        parsed = json.loads(meta.to_json())
        for p in parsed["predictorList"]:
            assert "pegaProperty" not in p
            assert p["dataType"] == "Numeric"  # always included

    def test_build_predictor_list_length_mismatch(self):
        from pdstools.infinity.resources.prediction_studio.local_model_utils import Metadata

        with pytest.raises(ValueError, match="Length of data_types"):
            Metadata.build_predictor_list(
                [".Customer.Age", ".Customer.Tenure"],
                data_types=["Numeric"],
            )


# ============================================================================
# Test: ONNXModel.from_pytorch
# ============================================================================


class TestONNXModelFromPyTorch:
    @pytest.mark.skipif(not _pytorch_available(), reason="torch not installed")
    def test_from_pytorch_fixed_shapes(self, simple_pytorch_model, dummy_input, tmp_onnx_path):
        from pdstools.infinity.resources.prediction_studio.local_model_utils import ONNXModel

        onnx_model = ONNXModel.from_pytorch(
            simple_pytorch_model,
            dummy_input,
            input_names=["features"],
            output_names=["prediction"],
            fixed_batch_size=True,
        )

        # All dimensions must be fixed integers
        for inp in onnx_model._model.graph.input:
            for dim in inp.type.tensor_type.shape.dim:
                assert dim.dim_param == "", f"Dynamic dim found: {dim.dim_param}"
                assert dim.dim_value >= 1

        # Save and reload
        import onnx

        onnx_model.save(str(tmp_onnx_path))
        reloaded = onnx.load(str(tmp_onnx_path))
        assert len(reloaded.graph.node) > 0

    @pytest.mark.skipif(not _pytorch_available(), reason="torch not installed")
    def test_from_pytorch_with_metadata(self, simple_pytorch_model, dummy_input):
        from pdstools.infinity.resources.prediction_studio.local_model_utils import (
            Metadata,
            ONNXModel,
            OutcomeType,
            Output,
            Predictor,
        )

        onnx_model = ONNXModel.from_pytorch(
            simple_pytorch_model,
            dummy_input,
            fixed_batch_size=True,
        )

        meta = Metadata(
            type=OutcomeType.BINARY,
            output=Output(label_name="output"),
            predictor_list=[Predictor(name=f"F{i}", index=i, input_name="input") for i in range(1, 5)],
        )
        onnx_model.add_metadata(meta)

        # Verify round-trip via get_metadata
        retrieved = onnx_model.get_metadata()
        assert retrieved is not None
        assert retrieved.type == OutcomeType.BINARY
        assert len(retrieved.predictor_list) == 4


# ============================================================================
# Test: Security
# ============================================================================


class TestModelSecurity:
    def test_clean_metadata_passes(self):
        from pdstools.infinity.resources.prediction_studio.model_security import (
            ModelSecurityValidator,
        )

        validator = ModelSecurityValidator()
        result = validator.validate_metadata(
            {
                "type": "binary",
                "predictorList": [{"name": "Age", "index": 1, "inputName": "inp"}],
                "output": {"labelName": "pred"},
            }
        )
        assert result.is_secure

    def test_unauthorized_key_flagged(self):
        from pdstools.infinity.resources.prediction_studio.model_security import (
            ModelSecurityValidator,
        )

        validator = ModelSecurityValidator()
        result = validator.validate_metadata(
            {
                "type": "binary",
                "predictorList": [],
                "output": {"labelName": "pred"},
                "maliciousKey": "value",
            }
        )
        assert not result.is_secure
        assert any("Unauthorized key" in i for i in result.issues)

    def test_extra_allowed_keys(self):
        from pdstools.infinity.resources.prediction_studio.model_security import (
            ModelSecurityValidator,
        )

        validator = ModelSecurityValidator(extra_allowed_keys={"customKey"})
        result = validator.validate_metadata(
            {
                "type": "binary",
                "predictorList": [],
                "output": {"labelName": "pred"},
                "customKey": "allowed",
            }
        )
        assert result.is_secure

    def test_sanitize_removes_bad_keys(self):
        from pdstools.infinity.resources.prediction_studio.model_security import (
            ModelSecurityValidator,
        )

        validator = ModelSecurityValidator()
        sanitized = validator.sanitize_metadata(
            {
                "type": "binary",
                "predictorList": [],
                "output": {"labelName": "x"},
                "badKey": "should be removed",
            }
        )
        assert "badKey" not in sanitized
        assert "type" in sanitized

    def test_compute_and_verify_hash(self, tmp_path):
        from pdstools.infinity.resources.prediction_studio.model_security import (
            compute_model_hash,
            verify_model_hash,
        )

        f = tmp_path / "dummy.bin"
        f.write_bytes(b"test data")
        h = compute_model_hash(f)
        assert len(h) == 64  # SHA-256 hex
        assert verify_model_hash(f, h)
        assert not verify_model_hash(f, "wrong_hash")
