"""
Lightweight security helpers for Pega ONNX models.

Provides:
- Extensible allow-list validation for ``pegaMetadata`` keys.
- Size-limit checks (predictor count, possible values, string length).
- SHA-256 hashing / verification of model files.

Usage
-----
>>> from pdstools.infinity.resources.prediction_studio.model_security import (
...     ModelSecurityValidator,
...     compute_model_hash,
...     verify_model_hash,
... )
>>> validator = ModelSecurityValidator()
>>> result = validator.validate_metadata(metadata_dict)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default allow-lists (callers may extend via constructor)
# ---------------------------------------------------------------------------

ALLOWED_PEGA_KEYS: set[str] = {
    "type",
    "predictorList",
    "output",
    "modelingTechnique",
    "internal",
    # Enterprise (Pega 24+)
    "pyFileSource",
    "pyObjective",
    "pyPredictMethodUsesNameValuePair",
    "pyRuleSet",
    "pyRuleSetVersion",
    "pyModelVersion",
    "pyCreatedBy",
    "pyCreatedDate",
    "pyLastModifiedDate",
    "pyTrainingDataset",
    "pyExperimentId",
    "pyParentModelId",
    "pyBaselineAUC",
    "pyBaselineAccuracy",
    "pyPerformanceThreshold",
}

ALLOWED_PREDICTOR_KEYS: set[str] = {
    "name",
    "index",
    "inputName",
    "dataType",
    "pegaProperty",
}

ALLOWED_OUTPUT_KEYS: set[str] = {
    "labelName",
    "scoreName",
    "possibleValues",
    "minValue",
    "maxValue",
}

# Size limits
MAX_STRING_LENGTH: int = 10_000
MAX_PREDICTOR_COUNT: int = 1_000
MAX_POSSIBLE_VALUES: int = 100


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class SecurityResult:
    """Simple validation result container."""

    def __init__(self) -> None:
        self.is_secure: bool = True
        self.issues: list[str] = []

    def _add(self, msg: str, *, fail: bool = False) -> None:
        self.issues.append(msg)
        if fail:
            self.is_secure = False

    def __str__(self) -> str:
        status = "passed" if self.is_secure else "FAILED"
        lines = [f"Security: {status}"]
        for issue in self.issues:
            lines.append(f"  - {issue}")
        return "\n".join(lines)


class ModelSecurityValidator:
    """Validate ``pegaMetadata`` dictionaries against an extensible allow-list.

    Parameters
    ----------
    extra_allowed_keys : set[str] | None
        Additional top-level pegaMetadata keys to accept.
    """

    def __init__(self, extra_allowed_keys: set[str] | None = None) -> None:
        self.allowed_keys = ALLOWED_PEGA_KEYS | (extra_allowed_keys or set())

    def validate_metadata(self, metadata: dict[str, Any]) -> SecurityResult:
        """Check *metadata* against the allow-list and size limits."""
        result = SecurityResult()

        # Top-level keys
        for key in metadata:
            if key not in self.allowed_keys:
                result._add(f"Unauthorized key: '{key}'", fail=True)

        # Predictor list
        predictors = metadata.get("predictorList", [])
        if len(predictors) > MAX_PREDICTOR_COUNT:
            result._add(
                f"Too many predictors ({len(predictors)} > {MAX_PREDICTOR_COUNT})",
                fail=True,
            )
        for i, pred in enumerate(predictors):
            for key in pred:
                if key not in ALLOWED_PREDICTOR_KEYS:
                    result._add(f"Unauthorized predictor key: predictorList[{i}].{key}")

        # Output
        output = metadata.get("output", {})
        if isinstance(output, dict):
            for key in output:
                if key not in ALLOWED_OUTPUT_KEYS:
                    result._add(f"Unauthorized output key: output.{key}")
            possible = output.get("possibleValues", [])
            if len(possible) > MAX_POSSIBLE_VALUES:
                result._add(
                    f"Too many possibleValues ({len(possible)} > {MAX_POSSIBLE_VALUES})",
                    fail=True,
                )

        # String length checks
        self._check_string_lengths(metadata, result)

        return result

    def sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of *metadata* with only allowed keys retained."""
        out: dict[str, Any] = {}
        for key, value in metadata.items():
            if key not in self.allowed_keys:
                continue
            if key == "predictorList" and isinstance(value, list):
                out[key] = [{k: v for k, v in p.items() if k in ALLOWED_PREDICTOR_KEYS} for p in value]
            elif key == "output" and isinstance(value, dict):
                out[key] = {k: v for k, v in value.items() if k in ALLOWED_OUTPUT_KEYS}
            else:
                out[key] = value
        return out

    @staticmethod
    def _check_string_lengths(data: Any, result: SecurityResult, path: str = "") -> None:
        if isinstance(data, dict):
            for k, v in data.items():
                ModelSecurityValidator._check_string_lengths(v, result, f"{path}.{k}" if path else k)
        elif isinstance(data, list):
            for i, v in enumerate(data):
                ModelSecurityValidator._check_string_lengths(v, result, f"{path}[{i}]")
        elif isinstance(data, str) and len(data) > MAX_STRING_LENGTH:
            result._add(
                f"String too long at {path}: {len(data)} > {MAX_STRING_LENGTH}",
                fail=True,
            )


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------


def compute_model_hash(model_path: str | Path) -> str:
    """Compute the SHA-256 hex digest of an ONNX model file."""
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_hash(model_path: str | Path, expected_hash: str) -> bool:
    """Return ``True`` if the file's SHA-256 matches *expected_hash*."""
    return compute_model_hash(model_path) == expected_hash
