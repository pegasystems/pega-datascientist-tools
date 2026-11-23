"""Detect and decode encoded AGB split conditions.

Pega stores trained AGB models with encoded splits such as ``"12 LT 40"``:
a predictor index from ``inputsEncoder``, an ``LT``/``EQ`` operator and a
bin index.  Prediction Studio exports decode these into readable splits
(``"Age < 42.5"``, ``"Color in { red, blue }"``, ``"Color is Missing"``).
This module converts the former into the latter so that both formats share
one analysis path.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from ._nodes import _traverse

SplitEncoding = Literal["encoded", "decoded", "none"]

ENCODER_PATHS: tuple[tuple[str, ...], ...] = (
    ("model", "inputsEncoder", "encoders"),
    ("model", "model", "inputsEncoder", "encoders"),
)

_ENCODED_SPLIT_RE = re.compile(r"^(?P<predictor>\d+) (?P<operator>LT|EQ) (?P<index>-?\d+)$")

# Symbol positions are encoded from 129 upwards; missing values sort below
# them, so ``LT 129`` isolates missing values.
_SYMBOL_OFFSET = 129


def _iter_splits(tree: dict) -> list[str]:
    splits = []
    stack = [tree]
    while stack:
        node = stack.pop()
        if "split" in node:
            splits.append(node["split"])
        stack.extend(node[side] for side in ("left", "right") if side in node)
    return splits


def split_encoding(trees: list[dict]) -> SplitEncoding:
    """Classify the split format used by a list of trees.

    Parameters
    ----------
    trees : list[dict]
        Boosted trees as nested dicts.

    Returns
    -------
    SplitEncoding
        ``"encoded"`` when every split is an encoded ``"<index> LT|EQ <bin>"``
        condition, ``"decoded"`` when none is, and ``"none"`` when the trees
        have no splits.

    Raises
    ------
    ValueError
        When the trees mix encoded and decoded splits.
    """
    splits = [split for tree in trees for split in _iter_splits(tree)]
    if not splits:
        return "none"
    encoded = [isinstance(split, str) and _ENCODED_SPLIT_RE.match(split) is not None for split in splits]
    if all(encoded):
        return "encoded"
    if not any(encoded):
        return "decoded"
    example = next(split for split, is_encoded in zip(splits, encoded, strict=True) if is_encoded)
    raise ValueError(
        f"The model mixes encoded splits (such as {example!r}) with decoded splits; cannot interpret it safely.",
    )


def find_encoders(model_json: Any) -> list[dict] | None:
    """Return the raw ``inputsEncoder.encoders`` list, or ``None`` if absent."""
    for path in ENCODER_PATHS:
        try:
            return _traverse(model_json, path)
        except (KeyError, TypeError, IndexError):
            continue
    return None


def _decode_numeric(name: str, encoder: dict, operator: str, index: int) -> str:
    if operator != "LT":
        raise ValueError(f"Unsupported operator {operator!r} for numeric predictor {name!r}; expected 'LT'.")
    summary_type = encoder.get("summaryType")
    summary = encoder.get("summary") or {}
    if summary_type == "INITIAL_SUMMARY":
        values = summary.get("initialValues") or []
        position = index
    elif summary_type == "QUANTILE_SUMMARY":
        values = [str(entry).split("=", 1)[0] for entry in summary.get("list") or []]
        position = index - 1
    else:
        raise ValueError(f"Unsupported quantile summary type {summary_type!r} for predictor {name!r}.")
    if not 0 <= position < len(values):
        raise ValueError(
            f"Bin index {index} for numeric predictor {name!r} is outside its {len(values)} encoded boundaries.",
        )
    return f"{name} < {values[position]}"


def _decode_symbolic(name: str, encoder: dict, operator: str, index: int) -> str:
    symbols: dict[int, str] = {}
    for entry in encoder.get("symbols") or []:
        value, sep, position = str(entry).rpartition("=")
        if not sep:
            raise ValueError(f"Malformed symbol {entry!r} for predictor {name!r}; expected 'value=index'.")
        try:
            symbols[int(position)] = value
        except ValueError as exc:
            raise ValueError(f"Malformed symbol {entry!r} for predictor {name!r}; expected 'value=index'.") from exc

    bin_index = index - _SYMBOL_OFFSET
    if operator == "LT" and bin_index == 0:
        return f"{name} is Missing"
    if operator == "LT":
        selected = [symbols[i] for i in sorted(symbols) if i < bin_index]
    else:
        selected = [symbols[bin_index]] if bin_index in symbols else []
    if bin_index < 0 or not selected:
        raise ValueError(
            f"Bin index {index} with operator {operator!r} selects no encoded value of symbolic predictor {name!r}.",
        )
    return f"{name} in {{ {', '.join(selected)} }}"


def decode_trees(trees: list[dict], encoders: list[dict]) -> list[dict]:
    """Return copies of ``trees`` with encoded splits made readable.

    Numeric ``LT`` splits become ``"<name> < <boundary>"``.  Symbolic ``LT``
    splits become ``"<name> in { ... }"`` with every value below the bin, or
    ``"<name> is Missing"`` for the first symbolic bin.  Symbolic ``EQ`` splits
    select a single value.  Gains, scores, sample counts and tree shapes are
    unchanged.

    Parameters
    ----------
    trees : list[dict]
        Boosted trees with encoded splits.
    encoders : list[dict]
        Raw ``inputsEncoder.encoders`` entries.

    Returns
    -------
    list[dict]
        Decoded trees; the input is not modified.

    Raises
    ------
    ValueError
        When a split references an unknown predictor, uses an unsupported
        operator or encoder, or refers to a bin the encoder doesn't define.
    """
    by_index: dict[int, tuple[str, dict]] = {}
    for entry in encoders:
        try:
            name = entry["key"]
            index = int(entry["value"]["index"])
            encoder = dict(entry["value"]["encoder"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed inputsEncoder entry: {entry!r}") from exc
        by_index[index] = (name, encoder)

    def decode_split(split: str) -> str:
        match = _ENCODED_SPLIT_RE.match(split)
        if match is None:  # pragma: no cover - guarded by split_encoding
            raise ValueError(f"Not an encoded split: {split!r}")
        predictor = int(match["predictor"])
        if predictor not in by_index:
            raise ValueError(f"Split {split!r} refers to predictor index {predictor}, which inputsEncoder lacks.")
        name, encoder = by_index[predictor]
        operator, index = match["operator"], int(match["index"])
        if "quantileArray" in encoder:
            return _decode_numeric(name, encoder["quantileArray"], operator, index)
        if "stringTranslator" in encoder:
            return _decode_symbolic(name, encoder["stringTranslator"], operator, index)
        raise ValueError(f"Unsupported encoder {sorted(encoder)!r} for predictor {name!r}.")

    def decode_node(node: dict) -> dict:
        decoded = dict(node)
        if "split" in decoded:
            decoded["split"] = decode_split(decoded["split"])
        for side in ("left", "right"):
            if side in decoded:
                decoded[side] = decode_node(decoded[side])
        return decoded

    return [decode_node(tree) for tree in trees]
