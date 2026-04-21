"""Tree primitives: split parsing, node dataclass, traversal helpers."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, cast

SplitOperator = Literal["<", ">", "==", "in", "is"]
_SPLIT_OPERATORS: frozenset[str] = frozenset({"<", ">", "==", "in", "is"})

# Anchored to a known operator vocabulary so predictor names containing spaces
# don't trip the parser. Predictor and value can be any non-empty string.
_SPLIT_RE = re.compile(
    r"^(?P<variable>.+?) (?P<operator><|>|==|in|is) (?P<value>.+)$",
)


@dataclass(frozen=True)
class Split:
    """A parsed tree split condition.

    Attributes
    ----------
    variable : str
        Predictor name being split on.
    operator : SplitOperator
        Comparison operator: ``"<"`` and ``">"`` for numeric thresholds,
        ``"=="`` for single-category equality, ``"in"`` for set membership,
        ``"is"`` for missing-value checks.
    value : float | str | tuple[str, ...]
        Right-hand side of the split.  ``float`` for numeric thresholds,
        ``tuple[str, ...]`` for ``in``-splits, ``str`` for ``==``/``is``.
    raw : str
        Original split string, useful for diagnostics or display.
    """

    variable: str
    operator: SplitOperator
    value: float | str | tuple[str, ...]
    raw: str

    @property
    def is_numeric(self) -> bool:
        return self.operator in {"<", ">"}

    @property
    def is_symbolic(self) -> bool:
        return self.operator in {"in", "==", "is"}


def parse_split(raw: str) -> Split:
    """Parse a tree-split string into a :class:`Split`.

    Examples
    --------
    >>> parse_split("Age < 42.5").operator
    '<'
    >>> sorted(parse_split("Color in { red, blue }").value)
    ['blue', 'red']
    >>> parse_split("Status is Missing").value
    'Missing'
    """
    match = _SPLIT_RE.match(raw)
    if match is None:
        raise ValueError(f"Cannot parse split: {raw!r}")
    variable = match.group("variable")
    op = match.group("operator")
    if op not in _SPLIT_OPERATORS:  # pragma: no cover - guarded by regex
        raise ValueError(f"Unsupported split operator {op!r} in {raw!r}")
    value_text = match.group("value").strip()

    value: float | str | tuple[str, ...]
    if op in {"<", ">"}:
        try:
            value = float(value_text)
        except ValueError:
            value = value_text
    elif op == "in" and value_text.startswith("{") and value_text.endswith("}"):
        inner = value_text[1:-1].strip()
        members = [m.strip() for m in inner.split(",")] if inner else []
        # Match legacy behaviour: count empty members (e.g. trailing commas).
        value = tuple(members)
    else:
        # Single-value 'in', 'is', '==' — keep as a string.
        value = value_text

    return Split(variable=variable, operator=cast(SplitOperator, op), value=value, raw=raw)


@dataclass(frozen=True)
class Node:
    """A single node in an AGB tree.

    All nodes carry a ``score`` (the leaf prediction or root prior).
    Internal nodes additionally carry a parsed :class:`Split` and a
    ``gain``.  Leaves have ``split=None`` and ``gain=0.0``.
    """

    depth: int
    score: float
    is_leaf: bool
    split: Split | None
    gain: float


def _iter_nodes(tree: dict, depth: int = 1) -> Iterator[Node]:
    """Yield every node in a tree in pre-order, root first.

    Single source of truth for tree traversal.  All metric computations
    consume this iterator instead of reimplementing recursion.
    """
    is_leaf = "split" not in tree
    split = None if is_leaf else parse_split(tree["split"])
    yield Node(
        depth=depth,
        score=tree.get("score", 0.0),
        is_leaf=is_leaf,
        split=split,
        gain=tree.get("gain", 0.0) if not is_leaf else 0.0,
    )
    if "left" in tree:
        yield from _iter_nodes(tree["left"], depth + 1)
    if "right" in tree:
        yield from _iter_nodes(tree["right"], depth + 1)


# Locations to try for the boosters/trees list inside the raw model JSON.
# Each entry is a tuple of dict keys to traverse.
_BOOSTER_PATHS: tuple[tuple[str | int, ...], ...] = (
    ("model", "boosters", 0, "trees"),
    ("model", "model", "boosters", 0, "trees"),
    ("model", "booster", "trees"),
    ("model", "model", "booster", "trees"),
)


def _traverse(d: Any, path: tuple) -> Any:
    """Walk a nested dict/list using the given key/index path; raise KeyError on miss."""
    cur = d
    for key in path:
        if isinstance(key, int):
            cur = cur[key]
        else:
            cur = cur[key]
    return cur
