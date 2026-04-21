"""ADM Gradient Boosting (AGB) model parsing, scoring, and diagnostics.

This module provides:

- :class:`Split` and :class:`Node` — small dataclasses describing a parsed
  split condition and a tree node.
- :class:`ADMTreesModel` — load and analyse a single AGB model.
- :class:`MultiTrees` — collection of snapshots of the same configuration
  over time.
- :class:`AGB` — Datamart helper for discovering and extracting AGB models.

Construction uses explicit factory classmethods
(``ADMTreesModel.from_file``, ``from_url``, ``from_datamart_blob``,
``from_dict``, and ``MultiTrees.from_datamart``).  The legacy
``ADMTrees(file, ...)`` polymorphic factory is still exported for
backward compatibility but is deprecated.
"""

from __future__ import annotations

__all__ = ["AGB", "ADMTrees", "ADMTreesModel", "MultiTrees", "Node", "Split"]

import base64
import collections
import json
import logging
import math
import multiprocessing
import re
import warnings
import zlib
from dataclasses import dataclass
from functools import cached_property
from math import exp
from pathlib import Path
from statistics import mean, median, stdev
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)
from collections.abc import Callable, Iterator

import polars as pl

from ..utils import cdh_utils
from ..utils.namespaces import MissingDependenciesException
from ..utils.types import QUERY

if TYPE_CHECKING:  # pragma: no cover
    import pydot

    from .ADMDatamart import ADMDatamart

logger = logging.getLogger(__name__)


# =============================================================================
# Split and Node primitives
# =============================================================================

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


# =============================================================================
# AGB — datamart-level discovery
# =============================================================================


class AGB:
    """Datamart helper for discovering and extracting AGB models.

    Reachable as ``ADMDatamart.agb``; not intended to be instantiated
    directly.
    """

    def __init__(self, datamart: ADMDatamart):
        self.datamart = datamart

    def discover_model_types(
        self,
        df: pl.LazyFrame,
        by: str = "Configuration",
    ) -> dict[str, str]:
        """Discover the type of model embedded in the ``Modeldata`` column.

        Groups by ``by`` (typically Configuration, since one model rule
        contains one model type) and decodes the first ``Modeldata`` blob
        per group to extract its ``_serialClass``.

        Parameters
        ----------
        df : pl.LazyFrame
            Datamart slice including ``Modeldata``.  Collected internally.
        by : str
            Grouping column.  ``Configuration`` is recommended.
        """
        if "Modeldata" not in df.collect_schema().names():
            raise ValueError(
                "Modeldata column not in the data. Please make sure to include it by setting 'subset' to False.",
            )

        def _get_type(val: str) -> str:
            return json.loads(zlib.decompress(base64.b64decode(val)))["_serialClass"]

        types = df.filter(pl.col("Modeldata").is_not_null()).group_by(by).agg(pl.col("Modeldata").first()).collect()
        return {row[by]: _get_type(row["Modeldata"]) for row in types.to_dicts()}

    def get_agb_models(
        self,
        last: bool = False,
        n_threads: int = 6,
        query: QUERY | None = None,
    ) -> dict[str, MultiTrees]:
        """Get all AGB models in the datamart, indexed by Configuration.

        Filters down to models whose ``_serialClass`` ends with
        ``GbModel`` and decodes them via :class:`MultiTrees`.

        Parameters
        ----------
        last : bool
            If True, use only the latest snapshot per model.
        n_threads : int
            Worker count for parallel blob decoding.
        query : QUERY | None
            Optional pre-filter applied before discovery.
        """
        df = self.datamart.aggregates.last(table="model_data") if last else self.datamart.model_data
        if df is None:
            raise ValueError(
                "Datamart has no model_data; cannot extract AGB models.",
            )
        df = cdh_utils._apply_query(df, query)
        if "Modeldata" not in df.collect_schema().names():
            raise ValueError(
                "Modeldata column not in the data. Please make sure to include it by setting 'subset' to False.",
            )
        types = self.discover_model_types(df, by="Configuration")
        agb_configs = [config for config, t in types.items() if t.endswith("GbModel")]
        logger.info("Found %d AGB configurations: %s", len(agb_configs), agb_configs)
        df_collected = df.filter(pl.col("Configuration").is_in(agb_configs)).collect()
        result: dict[str, MultiTrees] = {}
        for config in agb_configs:
            sub = df_collected.filter(pl.col("Configuration") == config)
            result[config] = MultiTrees.from_datamart(sub, n_threads=n_threads, configuration=config)
        return result


# =============================================================================
# ADMTrees — deprecated polymorphic factory
# =============================================================================


class ADMTrees:  # pragma: no cover
    """Deprecated polymorphic factory.

    Retained for backward compatibility with code that calls
    ``ADMTrees(file)`` and expects either an :class:`ADMTreesModel` or a
    ``dict[str, MultiTrees]`` back.  New code should call the explicit
    factory classmethods directly:

    - :meth:`ADMTreesModel.from_file`
    - :meth:`ADMTreesModel.from_url`
    - :meth:`ADMTreesModel.from_datamart_blob`
    - :meth:`ADMTreesModel.from_dict`
    - :meth:`MultiTrees.from_datamart`
    """

    def __new__(cls, file, n_threads: int = 6, **kwargs):
        warnings.warn(
            "ADMTrees(...) is deprecated; use ADMTreesModel.from_file / "
            "from_url / from_datamart_blob / from_dict, or "
            "MultiTrees.from_datamart / from_datamart_grouped for "
            "datamart DataFrames.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(file, pl.DataFrame):
            file = file.filter(pl.col("Modeldata").is_not_null())
            if len(file) > 1:
                # Multiple snapshots → return a dict for backward compatibility.
                return MultiTrees.from_datamart_grouped(file, n_threads=n_threads, **kwargs)
            return ADMTreesModel.from_datamart_blob(
                file.select("Modeldata").item(),
                **kwargs,
            )
        if isinstance(file, pl.Series):
            return ADMTreesModel.from_datamart_blob(file.item(), **kwargs)
        return ADMTreesModel._from_anything(file, **kwargs)


# =============================================================================
# ADMTreesModel
# =============================================================================


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


class ADMTreesModel:
    """Functions for ADM Gradient boosting

    ADM Gradient boosting models consist of multiple trees, which build
    upon each other in a 'boosting' fashion.  This class provides
    functions to extract data from these trees: the features on which
    the trees split, important values for these features, statistics
    about the trees, or visualising each individual tree.

    Construct via :meth:`from_file`, :meth:`from_url`,
    :meth:`from_datamart_blob`, or :meth:`from_dict`.  The legacy
    ``ADMTreesModel(file_path)`` constructor is still supported for
    backward compatibility.

    Notes
    -----
    The "save model" action in Prediction Studio exports a JSON file
    that this class can load directly.  The Datamart's ``pyModelData``
    column also contains this information, but compressed and with
    encoded split values; the "save model" button decompresses and
    decodes that data.

    """

    trees: dict
    """The full parsed model JSON."""

    model: list[dict]
    """The list of boosted trees (each a nested dict)."""

    raw_input: Any
    """The raw input used to construct this instance (path, bytes, or dict)."""

    learning_rate: float | None = None
    context_keys: list | None = None
    _properties: dict[str, Any]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, file: str | None = None, **kwargs):
        """Backward-compatible constructor.

        .. deprecated::
            Prefer the ``from_*`` classmethods (``from_file``,
            ``from_url``, ``from_datamart_blob``, ``from_dict``).  The
            string-dispatch constructor will be removed in a future
            release.
        """
        if file is None:
            return  # allow classmethods to populate self
        warnings.warn(
            "ADMTreesModel(file) is deprecated; use one of the explicit "
            "factory classmethods: ADMTreesModel.from_file / from_url / "
            "from_datamart_blob / from_dict.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._init_from_anything(file, **kwargs)

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> ADMTreesModel:
        """Build from an already-parsed model dict."""
        instance = cls.__new__(cls)
        instance.trees = data
        instance.raw_input = data
        instance._post_import_cleanup(decode=False, **kwargs)
        return instance

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> ADMTreesModel:
        """Load a model from a local JSON file (Prediction Studio "save model" output)."""
        path = Path(path)
        with path.open() as f:
            data = json.load(f)
        instance = cls.__new__(cls)
        instance.trees = data
        instance.raw_input = str(path)
        instance._post_import_cleanup(decode=False, **kwargs)
        return instance

    @classmethod
    def from_url(cls, url: str, *, timeout: float = 30.0, **kwargs) -> ADMTreesModel:
        """Load a model from a URL pointing at the JSON export.

        ``timeout`` is the per-request timeout in seconds (default 30).
        """
        import requests  # type: ignore[import-untyped]

        content = requests.get(url, timeout=timeout)
        content.raise_for_status()
        data = json.loads(content.content)
        instance = cls.__new__(cls)
        instance.trees = data
        instance.raw_input = url
        instance._post_import_cleanup(decode=False, **kwargs)
        return instance

    @classmethod
    def from_datamart_blob(cls, blob: str | bytes, **kwargs) -> ADMTreesModel:
        """Load from a base64-encoded zlib-compressed datamart ``Modeldata`` blob."""
        if isinstance(blob, str):
            raw_bytes = base64.b64decode(blob)
        else:
            raw_bytes = blob
        decompressed = zlib.decompress(raw_bytes)
        data = json.loads(decompressed)
        if not data.get("_serialClass", "").endswith("GbModel"):
            raise ValueError("Not an AGB model")
        instance = cls.__new__(cls)
        instance.trees = data
        instance.raw_input = blob
        instance._post_import_cleanup(decode=True, **kwargs)
        return instance

    def _init_from_anything(self, file: Any, **kwargs) -> None:
        instance = self._from_anything(file, **kwargs)
        # Mirror state onto self (legacy single-arg __init__ flow).
        self.trees = instance.trees
        self.model = instance.model
        self.raw_input = instance.raw_input
        self._properties = instance._properties
        self.learning_rate = instance.learning_rate
        self.context_keys = instance.context_keys

    @classmethod
    def _from_anything(cls, file: Any, **kwargs) -> ADMTreesModel:
        """Best-effort load from a string (path / URL / base64) or bytes.

        Dispatches based on input *shape*, not by trying every loader and
        catching exceptions:

        * ``dict``           → :meth:`from_dict`
        * ``bytes``          → :meth:`from_datamart_blob`
        * ``str`` starting with ``http://`` / ``https://`` → :meth:`from_url`
        * ``str`` that names an existing path → :meth:`from_file`
        * any other ``str``  → :meth:`from_datamart_blob` (assume base64)
        """
        if isinstance(file, dict):
            return cls.from_dict(file, **kwargs)
        if isinstance(file, bytes):
            return cls.from_datamart_blob(file, **kwargs)
        if isinstance(file, str):
            if file.startswith(("http://", "https://")):
                return cls.from_url(file, **kwargs)
            if Path(file).expanduser().is_file():
                return cls.from_file(file, **kwargs)
            # Last-resort: treat as a base64-encoded datamart blob.  If the
            # caller passed a non-existent path string, this surfaces the
            # decode error directly instead of a misleading "file not found".
            return cls.from_datamart_blob(file, **kwargs)
        raise TypeError(f"Unsupported input type: {type(file).__name__}")

    # ------------------------------------------------------------------
    # Internal post-load processing
    # ------------------------------------------------------------------

    def _decode_trees(self):  # pragma: no cover
        def quantile_decoder(encoder: dict, index: int):
            if encoder["summaryType"] == "INITIAL_SUMMARY":
                return encoder["summary"]["initialValues"][index]
            return encoder["summary"]["list"][index - 1].split("=")[0]

        def string_decoder(encoder: dict, index: int, sign):
            def set_types(split):
                logger.debug(split)
                split = split.rsplit("=", 1)
                split[1] = int(split[1])
                return tuple(reversed(split))

            valuelist = dict(sorted(set_types(i) for i in encoder["symbols"]))
            splitvalues = []
            for key, value in valuelist.items():
                logger.debug("string_decoder candidate: %s=%s (index=%s)", key, value, index)
                if int(key) == index - 129 and index == 129 and sign == "<":
                    splitvalues.append("Missing")
                    break
                if self._safe_numeric_compare(int(key), sign, index - 129):
                    splitvalues.append(value)
            return ", ".join(splitvalues)

        def decode_split(split: str):
            if not isinstance(split, str):
                return split
            logger.debug("Decoding split: %s", split)
            predictor, sign, splitval = split.split(" ")
            if sign == "LT":
                sign = "<"
            elif sign == "EQ":
                sign = "=="
            else:
                raise ValueError(
                    f"Unsupported split operator (only LT/EQ supported): {(predictor, sign, splitval)}",
                )
            variable = encoderkeys[int(predictor)]
            encoder = encoders[variable]
            variable_type = list(encoder["encoder"].keys())[0]
            to_decode = list(encoder["encoder"].values())[0]
            if variable_type == "quantileArray":
                val = quantile_decoder(to_decode, int(splitval))
            if variable_type == "stringTranslator":
                val = string_decoder(to_decode, int(splitval), sign=sign)
                if val == "Missing":
                    sign, val = "is", "Missing"
                else:
                    val = "{ " + val + " }"
                    sign = "in"
            logger.debug("Decoded split: %s %s %s", variable, sign, val)
            return f"{variable} {sign} {val}"

        try:
            encoders = self.trees["model"]["model"]["inputsEncoder"]["encoders"]
        except KeyError:
            encoders = self.trees["model"]["inputsEncoder"]["encoders"]
        encoderkeys: dict[int, str] = {}
        for encoder in encoders:
            encoderkeys[encoder["value"]["index"]] = encoder["key"]
        encoders = {encoder["key"]: encoder["value"] for encoder in encoders}

        def decode_all(ob, func):
            if isinstance(ob, collections.abc.Mapping):
                return {k: decode_all(v, func) for k, v in ob.items()}
            return func(ob)

        for i, model in enumerate(self.model):
            self.model[i] = decode_all(model, decode_split)
            logger.debug("Decoded tree %d", i)

    def _post_import_cleanup(self, decode: bool, **kwargs):
        if not hasattr(self, "model"):
            self.model = self._locate_boosters()

        if decode:  # pragma: no cover
            logger.debug("Decoding the tree splits.")
            self._decode_trees()

        if isinstance(self.trees, dict):
            self._properties = {k: v for k, v in self.trees.items() if k != "model"}
        else:  # pragma: no cover
            logger.debug("Could not extract the properties.")
            self._properties = {}

        config = self._properties.get("configuration", {}) or {}
        self.learning_rate = (config.get("parameters") or {}).get("learningRateEta")
        self.context_keys = config.get("contextKeys", kwargs.get("context_keys"))

        if self.model is None:  # pragma: no cover
            raise ValueError("Import unsuccessful: no boosters/trees found.")

    def _locate_boosters(self) -> list[dict]:
        """Find the boosters/trees list in the model JSON.

        Different Pega versions place the boosters at different paths;
        try them in order.
        """
        errors = []
        for path in _BOOSTER_PATHS:
            try:
                return _traverse(self.trees, path)
            except (KeyError, TypeError, IndexError) as exc:
                errors.append(f"{'.'.join(map(str, path))}: {exc}")
        raise ValueError(
            "Could not locate boosters/trees list in the model JSON. Tried paths: " + "; ".join(errors),
        )

    # ------------------------------------------------------------------
    # Safe condition evaluation (used during scoring)
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_numeric_compare(left: float, operator: str, right: float) -> bool:
        """Safely compare two numeric values without using ``eval()``."""
        if operator == "<":
            return left < right
        if operator == ">":
            return left > right
        if operator == "==":
            return left == right
        if operator == "<=":
            return left <= right
        if operator == ">=":
            return left >= right
        if operator == "!=":
            return left != right
        raise ValueError(f"Unsupported operator: {operator}")

    # Class-level dedupe so repeated per-row scoring failures don't spam logs.
    _safe_eval_seen_errors: set[tuple[str, str]] = set()

    @staticmethod
    def _safe_condition_evaluate(
        value: Any,
        operator: str,
        comparison_set: set | float | str | frozenset,
    ) -> bool:
        """Safely evaluate split conditions without using ``eval()``.

        Returns ``False`` on type-conversion errors after logging the
        first occurrence per (operator, error-type) pair at INFO level.
        Subsequent matching failures log at DEBUG only — we don't want
        per-row scoring to swamp the application logs, but the first
        failure for each error class is worth surfacing.
        """
        try:
            if operator == "in":
                return str(value).strip("'") in comparison_set  # type: ignore[operator]
            if operator == "<":
                return float(str(value).strip("'")) < float(comparison_set)  # type: ignore[arg-type]
            if operator == ">":
                return float(str(value).strip("'")) > float(comparison_set)  # type: ignore[arg-type]
            if operator == "==":
                return str(value).strip("'") == str(comparison_set)
            raise ValueError(f"Unsupported operator: {operator}")
        except (ValueError, TypeError) as e:
            key = (operator, type(e).__name__)
            if key in ADMTreesModel._safe_eval_seen_errors:
                logger.debug("Safe evaluation failed (%s %s): %s — returning False", operator, type(e).__name__, e)
            else:
                ADMTreesModel._safe_eval_seen_errors.add(key)
                logger.info(
                    "Safe scoring evaluation failed for %r %s %r: %s — returning "
                    "False. Subsequent failures with the same operator/error "
                    "type will be logged at DEBUG only.",
                    value,
                    operator,
                    comparison_set,
                    e,
                )
            return False

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @cached_property
    def metrics(self) -> dict[str, Any]:
        """Compute CDH_ADM005-style diagnostic metrics for this model.

        Returns a flat dictionary of key/value pairs aligned with the
        CDH_ADM005 telemetry event specification.  Metrics that cannot be
        computed from an exported model (e.g. saturation counts that
        require bin-level data) are omitted.

        See :meth:`metric_descriptions` for human-readable descriptions
        of every key.
        """
        return self._compute_metrics()

    @staticmethod
    def metric_descriptions() -> dict[str, str]:
        """Return a dictionary mapping metric names to human-readable descriptions."""
        return {
            # Properties-level
            "auc": "Area Under the ROC Curve — overall model discrimination power.",
            "success_rate": "Proportion of positive outcomes in the training data.",
            "factory_update_time": "Timestamp of the last factory (re)build of this model.",
            "response_positive_count": "Number of positive responses in training data.",
            "response_negative_count": "Number of negative responses in training data.",
            # Model complexity
            "number_of_tree_nodes": "Total node count across all trees (splits + leaves).",
            "tree_depth_max": "Maximum depth of any single tree in the ensemble.",
            "tree_depth_avg": "Average depth across all trees.",
            "tree_depth_std": "Standard deviation of tree depths — uniformity of tree complexity.",
            "number_of_trees": "Total number of boosting rounds (trees) in the model.",
            "number_of_stump_trees": "Trees with no splits (single root node). Stumps contribute no learned signal.",
            "avg_leaves_per_tree": "Average number of leaf nodes per tree — a proxy for tree complexity.",
            # Splits by predictor category
            "number_of_splits_on_ih_predictors": "Total splits on Interaction History (IH.*) predictors.",
            "number_of_splits_on_context_key_predictors": "Total splits on context-key predictors (py*, Param.*, *.Context.*).",
            "number_of_splits_on_other_predictors": "Total splits on customer/other predictors.",
            # Predictor counts
            "total_number_of_active_predictors": "Predictors that appear in at least one split.",
            "total_number_of_predictors": "All predictors known to the model (active or not).",
            "number_of_active_ih_predictors": "Active IH predictors (appear in splits).",
            "total_number_of_ih_predictors": "All IH predictors in the model configuration.",
            "number_of_active_context_key_predictors": "Active context-key predictors.",
            "number_of_active_symbolic_predictors": "Active symbolic (categorical) predictors.",
            "total_number_of_symbolic_predictors": "All symbolic predictors in configuration.",
            "number_of_active_numeric_predictors": "Active numeric (continuous) predictors.",
            "total_number_of_numeric_predictors": "All numeric predictors in configuration.",
            # Gain distribution
            "total_gain": "Sum of all split gains — total information gained by the ensemble.",
            "mean_gain_per_split": "Average gain per split node (analogous to XGBoost gain importance).",
            "median_gain_per_split": "Median gain — robust central tendency, less sensitive to outlier splits.",
            "max_gain_per_split": "Largest single split gain — identifies the most informative split.",
            "gain_std": "Standard deviation of gains — high values indicate a few dominant splits.",
            # Leaf scores
            "number_of_leaves": "Total leaf nodes across all trees.",
            "leaf_score_mean": "Average leaf score (log-odds contribution). Near zero means balanced.",
            "leaf_score_std": "Spread of leaf scores — wider spread means better discrimination.",
            "leaf_score_min": "Most negative leaf score.",
            "leaf_score_max": "Most positive leaf score.",
            # Split types
            "number_of_numeric_splits": "Splits using '<' (numeric/continuous thresholds).",
            "number_of_symbolic_splits": "Splits using 'in' or '==' (categorical membership).",
            "symbolic_split_fraction": "Fraction of splits that are symbolic (0–1).",
            "number_of_unique_splits": "Distinct split conditions across all trees.",
            "number_of_unique_predictors_split_on": "Number of distinct predictor variables used in splits.",
            "split_reuse_ratio": "Total splits / unique splits — how often the same condition recurs across trees.",
            "avg_symbolic_set_size": "Average number of categories in symbolic 'in { ... }' splits.",
            # Learning convergence
            "mean_abs_score_first_10": "Mean |root score| of the first 10 trees — initial correction magnitude.",
            "mean_abs_score_last_10": "Mean |root score| of the last 10 trees — late correction magnitude.",
            "score_decay_ratio": "Ratio last/first — values < 1 indicate convergence, >> 1 indicates instability.",
            "mean_gain_first_half": "Average gain in the first half of trees.",
            "mean_gain_last_half": "Average gain in the second half — lower values suggest convergence.",
            # Feature importance concentration
            "top_predictor_by_gain": "Predictor with the highest total gain.",
            "top_predictor_gain_share": "Fraction of total gain from the top predictor (0–1). High = dominance.",
            "predictor_gain_entropy": "Normalised Shannon entropy of gain distribution (0–1). Low = concentrated.",
            # Saturation (encoder-based only)
            "number_of_saturated_context_key_predictors": "Context-key predictors that have reached max bin capacity.",
            "number_of_saturated_symbolic_predictors": "Symbolic predictors at max bin capacity.",
            "max_saturation_rate_on_context_key_predictors": "Highest bin-fill percentage among context-key predictors.",
        }

    @staticmethod
    def _classify_predictor(name: str) -> str:
        """Classify a predictor as 'ih', 'context_key', or 'other'."""
        if name.startswith("IH."):
            return "ih"
        if name.startswith("py") or name.startswith("Param.") or ".Context." in name:
            return "context_key"
        return "other"

    def _compute_metrics(self) -> dict[str, Any]:
        """Walk all trees once and assemble the metrics dictionary."""
        var_ops: dict[str, set[str]] = collections.defaultdict(set)
        var_split_count: dict[str, int] = collections.defaultdict(int)
        var_total_gain: dict[str, float] = collections.defaultdict(float)
        depths: list[int] = []
        all_gains: list[float] = []
        leaf_scores: list[float] = []
        split_strings: list[str] = []
        symbolic_set_sizes: list[int] = []
        per_tree_gains: list[list[float]] = []
        root_scores: list[float] = []
        total_nodes = 0
        numeric_split_count = 0
        symbolic_split_count = 0
        leaf_count = 0
        stump_count = 0

        for tree in self.model:
            tree_max_depth = 0
            tree_gains: list[float] = []
            root_added = False
            has_split = False
            for node in _iter_nodes(tree):
                total_nodes += 1
                tree_max_depth = max(tree_max_depth, node.depth)
                if not root_added:
                    root_scores.append(node.score)
                    root_added = True

                if node.is_leaf:
                    leaf_count += 1
                    leaf_scores.append(node.score)
                    continue

                has_split = True
                split = node.split
                assert split is not None  # narrow for mypy
                split_strings.append(split.raw)
                var_ops[split.variable].add(split.operator)
                var_split_count[split.variable] += 1
                var_total_gain[split.variable] += node.gain

                if node.gain > 0:
                    all_gains.append(node.gain)
                    tree_gains.append(node.gain)

                if split.operator == "<":
                    numeric_split_count += 1
                elif split.operator in {"in", "=="}:
                    symbolic_split_count += 1
                    if split.operator == "in" and isinstance(split.value, tuple):
                        symbolic_set_sizes.append(len(split.value))

            depths.append(tree_max_depth)
            per_tree_gains.append(tree_gains)
            if not has_split:
                stump_count += 1

        n_trees = len(self.model)
        total_splits = numeric_split_count + symbolic_split_count

        # --- predictor-type classification ---------------------------------
        encoder_info = self._get_encoder_info()
        if encoder_info is not None:
            all_predictors = set(encoder_info.keys())
            all_numeric = {n for n, info in encoder_info.items() if info["type"] == "numeric"}
            all_symbolic = {n for n, info in encoder_info.items() if info["type"] == "symbolic"}
            active_numeric = all_numeric & set(var_ops)
            active_symbolic = all_symbolic & set(var_ops)
        else:
            all_predictors = None
            active_numeric = {v for v, ops in var_ops.items() if "<" in ops}
            active_symbolic = {v for v, ops in var_ops.items() if "in" in ops or "==" in ops}

        active_ih = {v for v in var_ops if self._classify_predictor(v) == "ih"}
        active_context_key = {v for v in var_ops if self._classify_predictor(v) == "context_key"}
        active_other = {v for v in var_ops if self._classify_predictor(v) == "other"}

        total_predictors: dict[str, str] | None = None
        try:
            total_predictors = self.predictors
        except (KeyError, AttributeError, TypeError, IndexError) as exc:  # pragma: no cover
            logger.debug("Could not read total predictors: %s", exc)

        if total_predictors is not None:
            all_ih = {n for n in total_predictors if self._classify_predictor(n) == "ih"}
            total_pred_count = len(total_predictors)
            if all_predictors is None:
                all_numeric = {
                    n for n, t in total_predictors.items() if t.lower() in {"numeric", "double", "integer", "float"}
                }
                all_symbolic = {
                    n for n, t in total_predictors.items() if t.lower() in {"symbolic", "string", "boolean"}
                }
        elif all_predictors is not None:
            all_ih = {n for n in all_predictors if self._classify_predictor(n) == "ih"}
            total_pred_count = len(all_predictors)
        else:
            all_ih = active_ih
            total_pred_count = len(var_ops)
            all_numeric = active_numeric
            all_symbolic = active_symbolic

        # --- saturation (encoder-only) -------------------------------------
        saturated_ctx = 0
        saturated_symbolic = 0
        max_saturation_ctx: float = 0.0
        if encoder_info is not None:
            for name, info in encoder_info.items():
                if info["max_bins"] is not None and info["max_bins"] > 0:
                    ratio = info["used_bins"] / info["max_bins"]
                    saturated = info["used_bins"] >= info["max_bins"]
                    cat = self._classify_predictor(name)
                    if cat == "context_key":
                        if saturated:
                            saturated_ctx += 1
                        max_saturation_ctx = max(max_saturation_ctx, ratio)
                    if info["type"] == "symbolic" and saturated:
                        saturated_symbolic += 1

        sum_all_gain = sum(var_total_gain.values())
        props = getattr(self, "_properties", {}) or {}
        training = props.get("trainingStats", {})

        m: dict[str, Any] = {}
        # Properties-level
        m["auc"] = props.get("auc") or props.get("performance")
        m["success_rate"] = props.get("successRate")
        m["factory_update_time"] = props.get("factoryUpdateTime")
        m["response_positive_count"] = training.get("positiveCount")
        m["response_negative_count"] = training.get("negativeCount")
        # Model complexity
        m["number_of_tree_nodes"] = total_nodes
        m["tree_depth_max"] = max(depths) if depths else 0
        m["tree_depth_avg"] = round(sum(depths) / len(depths), 2) if depths else 0.0
        m["tree_depth_std"] = round(stdev(depths), 2) if len(depths) >= 2 else 0.0
        m["number_of_trees"] = n_trees
        m["number_of_stump_trees"] = stump_count
        m["avg_leaves_per_tree"] = round(leaf_count / n_trees, 2) if n_trees > 0 else 0.0
        m["number_of_splits_on_ih_predictors"] = sum(var_split_count[v] for v in active_ih)
        m["number_of_splits_on_context_key_predictors"] = sum(var_split_count[v] for v in active_context_key)
        m["number_of_splits_on_other_predictors"] = sum(var_split_count[v] for v in active_other)
        # Predictor counts
        m["total_number_of_active_predictors"] = len(var_ops)
        m["total_number_of_predictors"] = total_pred_count
        m["number_of_active_ih_predictors"] = len(active_ih)
        m["total_number_of_ih_predictors"] = len(all_ih)
        m["number_of_active_context_key_predictors"] = len(active_context_key)
        m["number_of_active_symbolic_predictors"] = len(active_symbolic)
        m["total_number_of_symbolic_predictors"] = len(all_symbolic)
        m["number_of_active_numeric_predictors"] = len(active_numeric)
        m["total_number_of_numeric_predictors"] = len(all_numeric)
        # Gain distribution
        m["total_gain"] = round(sum(all_gains), 4) if all_gains else 0.0
        m["mean_gain_per_split"] = round(mean(all_gains), 4) if all_gains else 0.0
        m["median_gain_per_split"] = round(median(all_gains), 4) if all_gains else 0.0
        m["max_gain_per_split"] = round(max(all_gains), 4) if all_gains else 0.0
        m["gain_std"] = round(stdev(all_gains), 4) if len(all_gains) >= 2 else 0.0
        # Leaf scores
        m["number_of_leaves"] = leaf_count
        m["leaf_score_mean"] = round(mean(leaf_scores), 6) if leaf_scores else 0.0
        m["leaf_score_std"] = round(stdev(leaf_scores), 6) if len(leaf_scores) >= 2 else 0.0
        m["leaf_score_min"] = round(min(leaf_scores), 6) if leaf_scores else 0.0
        m["leaf_score_max"] = round(max(leaf_scores), 6) if leaf_scores else 0.0
        # Split types
        m["number_of_numeric_splits"] = numeric_split_count
        m["number_of_symbolic_splits"] = symbolic_split_count
        m["symbolic_split_fraction"] = round(symbolic_split_count / total_splits, 4) if total_splits > 0 else 0.0
        unique_splits = set(split_strings)
        m["number_of_unique_splits"] = len(unique_splits)
        m["number_of_unique_predictors_split_on"] = len(var_ops)
        m["split_reuse_ratio"] = round(len(split_strings) / len(unique_splits), 2) if unique_splits else 0.0
        m["avg_symbolic_set_size"] = round(mean(symbolic_set_sizes), 2) if symbolic_set_sizes else 0.0
        # Learning convergence
        window = min(10, n_trees)
        if window > 0:
            abs_root_first = [abs(s) for s in root_scores[:window]]
            abs_root_last = [abs(s) for s in root_scores[-window:]]
            m["mean_abs_score_first_10"] = round(mean(abs_root_first), 6)
            m["mean_abs_score_last_10"] = round(mean(abs_root_last), 6)
            m["score_decay_ratio"] = (
                round(m["mean_abs_score_last_10"] / m["mean_abs_score_first_10"], 4)
                if m["mean_abs_score_first_10"] > 0
                else 0.0
            )
        else:
            m["mean_abs_score_first_10"] = 0.0
            m["mean_abs_score_last_10"] = 0.0
            m["score_decay_ratio"] = 0.0
        half = n_trees // 2
        if half > 0:
            gains_first = [g for tg in per_tree_gains[:half] for g in tg]
            gains_last = [g for tg in per_tree_gains[half:] for g in tg]
            m["mean_gain_first_half"] = round(mean(gains_first), 4) if gains_first else 0.0
            m["mean_gain_last_half"] = round(mean(gains_last), 4) if gains_last else 0.0
        else:
            m["mean_gain_first_half"] = 0.0
            m["mean_gain_last_half"] = 0.0
        # Feature-importance concentration
        if var_total_gain and sum_all_gain > 0:
            top_var = max(var_total_gain, key=lambda k: var_total_gain[k])
            m["top_predictor_by_gain"] = top_var
            m["top_predictor_gain_share"] = round(var_total_gain[top_var] / sum_all_gain, 4)
            n_vars = len(var_total_gain)
            if n_vars > 1:
                entropy = 0.0
                for g in var_total_gain.values():
                    p = g / sum_all_gain
                    if p > 0:
                        entropy -= p * math.log2(p)
                max_entropy = math.log2(n_vars)
                m["predictor_gain_entropy"] = round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0
            else:
                m["predictor_gain_entropy"] = 0.0
        else:
            m["top_predictor_by_gain"] = None
            m["top_predictor_gain_share"] = 0.0
            m["predictor_gain_entropy"] = 0.0
        # Saturation
        if encoder_info is not None:
            m["number_of_saturated_context_key_predictors"] = saturated_ctx
            m["number_of_saturated_symbolic_predictors"] = saturated_symbolic
            m["max_saturation_rate_on_context_key_predictors"] = round(max_saturation_ctx * 100, 1)
        return m

    _ENCODER_PATHS: tuple[tuple[str, ...], ...] = (
        ("model", "inputsEncoder", "encoders"),
        ("model", "model", "inputsEncoder", "encoders"),
    )

    def _get_encoder_info(self) -> dict[str, dict[str, Any]] | None:
        """Extract predictor metadata from the inputsEncoder if present.

        Returns ``None`` when no encoder metadata is available (e.g. for
        exported/decoded models).
        """
        raw_encoders = None
        for path in self._ENCODER_PATHS:
            try:
                raw_encoders = _traverse(self.trees, path)
                break
            except (KeyError, TypeError, IndexError):
                continue

        if raw_encoders is None:
            return None

        result: dict[str, dict[str, Any]] = {}
        for entry in raw_encoders:
            name = entry["key"]
            value = entry["value"]
            encoder = value.get("encoder", {})
            encoder_type = next(iter(encoder), None)

            if encoder_type == "quantileArray":
                enc_data = encoder[encoder_type]
                summary = enc_data.get("summary", {})
                bins = summary.get("list", summary.get("initialValues", []))
                result[name] = {
                    "type": "numeric",
                    "used_bins": len(bins),
                    "max_bins": enc_data.get("maxNumberOfBins"),
                }
            elif encoder_type == "stringTranslator":
                enc_data = encoder[encoder_type]
                result[name] = {
                    "type": "symbolic",
                    "used_bins": len(enc_data.get("symbols", [])),
                    "max_bins": enc_data.get("maxNumberOfBins"),
                }
            else:
                result[name] = {"type": "unknown", "used_bins": 0, "max_bins": None}

        return result

    # ------------------------------------------------------------------
    # Cached views
    # ------------------------------------------------------------------

    @cached_property
    def predictors(self) -> dict[str, str] | None:
        logger.debug("Extracting predictors.")
        return self.get_predictors()

    @cached_property
    def tree_stats(self) -> pl.DataFrame:
        logger.debug("Calculating tree stats.")
        return self.get_tree_stats()

    @cached_property
    def splits_per_tree(self) -> dict[int, list[str]]:
        return self._splits_and_gains[0]

    @cached_property
    def gains_per_tree(self) -> dict[int, list[float]]:  # pragma: no cover
        return self._splits_and_gains[1]

    @cached_property
    def gains_per_split(self) -> pl.DataFrame:
        return self._splits_and_gains[2]

    @cached_property
    def grouped_gains_per_split(self) -> pl.DataFrame:
        logger.debug("Calculating grouped gains per split.")
        return self.get_grouped_gains_per_split()

    @cached_property
    def all_values_per_split(self) -> dict[str, set]:
        logger.debug("Calculating all values per split.")
        return self.get_all_values_per_split()

    @cached_property
    def splits_per_variable_type(self) -> tuple[list[collections.Counter], list[float]]:
        """Per-tree counts of splits grouped by predictor category.

        Equivalent to calling
        :meth:`compute_categorization_over_time` with no arguments.
        """
        logger.debug("Calculating splits per variable type.")
        return self.compute_categorization_over_time()

    # ------------------------------------------------------------------
    # Split parsing (deprecated helpers retained for backward compatibility)
    # ------------------------------------------------------------------

    def parse_split_values(self, value) -> tuple[str, str, set[str]]:
        """Parse a raw 'split' string into (variable, sign, value-set).

        .. deprecated::
            Prefer :func:`parse_split` which returns a typed
            :class:`Split` instance.  This shim is retained so existing
            callers keep working.
        """
        warnings.warn(
            "ADMTreesModel.parse_split_values is deprecated; use the "
            "module-level parse_split() which returns a typed Split.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(value, (tuple, pl.Series)):  # pragma: no cover
            value = value[0]
        try:
            split = parse_split(value)
        except ValueError:  # pragma: no cover
            return self._parse_legacy(value)
        if split.operator in {"<", ">", "=="}:
            return split.variable, split.operator, {str(split.value)}
        if split.operator == "is":
            return split.variable, "is", {str(split.value)}
        # 'in'
        members = split.value if isinstance(split.value, tuple) else (str(split.value),)
        return split.variable, "in", set(members)

    @staticmethod
    def parse_split_values_with_spaces(value) -> tuple[str, str, str]:  # pragma: no cover
        """Legacy stateful parser; kept for API compatibility.

        .. deprecated::
            Prefer :func:`parse_split`.
        """
        warnings.warn(
            "parse_split_values_with_spaces is deprecated; use parse_split().",
            DeprecationWarning,
            stacklevel=2,
        )
        splittypes = {">", "<", "in", "is", "=="}
        stage = "predictor"
        variable = ""
        sign = ""
        splitvalue = ""
        for item in value.split(" "):
            if item in splittypes:
                stage = "sign"
            if stage == "predictor":
                variable += " " + item
            elif stage == "values":
                splitvalue += item
            elif stage == "sign":
                sign = item
                stage = "values"
        return variable.strip(), sign, splitvalue

    def _parse_legacy(self, value) -> tuple[str, str, set[str]]:  # pragma: no cover
        var, sign, splitvalue = self.parse_split_values_with_spaces(value)
        if sign in {"<", ">", "=="}:
            return var, sign, {splitvalue}
        return var, sign, set(splitvalue.split(","))

    # ------------------------------------------------------------------
    # Predictor extraction
    # ------------------------------------------------------------------

    def get_predictors(self) -> dict[str, str] | None:
        """Extract predictor names and types from model metadata.

        Tries explicit metadata first (``configuration.predictors`` then
        ``predictors``); falls back to inferring from tree splits when
        neither is present.
        """
        config = self._properties.get("configuration") or {}
        predictors = config.get("predictors") or self._properties.get("predictors")
        if not predictors:
            return self._infer_predictors_from_splits()
        result = {}
        for predictor in predictors:
            if isinstance(predictor, str):  # pragma: no cover
                predictor = json.loads(predictor)
            result[predictor["name"]] = predictor["type"]
        return result

    def _infer_predictors_from_splits(self) -> dict[str, str] | None:
        """Infer predictor names + types by walking all tree splits."""
        var_ops: dict[str, set[str]] = collections.defaultdict(set)
        for tree in self.model:
            for node in _iter_nodes(tree):
                if node.split is not None:
                    var_ops[node.split.variable].add(node.split.operator)
        if not var_ops:
            logger.debug("No splits found, cannot infer predictors.")
            return None
        result: dict[str, str] = {}
        for name, ops in var_ops.items():
            if "<" in ops:
                result[name] = "numeric"
            else:
                result[name] = "symbolic"
        logger.debug("Inferred %d predictors from tree splits.", len(result))
        return result

    # ------------------------------------------------------------------
    # Splits / gains / tree-level views
    # ------------------------------------------------------------------

    @cached_property
    def _splits_and_gains(
        self,
    ) -> tuple[dict[int, list[str]], dict[int, list[float]], pl.DataFrame]:
        """Compute (splits_per_tree, gains_per_tree, gains_per_split) once.

        Backs the public ``splits_per_tree`` / ``gains_per_tree`` /
        ``gains_per_split`` properties via a single tree-walk per tree.
        Implemented as a ``cached_property`` rather than ``@lru_cache``
        because lru_cache holds a strong reference to ``self`` and would
        leak the entire ADMTreesModel instance for the lifetime of the
        cache.

        Zero-gain splits are kept (with ``gains == 0.0``) in
        ``gains_per_split`` so the per-split DataFrame is always aligned
        with ``splits_per_tree``.  ``gains_per_tree`` continues to keep
        only positive gains for backward compatibility.
        """
        # Touch predictors so legacy callers still see them populated.
        self.predictors

        splits_per_tree: dict[int, list[str]] = {}
        gains_per_tree: dict[int, list[float]] = {}
        all_splits: list[str] = []
        all_gains: list[float] = []
        all_predictors: list[str] = []
        for tree_id, tree in enumerate(self.model):
            tsplits: list[str] = []
            tgains_positive: list[float] = []
            for node in _iter_nodes(tree):
                if node.split is not None:
                    tsplits.append(node.split.raw)
                    all_splits.append(node.split.raw)
                    all_predictors.append(node.split.variable)
                    all_gains.append(node.gain)
                    if node.gain > 0:
                        tgains_positive.append(node.gain)
            splits_per_tree[tree_id] = tsplits
            gains_per_tree[tree_id] = tgains_positive

        gains_per_split = pl.DataFrame(
            {"split": all_splits, "gains": all_gains, "predictor": all_predictors},
        )
        return splits_per_tree, gains_per_tree, gains_per_split

    def get_grouped_gains_per_split(self) -> pl.DataFrame:
        """Gains per split, grouped by split string with helpful aggregates."""

        def _sign_and_values(raw: str) -> dict[str, Any]:
            split = parse_split(raw)
            sign = "in" if split.operator == "is" else split.operator
            if isinstance(split.value, tuple):
                values = set(split.value)
            else:
                values = {str(split.value)}
            return {"sign": sign, "values": values}

        return (
            self.gains_per_split.group_by("split", maintain_order=True)
            .agg(
                [
                    pl.first("predictor"),
                    pl.col("gains").implode(),
                    pl.col("gains").mean().alias("mean"),
                    pl.first("split").map_elements(_sign_and_values, return_dtype=pl.Object).alias("_parsed"),
                ],
            )
            .with_columns(
                sign=pl.col("_parsed").map_elements(lambda d: d["sign"], return_dtype=pl.Utf8),
                values=pl.col("_parsed").map_elements(lambda d: d["values"], return_dtype=pl.Object),
                n=pl.col("gains").list.len(),
            )
            .drop("_parsed")
        )

    def plot_splits_per_variable(self, subset: set | None = None, show: bool = True):
        """Box-plot of gains per split for each variable."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm")
        figlist = []
        for (name,), data in self.gains_per_split.group_by("predictor"):
            if subset is None or name in subset:
                fig = make_subplots()
                fig.add_trace(
                    go.Box(
                        x=data.get_column("split"),
                        y=data.get_column("gains"),
                        name="Gain",
                    ),
                )
                grouped = self.grouped_gains_per_split.filter(pl.col("predictor") == name)
                fig.add_trace(
                    go.Scatter(
                        x=grouped.select("split").to_series().to_list(),
                        y=grouped.select("n").to_series().to_list(),
                        name="Number of splits",
                        mode="lines+markers",
                    ),
                )
                fig.update_layout(
                    template="none",
                    title=f"Splits on {name}",
                    xaxis_title="Split",
                    yaxis_title="Number",
                )
                if show:
                    fig.show()  # pragma: no cover
                figlist.append(fig)
        return figlist

    def get_tree_stats(self) -> pl.DataFrame:
        """Generate a dataframe with useful stats for each tree."""
        rows = []
        for tree_id, tree in enumerate(self.model):
            depth = 0
            n_splits = 0
            gains: list[float] = []
            root_score = tree.get("score", 0.0)
            for node in _iter_nodes(tree):
                depth = max(depth, node.depth)
                if node.split is not None:
                    n_splits += 1
                    if node.gain > 0:
                        gains.append(node.gain)
            rows.append(
                {
                    "treeID": tree_id,
                    "score": root_score,
                    "depth": depth - 1,  # match legacy behaviour
                    "nsplits": n_splits,
                    "gains": gains,
                    "meangains": mean(gains) if gains else 0,
                },
            )
        return pl.from_dicts(rows)

    def get_all_values_per_split(self) -> dict[str, set]:
        """All distinct split values seen for each predictor."""
        splitvalues: dict[str, set] = {}
        for (name,), group in self.grouped_gains_per_split.group_by("predictor"):
            splitvalues.setdefault(name, set())
            for v in group.get_column("values").to_list():
                try:
                    splitvalues[name] = splitvalues[name].union(v)
                except Exception as exc:  # pragma: no cover
                    logger.debug("Could not union split values for %s: %s", name, exc)
        return splitvalues

    # ------------------------------------------------------------------
    # Tree representation, plotting, scoring
    # ------------------------------------------------------------------

    def get_tree_representation(self, tree_number: int) -> dict[int, dict]:
        """Build a flat node-id-keyed representation of one tree.

        Walks ``self.model[tree_number]`` in pre-order (left subtree
        before right) and returns a dict keyed by 1-based node id.

        Each entry has ``score``; internal nodes additionally carry
        ``split``, ``gain``, ``left_child`` and ``right_child``; non-root
        nodes carry ``parent_node``.

        This replaces an earlier implementation that mutated three
        accumulator parameters and relied on a final ``del`` to drop a
        spurious trailing entry.
        """
        nodes: dict[int, dict] = {}
        next_id = 0

        def visit(node: dict, parent_id: int | None) -> int:
            nonlocal next_id
            next_id += 1
            my_id = next_id
            info: dict = {"score": node["score"]}
            if parent_id is not None:
                info["parent_node"] = parent_id
            nodes[my_id] = info
            if "split" in node:
                info["split"] = node["split"]
                info["gain"] = node["gain"]
                info["left_child"] = visit(node["left"], my_id)
                info["right_child"] = visit(node["right"], my_id)
            return my_id

        visit(self.model[tree_number], parent_id=None)
        return nodes

    def plot_tree(
        self,
        tree_number: int,
        highlighted: dict | list | None = None,
        show: bool = True,
    ) -> pydot.Graph:
        """Plot the chosen decision tree."""
        try:
            import pydot
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["pydot"], "AGB", deps_group="adm")
        if isinstance(highlighted, dict):
            highlighted = self.get_visited_nodes(tree_number, highlighted)[0]
        else:  # pragma: no cover
            highlighted = highlighted or []
        nodes = self.get_tree_representation(tree_number)
        graph = pydot.Dot("my_graph", graph_type="graph", rankdir="BT")
        for key, node in nodes.items():
            color = "green" if key in highlighted else "white"
            label = f"Score: {node['score']}"
            if "split" in node:
                split_obj = parse_split(node["split"])
                if split_obj.operator == "in" and isinstance(split_obj.value, tuple):
                    members = split_obj.value
                    if len(members) <= 3:
                        members_label: str = str(set(members))
                    else:
                        totallen = len(self.all_values_per_split[split_obj.variable])
                        members_label = f"{list(members[:2]) + ['...']} ({len(members)}/{totallen})"
                    label += f"\nSplit: {split_obj.variable} in {members_label}\nGain: {node['gain']}"
                else:
                    label += f"\nSplit: {node['split']}\nGain: {node['gain']}"
                graph.add_node(
                    pydot.Node(
                        name=str(key),
                        label=label,
                        shape="box",
                        style="filled",
                        fillcolor=color,
                    )
                )
            else:
                graph.add_node(
                    pydot.Node(
                        name=str(key),
                        label=label,
                        shape="ellipse",
                        style="filled",
                        fillcolor=color,
                    )
                )
            if "parent_node" in node:
                graph.add_edge(pydot.Edge(str(key), str(node["parent_node"])))

        if show:  # pragma: no cover
            try:
                from IPython.display import Image, display
            except ImportError:
                raise ValueError(
                    "IPython not installed, please install it using your package manager (e.g. `pip install IPython`).",
                )
            try:
                display(Image(graph.create_png()))  # type: ignore[attr-defined]
            except FileNotFoundError as exc:
                logger.error(
                    "Dot/Graphviz not installed; please install it on your machine: %s",
                    exc,
                )
        return graph

    def get_visited_nodes(
        self,
        treeID: int,
        x: dict,
        save_all: bool = False,
    ) -> tuple[list, float, list]:
        """Trace the path through one tree for the given feature values."""
        tree = self.get_tree_representation(treeID)
        current_node_id = 1
        leaf = False
        visited: list[int] = []
        scores: list[dict] = []
        while not leaf:
            visited.append(current_node_id)
            current_node = tree[current_node_id]
            if "split" in current_node:
                split_obj = parse_split(current_node["split"])
                op = split_obj.operator
                try:
                    feature_value = x[split_obj.variable]
                except KeyError as exc:
                    raise KeyError(
                        f"Missing predictor {split_obj.variable!r} required by tree "
                        f"{treeID} at node {current_node_id}; provide it in `x`.",
                    ) from exc
                if op in {"in", "is"}:
                    splitvalue = f"'{feature_value}'"
                    op = "in"
                else:
                    splitvalue = feature_value
                if save_all:
                    scores.append({current_node["split"]: current_node["gain"]})
                # Resolve the right-hand side of the comparison.
                if op in {"<", ">"}:
                    rhs: Any = float(split_obj.value)  # type: ignore[arg-type]
                elif isinstance(split_obj.value, tuple):
                    rhs = set(split_obj.value)
                else:
                    rhs = split_obj.value
                if self._safe_condition_evaluate(splitvalue, op, rhs):
                    current_node_id = current_node["left_child"]
                else:
                    current_node_id = current_node["right_child"]
            else:
                leaf = True
        return visited, current_node["score"], scores

    def get_all_visited_nodes(self, x: dict) -> pl.DataFrame:
        """Score every tree against ``x`` and return per-tree visit info."""
        tree_ids: list[int] = []
        visited_nodes: list[list[int]] = []
        scores: list[float] = []
        splits: list[str] = []
        for tree_id in range(len(self.model)):
            tree_ids.append(tree_id)
            visits = self.get_visited_nodes(tree_id, x, save_all=True)
            visited_nodes.append(visits[0])
            scores.append(visits[1])
            splits.append(str(visits[2]))
        return pl.DataFrame(
            [tree_ids, visited_nodes, scores, splits],
            schema=["treeID", "visited_nodes", "score", "splits"],
        )

    def score(self, x: dict) -> float:
        """Compute the (sigmoid-normalised) propensity score for ``x``.

        Calls :meth:`get_visited_nodes` per tree and sums the resulting
        leaf scores; avoids building the full per-tree DataFrame that
        :meth:`get_all_visited_nodes` would produce.
        """
        total = 0.0
        for tree_id in range(len(self.model)):
            total += self.get_visited_nodes(tree_id, x)[1]
        return 1 / (1 + exp(-total))

    def plot_contribution_per_tree(self, x: dict, show: bool = True):
        """Plot the per-tree contribution toward the final propensity."""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm")

        scores = (
            self.get_all_visited_nodes(x)
            .sort("treeID")
            .with_row_index("row_idx")
            .with_columns(
                [
                    pl.col("score").cum_sum().alias("scoresum"),
                    (pl.col("score").cum_sum() / (pl.col("row_idx") + 1)).alias("mean"),
                    (1 / (1 + (-pl.col("score").cum_sum()).exp())).alias("propensity"),
                ],
            )
        )
        fig = px.scatter(
            scores,
            x="row_idx",
            y="score",
            template="none",
            title="Score contribution per tree, for single prediction",
            labels={"row_idx": "Tree", "score": "Score"},
        )
        fig["data"][0]["showlegend"] = True
        fig["data"][0]["name"] = "Individual scores"
        fig.add_trace(go.Scatter(x=scores["row_idx"], y=scores["mean"], name="Cumulative mean"))
        fig.add_trace(go.Scatter(x=scores["row_idx"], y=scores["propensity"], name="Propensity"))
        fig.add_trace(
            go.Scatter(
                x=[scores["row_idx"][-1]],
                y=[scores["propensity"][-1]],
                text=[scores["propensity"][-1]],
                mode="markers+text",
                textposition="top right",
                name="Final propensity",
            ),
        )
        fig.update_xaxes(zeroline=False)
        if show:
            fig.show()  # pragma: no cover
        return fig

    def predictor_categorization(self, x: str, context_keys: list | None = None) -> str:
        """Default predictor categorisation function."""
        context_keys = context_keys if context_keys is not None else self.context_keys
        if context_keys is None:
            context_keys = []  # pragma: no cover
        if len(x.split(".")) > 1:
            return x.split(".")[0]
        if x in context_keys:
            return x
        return "Primary"  # pragma: no cover

    def compute_categorization_over_time(
        self,
        predictor_categorization: Callable | None = None,
        context_keys: list | None = None,
    ) -> tuple[list[collections.Counter], list[float]]:
        """Per-tree categorisation counts plus per-tree absolute scores."""
        context_keys = context_keys if context_keys is not None else self.context_keys
        categorize = predictor_categorization or self.predictor_categorization
        per_tree: list[collections.Counter] = []
        for splits in self.splits_per_tree.values():
            counter: collections.Counter = collections.Counter()
            for split in splits:
                counter.update(
                    [categorize(parse_split(split).variable, context_keys)],
                )
            per_tree.append(counter)
        return per_tree, self.tree_stats.select("score").to_series().abs().to_list()

    def plot_splits_per_variable_type(
        self,
        predictor_categorization: Callable | None = None,
        **kwargs,
    ):
        """Stacked-area chart of categorised split counts per tree."""
        try:
            import plotly.express as px
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB", deps_group="adm")
        if predictor_categorization is not None:  # pragma: no cover
            to_plot = self.compute_categorization_over_time(predictor_categorization)[0]
        else:
            to_plot = self.splits_per_variable_type[0]
        df = pl.DataFrame(to_plot)
        df = df.select(sorted(df.columns))
        fig = px.area(
            df,
            title="Variable types per tree",
            labels={"index": "Tree number", "value": "Number of splits"},
            template="none",
            **kwargs,
        )
        fig.layout["updatemenus"] += (
            dict(
                type="buttons",
                direction="left",
                active=0,
                buttons=[
                    dict(
                        args=[
                            {"groupnorm": None},
                            {"yaxis": {"title": "Number of splits"}},
                        ],
                        label="Absolute",
                        method="update",
                    ),
                    dict(
                        args=[
                            {"groupnorm": "percent"},
                            {"yaxis": {"title": "Percentage of splits"}},
                        ],
                        label="Relative",
                        method="update",
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.3,
                yanchor="top",
            ),
        )
        return fig


# =============================================================================
# MultiTrees
# =============================================================================


@dataclass
class MultiTrees:
    """A collection of :class:`ADMTreesModel` snapshots indexed by timestamp.

    Construct via :meth:`from_datamart`.
    """

    trees: dict[str, ADMTreesModel]
    model_name: str | None = None
    context_keys: list | None = None

    def __repr__(self) -> str:
        mod = "" if self.model_name is None else f" for {self.model_name}"
        keys = list(self.trees.keys())
        if not keys:  # pragma: no cover
            return f"MultiTrees object{mod}, empty"
        return f"MultiTrees object{mod}, with {len(self)} trees ranging from {keys[0]} to {keys[-1]}"

    def __getitem__(self, index: int | str) -> ADMTreesModel:
        """Return the :class:`ADMTreesModel` at ``index``.

        Integer indices select by insertion order; string indices select
        by snapshot timestamp.  Use :meth:`items` if you need both keys
        and values together.
        """
        if isinstance(index, int):
            return list(self.trees.values())[index]
        return self.trees[index]

    def __len__(self) -> int:
        return len(self.trees)

    def items(self):
        """Iterate ``(timestamp, model)`` pairs in insertion order."""
        return self.trees.items()

    def values(self):
        """Iterate :class:`ADMTreesModel` instances in insertion order."""
        return self.trees.values()

    def keys(self):
        """Iterate snapshot timestamps in insertion order."""
        return self.trees.keys()

    def __iter__(self):
        return iter(self.trees)

    def __add__(self, other: MultiTrees | ADMTreesModel) -> MultiTrees:
        if isinstance(other, MultiTrees):
            return MultiTrees(
                trees={**self.trees, **other.trees},
                model_name=self.model_name or other.model_name,
                context_keys=self.context_keys or other.context_keys,
            )
        if isinstance(other, ADMTreesModel):
            timestamp = other.metrics.get("factory_update_time")
            if not timestamp:
                raise ValueError(
                    "Cannot add ADMTreesModel to MultiTrees: model has no "
                    "'factory_update_time' to use as a snapshot key. Add it "
                    "to a fresh MultiTrees with an explicit timestamp key.",
                )
            return MultiTrees(
                trees={**self.trees, str(timestamp): other},
                model_name=self.model_name,
                context_keys=self.context_keys,
            )
        return NotImplemented  # pragma: no cover

    @property
    def first(self) -> ADMTreesModel:
        return self[0]

    @property
    def last(self) -> ADMTreesModel:
        return self[-1]

    @classmethod
    def from_datamart(
        cls,
        df: pl.DataFrame,
        n_threads: int = 1,
        configuration: str | None = None,
    ) -> MultiTrees:
        """Decode every Modeldata blob in ``df`` for a single configuration.

        Returns one :class:`MultiTrees` containing one
        :class:`ADMTreesModel` per snapshot.

        Parameters
        ----------
        df : pl.DataFrame
            Datamart slice.  Must contain ``Modeldata``, ``SnapshotTime``
            and ``Configuration`` columns and cover exactly one
            Configuration.  Use :meth:`from_datamart_grouped` if ``df``
            spans multiple configurations.
        n_threads : int
            Worker count for parallel base64+zlib decoding.
        configuration : str | None
            Optional explicit Configuration name; required if ``df``
            doesn't already contain a single Configuration.
        """
        decoded = cls._decode_datamart_frame(df, n_threads=n_threads)
        configs = {cfg for cfg, _, _ in decoded}
        if configuration is not None:
            decoded = [(cfg, ts, mdl) for (cfg, ts, mdl) in decoded if cfg == configuration]
            chosen = configuration
        elif len(configs) == 1:
            chosen = next(iter(configs))
        else:
            raise ValueError(
                f"from_datamart received {len(configs)} configurations "
                f"({sorted(configs)!r}); pass `configuration=` to pick one, "
                "or call from_datamart_grouped to get one MultiTrees per config.",
            )
        trees = {ts: mdl for _, ts, mdl in decoded}
        return cls(trees=trees, model_name=chosen)

    @classmethod
    def from_datamart_grouped(
        cls,
        df: pl.DataFrame,
        n_threads: int = 1,
    ) -> dict[str, MultiTrees]:
        """Decode every Modeldata blob in ``df``, grouped by Configuration.

        Returns a mapping of configuration name to :class:`MultiTrees`.
        Use :meth:`from_datamart` instead when the input has only one
        configuration.
        """
        decoded = cls._decode_datamart_frame(df, n_threads=n_threads)
        per_config: dict[str, dict[str, ADMTreesModel]] = {}
        for cfg, ts, mdl in decoded:
            per_config.setdefault(cfg, {})[ts] = mdl
        return {cfg: cls(trees=trees, model_name=cfg) for cfg, trees in per_config.items()}

    @staticmethod
    def _decode_datamart_frame(
        df: pl.DataFrame,
        n_threads: int = 1,
    ) -> list[tuple[str, str, ADMTreesModel]]:
        """Decode every blob in ``df`` and return ``(config, timestamp, model)`` rows."""
        df = df.filter(pl.col("Modeldata").is_not_null()).select(
            # Format SnapshotTime explicitly — strip_chars_end strips a *set*
            # of characters, not a literal suffix, so it would mangle
            # timestamps ending in 0 (e.g. 12:30:20 → 12:30:2).
            pl.col("SnapshotTime").dt.round("1s").dt.strftime("%Y-%m-%d %H:%M:%S"),
            pl.col("Modeldata").str.decode("base64"),
            pl.col("Configuration").cast(pl.Utf8),
        )
        if len(df) > 50 and n_threads == 1:
            logger.info(
                "Decoding %d models; setting n_threads higher may speed this up.",
                len(df),
            )
        configs = df["Configuration"].to_list()
        timestamps = df["SnapshotTime"].to_list()
        blobs = df["Modeldata"].to_list()
        try:
            from tqdm import tqdm

            iterable: Any = tqdm(blobs)
        except ImportError:  # pragma: no cover
            iterable = blobs
        with multiprocessing.Pool(n_threads) as p:
            decoder = map if n_threads < 2 else p.imap
            models = list(decoder(ADMTreesModel.from_datamart_blob, iterable))
        return list(zip(configs, timestamps, models, strict=True))

    def compute_over_time(
        self,
        predictor_categorization: Callable | None = None,
    ) -> pl.DataFrame:
        """Return per-tree categorisation counts across snapshots, with a
        ``SnapshotTime`` column per row.
        """
        outdf = []
        for timestamp, tree in self.trees.items():
            if predictor_categorization is not None:
                to_plot = tree.compute_categorization_over_time(
                    predictor_categorization,
                )[0]
            else:
                to_plot = tree.splits_per_variable_type[0]
            outdf.append(
                pl.DataFrame(to_plot).with_columns(
                    SnapshotTime=pl.lit(timestamp).str.to_datetime(format="%Y-%m-%d %H:%M:%S").dt.date(),
                ),
            )
        return pl.concat(outdf, how="diagonal")
