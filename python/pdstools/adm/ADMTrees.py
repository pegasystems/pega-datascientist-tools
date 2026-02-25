from __future__ import annotations

__all__ = ["AGB", "ADMTrees", "ADMTreesModel", "MultiTrees"]

import base64
import collections
import copy
import functools
import json
import logging
import math
import multiprocessing
import operator
import zlib
from dataclasses import dataclass
from functools import cached_property, lru_cache
from math import exp
from statistics import mean, median, stdev
from typing import (
    TYPE_CHECKING,
    Any,
)

import polars as pl

from ..utils import cdh_utils
from ..utils.namespaces import MissingDependenciesException
from ..utils.types import QUERY

if TYPE_CHECKING:  # pragma: no cover
    import pydot  # type: ignore[import-untyped]

    from .ADMDatamart import ADMDatamart

logger = logging.getLogger(__name__)


class AGB:
    def __init__(self, datamart: ADMDatamart):  # pragma: no cover
        self.datamart = datamart

    def discover_model_types(
        self,
        df: pl.LazyFrame,
        by: str = "Configuration",
    ) -> dict:  # pragma: no cover
        """Discovers the type of model embedded in the pyModelData column.

        By default, we do a group_by Configuration, because a model rule can only
        contain one type of model. Then, for each configuration, we look into the
        pyModelData blob and find the _serialClass, returning it in a dict.

        Parameters
        ----------
        df: pl.LazyFrame
            The dataframe to search for model types
        by: str
            The column to look for types in. Configuration is recommended.
        allow_collect: bool, default = False
            set to True to allow discovering modelTypes, even if in lazy strategy.
            It will fetch one modelData string per configuration.

        """
        if "Modeldata" not in df.columns:
            raise ValueError(
                "Modeldata column not in the data. "
                "Please make sure to include it by setting 'subset' to False.",
            )

        def _get_type(val):
            import base64
            import zlib

            return next(
                line.split('"')[-2].split(".")[-1]
                for line in zlib.decompress(base64.b64decode(val)).decode().split("\n")
                if line.startswith('  "_serialClass"')
            )

        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        types = (
            df.filter(pl.col("Modeldata").is_not_null())
            .group_by(by)
            .agg(pl.col("Modeldata").last())
            .collect()
            .with_columns(pl.col("Modeldata").map_elements(lambda v: _get_type(v)))
            .to_dicts()
        )
        return {key: value for key, value in [i.values() for i in types]}

    def get_agb_models(
        self,
        last: bool = False,
        by: str = "Configuration",
        n_threads: int = 1,
        query: QUERY | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> ADMTrees:  # pragma: no cover
        """Method to automatically extract AGB models.

        Recommended to subset using the querying functionality
        to cut down on execution time, because it checks for each
        model ID. If you only have AGB models remaining after the query,
        it will only return proper AGB models.

        Parameters
        ----------
        last: bool, default = False
            Whether to only look at the last snapshot for each model
        by: str, default = 'Configuration'
            Which column to determine unique models with
        n_threads: int, default = 6
            The number of threads to use for extracting the models.
            Since we use multithreading, setting this to a reasonable value
            helps speed up the import.
        query: Optional[Union[pl.Expr, list[pl.Expr], str, dict[str, list]]]
            Please refer to :meth:`._apply_query`
        verbose: bool, default = False
            Whether to print out information while importing

        """
        df = self.datamart.model_data
        if df is None:
            raise ValueError("No model data available to get agb models.")
        if query is not None:
            df = cdh_utils._apply_query(df, query)

        model_types = self.discover_model_types(df)
        agb_models = [
            model for model, type in model_types.items() if type.endswith("GbModel")
        ]
        logger.info(f"Found AGB models: {agb_models}")
        df = df.filter(pl.col("Configuration").is_in(agb_models))
        if df.select(pl.col("ModelID").n_unique()).collect().item() == 0:
            raise ValueError("No models found.")

        if last:
            return ADMTrees(
                self.datamart.aggregates.last(data=df),
                n_threads=n_threads,
                verbose=verbose,
                **kwargs,
            )
        return ADMTrees(
            df.select("Configuration", "SnapshotTime", "Modeldata").collect(),
            n_threads=n_threads,
            verbose=verbose,
            **kwargs,
        )


class ADMTrees:  # pragma: no cover
    def __new__(cls, file, n_threads=6, verbose=True, **kwargs):
        if isinstance(file, pl.DataFrame):
            logger.info("DataFrame supplied.")
            file = file.filter(pl.col("Modeldata").is_not_null())
            if len(file) > 1:
                logger.info("Multiple models found, so creating MultiTrees")
                if verbose:
                    print(
                        f"AGB models found: {file.select(pl.col('Configuration').unique())}",
                    )
                return cls.get_multi_trees(
                    file=file,
                    n_threads=n_threads,
                    verbose=verbose,
                    **kwargs,
                )
            logger.info("One model found, so creating ADMTrees")
            return ADMTrees(file.select("Modeldata").item(), **kwargs)
        if isinstance(file, pl.Series):
            logger.info("One model found, so creating ADMTrees")
            logger.debug(file.select(pl.col("Modeldata")))
            return ADMTrees(file.select(pl.col("Modeldata")), **kwargs)

        logger.info("No need to extract from DataFrame, regular import")
        return ADMTreesModel(file, **kwargs)

    @staticmethod
    def get_multi_trees(file: pl.DataFrame, n_threads=1, verbose=True, **kwargs):
        out = {}
        df = file.filter(pl.col("Modeldata").is_not_null()).select(
            pl.col("SnapshotTime")
            .dt.round("1s")
            .cast(pl.Utf8)
            .str.strip_chars_end(".000000000"),
            pl.col("Modeldata").str.decode("base64"),
            pl.col("Configuration").cast(pl.Utf8),
        )
        if len(df) > 50 and n_threads == 1 and verbose:
            print(
                f"""Decoding {len(df)} models,
            setting n_threads to a higher value may speed up processing time.""",
            )
        df2 = df.select(
            pl.concat_list(["Configuration", "SnapshotTime"]),
            "Modeldata",
        ).to_dict()

        try:
            from tqdm import tqdm

            iterable = tqdm(df2["Modeldata"])
        except ImportError:
            iterable = df2["Modeldata"]

        with multiprocessing.Pool(n_threads) as p:
            f: Any = map if n_threads < 2 else p.imap
            out = dict(
                zip(
                    map(tuple, df2["Configuration"].to_list()),
                    list(f(ADMTrees, iterable)),
                ),
            )
        dict_per_config: dict[Any, Any] = {key[0]: {} for key in out.keys()}
        for (configuration, timestamp), value in out.items():
            dict_per_config[configuration][timestamp] = value
        return {
            key: MultiTrees(value, model_name=key)
            for key, value in dict_per_config.items()
        }


class ADMTreesModel:
    """Functions for ADM Gradient boosting

    ADM Gradient boosting models consist of multiple trees,
    which build upon each other in a 'boosting' fashion.
    This class provides some functions to extract data from
    these trees, such as the features on which the trees split,
    important values for these features, statistics about the trees,
    or visualising each individual tree.

    Parameters
    ----------
    file: str
        The input file as a json (see notes)

    Attributes
    ----------
    trees: dict
    properties: dict
    learning_rate: float
    model: dict
    treeStats: dict
    splitsPerTree: dict
    gainsPerTree: dict
    gainsPerSplit: pl.DataFrame
    groupedGainsPerSplit: dict
    predictors: set
    allValuesPerSplit: dict

    Notes
    -----
    The input file is the extracted json file of the 'save model'
    action in Prediction Studio. The Datamart column 'pyModelData'
    also contains this information, but it is compressed and
    the values for each split is encoded. Using the 'save model'
    button, only that data is decompressed and decoded.

    """

    def __init__(self, file: str, **kwargs):
        logger.info("Reading model...")
        self._read_model(file, **kwargs)
        self.nospaces = True
        if self.trees is None:  # pragma: no cover
            raise ValueError("Import unsuccessful.")

    def _read_model(self, file, **kwargs):
        def _import(file):
            logger.info("Trying regular import.")
            with open(file) as f:
                file = json.load(f)
            logger.info("Regular import successful.")

            return file

        def read_url(file):  # pragma: no cover
            logger.info("Trying to read from URL.")
            import requests

            file = requests.get(file).content
            logger.info("Import from URL successful.")
            return file

        def decode_string(file):  # pragma: no cover
            logger.info("Trying to decompress the string.")
            file = zlib.decompress(base64.b64decode(file))
            logger.info("Decompressing string successful.")
            return file

        decode = kwargs.pop("decode", False)

        if isinstance(file, str):
            try:  # pragma: no cover
                self.trees = json.loads(decode_string(file))
                if not self.trees["_serialClass"].endswith("GbModel"):
                    return ValueError("Not an AGB model")
                decode = True
                logger.info("Model needs to be decoded")
            except Exception:
                logger.info("Decoding failed, exception:", exc_info=True)
                try:
                    self.trees = _import(file)
                    logger.info("Regular export, no need for decoding")
                except Exception:  # pragma: no cover
                    logger.info("Regular import failed, exception:", exc_info=True)
                    try:
                        self.trees = json.loads(read_url(file))
                        logger.info("Read model from URL")
                    except Exception:  # pragma: no cover
                        logger.info(
                            "Reading from URL failed, exception:",
                            exc_info=True,
                        )
                        msg = (
                            "Could not import the AGB model.\n"
                            "Please check if your model export is a valid format (json or base64 encoded). \n"
                            "Also make sure you're using Pega version 8.7.3 or higher, "
                            "as the export format from before that isn't supported."
                        )

                        raise ValueError(msg)
        elif isinstance(file, bytes):  # pragma: no cover
            self.trees = json.loads(zlib.decompress(file))
            if not self.trees["_serialClass"].endswith("GbModel"):
                return ValueError("Not an AGB model")
            decode = True
            logger.info("Model needs to be decoded")
        elif isinstance(file, dict):  # pragma: no cover
            logger.info("dict supplied, so no reading required")
            self.trees = file
        self.raw_model = copy.deepcopy(file)

        self._post_import_cleanup(decode=decode, **kwargs)

    def _decode_trees(self):  # pragma: no cover
        def quantile_decoder(encoder: dict, index: int, verbose=False):
            if encoder["summaryType"] == "INITIAL_SUMMARY":
                return encoder["summary"]["initialValues"][
                    index
                ]  # Note: could also be index-1
            return encoder["summary"]["list"][index - 1].split("=")[0]

        def string_decoder(encoder: dict, index: int, sign, verbose=False):
            def set_types(split):
                logger.debug(split)
                split = split.rsplit("=", 1)
                split[1] = int(split[1])
                return tuple(reversed(split))

            valuelist = collections.OrderedDict(
                sorted([set_types(i) for i in encoder["symbols"]]),
            )
            splitvalues = list()
            for key, value in valuelist.items():
                if verbose:
                    print(key, value, index)
                if int(key) == index - 129 and index == 129 and sign == "<":
                    splitvalues.append("Missing")
                    break
                if self._safe_numeric_compare(int(key), sign, index - 129):
                    splitvalues.append(value)
                else:
                    pass
            output = ", ".join(splitvalues)
            return output

        def decode_split(split: str, verbose=False):
            if not isinstance(split, str):
                return split
            logger.debug(f"Decoding split: {split}")
            predictor, sign, splitval = split.split(" ")

            if sign == "LT":
                sign = "<"
            elif sign == "EQ":
                sign = "=="
            else:
                print("For now, only supporting less than and equality splits")
                raise ValueError((predictor, sign, splitval))
            variable = encoderkeys[int(predictor)]
            encoder = encoders[variable]
            variable_type = list(encoder["encoder"].keys())[0]
            to_decode = list(encoder["encoder"].values())[0]
            if variable_type == "quantileArray":
                val = quantile_decoder(to_decode, int(splitval))
            if variable_type == "stringTranslator":
                val = string_decoder(
                    to_decode,
                    int(splitval),
                    sign=sign,
                    verbose=verbose,
                )
                if val == "Missing":
                    sign, val = "is", "Missing"
                else:
                    val = "{ " + val + " }"
                    sign = "in"
            logger.debug(f"Decoded split: {variable} {sign} {val}")
            return f"{variable} {sign} {val}"

        try:
            encoders = self.trees["model"]["model"]["inputsEncoder"]["encoders"]
        except KeyError:
            encoders = self.trees["model"]["inputsEncoder"]["encoders"]
        encoderkeys = collections.OrderedDict()
        for encoder in encoders:
            encoderkeys[encoder["value"]["index"]] = encoder["key"]
        encoders = {encoder["key"]: encoder["value"] for encoder in encoders}

        def decode_all_trees(ob, func):
            if isinstance(ob, collections.abc.Mapping):
                return {k: decode_all_trees(v, func) for k, v in ob.items()}
            return func(ob)

        for i, model in enumerate(self.model):
            self.model[i] = decode_all_trees(model, decode_split)
            logger.debug(f"Decoded tree {i}")

    def _post_import_cleanup(self, decode, **kwargs):
        if not hasattr(self, "model"):
            logger.info("Adding model tag")

            try:
                self.model = self.trees["model"]["boosters"][0]["trees"]
            except Exception as e1:
                try:
                    self.model = self.trees["model"]["model"]["boosters"][0]["trees"]
                except Exception as e2:  # pragma: no cover
                    try:
                        self.model = self.trees["model"]["booster"]["trees"]
                    except Exception as e3:
                        try:
                            self.model = self.trees["model"]["model"]["booster"][
                                "trees"
                            ]
                        except Exception as e4:
                            raise (e1, e2, e3, e4)

        if decode:  # pragma: no cover
            logger.info("Decoding the tree splits.")
            self._decode_trees()

        try:
            self._properties = {
                prop[0]: prop[1] for prop in self.trees.items() if prop[0] != "model"
            }
        except Exception:  # pragma: no cover
            logger.info("Could not extract the properties.")
            self._properties = {}

        try:
            self.learning_rate = self._properties["configuration"]["parameters"][
                "learningRateEta"
            ]
        except Exception:  # pragma: no cover
            logger.info("Could not find the learning rate in the model.")

        try:
            self.context_keys = self._properties["configuration"]["contextKeys"]
        except Exception:  # pragma: no cover
            logger.info("Could not find context keys.")
            self.context_keys = kwargs.get("context_keys")

    def _depth(self, d: dict) -> int:
        """Calculates the depth of the tree, used in TreeStats."""
        if isinstance(d, dict):
            return 1 + (max(map(self._depth, d.values())) if d else 0)
        return 0

    def _safe_numeric_compare(
        self,
        left: float,
        operator: str,
        right: float,
    ) -> bool:
        """Safely compare two numeric values without using eval().

        This method replaces dangerous eval() calls with safe numeric comparisons.

        Parameters
        ----------
        left : Union[int, float]
            Left operand for comparison
        operator : str
            Comparison operator as string ('<', '>', '==', '<=', '>=', '!=')
        right : Union[int, float]
            Right operand for comparison

        Returns
        -------
        bool
            Result of the comparison

        Raises
        ------
        ValueError
            If operator is not supported

        """
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

    def _safe_condition_evaluate(
        self,
        value: Any,
        operator: str,
        comparison_set: set | float | str,
    ) -> bool:
        """Safely evaluate conditions without using eval().

        This method replaces dangerous eval() calls with safe condition evaluation.

        Parameters
        ----------
        value : Any
            The value to test
        operator : str
            The operator ('in', '<', '>', '==')
        comparison_set : Union[set, float, str]
            The value or set to compare against

        Returns
        -------
        bool
            Result of the condition evaluation

        """
        try:
            if operator == "in":
                return str(value).strip("'") in comparison_set
            if operator == "<":
                return float(str(value).strip("'")) < float(comparison_set)
            if operator == ">":
                return float(str(value).strip("'")) > float(comparison_set)
            if operator == "==":
                return str(value).strip("'") == str(comparison_set)
            raise ValueError(f"Unsupported operator: {operator}")
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Safe evaluation failed for {value} {operator} {comparison_set}: {e}",
            )
            return False

    @cached_property
    def metrics(self) -> dict[str, Any]:
        """Compute CDH_ADM005-style diagnostic metrics for this model.

        Returns a flat dictionary of key/value pairs aligned with the
        CDH_ADM005 telemetry event specification.  Metrics that cannot be
        computed from an exported model (e.g. saturation counts that
        require bin-level data) are omitted.

        See Also
        --------
        Pega CDH_ADM005 telemetry event specification in the Pega platform documentation.

        Returns
        -------
        dict[str, Any]
            Metric name → value mapping.

        """
        return self._compute_metrics()

    @staticmethod
    def metric_descriptions() -> dict[str, str]:
        """Return a dictionary mapping metric names to human-readable descriptions.

        These descriptions document every metric returned by the
        :attr:`metrics` property. They can be used programmatically to
        annotate reports or plots.

        Returns
        -------
        dict[str, str]
            Metric name → one-line description.

        """
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

    def _compute_metrics(self) -> dict[str, Any]:
        """Walk the trees once to gather all diagnostic metrics.

        For the full list of returned keys and their descriptions, see
        :meth:`metric_descriptions`.

        For exported (decoded) models, predictor types are inferred from
        split operators (``<`` → numeric, ``in``/``==`` → symbolic).
        For encoded models (from datamart blobs with ``inputsEncoder``),
        the encoder metadata provides authoritative type information.
        """

        # --- helpers -------------------------------------------------------
        def _count_nodes(tree: dict) -> int:
            c = 1
            if "left" in tree:
                c += _count_nodes(tree["left"])
            if "right" in tree:
                c += _count_nodes(tree["right"])
            return c

        def _tree_depth(tree: dict) -> int:
            d = 1
            dl = _tree_depth(tree["left"]) if "left" in tree else 0
            dr = _tree_depth(tree["right"]) if "right" in tree else 0
            return d + max(dl, dr)

        def _walk_tree(
            tree: dict,
            var_ops: dict[str, set[str]],
            var_split_count: dict[str, int],
            all_gains: list[float],
            leaf_scores: list[float],
            split_strings: list[str],
            numeric_split_count: list[int],
            symbolic_split_count: list[int],
            symbolic_set_sizes: list[int],
            leaf_count: list[int],
        ) -> None:
            """Single-pass recursive walk collecting all per-node data."""
            is_leaf = "split" not in tree
            if is_leaf:
                leaf_scores.append(tree["score"])
                leaf_count[0] += 1
                return

            # --- internal (split) node ---
            split_str = tree["split"]
            gain = tree.get("gain", 0.0)
            split_strings.append(split_str)

            if gain > 0:
                all_gains.append(gain)

            parts = split_str.split(" ", 2)
            var = parts[0]
            op = parts[1] if len(parts) > 1 else ""
            var_ops[var].add(op)
            var_split_count[var] += 1

            if op == "<":
                numeric_split_count[0] += 1
            elif op in {"in", "=="}:
                symbolic_split_count[0] += 1
                # Count set size for "in { a, b, c }" splits
                if op == "in" and len(parts) > 2:
                    set_body = parts[2].strip()
                    if set_body.startswith("{") and set_body.endswith("}"):
                        inner = set_body[1:-1].strip()
                        n_items = len(inner.split(",")) if inner else 0
                        symbolic_set_sizes.append(n_items)

            if "left" in tree:
                _walk_tree(
                    tree["left"],
                    var_ops,
                    var_split_count,
                    all_gains,
                    leaf_scores,
                    split_strings,
                    numeric_split_count,
                    symbolic_split_count,
                    symbolic_set_sizes,
                    leaf_count,
                )
            if "right" in tree:
                _walk_tree(
                    tree["right"],
                    var_ops,
                    var_split_count,
                    all_gains,
                    leaf_scores,
                    split_strings,
                    numeric_split_count,
                    symbolic_split_count,
                    symbolic_set_sizes,
                    leaf_count,
                )

        def _classify_predictor(name: str) -> str:
            """Classify a predictor as 'ih', 'context_key', or 'other'."""
            if name.startswith("IH."):
                return "ih"
            if (
                name.startswith("py")
                or name.startswith("Param.")
                or ".Context." in name
            ):
                return "context_key"
            return "other"

        # --- walk all trees (single pass) ----------------------------------
        var_ops: dict[str, set[str]] = collections.defaultdict(set)
        var_split_count: dict[str, int] = collections.defaultdict(int)
        depths: list[int] = []
        total_nodes = 0
        all_gains: list[float] = []
        leaf_scores: list[float] = []
        split_strings: list[str] = []
        numeric_split_count = [0]
        symbolic_split_count = [0]
        symbolic_set_sizes: list[int] = []
        leaf_count = [0]
        stump_count = 0
        per_tree_gains: list[list[float]] = []  # gains per tree for convergence
        root_scores: list[float] = []  # root score per tree for convergence

        for tree in self.model:
            total_nodes += _count_nodes(tree)
            depths.append(_tree_depth(tree))
            root_scores.append(tree["score"])

            # Track per-tree gains for convergence metrics
            pre_len = len(all_gains)
            _walk_tree(
                tree,
                var_ops,
                var_split_count,
                all_gains,
                leaf_scores,
                split_strings,
                numeric_split_count,
                symbolic_split_count,
                symbolic_set_sizes,
                leaf_count,
            )
            tree_gains = all_gains[pre_len:]
            per_tree_gains.append(tree_gains)
            if "split" not in tree:
                stump_count += 1

        n_trees = len(self.model)
        total_splits = numeric_split_count[0] + symbolic_split_count[0]

        # --- encoder-based classification (when available) -----------------
        encoder_info = self._get_encoder_info()

        if encoder_info is not None:
            all_predictors = set(encoder_info.keys())
            all_numeric = {
                n for n, info in encoder_info.items() if info["type"] == "numeric"
            }
            all_symbolic = {
                n for n, info in encoder_info.items() if info["type"] == "symbolic"
            }
            active_numeric = all_numeric & set(var_ops)
            active_symbolic = all_symbolic & set(var_ops)
        else:
            all_predictors = None
            active_numeric = {v for v, ops in var_ops.items() if "<" in ops}
            active_symbolic = {
                v for v, ops in var_ops.items() if "in" in ops or "==" in ops
            }

        active_ih = {v for v in var_ops if _classify_predictor(v) == "ih"}
        active_context_key = {
            v for v in var_ops if _classify_predictor(v) == "context_key"
        }
        active_other = {v for v in var_ops if _classify_predictor(v) == "other"}

        # total predictor counts from the configuration if available
        total_predictors: dict[str, str] | None = None
        try:
            total_predictors = self.predictors
        except Exception:
            pass

        if total_predictors is not None:
            all_ih = {n for n in total_predictors if _classify_predictor(n) == "ih"}
            total_pred_count = len(total_predictors)
            if all_predictors is None:
                all_numeric = {
                    n
                    for n, t in total_predictors.items()
                    if t.lower() in {"numeric", "double", "integer", "float"}
                }
                all_symbolic = {
                    n
                    for n, t in total_predictors.items()
                    if t.lower() in {"symbolic", "string", "boolean"}
                }
        elif all_predictors is not None:
            all_ih = {n for n in all_predictors if _classify_predictor(n) == "ih"}
            total_pred_count = len(all_predictors)
        else:
            all_ih = active_ih
            total_pred_count = len(var_ops)
            all_numeric = active_numeric
            all_symbolic = active_symbolic

        # --- saturation metrics (encoder-only) -----------------------------
        saturated_ctx = 0
        saturated_symbolic = 0
        max_saturation_ctx: float = 0.0
        if encoder_info is not None:
            for name, info in encoder_info.items():
                if info["max_bins"] is not None and info["max_bins"] > 0:
                    ratio = info["used_bins"] / info["max_bins"]
                    saturated = info["used_bins"] >= info["max_bins"]
                    cat = _classify_predictor(name)
                    if cat == "context_key":
                        if saturated:
                            saturated_ctx += 1
                        max_saturation_ctx = max(max_saturation_ctx, ratio)
                    if info["type"] == "symbolic" and saturated:
                        saturated_symbolic += 1

        # --- feature importance by gain ------------------------------------
        var_total_gain: dict[str, float] = collections.defaultdict(float)
        for tree in self.model:
            self._accumulate_gain(tree, var_total_gain)
        sum_all_gain = sum(var_total_gain.values())

        # --- training stats from properties --------------------------------
        props = getattr(self, "_properties", {}) or {}
        training = props.get("trainingStats", {})

        # === assemble metrics dict =========================================
        m: dict[str, Any] = {}

        # Properties-level metrics
        m["auc"] = props.get("auc", props.get("performance"))
        m["success_rate"] = props.get("successRate")
        m["factory_update_time"] = props.get("factoryUpdateTime")

        # Data quality
        m["response_positive_count"] = training.get("positiveCount")
        m["response_negative_count"] = training.get("negativeCount")

        # --- Model complexity ----------------------------------------------
        m["number_of_tree_nodes"] = total_nodes
        m["tree_depth_max"] = max(depths) if depths else 0
        m["tree_depth_avg"] = round(sum(depths) / len(depths), 2) if depths else 0.0
        m["tree_depth_std"] = round(stdev(depths), 2) if len(depths) >= 2 else 0.0
        m["number_of_trees"] = n_trees
        m["number_of_stump_trees"] = stump_count
        m["avg_leaves_per_tree"] = (
            round(leaf_count[0] / n_trees, 2) if n_trees > 0 else 0.0
        )
        m["number_of_splits_on_ih_predictors"] = sum(
            var_split_count[v] for v in active_ih
        )
        m["number_of_splits_on_context_key_predictors"] = sum(
            var_split_count[v] for v in active_context_key
        )
        m["number_of_splits_on_other_predictors"] = sum(
            var_split_count[v] for v in active_other
        )

        # --- Predictor info ------------------------------------------------
        m["total_number_of_active_predictors"] = len(var_ops)
        m["total_number_of_predictors"] = total_pred_count
        m["number_of_active_ih_predictors"] = len(active_ih)
        m["total_number_of_ih_predictors"] = len(all_ih)
        m["number_of_active_context_key_predictors"] = len(active_context_key)
        m["number_of_active_symbolic_predictors"] = len(active_symbolic)
        m["total_number_of_symbolic_predictors"] = len(all_symbolic)
        m["number_of_active_numeric_predictors"] = len(active_numeric)
        m["total_number_of_numeric_predictors"] = len(all_numeric)

        # --- Gain distribution ---------------------------------------------
        m["total_gain"] = round(sum(all_gains), 4) if all_gains else 0.0
        m["mean_gain_per_split"] = round(mean(all_gains), 4) if all_gains else 0.0
        m["median_gain_per_split"] = round(median(all_gains), 4) if all_gains else 0.0
        m["max_gain_per_split"] = round(max(all_gains), 4) if all_gains else 0.0
        m["gain_std"] = round(stdev(all_gains), 4) if len(all_gains) >= 2 else 0.0

        # --- Leaf scores ---------------------------------------------------
        m["number_of_leaves"] = leaf_count[0]
        m["leaf_score_mean"] = round(mean(leaf_scores), 6) if leaf_scores else 0.0
        m["leaf_score_std"] = (
            round(stdev(leaf_scores), 6) if len(leaf_scores) >= 2 else 0.0
        )
        m["leaf_score_min"] = round(min(leaf_scores), 6) if leaf_scores else 0.0
        m["leaf_score_max"] = round(max(leaf_scores), 6) if leaf_scores else 0.0

        # --- Split types ---------------------------------------------------
        m["number_of_numeric_splits"] = numeric_split_count[0]
        m["number_of_symbolic_splits"] = symbolic_split_count[0]
        m["symbolic_split_fraction"] = (
            round(symbolic_split_count[0] / total_splits, 4)
            if total_splits > 0
            else 0.0
        )
        unique_splits = set(split_strings)
        m["number_of_unique_splits"] = len(unique_splits)
        m["number_of_unique_predictors_split_on"] = len(var_ops)
        m["split_reuse_ratio"] = (
            round(len(split_strings) / len(unique_splits), 2) if unique_splits else 0.0
        )
        m["avg_symbolic_set_size"] = (
            round(mean(symbolic_set_sizes), 2) if symbolic_set_sizes else 0.0
        )

        # --- Learning convergence ------------------------------------------
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
            gains_first = [g for tree_g in per_tree_gains[:half] for g in tree_g]
            gains_last = [g for tree_g in per_tree_gains[half:] for g in tree_g]
            m["mean_gain_first_half"] = (
                round(mean(gains_first), 4) if gains_first else 0.0
            )
            m["mean_gain_last_half"] = round(mean(gains_last), 4) if gains_last else 0.0
        else:
            m["mean_gain_first_half"] = 0.0
            m["mean_gain_last_half"] = 0.0

        # --- Feature importance concentration ------------------------------
        if var_total_gain and sum_all_gain > 0:
            top_var = max(var_total_gain, key=var_total_gain.get)
            m["top_predictor_by_gain"] = top_var
            m["top_predictor_gain_share"] = round(
                var_total_gain[top_var] / sum_all_gain,
                4,
            )
            # Shannon entropy (normalised to [0, 1])
            n_vars = len(var_total_gain)
            if n_vars > 1:
                entropy = 0.0
                for g in var_total_gain.values():
                    p = g / sum_all_gain
                    if p > 0:
                        entropy -= p * math.log2(p)
                max_entropy = math.log2(n_vars)
                m["predictor_gain_entropy"] = (
                    round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0
                )
            else:
                m["predictor_gain_entropy"] = 0.0
        else:
            m["top_predictor_by_gain"] = None
            m["top_predictor_gain_share"] = 0.0
            m["predictor_gain_entropy"] = 0.0

        # --- Saturation (only when encoder metadata is available) ----------
        if encoder_info is not None:
            m["number_of_saturated_context_key_predictors"] = saturated_ctx
            m["number_of_saturated_symbolic_predictors"] = saturated_symbolic
            m["max_saturation_rate_on_context_key_predictors"] = round(
                max_saturation_ctx * 100,
                1,
            )

        return m

    @staticmethod
    def _accumulate_gain(tree: dict, var_gain: dict[str, float]) -> None:
        """Recursively accumulate gain per predictor variable."""
        if "split" in tree:
            var_name = tree["split"].split(" ", 1)[0]
            var_gain[var_name] += tree.get("gain", 0.0)
        if "left" in tree:
            ADMTreesModel._accumulate_gain(tree["left"], var_gain)
        if "right" in tree:
            ADMTreesModel._accumulate_gain(tree["right"], var_gain)

    def _get_encoder_info(self) -> dict[str, dict[str, Any]] | None:
        """Extract predictor metadata from the inputsEncoder if present.

        Returns a dict mapping predictor name to a dict with keys:
        - ``type``: ``"numeric"`` or ``"symbolic"``
        - ``used_bins``: number of bins currently used
        - ``max_bins``: maximum bins allowed (``None`` if unknown)

        Returns ``None`` if no encoder metadata is available (e.g. for
        exported/decoded models).
        """
        # Try different locations where encoders may live
        raw_encoders = None
        for path in [
            lambda: self.trees["model"]["inputsEncoder"]["encoders"],
            lambda: self.trees["model"]["model"]["inputsEncoder"]["encoders"],
        ]:
            try:
                raw_encoders = path()
                break
            except (KeyError, TypeError):
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
                result[name] = {
                    "type": "unknown",
                    "used_bins": 0,
                    "max_bins": None,
                }

        return result

    @cached_property
    def predictors(self):
        logger.info("Extracting predictors.")
        return self.get_predictors()

    @cached_property
    def tree_stats(self):
        logger.info("Calculating tree stats.")
        return self.get_tree_stats()

    @cached_property
    def splits_per_tree(self):
        return self.get_gains_per_split()[0]

    @cached_property
    def gains_per_tree(self):  # pragma: no cover
        return self.get_gains_per_split()[1]

    @cached_property
    def gains_per_split(self):
        return self.get_gains_per_split()[2]

    @cached_property
    def grouped_gains_per_split(self):
        logger.info("Calculating grouped gains per split.")
        return self.get_grouped_gains_per_split()

    @cached_property
    def all_values_per_split(self):
        logger.info("Calculating all values per split.")
        return self.get_all_values_per_split()

    @cached_property
    def splits_per_variable_type(self, **kwargs):
        logger.info("Calculating splits per variable type.")
        return self.compute_categorization_over_time(
            predictorCategorization=kwargs.pop("predictorCategorization", None),
        )

    def parse_split_values(self, value) -> tuple[str, str, str]:
        """Parses the raw 'split' string into its three components.

        Once the split is parsed, Python can use it to evaluate.

        Parameters
        ----------
        value: str
            The raw 'split' string

        Returns
        -------
            tuple[str, str, str]
            The variable on which the split is done,
            The direction of the split (< or 'in')
            The value on which to split

        """
        if isinstance(value, (tuple, pl.Series)):  # pragma: no cover
            value = value[0]
        if self.nospaces:  # pragma: no cover
            variable, sign, *splitvalue = value.split(" ")
            if sign not in {">", "<", "in", "is", "=="}:  # pragma: no cover
                self.nospaces = False
                variable, sign, *splitvalue = self.parse_split_values_with_spaces(value)
        else:  # pragma: no cover
            variable, sign, *splitvalue = self.parse_split_values_with_spaces(value)

        if len(splitvalue) == 1 and isinstance(splitvalue, list):
            splitvalue = splitvalue[0]

        if sign in {"<", ">", "=="} or splitvalue == "Missing":
            splitvalue = {splitvalue}
        else:
            splitvalue = "".join(splitvalue[1:-1])
            splitvalue = set(splitvalue.split(","))
        return variable, sign, splitvalue

    @staticmethod
    def parse_split_values_with_spaces(
        value,
    ) -> tuple[str, str, str]:  # pragma: no cover
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
        variable = variable.strip()
        return variable, sign, splitvalue

    def get_predictors(self) -> dict | None:
        """Extract predictor names and types from model metadata.

        Tries to find predictor metadata from the ``configuration``
        section of the JSON.  Models exported via the Prediction Studio
        "Save Model" button include a ``configuration`` key with an
        explicit predictor list.  However, models exported in the newer
        format (e.g. via automated pipelines or newer Pega versions) may
        omit the ``configuration`` section entirely, containing only
        ``type``, ``modelVersion``, ``algorithm``, ``trainingStats``,
        ``auc``, etc. at the top level.  In that case, predictor names
        and types are inferred from the tree split nodes instead.
        """
        self.nospaces = True
        try:
            predictors = self._properties["configuration"]["predictors"]
        except Exception:  # pragma: no cover
            try:
                predictors = self._properties["predictors"]
            except Exception:
                try:
                    predictors = []
                    for i in self._properties.split("=")[4].split(
                        "com.pega.decision.adm.client.PredictorInfo: ",
                    ):
                        if i.startswith("{"):
                            if i.endswith("ihSummaryPredictors"):
                                predictors += [i.split("], ihSummaryPredictors")[0]]
                            else:
                                predictors += [i[:-2]]

                except Exception:
                    # No explicit predictor metadata available — infer
                    # from tree splits (see docstring above).
                    return self._infer_predictors_from_splits()
        predictors_dict = {}
        for predictor in predictors:
            if isinstance(predictor, str):  # pragma: no cover
                predictor = json.loads(predictor)
            predictors_dict[predictor["name"]] = predictor["type"]
        return predictors_dict

    def _infer_predictors_from_splits(self) -> dict | None:
        """Infer predictor names and types from tree split nodes.

        When no explicit predictor metadata is available (e.g. in exported
        models without a ``configuration`` section), we walk the trees and
        derive predictor names from splits. The operator determines the
        type: ``<`` → numeric, ``in``/``==`` → symbolic.

        Returns
        -------
        dict[str, str] | None
            Mapping of predictor name → type ("numeric" or "symbolic").

        """
        var_ops: dict[str, set[str]] = collections.defaultdict(set)

        def _collect(tree: dict):
            if "split" in tree:
                parts = tree["split"].split(" ", 2)
                if len(parts) >= 2:
                    var_ops[parts[0]].add(parts[1])
            if "left" in tree:
                _collect(tree["left"])
            if "right" in tree:
                _collect(tree["right"])

        for tree in self.model:
            _collect(tree)

        if not var_ops:
            logger.info("No splits found, cannot infer predictors.")
            return None

        result: dict[str, str] = {}
        for name, ops in var_ops.items():
            if "<" in ops:
                result[name] = "numeric"
            elif "in" in ops or "==" in ops:
                result[name] = "symbolic"
            else:
                result[name] = "symbolic"

        logger.info(f"Inferred {len(result)} predictors from tree splits.")
        return result

    @lru_cache
    def get_gains_per_split(
        self,
    ) -> tuple[
        dict,
        dict,
        pl.DataFrame,
    ]:
        """Function to compute the gains of each split in each tree."""
        self.predictors

        splitsPerTree = {
            treeID: self.get_splits_recursively(tree=tree, splits=[], gains=[])[0]
            for treeID, tree in enumerate(self.model)
        }
        gainsPerTree = {
            treeID: self.get_splits_recursively(tree=tree, splits=[], gains=[])[1]
            for treeID, tree in enumerate(self.model)
        }
        splitlist = [value for value in splitsPerTree.values() if value != []]
        gainslist = [value for value in gainsPerTree.values() if value != []]
        total_split_list: list = functools.reduce(operator.iconcat, splitlist, [])
        total_gains_list: list = functools.reduce(operator.iconcat, gainslist, [])
        gainsPerSplit = pl.DataFrame(
            list(zip(total_split_list, total_gains_list)),
            schema=["split", "gains"],
            orient="row",
        )
        gainsPerSplit = gainsPerSplit.with_columns(
            predictor=pl.col("split").map_elements(
                lambda x: self.parse_split_values(x)[0],
                return_dtype=pl.Utf8,
            ),
        )
        return splitsPerTree, gainsPerTree, gainsPerSplit

    def get_grouped_gains_per_split(self) -> pl.DataFrame:
        """Function to get the gains per split, grouped by split.

        It adds some additional information, such as the possible values,
        the mean gains, and the number of times the split is performed.
        """
        return (
            self.gains_per_split.group_by("split", maintain_order=True)
            .agg(
                [
                    pl.first("predictor"),
                    pl.col("gains").implode(),
                    pl.col("gains").mean().alias("mean"),
                    pl.first("split")
                    .map_elements(
                        lambda x: self.parse_split_values(x)[1],
                        return_dtype=pl.Utf8,
                    )
                    .alias("sign"),
                    pl.first("split")
                    .map_elements(
                        lambda x: self.parse_split_values(x)[2],
                        return_dtype=pl.Object,
                    )
                    .alias("values"),
                ],
            )
            .with_columns(n=pl.col("gains").list.len())
        )

    def get_splits_recursively(
        self,
        tree: dict,
        splits: list,
        gains: list,
    ) -> tuple[list, list]:
        """Recursively finds splits and their gains for each node.

        By Python's mutatable list mechanic, the easiest way to achieve
        this is to explicitly supply the function with empty lists.
        Therefore, the 'splits' and 'gains' parameter expect
        empty lists when initially called.

        Parameters
        ----------
        tree: dict
        splits: list
        gains: list

        Returns
        -------
            tuple[list, list]
            Each split, and its corresponding gain

        """
        for key, value in tree.items():
            if key == "gain":
                if value > 0:
                    gains.append(value)
            if key == "split":
                splits.append(value)
            if key in {"left", "right"}:
                self.get_splits_recursively(
                    tree=dict(value),
                    splits=splits,
                    gains=gains,
                )
        return splits, gains

    def plot_splits_per_variable(self, subset: set | None = None, show=True):
        """Plots the splits for each variable in the tree.

        Parameters
        ----------
        subset: Optional[set]
            Optional parameter to subset the variables to plot
        show: bool
            Whether to display each plot

        Returns
        -------
        plt.figure

        """
        try:
            import plotly.graph_objects as go  # type: ignore[import-untyped]
            from plotly.subplots import make_subplots  # type: ignore[import-untyped]
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB")
        figlist = []
        for (name,), data in self.gains_per_split.group_by("predictor"):
            if (subset is not None and name in subset) or subset is None:
                fig = make_subplots()
                fig.add_trace(
                    go.Box(
                        x=data.get_column("split"),
                        y=data.get_column("gains"),
                        name="Gain",
                    ),
                )
                fig.add_trace(
                    go.Scatter(
                        x=self.grouped_gains_per_split.filter(
                            pl.col("predictor") == name,
                        )
                        .select("split")
                        .to_series()
                        .to_list(),
                        y=self.grouped_gains_per_split.filter(
                            pl.col("predictor") == name,
                        )
                        .select("n")
                        .to_series()
                        .to_list(),
                        name="Number of splits",
                        mode="lines+markers",
                    ),
                )
                # fig.update_xaxes(
                #     categoryorder="array",
                #     categoryarray=self.groupedGainsPerSplit.filter(
                #         pl.col("predictor") == name
                #     )
                #     .select("split")
                #     .to_series()
                #     .to_list(),
                # )

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
        """Generate a dataframe with useful stats for each tree"""
        stats: dict[str, list] = {
            k: [] for k in ["treeID", "score", "depth", "nsplits", "gains", "meangains"]
        }
        for treeID, tree in enumerate(self.model):
            stats["treeID"].append(treeID)
            stats["score"].append(tree["score"])
            stats["depth"].append(self._depth(tree) - 1)
            splits, gains = self.get_splits_recursively(tree, splits=[], gains=[])
            stats["nsplits"].append(len(splits))
            stats["gains"].append(gains)
            meangains = mean(gains) if len(gains) > 0 else 0
            stats["meangains"].append(meangains)

        return pl.from_dict(stats)

    def get_all_values_per_split(self) -> dict:
        """Generate a dictionary with the possible values for each split"""
        splitvalues: dict = {}
        for (name,), group in self.grouped_gains_per_split.group_by("predictor"):
            if name not in splitvalues.keys():
                splitvalues[name] = set()
            splitvalue = group.get_column("values").to_list()
            try:
                for i in splitvalue:
                    splitvalues[name] = splitvalues[name].union(i)
            except Exception as e:  # pragma: no cover
                print(e)
        return splitvalues

    def get_nodes_recursively(
        self,
        tree: dict,
        nodelist: dict,
        counter: list,
        childs: dict,
    ) -> tuple[dict, dict]:
        """Recursively walks through each node, used for tree representation.

        Again, nodelist, counter and childs expects
        empty dict, dict and list parameters.

        Parameters
        ----------
        tree: dict
        nodelist: dict
        counter: dict
        childs: list

        Returns
        -------
            tuple[dict, list]
            The dictionary of nodes and the list of child nodes

        """
        checked = False

        for key, value in tree.items():
            if key in {"left", "right"}:
                nodelist[len(counter) + 1], _ = self.get_nodes_recursively(
                    value,
                    nodelist,
                    counter,
                    childs,
                )

            else:
                nodelist[len(counter) + 1] = {}

                if key == "score":
                    counter.append(len(counter) + 1)

                nodelist[len(counter)][key] = value

            if key == "split":
                childs[len(counter)] = {"left": 0, "right": 0}

            if not checked:
                for node, children in reversed(list(childs.items())):
                    if children["left"] == 0:
                        childs[node]["left"] = len(counter)
                        break
                    if children["right"] == 0:
                        childs[node]["right"] = len(counter)
                        break
                if len(counter) > 1:
                    nodelist[len(counter)]["parent_node"] = node
                checked = True

        return nodelist, childs

    @staticmethod
    def _fill_child_node_ids(nodeinfo: dict, childs: dict) -> dict:
        """Utility function to add child info to nodes"""
        for ID, children in childs.items():
            nodeinfo[ID]["left_child"] = children["left"]
            nodeinfo[ID]["right_child"] = children["right"]
        return nodeinfo

    def get_tree_representation(self, tree_number: int) -> dict:
        """Generates a more usable tree representation.

        In this tree representation, each node has an ID,
        and its attributes are the attributes,
        with parent and child nodes added as well.

        Parameters
        ----------
        tree_number: int
            The number of the tree, in order of the original json

        returns: dict

        """
        tree = self.model[tree_number]
        nodeinfo, childs = self.get_nodes_recursively(
            tree,
            nodelist={},
            childs={},
            counter=[],
        )
        tree = self._fill_child_node_ids(nodeinfo, childs)
        # This is a very crude fix
        # It seems to add the entire tree as the last key value
        # Will work on a better fix in the future
        # For now, removing the last key seems the easiest solution
        del tree[list(tree.keys())[-1]]
        return tree

    def plot_tree(
        self,
        tree_number: int,
        highlighted: dict | list | None = None,
        show=True,
    ) -> pydot.Graph:
        """Plots the chosen decision tree.

        Parameters
        ----------
        tree_number: int
            The number of the tree to visualise
        highlighted: Optional[dict, list]
            Optional parameter to highlight nodes in green
            If a dictionary, it expects an 'x': i.e., features
            with their corresponding values.
            If a list, expects a list of node IDs for that tree.

        Returns
        -------
        pydot.Graph

        """
        try:
            import pydot
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["pydot"], "AGB")
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
                split = node["split"]
                variable, sign, values = self.parse_split_values(split)
                if sign == "in":
                    if len(values) <= 3:
                        labelname = values
                    else:
                        totallen = len(self.all_values_per_split[variable])
                        labelname = (
                            f"{list(values)[0:2] + ['...']} ({len(values)}/{totallen})"
                        )
                    label += f"\nSplit: {variable} in {labelname}\nGain: {node['gain']}"

                else:
                    label += f"\nSplit: {node['split']}\nGain: {node['gain']}"

                graph.add_node(
                    pydot.Node(
                        name=key,
                        label=label,
                        shape="box",
                        style="filled",
                        fillcolor=color,
                    ),
                )
            else:
                graph.add_node(
                    pydot.Node(
                        name=key,
                        label=label,
                        shape="ellipse",
                        style="filled",
                        fillcolor=color,
                    ),
                )
            if "parent_node" in node:
                graph.add_edge(pydot.Edge(key, node["parent_node"]))

        if show:  # pragma: no cover
            try:
                from IPython.display import Image, display
            except ImportError:
                raise ValueError(
                    "IPython not installed, please install it using `pip install IPython`.",
                )
            try:
                display(Image(graph.create_png()))  # pragma: no cover
            except FileNotFoundError as e:
                print(
                    "Dot/Graphviz not installed. Please install it to your machine.",
                    e,
                )
        return graph

    def get_visited_nodes(
        self,
        treeID: int,
        x: dict,
        save_all: bool = False,
    ) -> tuple[list, float, list]:
        """Finds all visited nodes for a given tree, given an x

        Parameters
        ----------
        treeID: int
            The ID of the tree
        x: dict
            Features to split on, with their values
        save_all: bool, default = False
            Whether to save all gains for each individual split

        Returns
        -------
        list, float, list
            The list of visited nodes,
            The score of the final leaf node,
            The gains for each split in the visited nodes

        """
        tree = self.get_tree_representation(treeID)
        current_node_id = 1
        leaf = False
        visited = []
        scores = []
        while not leaf:
            visited += [current_node_id]
            current_node = tree[current_node_id]
            if "split" in current_node:
                variable, type, split = self.parse_split_values(current_node["split"])
                splitvalue = f"'{x[variable]}'" if type in {"in", "is"} else x[variable]
                type = "in" if type == "is" else type
                if save_all:
                    scores += [{current_node["split"]: current_node["gain"]}]
                if type in {"<", ">"} and isinstance(split, set):
                    split = float(split.pop())
                if self._safe_condition_evaluate(splitvalue, type, split):
                    current_node_id = current_node["left_child"]
                else:
                    current_node_id = current_node["right_child"]
            else:
                leaf = True
        return visited, current_node["score"], scores

    def get_all_visited_nodes(self, x: dict) -> pl.DataFrame:
        """Loops through each tree, and records the scoring info

        Parameters
        ----------
        x: dict
            Features to split on, with their values

        Returns
        -------
            pl.DataFrame

        """
        tree_ids, visited_nodes, score, splits = [], [], [], []
        for treeID in range(len(self.model)):
            tree_ids.append(treeID)
            visits = self.get_visited_nodes(treeID, x, save_all=True)
            visited_nodes.append(visits[0])
            score.append(visits[1])
            splits.append(visits[2])
        df = pl.DataFrame(
            [tree_ids, visited_nodes, score, [str(path) for path in splits]],
            schema=["treeID", "visited_nodes", "score", "splits"],
        )
        return df

    def score(self, x: dict) -> float:
        """Computes the score for a given x"""
        score = self.get_all_visited_nodes(x)["score"].sum()
        return 1 / (1 + exp(-score))

    def plot_contribution_per_tree(self, x: dict, show=True):
        """Plots the contribution of each tree towards the final propensity."""
        try:
            import plotly.express as px  # type: ignore[import-untyped]
            import plotly.graph_objects as go
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB")

        # Sort by treeID and add row index for plotting order
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
        fig.add_trace(
            go.Scatter(x=scores["row_idx"], y=scores["mean"], name="Cumulative mean"),
        )
        fig.add_trace(
            go.Scatter(x=scores["row_idx"], y=scores["propensity"], name="Propensity"),
        )
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

    def predictor_categorization(self, x: str, context_keys=None):
        context_keys = context_keys if context_keys is not None else self.context_keys
        if context_keys is None:
            context_keys = set()  # pragma: no cover
        if len(x.split(".")) > 1:
            return x.split(".")[0]
        if x in context_keys:
            return x
        return "Primary"  # pragma: no cover

    def compute_categorization_over_time(
        self,
        predictorCategorization=None,
        context_keys=None,
    ):
        context_keys = context_keys if context_keys is not None else self.context_keys
        predictorCategorization = (
            predictorCategorization
            if predictorCategorization is not None
            else self.predictor_categorization
        )
        splitsPerTree = list()
        for splits in self.splits_per_tree.values():
            counter = collections.Counter()
            for split in splits:
                counter.update(
                    [
                        predictorCategorization(
                            self.parse_split_values(split)[0],
                            context_keys,
                        ),
                    ],
                )
            splitsPerTree.append(counter)
        return splitsPerTree, self.tree_stats.select(
            "score",
        ).to_series().abs().to_list()

    def plot_splits_per_variable_type(self, predictor_categorization=None, **kwargs):
        try:
            import plotly.express as px
        except ImportError:  # pragma: no cover
            raise MissingDependenciesException(["plotly"], "AGB")
        if predictor_categorization is not None:  # pragma: no cover
            to_plot = self.compute_categorization_over_time(predictor_categorization)[0]
        else:
            to_plot = self.splits_per_variable_type[0]
        # sort column names
        df = pl.DataFrame(to_plot).select(sorted(pl.col("*")))

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
                buttons=list(
                    [
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
                ),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.3,
                yanchor="top",
            ),
        )
        return fig


@dataclass
class MultiTrees:  # pragma: no cover
    trees: dict
    model_name: str | None = None
    context_keys: list | None = None

    def __repr__(self):
        mod = "" if self.model_name is None else f" for {self.model_name}"
        return repr(
            f"MultiTree object{mod}, with {len(self)} trees ranging from {list(self.trees.keys())[0]} to {list(self.trees.keys())[-1]}",
        )

    def __getitem__(self, index):
        if isinstance(index, int):
            return list(self.trees.items())[index]
        if isinstance(index, pl.datetime):
            return self.trees[index]

    def __len__(self):
        return len(self.trees)

    def __add__(self, other):
        if isinstance(other, MultiTrees):
            return MultiTrees({**self.trees, **other.trees})
        if isinstance(other, ADMTreesModel):
            return MultiTrees(
                {
                    **self.trees,
                    pl.datetime(other.metrics["factory_update_time"]): other,
                },
            )

    @property
    def first(self):
        return self[0]

    @property
    def last(self):
        return self[-1]

    def compute_over_time(self, predictor_categorization=None):
        outdf = []
        for timestamp, tree in self.trees.items():
            to_plot = tree.splitsPerVariableType[0]
            if predictor_categorization is not None:
                to_plot = tree.computeCategorizationOverTime(predictor_categorization)[
                    0
                ]
            outdf.append(
                pl.DataFrame(to_plot).with_columns(
                    SnapshotTime=pl.lit(timestamp).str.to_date(format="%Y-%m-%d %X"),
                ),
            )

        return pl.concat(outdf, how="diagonal")
