"""Interaction History analysis for Pega CDH."""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar, Literal

import polars as pl

from ..pega_io.File import read_ds_export
from ..utils import cdh_utils
from ..utils.cdh_utils import (
    _apply_query,
    _polars_capitalize,
    parse_pega_date_time_formats,
)
from ..utils.pega_outcomes import resolve_outcome_labels as _resolve_outcome_labels
from . import Schema
from .Aggregates import Aggregates
from .Plots import Plots
from .Schema import REQUIRED_IH_COLUMNS

if TYPE_CHECKING:
    import datetime
    import os

    from ..utils.types import QUERY

logger = logging.getLogger(__name__)


class IH:
    """Analyze Interaction History data from Pega CDH.

    The IH class provides analysis and visualization capabilities for
    customer interaction data from Pega's Customer Decision Hub. It supports
    engagement, conversion, and open rate metrics through customizable
    outcome label mappings.

    See Also
    --------
    pdstools.adm.ADMDatamart : For ADM model analysis.
    pdstools.impactanalyzer.ImpactAnalyzer : For Impact Analyzer experiments.

    Examples
    --------
    >>> from pdstools import IH
    >>> ih = IH.from_ds_export("interaction_history.zip")
    >>> ih.aggregates.summary_by_channel().collect()
    >>> ih.plot.response_count_trend()

    """

    data: pl.LazyFrame
    """The underlying interaction history data."""

    outcome_labels_used: dict | None
    aggregates: Aggregates
    """Aggregation methods accessor."""

    plot: Plots
    """Plot accessor for visualization methods."""

    positive_outcome_labels: ClassVar[dict[str, list[str]]] = {
        "Engagement": ["Accepted", "Accept", "Clicked", "Click"],
        "Conversion": ["Conversion"],
        "OpenRate": ["Opened", "Open"],
    }
    """Mapping of metric types to positive outcome labels."""

    negative_outcome_labels: ClassVar[dict[str, list[str]]] = {
        "Engagement": ["Impression", "Impressed", "Pending", "NoResponse"],
        "Conversion": ["Impression", "Pending"],
        "OpenRate": ["Impression", "Pending"],
    }
    """Mapping of metric types to negative outcome labels."""

    def __init__(self, data: pl.LazyFrame):
        """Initialize an IH instance.

        Parameters
        ----------
        data : pl.LazyFrame
            Interaction history data. Column names will be automatically
            capitalized using Pega naming conventions.

        Raises
        ------
        ValueError
            If any column listed in
            :data:`pdstools.ih.Schema.REQUIRED_IH_COLUMNS` is absent.

        Notes
        -----
        Use the class methods :meth:`from_ds_export` or :meth:`from_mock_data`
        to create instances from data sources.

        """
        self.data = self._validate_ih_data(data)
        self.outcome_labels_used = self._scan_outcome_labels()
        self.aggregates = Aggregates(ih=self)
        self.plot = Plots(ih=self)

    @staticmethod
    def _validate_ih_data(df: pl.LazyFrame) -> pl.LazyFrame:
        """Internal method to validate and normalise IH data.

        Mirrors the ``_validate_*_data`` pattern used by
        :class:`pdstools.adm.ADMDatamart`. Capitalises Pega-style column
        names (stripping ``py``/``px`` prefixes), checks required columns
        are present, and casts columns to the dtypes declared on
        :class:`pdstools.ih.Schema.IHInteraction` (which auto-parses
        string ``OutcomeTime`` values into ``pl.Datetime``).

        Parameters
        ----------
        df : pl.LazyFrame
            Raw interaction history data.

        Returns
        -------
        pl.LazyFrame
            Capitalised and dtype-normalised frame.

        Raises
        ------
        ValueError
            If any column listed in
            :data:`pdstools.ih.Schema.REQUIRED_IH_COLUMNS` is absent.
        """
        df = _polars_capitalize(df)
        missing = set(REQUIRED_IH_COLUMNS).difference(df.collect_schema().names())
        if missing:
            raise ValueError(f"Missing required IH columns: {sorted(missing)}")
        return cdh_utils._apply_schema_types(df, Schema.IHInteraction)

    def _scan_outcome_labels(self) -> dict | None:
        """Scan data for channel/outcome combinations and resolve defaults.

        Returns None if the required columns are not present.
        """
        schema_names = self.data.collect_schema().names()
        if not {"Channel", "Direction", "Outcome"}.issubset(schema_names):
            return None
        scan = (
            self.data.select("Channel", "Direction", "Outcome")
            .unique()
            .group_by("Channel", "Direction")
            .agg(pl.col("Outcome").sort())
            .collect()
        )
        return _resolve_outcome_labels({f"{row[0]}/{row[1]}": row[2] for row in scan.iter_rows()})

    @classmethod
    def from_ds_export(
        cls,
        ih_filename: os.PathLike | str,
        *,
        query: QUERY | None = None,
    ) -> "IH":
        """Create an IH instance from a Pega Dataset Export.

        Parameters
        ----------
        ih_filename : os.PathLike or str
            Path to the dataset export file (parquet, csv, ndjson, or zip).
        query : QUERY, optional
            Polars expression to filter the data. Default is None.

        Returns
        -------
        IH
            Initialized IH instance.

        Examples
        --------
        >>> ih = IH.from_ds_export("Data-pxStrategyResult_pxInteractionHistory.zip")
        >>> ih.data.collect_schema()

        """
        data_raw = read_ds_export(ih_filename)
        if data_raw is None:
            raise ValueError(f"Could not read file: {ih_filename}")
        data = data_raw.with_columns(
            pxOutcomeTime=parse_pega_date_time_formats("pxOutcomeTime"),
        )
        if query is not None:
            data = _apply_query(data, query=query)

        return IH(data)

    @classmethod
    def from_s3(
        cls,
        bucket: str,
        key: str,
        *,
        region: str | None = None,
        boto3_client=None,
        query: QUERY | None = None,
    ) -> "IH":
        """Create an IH instance from a single object stored in S3.

        Downloads the interaction-history export from the given S3
        bucket to a temporary directory, then delegates to
        :meth:`from_ds_export` for parsing.

        Parameters
        ----------
        bucket : str
            Name of the S3 bucket holding the export file.
        key : str
            S3 object key for the interaction-history export file.
        region : str or None, optional
            AWS region name. Ignored if ``boto3_client`` is provided.
        boto3_client : optional
            Pre-configured ``boto3`` S3 client. Use this to inject custom
            credentials, endpoints, or sessions. When omitted, a default
            client is created via ``boto3.client("s3", region_name=region)``.
        query : QUERY, optional
            Polars expression to filter the data. Default is None.

        Returns
        -------
        IH
            Initialized IH instance.

        Examples
        --------
        >>> from pdstools import IH
        >>> ih = IH.from_s3(
        ...     bucket="my-pega-exports",
        ...     key="ih/Data-pxStrategyResult_pxInteractionHistory.zip",
        ... )

        Note
        ----
        ``boto3`` is an optional dependency; install the ``pega_io`` extra
        (or install ``boto3`` directly) before calling this method.

        See Also
        --------
        IH.from_ds_export : Underlying parser for downloaded files.

        """
        if boto3_client is None:
            try:
                import boto3
            except ImportError as err:
                from ..utils.namespaces import MissingDependenciesException

                raise MissingDependenciesException(
                    ["boto3"],
                    namespace="IH.from_s3",
                    deps_group="pega_io",
                ) from err
            boto3_client = boto3.client("s3", region_name=region)

        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            basename = os.path.basename(key) or key.replace("/", "_")
            local_path = os.path.join(tmp_dir, basename)
            boto3_client.download_file(bucket, key, local_path)
            return cls.from_ds_export(local_path, query=query)

    @classmethod
    def from_mock_data(
        cls,
        days: int = 182,
        n: int = 100000,
        *,
        seed: int | None = 42,
        end: datetime.date | Literal["canonical"] | None = None,
        scale: float = 1.0,
    ) -> IH:
        """Create an IH instance from the shared synthetic sample world.

        Samples ``n`` decisions from the same world as
        :meth:`pdstools.adm.ADMDatamart.from_mock_data` and
        :meth:`pdstools.prediction.Prediction.from_mock_data`, so engagement
        per channel matches the model and prediction data. Web and Mobile
        (inbound) record Impression and Clicked; Email and Call Center
        (outbound) record Pending and Accepted, with Email Opened in between.
        Every offered action is an exposure carrying its Impact Analyzer
        control group (``MktValue``, ``MktType``), model-control cohort
        (``ModelControlGroup``), propensities and accepted ``Value``. A
        separate Conversion-Test/Control experiment adds Conversion outcomes.

        Parameters
        ----------
        days : int, default 182
            Number of days in the window.
        n : int, default 100000
            Number of interactions (decisions) to sample.
        seed : int or None, default 42
            Seed for reproducible data; ``None`` draws a random seed.
        end : date, "canonical" or None, default None
            Last day of the window, resolved to the Sunday on or before it.
            ``None`` uses the most recent completed Sunday, ``"canonical"``
            pins Sunday 2000-07-02.
        scale : float, default 1.0
            Multiplier on world response volumes.

        Returns
        -------
        IH
            IH instance with one row per outcome.

        Examples
        --------
        >>> ih = IH.from_mock_data(n=10000, end="canonical")
        >>> ih.aggregates.summary_by_channel().collect()

        """
        from ..sample_data._ih import interaction_history, outcome_labels
        from ..sample_data._world import sample_world

        world = sample_world(seed=seed, days=days, end=end, scale=scale)
        ih = cls(interaction_history(world, int(n)))
        ih.outcome_labels_used = outcome_labels(world)
        return ih

    def get_sequences(
        self,
        positive_outcome_label: str,
        level: str,
        outcome_column: str,
        customerid_column: str,
    ) -> tuple[
        list[tuple[str, ...]],
        list[tuple[int, ...]],
        list[defaultdict],
        list[defaultdict],
    ]:
        """Extract customer action sequences for PMI analysis.

        Processes customer interaction data to produce action sequences,
        outcome labels, and frequency counts needed for Pointwise Mutual
        Information (PMI) calculations.

        Parameters
        ----------
        positive_outcome_label : str
            Outcome label marking the target event (e.g., "Conversion").
        level : str
            Column name containing the action/offer/treatment.
        outcome_column : str
            Column name containing the outcome label.
        customerid_column : str
            Column name identifying unique customers.

        Returns
        -------
        customer_sequences : list[tuple[str, ...]]
            Action sequences per customer.
        customer_outcomes : list[tuple[int, ...]]
            Binary outcomes (1=positive, 0=other) per sequence position.
        count_actions : list[defaultdict]
            Action frequency counts:
            - [0]: First element counts in bigrams
            - [1]: Second element counts in bigrams
        count_sequences : list[defaultdict]
            Sequence frequency counts:
            - [0]: All bigrams
            - [1]: ≥3-grams ending with positive outcome
            - [2]: Bigrams ending with positive outcome
            - [3]: Unique n-grams per customer

        See Also
        --------
        calculate_pmi : Compute PMI scores from sequence counts.
        pmi_overview : Generate PMI analysis summary.

        """
        cols = [customerid_column, level, outcome_column]

        df = (
            self.data.select(*cols, "OutcomeTime").sort(customerid_column, "OutcomeTime", maintain_order=True).collect()
        )

        count_actions: list[defaultdict[tuple[str, ...], int]] = [defaultdict(int), defaultdict(int)]
        count_sequences: list[defaultdict[tuple[str, ...], int]] = [
            defaultdict(int),
            defaultdict(int),
            defaultdict(int),
            defaultdict(int),
        ]
        customer_sequences = []
        customer_outcomes = []

        # Iterate over customers
        for _user_id, user_df in df.group_by(customerid_column):
            user_actions = user_df[level].to_list()
            outcome_actions = user_df[outcome_column].to_list()

            outcome_actions = [1 if action == positive_outcome_label else 0 for action in outcome_actions]

            if len(user_actions) < 2:
                continue

            if 1 not in outcome_actions:
                continue

            customer_sequences.append(tuple(user_actions))
            customer_outcomes.append(tuple(outcome_actions))

        def accumulate_counts(sequences, outcomes):
            # Iterate over positive end positions j only, and use the
            # closed-form multiplicity of each (bigram, ending j) pair to
            # increment the counters directly — no intermediate ngram lists.
            # Equivalence to the original triple-list implementation is
            # locked in by python/tests/ih/test_IH_get_sequences.py.
            for seq, out in zip(sequences, outcomes, strict=False):
                ngrams_seen = set()

                positive_positions = [j for j, o in enumerate(out) if o == 1]
                for j in positive_positions:
                    if j < 1:
                        # Need at least a bigram (n >= 2 => start index j-1 >= 0).
                        continue

                    # Ending bigram seq[j-1:j+1] — multiplicity j in bigrams_all,
                    # 1 in bigrams (n=2 only).
                    ending = seq[j - 1 : j + 1]
                    count_sequences[0][ending] += j
                    count_actions[0][(seq[j - 1],)] += j
                    count_actions[1][(seq[j],)] += j
                    count_sequences[2][ending] += 1
                    if ending not in ngrams_seen:
                        count_sequences[3][ending] += 1
                        ngrams_seen.add(ending)

                    # Earlier bigrams seq[k:k+2] for k = 0..j-2 — closed-form
                    # multiplicity (k + 1) in bigrams_all.
                    for k in range(j - 1):
                        bigram = seq[k : k + 2]
                        mult = k + 1
                        count_sequences[0][bigram] += mult
                        count_actions[0][(seq[k],)] += mult
                        count_actions[1][(seq[k + 1],)] += mult

                    # ≥3-grams ending at j: seq[j-n+1:j+1] for n = 3..j+1.
                    for n in range(3, j + 2):
                        ngram = seq[j - n + 1 : j + 1]
                        count_sequences[1][ngram] += 1
                        if ngram not in ngrams_seen:
                            count_sequences[3][ngram] += 1
                            ngrams_seen.add(ngram)

        accumulate_counts(customer_sequences, customer_outcomes)

        return customer_sequences, customer_outcomes, count_actions, count_sequences

    @staticmethod
    def calculate_pmi(
        count_actions: list[defaultdict],
        count_sequences: list[defaultdict],
    ) -> dict[tuple[str, ...], float | dict[str, float | dict[tuple[str, ...], float]]]:
        """Compute PMI scores for action sequences.

        Calculates Pointwise Mutual Information scores for bigrams and
        higher-order n-grams. Higher values indicate more informative
        or surprising action sequences.

        Parameters
        ----------
        count_actions : list[defaultdict]
            Action frequency counts from :meth:`get_sequences`.
        count_sequences : list[defaultdict]
            Sequence frequency counts from :meth:`get_sequences`.

        Returns
        -------
        dict[tuple[str, ...], float | dict]
            PMI scores for sequences:
            - Bigrams: Direct PMI value (float)
            - N-grams (n≥3): dict with 'average_pmi' and 'links' (constituent bigram PMIs)

        See Also
        --------
        get_sequences : Extract sequences for PMI analysis.
        pmi_overview : Generate PMI analysis summary.

        Notes
        -----
        Bigram PMI is calculated as:

        .. math::

            PMI(a, b) = \\log_2 \\frac{P(a, b)}{P(a) \\cdot P(b)}

        N-gram PMI is the average of constituent bigram PMIs.

        """
        # corpus size (number of action tokens)
        corpus = sum(len(k) * v for k, v in count_sequences[1].items()) + sum(
            len(k) * v for k, v in count_sequences[2].items()
        )

        bigrams_pmi: dict[tuple[str, ...], float] = {}

        # bigram PMI
        for bigram, bigram_count in count_sequences[0].items():
            first_count = count_actions[0][(bigram[0],)]
            second_count = count_actions[1][(bigram[1],)]

            pmi = math.log2(
                (bigram_count / corpus) / ((first_count / corpus) * (second_count / corpus)),
            )

            bigrams_pmi[bigram] = pmi

        # n‑gram PMI
        customer_bigrams = {key: bigrams_pmi[key] for key in count_sequences[2] if key in bigrams_pmi}

        ngrams_pmi: dict[tuple[str, ...], float | dict[str, float | dict[tuple[str, ...], float]]] = dict(
            customer_bigrams
        )

        for seq in count_sequences[1].keys():
            links = [seq[i : i + 2] for i in range(len(seq) - 1)]
            values = []
            link_info: dict[tuple[str, ...], float] = {}

            for link in links:
                if link in bigrams_pmi:
                    value = bigrams_pmi[link]
                    values.append(value)
                    link_info[link] = value

            average = sum(values) / len(values) if values else 0

            ngrams_pmi[seq] = {"average_pmi": average, "links": link_info}

        return ngrams_pmi

    @staticmethod
    def pmi_overview(
        ngrams_pmi: dict[tuple[str, ...], float | dict],
        count_sequences: list[defaultdict],
        customer_sequences: list[tuple[str, ...]],
        customer_outcomes: list[tuple[int, ...]],
    ) -> pl.DataFrame:
        """Generate PMI analysis summary DataFrame.

        Creates a summary of action sequences ranked by their significance
        in predicting positive outcomes.

        Parameters
        ----------
        ngrams_pmi : dict[tuple[str, ...], float | dict]
            PMI scores from :meth:`calculate_pmi`.
        count_sequences : list[defaultdict]
            Sequence frequency counts from :meth:`get_sequences`.
        customer_sequences : list[tuple[str, ...]]
            Customer action sequences from :meth:`get_sequences`.
        customer_outcomes : list[tuple[int, ...]]
            Customer outcome sequences from :meth:`get_sequences`.

        Returns
        -------
        pl.DataFrame
            Summary DataFrame with columns:

            - **Sequence**: Action sequence tuple
            - **Length**: Number of actions in sequence
            - **Avg PMI**: Average PMI value
            - **Frequency**: Total occurrence count
            - **Unique freq**: Unique customer count
            - **Score**: PMI × log(Frequency), sorted descending

        See Also
        --------
        get_sequences : Extract sequences for analysis.
        calculate_pmi : Compute PMI scores.

        Examples
        --------
        >>> seqs, outs, actions, counts = ih.get_sequences(
        ...     "Conversion", "pyName", "pyOutcome", "pxInteractionID"
        ... )
        >>> pmi = IH.calculate_pmi(actions, counts)
        >>> IH.pmi_overview(pmi, counts, seqs, outs)

        """
        data: list[dict[str, object]] = []
        freq_all = count_sequences[1] | count_sequences[2]

        for seq, pmi_val in ngrams_pmi.items():
            if isinstance(pmi_val, dict):
                avg_pmi: float = pmi_val["average_pmi"]
            else:
                avg_pmi = pmi_val

            count = freq_all.get(seq, 0)

            if count <= 1:
                continue

            data.append(
                {
                    "Sequence": seq,
                    "Length": len(seq),
                    "Avg PMI": avg_pmi,
                    "Frequency": count,
                    "Unique freq": count_sequences[3][seq],
                    "Score": avg_pmi * math.log(count),
                },
            )

        return (
            pl.DataFrame(data)
            .sort("Score", descending=True)
            .with_columns(pl.col("Score").round(3), pl.col("Avg PMI").round(3))
        )
