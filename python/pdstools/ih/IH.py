"""Interaction History analysis for Pega CDH."""

import datetime
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import polars as pl
import polars.selectors as cs

from ..pega_io.File import read_ds_export
from ..utils.cdh_utils import (
    _apply_query,
    _polars_capitalize,
    parse_pega_date_time_formats,
)
from ..utils.types import QUERY
from .Aggregates import Aggregates
from .Plots import Plots


class IH:
    """Analyze Interaction History data from Pega CDH.

    The IH class provides analysis and visualization capabilities for
    customer interaction data from Pega's Customer Decision Hub. It supports
    engagement, conversion, and open rate metrics through customizable
    outcome label mappings.

    Attributes
    ----------
    data : pl.LazyFrame
        The underlying interaction history data.
    aggregates : Aggregates
        Aggregation methods accessor.
    plot : Plots
        Plot accessor for visualization methods.
    positive_outcome_labels : dict
        Mapping of metric types to positive outcome labels.
    negative_outcome_labels : dict
        Mapping of metric types to negative outcome labels.

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

    positive_outcome_labels: Dict[str, List[str]] = {
        "Engagement": ["Accepted", "Accept", "Clicked", "Click"],
        "Conversion": ["Conversion"],
        "OpenRate": ["Opened", "Open"],
    }
    """Mapping of metric types to positive outcome labels."""

    negative_outcome_labels: Dict[str, List[str]] = {
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

        Notes
        -----
        Use the class methods :meth:`from_ds_export` or :meth:`from_mock_data`
        to create instances from data sources.
        """
        self.data = _polars_capitalize(data)
        self.aggregates = Aggregates(ih=self)
        self.plot = Plots(ih=self)

    @classmethod
    def from_ds_export(
        cls,
        ih_filename: Union[os.PathLike, str],
        query: Optional[QUERY] = None,
    ) -> "IH":
        """Create an IH instance from a Pega Dataset Export.

        Parameters
        ----------
        ih_filename : Union[os.PathLike, str]
            Path to the dataset export file (parquet, csv, ndjson, or zip).
        query : Optional[QUERY], optional
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
        data = read_ds_export(ih_filename).with_columns(
            pxOutcomeTime=parse_pega_date_time_formats("pxOutcomeTime")
        )
        if query is not None:
            data = _apply_query(data, query=query)

        return IH(data)

    @classmethod
    def from_s3(cls) -> "IH":
        """Create an IH instance from S3 data.

        .. note::
            Not implemented yet. Please let us know if you would like this!

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("from_s3 is not yet implemented")

    @classmethod
    def from_mock_data(cls, days: int = 90, n: int = 100000) -> "IH":
        """Create an IH instance with synthetic sample data.

        Generates realistic interaction history data for testing and
        demonstration purposes. Includes inbound (Web) and outbound (Email)
        channels with configurable propensities and model noise.

        Parameters
        ----------
        days : int, default 90
            Number of days of data to generate.
        n : int, default 100000
            Number of interaction records to generate.

        Returns
        -------
        IH
            IH instance with synthetic data.

        Examples
        --------
        >>> ih = IH.from_mock_data(days=30, n=10000)
        >>> ih.data.select("pyChannel").collect().unique()
        """
        n = int(n)

        n_actions = 10
        click_avg_duration_minutes = 2
        accept_avg_duration_minutes = 30
        convert_over_accept_click_rate_test = 0.5
        convert_over_accept_click_rate_control = 0.3
        convert_avg_duration_days = 2
        inbound_base_propensity = 0.02
        outbound_base_propensity = 0.01
        inbound_modelnoise_NaiveBayes = (
            0.2  # relative amount of extra noise added to models
        )
        inbound_modelnoise_GradientBoost = 0.0
        outbound_modelnoise_NaiveBayes = 0.3
        outbound_modelnoise_GradientBoost = 0.1

        now = datetime.datetime.now()

        def thompson_sampler(propensity, responses=10000):
            # Using the same approach as we're doing in NBAD at the moment
            a = responses * propensity
            b = responses * (1 - propensity)

            sampled = random.betavariate(a, b)

            # print(
            #     f"{a} {b} {propensity} {sampled} {abs((sampled-propensity)/propensity)}"
            # )

            return sampled

        # TODO maybe this should be changed in PDS tools - w/o __TimeStamp__ flag
        # def to_prpc_time_str(__TimeStamp__):
        #     return to_prpc_date_time(__TimeStamp__)[0:15]

        ih_fake_impressions = pl.DataFrame(
            {
                "pxInteractionID": [str(int(1e9 + i)) for i in range(n)],
                "pyChannel": random.choices(
                    ["Web", "Email"], k=n
                ),  # Direction will be derived from this later
                "pyIssue": random.choices(
                    ["Acquisition", "Retention", "Risk", "Service"], k=n
                ),
                "pyGroup": random.choices(
                    [
                        "Pension",
                        "Lending",
                        "Mortgages",
                        "Investments",
                        "Insurance",
                        "Savings",
                    ],
                    weights=[1, 1, 2, 2, 3, 3],
                    k=n,
                ),
                "pyName": random.choices(
                    range(1, 1 + n_actions),
                    weights=reversed(range(1, 1 + n_actions)),
                    k=n,
                ),  # nr will be appended to group name to form action name
                "pyTreatment": [
                    random.randint(1, 2) for _ in range(n)
                ],  # nr will be appended to group/channel
                # https://stackoverflow.com/questions/40351791/how-to-hash-strings-into-a-float-in-01
                "ExperimentGroup": [
                    "Conversion-Test",
                    "Conversion-Test",
                    "Conversion-Control",
                    "Conversion-Control",
                ]
                * int(n / 4),
                "pyModelTechnique": [
                    "NaiveBayes",
                    "GradientBoost",
                    "NaiveBayes",
                    "GradientBoost",
                ]
                * int(n / 4),
                "pxOutcomeTime": [
                    (now - datetime.timedelta(days=i * days / n)) for i in range(n)
                ],
                "Temp.ClickDurationMinutes": [
                    random.uniform(0, 2 * click_avg_duration_minutes) for i in range(n)
                ],
                "Temp.AcceptDurationMinutes": [
                    random.uniform(0, 2 * accept_avg_duration_minutes) for i in range(n)
                ],
                "Temp.ConvertDurationDays": [
                    random.uniform(0, 2 * convert_avg_duration_days) for i in range(n)
                ],
                "Temp.RandomUniform": [random.uniform(0, 1) for i in range(n)],
            }
        ).with_columns(
            pyDirection=pl.when(pl.col("pyChannel") == "Web")
            .then(pl.lit("Inbound"))
            .otherwise(pl.lit("Outbound")),
            pyName=pl.format("{}_{}", pl.col("pyGroup"), pl.col("pyName")),
            pyTreatment=pl.format(
                "{}_{}_{}Treatment{}",
                pl.col("pyGroup"),
                pl.col("pyName"),
                pl.col("pyChannel"),
                pl.col("pyTreatment"),
            ),
            pyOutcome=pl.when(pl.col.pyChannel == "Web")
            .then(pl.lit("Impression"))
            .otherwise(pl.lit("Pending")),
        )

        action_basepropensities = (
            ih_fake_impressions.group_by(["pyName"])
            .agg(Count=pl.len())
            .sort("Count", descending=True)
            .with_row_index("Index")
            .with_columns(
                (1.0 / (5 + pl.col("Index"))).alias("Temp.Zipf"),
            )
        )

        ih_fake_impressions = (
            ih_fake_impressions.join(
                action_basepropensities.select(["pyName", "Temp.Zipf"]), on=["pyName"]
            )
            .with_columns(
                pl.col("Temp.Zipf")
                .mean()
                .over(["pyChannel", "pyDirection"])
                .alias("Temp.ZipfMean"),
                pl.when(pl.col("pyDirection") == "Inbound")
                .then(pl.lit(inbound_base_propensity))
                .otherwise(pl.lit(outbound_base_propensity))
                .alias("Temp.ChannelBasePropensity"),
            )
            .with_columns(
                BasePropensity=pl.col("Temp.Zipf")
                * pl.col("Temp.ChannelBasePropensity")
                / pl.col("Temp.ZipfMean")
            )
            .with_columns(
                pyPropensity=pl.col("BasePropensity").map_elements(
                    thompson_sampler, pl.Float64
                )
            )
        )

        # Add artificial noise to the models to manipulate some scenarios
        ih_fake_impressions = ih_fake_impressions.with_columns(
            pl.when(
                (pl.col.pyModelTechnique == "NaiveBayes")
                & (pl.col.pyDirection == "Inbound")
            )
            .then(pl.col("Temp.ChannelBasePropensity") * inbound_modelnoise_NaiveBayes)
            .when(
                (pl.col.pyModelTechnique == "GradientBoost")
                & (pl.col.pyDirection == "Inbound")
            )
            .then(
                pl.col("Temp.ChannelBasePropensity") * inbound_modelnoise_GradientBoost
            )
            .when(
                (pl.col.pyModelTechnique == "NaiveBayes")
                & (pl.col.pyDirection == "Outbound")
            )
            .then(pl.col("Temp.ChannelBasePropensity") * outbound_modelnoise_NaiveBayes)
            .when(
                (pl.col.pyModelTechnique == "GradientBoost")
                & (pl.col.pyDirection == "Outbound")
            )
            .then(
                pl.col("Temp.ChannelBasePropensity") * outbound_modelnoise_GradientBoost
            )
            .otherwise(pl.lit(0.0))
            .alias("Temp.ExtraModelNoise")
        )

        ih_fake_clicks = (
            ih_fake_impressions.filter(pl.col.pyDirection == "Inbound")
            .filter(
                pl.col("Temp.RandomUniform")
                < (pl.col("pyPropensity") + pl.col("Temp.ExtraModelNoise"))
            )
            .with_columns(
                pxOutcomeTime=pl.col.pxOutcomeTime
                + pl.duration(minutes=pl.col("Temp.ClickDurationMinutes")),
                pyOutcome=pl.lit("Clicked"),
            )
        )
        ih_fake_accepts = (
            ih_fake_impressions.filter(pl.col.pyDirection == "Outbound")
            .filter(
                pl.col("Temp.RandomUniform")
                < (pl.col("pyPropensity") + pl.col("Temp.ExtraModelNoise"))
            )
            .with_columns(
                pxOutcomeTime=pl.col.pxOutcomeTime
                + pl.duration(minutes=pl.col("Temp.AcceptDurationMinutes")),
                pyOutcome=pl.lit("Accepted"),
            )
        )

        def create_fake_converts(df, group, fraction):
            return (
                df.filter(pl.col("ExperimentGroup") == group)
                .sample(fraction=fraction)
                .with_columns(
                    pxOutcomeTime=pl.col("pxOutcomeTime")
                    + pl.duration(days=pl.col("Temp.ConvertDurationDays")),
                    pyOutcome=pl.lit("Conversion"),
                )
            )

        ih_data = (
            pl.concat(
                [
                    ih_fake_impressions,
                    ih_fake_clicks,
                    ih_fake_accepts,
                    create_fake_converts(
                        ih_fake_clicks,
                        "Conversion-Test",
                        convert_over_accept_click_rate_test,
                    ),
                    create_fake_converts(
                        ih_fake_accepts,
                        "Conversion-Test",
                        convert_over_accept_click_rate_test,
                    ),
                    create_fake_converts(
                        ih_fake_clicks,
                        "Conversion-Control",
                        convert_over_accept_click_rate_control,
                    ),
                    create_fake_converts(
                        ih_fake_accepts,
                        "Conversion-Control",
                        convert_over_accept_click_rate_control,
                    ),
                ]
            )
            .filter(pl.col("pxOutcomeTime") <= pl.lit(now))
            .drop(cs.starts_with("Temp."))
            .sort("pxInteractionID", "pxOutcomeTime")
        )

        return IH(ih_data.lazy())

    def get_sequences(
        self,
        positive_outcome_label: str,
        level: str,
        outcome_column: str,
        customerid_column: str,
    ) -> Tuple[
        List[Tuple[str, ...]],
        List[Tuple[int, ...]],
        List[defaultdict],
        List[defaultdict],
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
        customer_sequences : List[Tuple[str, ...]]
            Action sequences per customer.
        customer_outcomes : List[Tuple[int, ...]]
            Binary outcomes (1=positive, 0=other) per sequence position.
        count_actions : List[defaultdict]
            Action frequency counts:
            - [0]: First element counts in bigrams
            - [1]: Second element counts in bigrams
        count_sequences : List[defaultdict]
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

        df = self.data.select(cols).sort([customerid_column]).collect()

        count_actions = [defaultdict(int), defaultdict(int)]
        count_sequences = [
            defaultdict(int),
            defaultdict(int),
            defaultdict(int),
            defaultdict(int),
        ]
        customer_sequences = []
        customer_outcomes = []

        # Iterate over customers
        for user_id, user_df in df.group_by(customerid_column):
            user_actions = user_df[level].to_list()
            outcome_actions = user_df[outcome_column].to_list()

            outcome_actions = [
                1 if action == positive_outcome_label else 0
                for action in outcome_actions
            ]

            if len(user_actions) < 2:
                continue

            if 1 not in outcome_actions:
                continue

            customer_sequences.append(tuple(user_actions))
            customer_outcomes.append(tuple(outcome_actions))

        def ngrams_and_bigrams(sequences, outcomes):
            ngrams = []
            bigrams = []
            bigrams_all = []

            for seq, out in zip(sequences, outcomes):
                ngrams_seen = set()

                for n in range(2, len(seq) + 1):
                    for i in range(len(seq) - n + 1):
                        ngram = seq[i : i + n]
                        ngram_outcomes = out[i : i + n]

                        if ngram_outcomes[-1] == 1:  # Ends in positive_outcome_label
                            if len(ngram) == 2:
                                bigrams_all.append(ngram)
                                bigrams.append(ngram)
                            else:
                                ngrams.append(ngram)
                                for j in range(len(ngram) - 1):
                                    bigrams_all.append(ngram[j : j + 2])

                            if ngram not in ngrams_seen:
                                count_sequences[3][ngram] += 1
                                ngrams_seen.add(ngram)

            return ngrams, bigrams, bigrams_all

        ngrams, bigrams, bigrams_all = ngrams_and_bigrams(
            customer_sequences, customer_outcomes
        )

        # Frequency tables
        for seq in ngrams:
            count_sequences[1][seq] += 1

        for bigram in bigrams_all:
            count_sequences[0][bigram] += 1
            count_actions[0][(bigram[0],)] += 1
            count_actions[1][(bigram[1],)] += 1

        for bigram in bigrams:
            count_sequences[2][bigram] += 1

        return customer_sequences, customer_outcomes, count_actions, count_sequences

    @staticmethod
    def calculate_pmi(
        count_actions: List[defaultdict],
        count_sequences: List[defaultdict],
    ) -> Dict[Tuple[str, ...], Union[float, Dict[str, Union[float, Dict]]]]:
        """Compute PMI scores for action sequences.

        Calculates Pointwise Mutual Information scores for bigrams and
        higher-order n-grams. Higher values indicate more informative
        or surprising action sequences.

        Parameters
        ----------
        count_actions : List[defaultdict]
            Action frequency counts from :meth:`get_sequences`.
        count_sequences : List[defaultdict]
            Sequence frequency counts from :meth:`get_sequences`.

        Returns
        -------
        Dict[Tuple[str, ...], Union[float, Dict]]
            PMI scores for sequences:
            - Bigrams: Direct PMI value (float)
            - N-grams (n≥3): Dict with 'average_pmi' and 'links' (constituent bigram PMIs)

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

        bigrams_pmi = {}

        # bigram PMI
        for bigram, bigram_count in count_sequences[0].items():
            first_count = count_actions[0][(bigram[0],)]
            second_count = count_actions[1][(bigram[1],)]

            pmi = math.log2(
                (bigram_count / corpus)
                / ((first_count / corpus) * (second_count / corpus))
            )

            bigrams_pmi[bigram] = pmi

        # n‑gram PMI
        customer_bigrams = {
            key: bigrams_pmi[key] for key in count_sequences[2] if key in bigrams_pmi
        }

        ngrams_pmi = dict(customer_bigrams)

        for seq in count_sequences[1].keys():
            links = [seq[i : i + 2] for i in range(len(seq) - 1)]
            values = []
            link_info = {}

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
        ngrams_pmi: Dict[Tuple[str, ...], Union[float, Dict]],
        count_sequences: List[defaultdict],
        customer_sequences: List[Tuple[str, ...]],
        customer_outcomes: List[Tuple[int, ...]],
    ) -> pl.DataFrame:
        """Generate PMI analysis summary DataFrame.

        Creates a summary of action sequences ranked by their significance
        in predicting positive outcomes.

        Parameters
        ----------
        ngrams_pmi : Dict[Tuple[str, ...], Union[float, Dict]]
            PMI scores from :meth:`calculate_pmi`.
        count_sequences : List[defaultdict]
            Sequence frequency counts from :meth:`get_sequences`.
        customer_sequences : List[Tuple[str, ...]]
            Customer action sequences from :meth:`get_sequences`.
        customer_outcomes : List[Tuple[int, ...]]
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
            if len(seq) > 2:
                pmi_val = pmi_val["average_pmi"]

            count = freq_all.get(seq, 0)

            if count <= 1:
                continue

            data.append(
                {
                    "Sequence": seq,
                    "Length": len(seq),
                    "Avg PMI": pmi_val,
                    "Frequency": count,
                    "Unique freq": count_sequences[3][seq],
                    "Score": pmi_val * math.log(count),
                }
            )

        return (
            pl.DataFrame(data)
            .sort("Score", descending=True)
            .with_columns(pl.col("Score").round(3), pl.col("Avg PMI").round(3))
        )
