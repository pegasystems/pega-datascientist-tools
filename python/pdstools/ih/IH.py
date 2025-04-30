import datetime
import os
import random
from typing import Dict, List, Optional, Union
from collections import defaultdict
import math
import polars as pl
import polars.selectors as cs

from .Aggregates import Aggregates
from .Plots import Plots
from ..utils.cdh_utils import _polars_capitalize, _apply_query
from ..utils.types import QUERY
from ..pega_io.File import read_ds_export


class IH:
    data: pl.LazyFrame
    positive_outcome_labels: Dict[str, List[str]]

    def __init__(self, data: pl.LazyFrame):
        self.data = _polars_capitalize(data)

        self.aggregates = Aggregates(ih=self)
        self.plot = Plots(ih=self)
        self.positive_outcome_labels = {
            "Engagement": ["Accepted", "Accept", "Clicked", "Click"],
            "Conversion": ["Conversion"],
            "OpenRate": ["Opened", "Open"],
        }
        self.negative_outcome_labels = {
            "Engagement": [
                "Impression",
                "Impressed",
                "Pending",
                "NoResponse",
            ],
            "Conversion": ["Impression", "Pending"],
            "OpenRate": ["Impression", "Pending"],
        }

    @classmethod
    def from_ds_export(
        cls,
        ih_filename: Union[os.PathLike, str],
        query: Optional[QUERY] = None,
    ):
        """Create an IH instance from a file with Pega Dataset Export

        Parameters
        ----------
        ih_filename : Union[os.PathLike, str]
            The full path to the dataset files
        query : Optional[QUERY], optional
            An optional argument to filter out selected data, by default None

        Returns
        -------
        IH
            The properly initialized IH object

        """
        data = read_ds_export(ih_filename).with_columns(
            # TODO this should come from some polars func in utils
            pl.col("pxOutcomeTime").str.strptime(pl.Datetime, "%Y%m%dT%H%M%S%.3f %Z")
        )
        if query is not None:
            data = _apply_query(data, query=query)

        return IH(data)

    @classmethod
    def from_s3(cls):
        """Not implemented yet. Please let us know if you would like this functionality!"""
        ...

    @classmethod
    def from_mock_data(cls, days=90, n=100000):
        """Initialize an IH instance with sample data

        Parameters
        ----------
        days : number of days, defaults to 90 days
        n : number of interaction data records, defaults to 100k

        Returns
        -------
        IH
            The properly initialized IH object
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
    ) -> tuple[
        list[tuple[str, ...]],
        list[tuple[int, ...]],
        list[defaultdict[tuple[str], int]],
        list[defaultdict[tuple[str, ...], int]],
    ]:
        """
        Generates customer sequences, outcome labels, counts
        needed for PMI (Pointwise Mutual Information) calculations.

        This function processes customer interaction data to produce:
        1. Action sequences per customer.
        2. Corresponding binary outcome sequences (1 for positive outcome, 0 otherwise).
        3. Counts of bigrams and ≥3-grams that end with a positive outcome.
        4. Counts of all possible bigrams within that corpus.

        Parameters
        ----------
        positive_outcome_label : str
            The outcome label that marks the final event in a sequence.
        level : str
            Column name that contains the action (offer / treatment).
        outcome_column : str
            Column name that contains the outcome label.
        customerid_column : str
            Column name that identifies a unique customer / subject.

        Returns
        -------
        customer_sequences: list[tuple[str, ...]]
            Sequences of actions per customer.
        customer_outcomes: list[tuple[int, ...]]
            Binary outcomes (0 or 1) for each customer action sequence.
        count_actions: list[defaultdict[tuple[str], int]]
            Actions frequency counts.
            Index 0 = count of first element in all bigrams
            Index 1 = count of second element in all bigrams
        count_sequences: list[defaultdict[tuple[str, …], int]] 
            Sequence frequency counts.
            Index 0 = bigrams (all)
            Index 1 = ≥3-grams that end with positive outcome
            Index 2 = bigrams that end with positive outcome
            Index 3 = unique ngrams per customer
        """
        cols = [customerid_column, level, outcome_column]

        df = (
            self.data
            .select(cols)
            .sort([customerid_column])
            .collect()
        )

        count_actions = [defaultdict(int), defaultdict(int)]
        count_sequences = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
        customer_sequences = []
        customer_outcomes = []

        # Iterate over customers
        for user_id, user_df in df.group_by(customerid_column):
            user_actions = user_df[level].to_list()
            outcome_actions = user_df[outcome_column].to_list()

            outcome_actions = [1 if action == positive_outcome_label else 0 for action in outcome_actions]

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
                        ngram = seq[i:i + n]
                        ngram_outcomes = out[i:i + n]

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

        ngrams, bigrams, bigrams_all = ngrams_and_bigrams(customer_sequences, customer_outcomes)

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
        count_actions: list[defaultdict[tuple[str], int]],
        count_sequences: list[defaultdict[tuple[str, ...], int]],
    ) -> tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, ...], float]
    ]:
        """
        Computes PMI scores for n-grams (n ≥ 2) in customer action sequences. 
        Returns an unsorted dictionary mapping sequences to their PMI values, providing insights into significant action associations.

        Bigrams values are calculated by PMI.
        N-gram values are computed by averaging the PMI of their constituent bigrams.
        Higher values indicate more informative or surprising paths.

        Parameters
        -------
        count_actions: list[defaultdict[tuple[str], int]] 
            Actions frequency counts. 
            Index 0 = count of first element in all bigrams
            Index 1 = count of second element in all bigrams
        count_sequences: list[defaultdict[tuple[str, …], int]]  
            Sequence frequency counts.
            Index 0 = bigrams (all)
            Index 1 = ≥3-grams that end with positive outcome
            Index 2 = bigrams that end with positive outcome
            Index 3 = unique ngrams per customer

        Returns
        -------
        ngrams_pmi: dict[tuple[str, ...], float | dict[str, float | dict[tuple[str, str], float]]]
            Dictionary containing PMI information for bigrams and n-grams.
            For bigrams, the value is a float representing the PMI value.
            For higher-order n-grams, the value is a dictionary with:
                - 'average_pmi: The average PMI value.
                - 'links': A dictionary mapping each constituent bigram to its PMI value.
        """
        # corpus size (number of action tokens)
        corpus = (
            sum(len(k) * v for k, v in count_sequences[1].items())
            + sum(len(k) * v for k, v in count_sequences[2].items())
        )

        bigrams_pmi = {}

        # bigram PMI
        for bigram, bigram_count in count_sequences[0].items():
            first_count = count_actions[0][(bigram[0],)]
            second_count = count_actions[1][(bigram[1],)]

            pmi = math.log2((bigram_count / corpus) / ((first_count / corpus) * (second_count / corpus)))

            bigrams_pmi[bigram] = pmi

        # n‑gram PMI
        customer_bigrams = {key: bigrams_pmi[key] for key in count_sequences[2] if key in bigrams_pmi}

        ngrams_pmi = dict(customer_bigrams)

        for seq in count_sequences[1].keys():
            links = [seq[i:i+2] for i in range(len(seq) - 1)]
            values = []
            link_info = {}

            for link in links:
                if link in bigrams_pmi:
                    value = bigrams_pmi[link]
                    values.append(value)
                    link_info[link] = value

            average = sum(values) / len(values) if values else 0

            ngrams_pmi[seq] = {
                "average_pmi": average,
                "links": link_info
            }

        return ngrams_pmi
    
    @staticmethod
    def pmi_overview(
        ngrams_pmi: Dict[str, Dict[str, Union[Dict[str, float], float]]],
        count_sequences: list[defaultdict[tuple[str, ...], int]],
        customer_sequences: list[tuple[str, ...]],
        customer_outcomes: list[tuple[int, ...]],
    ) -> pl.DataFrame:
        """
        Analyzes customer sequences to identify patterns linked to positive outcomes. Returns a sorted Polars DataFrame of significant n-grams

        Parameters
        ----------
        ngrams_pmi: dict[tuple[str, ...], float | dict[str, float | dict[tuple[str, str], float]]]
            Dictionary containing PMI information for bigrams and n-grams.
            For bigrams, the value is a float representing the PMI value.
            For higher-order n-grams, the value is a dictionary with:
                - 'average_pmi: The average PMI value.
                - 'links': A dictionary mapping each constituent bigram to its PMI value.

        count_sequences : list[defaultdict[tuple[str, ...], int]]
            Sequence frequency counts.
            Index 1 = ≥3-grams ending in positive outcome.
            Index 2 = bigrams ending in positive outcome.

        customer_sequences : list[tuple[str, ...]]
            Sequences of actions per customer.

        customer_outcomes : list[tuple[int, ...]]
            Binary outcomes (0 or 1) for each customer action sequence.

        Returns
        -------
        pl.DataFrame
            DataFrame containing:
            - 'Sequence': the action sequence
            - 'Length': number of actions
            - 'Avg PMI': average PMI value
            - 'Frequency': number of times the sequence appears
            - 'Unique freq': number of unique customers who had this sequence ending in a positive outcome
            - 'Score': Avg PMI x log(Frequency), sorted descending
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
                    "Sequence":  seq,
                    "Length":    len(seq),
                    "Avg PMI":   pmi_val,
                    "Frequency": count,
                    "Unique freq": count_sequences[3][seq],
                    "Score":     pmi_val * math.log(count),
                }
            )

        return (
            pl.DataFrame(data)
            .sort("Score", descending=True)
            .with_columns(pl.col("Score").round(3), pl.col("Avg PMI").round(3))
        )
