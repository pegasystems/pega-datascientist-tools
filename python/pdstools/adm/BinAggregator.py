import polars as pl
import numpy as np
import plotly.express as px
from plotly.graph_objects import Figure
from typing import Union, Optional, Literal

from .. import ADMDatamart
from ..utils.cdh_utils import lift  # only temp needed

# from IPython.display import display # for better display in notebooks rather than print of dataframes


class BinAggregator:
    """
    A class to generate rolled up insights from ADM predictor binning.
    """

    def __init__(self, dm: ADMDatamart, query: pl.Expr = None) -> None:
        data = dm.last("combinedData").lazy()
        if query is not None:
            # print(f"Query: {query}")
            data = data.filter(query)
        self.all_predictorbinning = self.normalize_all_binnings(data)

    def roll_up(
        self,
        predictors: Union[str, list],
        n: int = 10,
        distribution: Literal["lin", "log"] = "lin",
        boundaries: Optional[Union[float, list]] = None,
        symbols: Optional[Union[str, list]] = None,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        aggregation: Optional[str] = None,
        as_numeric: Optional[bool] = None,
        return_df: bool = False,
        verbose: bool = False,
    ) -> Union[pl.DataFrame, Figure]:
        """Roll up a predictor across all the models defined when creating the class.

        Predictors can be both numeric and symbolic (also called 'categorical'). You
        can aggregate the same predictor across different sets of models by specifying
        a column name in the aggregation argument.

        Parameters
        ----------
        predictors : str | list
            Name of the predictor to roll up. Multiple predictors can be passed in as
            a list.
        n : int, optional
            Number of bins (intervals or symbols) to generate, by default 10. Any
            custom intervals or symbols specified with the 'musthave' argument will
            count towards this number as well. For symbolic predictors can be None,
            which means unlimited.
        distribution : str, optional
            For numeric predictors: the way the intervals are constructed. By default
            "lin" for an evenly-spaced distribution, can be set to "log" for a long
            tailed distribution (for fields like income).
        boundaries : float | list, optional
            For numeric predictors: one value, or a list of the numeric values to
            include as interval boundaries. They will be used at the front of the
            automatically created intervals. By default None, all intervals are
            created automatically.
        symbols : str | list, optional
            For symbolic predictors, any symbol(s) that
            must be included in the symbol list in the generated binning. By default None.
        minimum : float, optional
            Minimum value for numeric predictors, by default None. When None the
            minimum is taken from the binning data of the models.
        maximum : float, optional
            Maximum value for numeric predictors, by default None. When None the
            maximum is taken from the binning data of the models.
        aggregation : str, optional
            Optional column name in the data to aggregate over, creating separate
            aggregations for each of the different values. By default None.
        as_numeric : bool, optional
            Optional override for the type of the predictor, so to be able to
            override in the (exceptional) situation that a predictor with the same
            name is numeric in some and symbolic in some other models. By default None
            which means the type is taken from the first predictor in the data.
        return_df : bool, optional
            Return the underlying binning instead of a plot.
        verbose : bool, optional
            Show detailed debug information while executing, by default False

        Returns
        -------
        pl.DataFrame | Figure
            By default returns a nicely formatted plot. When 'return_df' is set
            to True, it returns the actual binning with the lift aggregated over
            all the models, optionally per predictor and per set of models.
        """
        if not isinstance(predictors, list):
            predictors = [predictors]

        all_binnings = []
        for predictor in predictors:
            is_numeric = (
                as_numeric
                if as_numeric is not None
                else (
                    self.all_predictorbinning.filter(
                        pl.col("PredictorName") == predictor
                    )
                    .select(pl.first("isNumeric"))
                    .collect()
                    .item()
                )
            )  # seems expensive

            if is_numeric:
                empty_numeric_binning = self.create_empty_numbinning(
                    predictor=predictor,
                    n=n,
                    distribution=distribution,
                    boundaries=(
                        []
                        if boundaries is None
                        else (
                            boundaries if isinstance(boundaries, list) else [boundaries]
                        )
                    ),
                    minimum=minimum,
                    maximum=maximum,
                )
                symbol_list = None
            else:
                empty_numeric_binning = None
                symbol_list = self.create_symbol_list(
                    predictor=predictor,
                    n_symbols=n,
                    musthave_symbols=(
                        []
                        if symbols is None
                        else (symbols if isinstance(symbols, list) else [symbols])
                    ),
                )

            if aggregation is None:
                all_topics = ["All"]
            else:
                all_topics = (
                    self.all_predictorbinning.select(aggregation)
                    .unique()
                    .filter(pl.col(aggregation).is_not_null())
                    .collect()[aggregation]
                    .sort()
                    .to_list()
                )
            for topic in all_topics:
                if verbose:
                    print(f"Topic: {topic}, predictor: {predictor}")
                    if is_numeric:
                        print(empty_numeric_binning)
                    else:
                        print(f"Symbols: {', '.join(symbol_list)}")

                if aggregation is None:
                    ids = (
                        self.all_predictorbinning.select(
                            pl.col("ModelID").unique().sort()
                        )
                        .collect()["ModelID"]
                        .to_list()
                    )
                else:
                    ids = (
                        self.all_predictorbinning.filter(pl.col(aggregation) == topic)
                        .select(pl.col("ModelID").unique().sort())
                        .collect()["ModelID"]
                        .to_list()
                    )

                if is_numeric:
                    cum_binning = self.accumulate_num_binnings(
                        predictor,
                        ids,
                        empty_numeric_binning.clone(),
                        verbose=verbose,
                    )
                else:
                    cum_binning = self.accumulate_sym_binnings(
                        predictor,
                        ids,
                        symbol_list,
                        verbose=verbose,
                    )

                cum_binning = cum_binning.with_columns(
                    pl.lit(topic).alias(
                        aggregation if aggregation is not None else "Topic"
                    ),
                )

                all_binnings.append(cum_binning)

        if return_df:
            return pl.concat(all_binnings, how="vertical_relaxed")
        else:
            return self.plot_lift_binning(
                pl.concat(all_binnings, how="vertical_relaxed")
            )

    def accumulate_num_binnings(
        self, predictor, modelids, target_binning, verbose=False
    ) -> pl.DataFrame:
        for id in modelids:
            if verbose:
                print(f"Model ID: {id}")
            source_binning = self.get_source_numbinning(predictor, id)

            # TODO consider quick escape if all of source binning is empty (sum of BinResponses)

            if verbose:
                print(source_binning)

                # fig = self.plot_binning_attribution(source_binning, target_binning)
                # fig.update_layout(title=f"{predictor}, model={id}")
                # fig.show()

            target_binning = self.combine_two_numbinnings(
                source_binning, target_binning, verbose=verbose
            )

            if verbose:
                print(target_binning)

                # fig = self.plot_lift_binning(target_binning)
                # fig.update_layout(width=800, height=300, showlegend=False)
                # fig.show()

        if verbose:
            print(target_binning)

            # fig = self.plot_lift_binning(target_binning)
            # fig.update_layout(
            #     width=800,
            #     height=300,
            # )
            # fig.show()

        return target_binning

    def create_symbol_list(
        self,
        predictor,
        n_symbols,
        musthave_symbols,
    ) -> list:
        symbol_frequency = (
            self.all_predictorbinning.filter(pl.col("Type") != "numeric")
            .filter(pl.col("PredictorName") == predictor)
            .filter(pl.col("BinSymbol") != "NON-MISSING")
            .filter(pl.col("Symbol").is_not_null())
            .explode("Symbol")
            .group_by("Symbol")
            .agg(
                Frequency=pl.sum("BinResponses"),
            )
            .sort("Frequency", descending=True)
        ).collect()

        ordered_symbols = [
            col
            for col in symbol_frequency["Symbol"].to_list()
            if col not in set(musthave_symbols)
        ]

        if n_symbols is None:
            return musthave_symbols + ordered_symbols
        else:
            return (musthave_symbols + ordered_symbols)[slice(n_symbols)]

    def accumulate_sym_binnings(
        self,
        predictor,
        modelids,
        symbollist,
        verbose=False,
    ) -> pl.DataFrame:
        # All the bins for the given predictor, for the given models
        symbins = (
            self.all_predictorbinning.filter(pl.col("Type") != "numeric")
            .filter(pl.col("PredictorName") == predictor)
            .filter(pl.col("BinSymbol") != "NON-MISSING")
            .filter(pl.col("ModelID").is_in(modelids))
        )

        # Get the lift of the residuals per model. This will be used for the symbols that did not appear in the model.
        # Assuming all symbolic bins have a residual bin - I believe they do.
        lift_residual_bins = symbins.filter(pl.col("BinType") == "RESIDUAL").select(
            pl.col("ModelID"), pl.col("Lift").alias("Lift_RESIDUAL")
        )

        # Explode symbol list into separate rows
        symbins_long = (
            symbins.join(lift_residual_bins, on="ModelID", how="left")
            .explode("Symbol")
            .filter(pl.col("Symbol").is_in(symbollist))
            .collect()
        )

        # Pivot and fill the missing values with the lift of the residual bins
        symbins_pivot = symbins_long.pivot(
            values=["Lift"],
            index=["ModelID", "Lift_RESIDUAL"],
            columns="Symbol",
        ).sort("ModelID")  # just for debugging/transparency

        if verbose:
            print("Pivot table:")
            print(symbins_pivot)

        # Add columns for symbols that may be missing completely (must-have symbols not present in the models)
        symbins_pivot = symbins_pivot.with_columns(
            [
                pl.lit(0.0).alias(col)
                for col in symbollist
                if col not in set(symbins_pivot.columns)
            ]
        ).select(
            [pl.lit(predictor).alias("PredictorName")]
            +
            # Fill nulls with residual lift - not all models always have all symbols
            [
                # (pl.when(pl.col(col).is_null()).then(pl.col("Lift_RESIDUAL")).otherwise(pl.col(col))).mean()
                pl.when(pl.col(col).is_null())
                .then(pl.col("Lift_RESIDUAL"))
                .otherwise(pl.col(col))
                .alias(col)
                for col in symbollist
            ]
        )

        if verbose:
            print("Pivot table with additional columns and residuals filled in:")
            print(symbins_pivot)

        # melt the pivot back to long form, but now all symbols are present for all models
        molten = symbins_pivot.melt(
            id_vars="PredictorName", variable_name="Symbol", value_name="Lift"
        )
        if "Lift" not in molten.columns:
            molten = molten.with_columns(Lift=pl.lit(0.0))

        aggregate_binning = (
            molten
            # take the mean lift for each symbol
            .group_by(["PredictorName", "Symbol"])
            .agg(Lift=pl.mean("Lift"))
            # join back in the coverage and response counts from the models
            .join(
                symbins_long.group_by("Symbol").agg(
                    BinResponses=pl.sum("BinResponses"),
                    BinCoverage=pl.count(),  # nr of models that have this symbol
                ),
                on="Symbol",
                how="left",
            )
            # add a bin index
            .sort("Lift", descending=True)
            .with_row_count(name="BinIndex", offset=1)
            # put columns in exact same order as num binning
            .select(
                "PredictorName",
                "BinIndex",
                pl.lit(0.0).alias(
                    "BinLowerBound"
                ),  # for consistency with the num binning
                pl.lit(0.0).alias(
                    "BinUpperBound"
                ),  # for consistency with the num binning
                pl.col("Symbol").alias("BinSymbol"),
                "Lift",
                pl.col("BinResponses").fill_null(0),
                pl.col("BinCoverage").fill_null(0),
                pl.lit(symbins_pivot.select(pl.count()).item()).alias("Models"),
            )
        )

        # Normalize the lift: sum of lift over all bins should be zero by definition
        # if normalize:
        #     aggregate_binning = aggregate_binning.with_columns(
        #         (pl.col("Lift") - pl.sum("Lift") / pl.count()).alias("Lift"),
        #     )

        return aggregate_binning

    def normalize_all_binnings(self, combined_dm: pl.LazyFrame) -> pl.LazyFrame:
        """
        Prepare all predictor binning

        Fix up the boundaries for numeric bins and parse the bin labels
        into clean lists for symbolics.
        """
        numberRegExp = r"[+\-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?"  # matches numbers also with scientific notation

        binnings = (
            combined_dm.filter(pl.col("EntryType") != "Classifier")
            .filter(pl.col("BinType") != "MISSING")  # ignore those on purpose
            .with_columns(
                (pl.col("Lift") - 1.0).alias("Lift"),
                (pl.col("Type") == "numeric").alias("isNumeric"),
            )
            .with_columns(
                # The min/max from the boundaries often are too wide. The
                # Contents field has a better min/max to use for the numeric range.
                (
                    pl.when(pl.col("isNumeric"))
                    .then(
                        pl.col("Contents")
                        .cast(pl.Utf8)
                        .str.extract_all(numberRegExp)
                        .list.eval(pl.element().cast(pl.Float64))
                    )
                    .otherwise(pl.lit(None))
                ).alias("NumBoundaries")
            )
            .with_columns(
                [
                    # Replacing the lowest and highest bounds with the min/max found in the contents field.
                    pl.when(
                        pl.col("isNumeric") & (pl.col("BinSymbol") == "NON-MISSING")
                    )
                    .then(pl.col("NumBoundaries").list.get(0))
                    .when(
                        pl.col("isNumeric")
                        & (
                            pl.col("BinLowerBound").cast(pl.Float64)
                            < pl.col("NumBoundaries").list.get(0)
                        )
                    )
                    .then(pl.col("NumBoundaries").list.get(0))
                    .otherwise(pl.col("BinLowerBound"))
                    .alias("BinLowerBound"),
                    pl.when(
                        pl.col("isNumeric") & (pl.col("BinSymbol") == "NON-MISSING")
                    )
                    .then(pl.col("NumBoundaries").list.get(1))
                    .when(
                        pl.col("isNumeric")
                        & (
                            pl.col("BinUpperBound").cast(pl.Float64)
                            > pl.col("NumBoundaries").list.get(1)
                        )
                    )
                    .then(pl.col("NumBoundaries").list.get(1))
                    .otherwise(pl.col("BinUpperBound"))
                    .alias("BinUpperBound"),
                ]
            )
            .with_columns(
                pl.col("BinLowerBound").cast(pl.Float64),
                pl.col("BinUpperBound").cast(pl.Float64),
                pl.col("BinIndex").cast(pl.Int64),
            )
            .with_columns(
                # Split concatenated bin symbols into list
                pl.when(
                    pl.col("isNumeric") | (pl.col("BinSymbol") == "Remaining symbols")
                )
                .then(pl.lit(None))
                .otherwise(
                    pl.col("BinSymbol")
                    .str.split(by=",")
                    .list.eval(pl.element().str.strip_chars())
                )
                .alias("Symbol")
            )
            .with_columns(
                # Number of different symbols in the list
                pl.when(pl.col("isNumeric"))
                .then(pl.lit(None))
                .otherwise(
                    pl.col("Symbol").map_elements(
                        lambda x: len(x), return_dtype=pl.UInt32
                    )
                )
                .alias("NSymbols")
            )
            .with_columns(
                # Bin response count per symbol
                (
                    (pl.col("BinPositives") + pl.col("BinNegatives"))
                    / pl.col("NSymbols")
                ).alias("BinResponses")
            )
        ).sort(["ModelID", "PredictorName", "BinIndex"])

        return binnings

    def create_empty_numbinning(
        self,
        predictor: str,
        n: int,
        distribution: str = "lin",
        boundaries: Optional[list] = None,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> pl.DataFrame:
        # take min/max across all models
        bins_minmax = (
            self.all_predictorbinning.filter(pl.col("Type") == "numeric")
            .filter(pl.col("PredictorName") == predictor)
            .filter(pl.col("BinLowerBound").is_not_null())
            .filter(pl.col("BinUpperBound").is_not_null())
            .group_by("ModelID", "PredictorName")
            .agg(
                (
                    pl.min("BinLowerBound").alias("Minimum"),
                    pl.max("BinUpperBound").alias("Maximum"),
                    # pl.max("BinIndex").alias("Bins"),
                )
            )
            .collect()
        )
        if minimum is None:
            minimum = bins_minmax.select(pl.col("Minimum").min()).item()
        if maximum is None:
            maximum = bins_minmax.select(pl.col("Maximum").max()).item()
        if boundaries is None:
            boundaries = []

        def create_additional_intervals(
            n: int, distribution: str, boundaries: list, minimum: float, maximum: float
        ):
            num_additional_intervals = n - len(boundaries) + 2
            if num_additional_intervals < 1:
                return []
            if distribution == "lin":
                return np.linspace(
                    boundaries[-1], maximum, num=num_additional_intervals
                ).tolist()[1:]
            elif distribution == "log":
                if boundaries[-1] < 0:
                    raise Exception(
                        "Cant create log distribution with min < 0", boundaries[-1]
                    )
                if boundaries[-1] > 0.0:
                    return np.logspace(
                        np.log10(boundaries[-1]),
                        np.log10(maximum),
                        num=num_additional_intervals,
                    ).tolist()[1:]
            else:
                raise Exception("Invalid distribution", distribution)

        if (len(boundaries) > 0 and minimum < boundaries[0]) or (len(boundaries) == 0):
            boundaries.insert(0, minimum)

        if maximum > boundaries[-1]:
            boundaries = boundaries + create_additional_intervals(
                n, distribution, boundaries, minimum, maximum
            )

        # Bit of a hack here but if minimum = maximum = 0 otherwise errors out with no binning
        if len(boundaries) == 1:
            boundaries = boundaries * 2

        boundaries = np.array(boundaries, dtype=np.float64)  # just for casting

        target_binning = pl.DataFrame(
            {
                "PredictorName": predictor,
                "BinIndex": list(range(1, len(boundaries))),
                "BinLowerBound": boundaries[:-1],
                "BinUpperBound": boundaries[1:],
            }
        ).with_columns(
            # Creating a simple representation of the intervals
            # TODO: the round(2) should be generalized
            pl.when(pl.col("BinIndex") < len(boundaries))
            .then(pl.format("<{}", pl.col("BinUpperBound").round(2)))
            .otherwise(pl.format("<={}", pl.col("BinUpperBound").round(2)))
            .alias("BinSymbol"),
            pl.lit(0.0).alias("Lift"),
            pl.lit(0.0).alias("BinResponses"),
            pl.lit(0.0).alias("BinCoverage"),
            pl.lit(0).alias("Models"),
        )

        return target_binning

    def get_source_numbinning(self, predictor: str, modelid: str) -> pl.DataFrame:
        is_model_immature = (
            (pl.sum("BinPositives") + pl.sum("BinNegatives")) < 200
        ) | (pl.max("BinIndex") < 2)

        return (
            self.all_predictorbinning.filter(pl.col("Type") == "numeric")
            .filter(pl.col("PredictorName") == predictor)
            .filter(pl.col("ModelID") == modelid)
            .select(
                "ModelID",
                "PredictorName",
                "BinIndex",
                "BinType",
                "BinLowerBound",
                "BinUpperBound",
                "BinSymbol",
                "Lift",
                # Set bin response count to 0 for immature models
                pl.when(is_model_immature)
                .then(pl.lit(0.0))
                .otherwise(pl.col("BinPositives") + pl.col("BinNegatives"))
                .alias("BinResponses"),
            )
            .collect()
        )

    def combine_two_numbinnings(
        self, source: pl.DataFrame, target: pl.DataFrame, verbose=False
    ) -> pl.DataFrame:
        class Interval:
            def __init__(self, lo, hi):
                self.lo_ = lo
                self.hi_ = hi

            def __str__(self):
                return f"[{self.lo_:.2f} - {self.hi_:.2f}]"

        source_index = target_index = 0
        while source_index < source.shape[0] and target_index < target.shape[0]:
            source_lo = source[source_index, "BinLowerBound"]
            source_hi = source[source_index, "BinUpperBound"]
            target_lo = target[target_index, "BinLowerBound"]
            target_hi = target[target_index, "BinUpperBound"]

            overlap = min(source_hi, target_hi) - max(source_lo, target_lo)
            if overlap > 0:
                source_lift = source[source_index, "Lift"]
                source_binresponses = source[source_index, "BinResponses"]

                target_lift = target[target_index, "Lift"]
                target_total_attribution = target[target_index, "BinCoverage"]
                target_binresponses = target[target_index, "BinResponses"]

                source_fraction = overlap / float(source_hi - source_lo)
                target_attribution = overlap / float(target_hi - target_lo)

                target[target_index, "Lift"] = (
                    (target_total_attribution * target_lift)
                    + (target_attribution * source_lift)
                ) / (target_total_attribution + target_attribution)
                target[target_index, "BinResponses"] = (
                    target_binresponses + source_fraction * source_binresponses
                )
                target[target_index, "BinCoverage"] = (
                    target_total_attribution + target_attribution
                )

                if verbose:
                    print(
                        f"Attribution from {source[source_index, 'ModelID']}:{source[source_index, 'BinIndex']} {Interval(source_lo, source_hi)} to {target[target_index, 'BinIndex']} {Interval(target_lo, target_hi)}: lift attribution={target_attribution} response attribution={source_fraction}"
                    )
                    # print(
                    #     f"(({target_total_attribution} * {target_lift}) + ({target_attribution} * {source_lift})) / ({target_total_attribution} + {target_attribution}) = {target[target_index, 'Lift']}"
                    # )
            else:
                if verbose:
                    print(
                        f"Attribution from {source[source_index, 'ModelID']}:{source[source_index, 'BinIndex']} {Interval(source_lo, source_hi)} to {target[target_index, 'BinIndex']} {Interval(target_lo, target_hi)}: None"
                    )

            if source_hi >= target_hi:
                target_index = target_index + 1
            if target_hi >= source_hi:
                source_index = source_index + 1

        # Normalize the lift: sum of lift over all bins should be zero by definition
        # if normalize:
        #     target = target.with_columns(
        #         (pl.col("Lift") - pl.sum("Lift") / pl.count()).alias("Lift")
        #     )

        model_count = 1 + target[0, "Models"]

        return (
            # ok bizarre construct, I really just want to replace Models with an incremented value (same for all rows)
            target.drop("Models").with_columns(
                pl.repeat(model_count, pl.count()).alias("Models")
            )
        )

    def plot_binning_attribution(
        self, source: pl.DataFrame, target: pl.DataFrame
    ) -> Figure:
        # create "long" dataframe with the upper and lower bounds of source and target
        boundaries_data = (
            pl.DataFrame(
                {
                    "boundary": source["BinLowerBound"],
                    "bin": source["BinIndex"],
                    "bound": "lower",
                    "binning": "source",
                }
            )
            .vstack(
                pl.DataFrame(
                    {
                        "boundary": source["BinUpperBound"],
                        "bin": source["BinIndex"],
                        "bound": "upper",
                        "binning": "source",
                    }
                )
            )
            .vstack(
                pl.DataFrame(
                    {
                        "boundary": target["BinLowerBound"],
                        "bin": target["BinIndex"],
                        "bound": "lower",
                        "binning": "target",
                    }
                )
            )
            .vstack(
                pl.DataFrame(
                    {
                        "boundary": target["BinUpperBound"],
                        "bin": target["BinIndex"],
                        "bound": "upper",
                        "binning": "target",
                    }
                )
            )
            .filter(pl.col("boundary").is_not_null())
        )

        fig = px.line(
            boundaries_data.to_pandas(),
            x="boundary",
            y="binning",
            markers="both",
            color="binning",
            template="plotly_white",
            # log_x=True,
            width=1400,
            height=250,
        )

        # add horizontal lines to indicate bounds
        for x in boundaries_data.filter(pl.col("binning") == "source")["boundary"]:
            fig.add_vline(x=x, line_width=0.5, line_dash="dash", line_color="blue")

        # add labels for boundaries in source binning
        for i in range(source.shape[0]):
            if source[i, "BinLowerBound"] is not None:
                fig.add_annotation(
                    x=(source[i, "BinLowerBound"] + source[i, "BinUpperBound"]) / 2,
                    y="source",
                    text=f"Bin {source[i, 'BinIndex']}<br><br>Lift: {100.0*source[i, 'Lift']:0.2f}%",
                    showarrow=False,
                    yshift=0,
                    font=dict(color="blue"),
                )

        # add labels for boundaries in target binning
        for i in range(target.shape[0]):
            if target[i, "BinLowerBound"] is not None:
                fig.add_annotation(
                    x=(target[i, "BinLowerBound"] + target[i, "BinUpperBound"]) / 2,
                    y="target",
                    text=f"Bin {target[i, 'BinIndex']}",
                    showarrow=False,
                    yshift=-10,
                    font=dict(color="red"),
                )

        return fig.update_xaxes(title="")

    # "Philip Mann" plot with simple red/green lift bars relative to base propensity
    # TODO currently shared between ModelReport.qmd and BinAggregator.py and
    # copied into plot_base - move over to that version once PDS tools version got bumped
    def plotBinningLift(
        self,
        binning,
        col_facet=None,
        row_facet=None,
        custom_data=["PredictorName", "BinSymbol"],
        return_df=False,
    ) -> Union[pl.DataFrame, Figure]:
        if not isinstance(binning, pl.LazyFrame):
            binning = binning.lazy()

        # Add Lift column if not present
        if "Lift" not in binning.collect_schema().names():
            binning = binning.with_columns(
                (lift(pl.col("BinPositives"), pl.col("BinNegatives")) - 1.0).alias(
                    "Lift"
                )
            )

        # Optionally a shading expression
        if "BinPositives" in binning.collect_schema().names():
            shading_expr = pl.col("BinPositives") <= 5
        else:
            shading_expr = pl.lit(False)

        pm_plot_binning_table = (
            # binning.select(
            #     pl.col(["PredictorName", "BinIndex", "BinSymbol", "BinPositives", "Lift"]),
            #     # add back bin reponses now?
            #     (lift(pl.col("BinPositives"), pl.col("BinNegatives")) - 1.0), # Pega starts lift at 0.0
            # )
            binning.with_columns(
                pl.when((pl.col("Lift") >= 0.0) & shading_expr.not_())
                .then(pl.lit("pos"))
                .when((pl.col("Lift") >= 0.0) & shading_expr)
                .then(pl.lit("pos_shaded"))
                .when((pl.col("Lift") < 0.0) & shading_expr.not_())
                .then(pl.lit("neg"))
                .otherwise(pl.lit("neg_shaded"))
                .alias("Direction"),
            )
            .sort(["PredictorName", "BinIndex"])
            .collect()
        )

        # Abbreviate possibly very long bin labels
        # TODO generalize this, use it in the standard bin plot as well
        # and make sure the resulting labels are unique - with just the
        # truncate they are not necessarily unique
        pm_plot_binning_table = pm_plot_binning_table.with_columns(
            pl.Series(
                "BinSymbolAbbreviated",
                [
                    (s[:25] + "...") if len(s) > 25 else s
                    for s in pm_plot_binning_table["BinSymbol"].to_list()
                ],
            )
        )

        fig = px.bar(
            data_frame=pm_plot_binning_table.to_pandas(),
            x="Lift",
            y="BinSymbolAbbreviated",
            color="Direction",
            color_discrete_map={
                "neg": "#A01503",
                "pos": "#5F9F37",
                "neg_shaded": "#DAA9AB",
                "pos_shaded": "#C5D9B7",
            },
            orientation="h",
            template="pega",
            custom_data=custom_data,
            facet_col=col_facet,
            facet_row=row_facet,
            facet_col_wrap=3,  # will be ignored when there is a row facet
        )
        fig.update_traces(
            hovertemplate="<br>".join(
                ["<b>%{customdata[0]}</b>", "%{customdata[1]}", "<b>Lift: %{x:.2%}</b>"]
            )
        )
        fig.add_vline(x=0, line_color="black")

        fig.update_layout(
            showlegend=False,
            title="Propensity Lift",
            hovermode="y",
        )
        fig.update_xaxes(title="", tickformat=",.2%")
        fig.update_yaxes(
            type="category",
            categoryorder="array",
            # abbreviate possibly lengthy symbol labels
            categoryarray=pm_plot_binning_table["BinSymbolAbbreviated"],
            automargin=True,
            autorange="reversed",
            title="",
            dtick=1,  # show all bins
            matches=None,  # allow independent y-labels if there are row facets
        )
        fig.for_each_annotation(
            lambda a: a.update(text=a.text.split("=")[-1])
        )  # split plotly facet label, show only right side

        if return_df:
            return pm_plot_binning_table
        else:
            return fig

    def plot_lift_binning(self, binning: pl.DataFrame) -> Figure:
        if (
            binning.columns[-1] == "Models"
        ):  # assuming Models is always the last column when not rolled up
            # printing a single binning, not rolled up
            topic = None
            model_facet = None
            predictor_facet = None
            n_models = binning.select(pl.first("Models")).item()
        else:
            # printing rolled up binning
            topic = binning.columns[-1]  # assuming the roll-up column is the last one
            model_facet = (
                # if there are multiple topics, take the topic otherwise assume not rolled up over a topic
                topic
                if binning.select(pl.col(topic).n_unique() > 1).item()
                else None
            )
            predictor_facet = (
                # check if there are multiple predictors
                "PredictorName"
                if binning.select(pl.col("PredictorName").n_unique() > 1).item()
                else None
            )
            n_models = (
                binning.group_by(topic)
                .agg(pl.first("Models"))
                .select(pl.sum("Models"))
                .item()
            )

        n_channels = (
            self.all_predictorbinning.select(["Channel", "Direction"])
            .unique()
            .collect()
            .shape[0]
        )

        fig = self.plotBinningLift(
            binning.with_columns(
                (pl.col("BinCoverage") / pl.col("Models")).alias("RelativeBinCoverage")
            ),
            col_facet=model_facet,
            row_facet=predictor_facet,
            custom_data=[
                "PredictorName",
                "BinSymbol",
                "RelativeBinCoverage",
                "BinResponses",
            ],
        )
        fig.update_traces(
            hovertemplate="<br>".join(
                [
                    "<b>%{customdata[0]}</b>",
                    "<b>%{customdata[1]}</b>",
                    "<b>Lift: %{x:.2%}</b>",
                    "Coverage: %{customdata[2]:.2%}",
                    "Attributed Responses: %{customdata[3]:.2f}",
                ]
            )
        )

        if predictor_facet is None:
            pred = binning.select(pl.first("PredictorName")).item()
            if model_facet is None:
                title = f"Propensity lift of <b>{pred}</b><br>in {n_models} models across {n_channels} channels"
            else:
                title = f"Propensity lift of <b>{pred}</b> in {n_models} models across {n_channels} channels, split by <b>{topic}</b>"
        else:
            if model_facet is None:
                title = f"Propensity lift in {n_models} models across {n_channels} channels, split by <b>{topic}</b>"
            else:
                title = (
                    f"Propensity lift in {n_models} models across {n_channels} channels"
                )

        fig.update_layout(
            title=title,
        )

        return fig
