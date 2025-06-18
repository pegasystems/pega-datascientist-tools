__all__ = ["GradientBoostGlobalExplanations", "DataLoader", "Plotter"]

import json
import logging
import pathlib
import shutil
from datetime import datetime, timedelta
from enum import Enum
from glob import glob
from importlib_resources import files
from typing import List, Literal, TypedDict, get_args

import duckdb
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from .resources import queries as queries_data

logger = logging.getLogger(__name__)


class _PREDICTOR_TYPE(Enum):
    NUMERIC: str = "NUMERIC"
    SYMBOLIC: str = "SYMBOLIC"


class _TABLE_NAME(Enum):
    NUMERIC = "numeric"
    SYMBOLIC = "symbolic"
    NUMERIC_OVERALL = "numeric_overall"
    SYMBOLIC_OVERALL = "symbolic_overall"
    CREATE = "create"


ContextInfo = TypedDict("ContextInfo", {"context_key": str, "context_value": str})


class GradientBoostGlobalExplanations:
    """
    Process and explore explanation data for Adaptive Gradient Boost models.

    Class is initialied with data location, which should point to the location of the model's explanation parquet files downloaded from the explanations file repository.
    These parquet files can then be processed to create aggregates to explain the contribution of different predictors on a global level.

    Parameters
    ----------
    data_folder: str
        The path of the folder containing the model explanation parquet files for processing.
    model_name : str, optional
        The name of the model rule. Will be used to identify files in the data folder and to validate that the correct files are being processed.
    end_date : datetime, optional, default = datetime.today()
        Defines the end date of the duration over which aggregates will be collected.
    start_date : datetime, optional, default = end_date - timedelta(7)
        Defines the start date of the duration over which aggregaates wille be collected.
    batch_limit : int, optional, default = 10
        The number of context key partitions to aggregate per batch.
    memory_limit: int, optional, default 2
        The maximum memory duckdb is allowed to allocate in GB.
    thread_count: int, optional, default 4
        The number of threads to be used by duckdb
    output_folder: str, optional, default = .tmp/out
        The folder location where the data aggregates will be written.
    overwrite: bool, optional, default = FALSE
        Flag if files in output folder should be overwritten. If FALSE then the output folder must be empty before aggregates can be processed.
    progress_bar: str, optional, default = FALSE
        Flag to toggle the progress bar for duck db queries
    """

    SEP = ", "
    LEFT_PREFIX = "l"
    RIGHT_PREFIX = "r"

    def __init__(
        self,
        data_folder: str,
        model_name: str = "",
        end_date: datetime = datetime.today(),
        start_date: datetime = None,
        output_folder: str = ".tmp/out",
        overwrite: bool = False,
        progress_bar: bool = False,
        batch_limit: int = 10,
        memory_limit: int = 2,
        thread_count: int = 4,
    ):
        self.conn = duckdb.connect(database=":memory:")
        self.model_name = model_name

        self.end_date = end_date
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=7)
        else:
            self.start_date = start_date

        self.data_folder = data_folder

        self.batch_limit = batch_limit
        self.memory_limit = memory_limit
        self.thread_count = thread_count

        self.output_folder = output_folder
        self.progress_bar = progress_bar

        self.selected_files = self._populate_selected_files()

        self.contexts = None

        output_path = pathlib.Path(self.output_folder)
        if output_path.exists() and output_path.is_dir() and overwrite:
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=overwrite)

    def process(self):
        """Process explanation parquet files and save calculated aggregates.

        This method reads the explanation data from the provided location and creates
        aggregates for multiple contexts which are used to create global explanation plots.

        The different context aggregates are as follows:
        i) Overall Numeric Predictor Contributions
            The average contribution towards predicted model propensity for each numeric predictor value decile.
        ii) Overal Symbolic Predictor Contributions
            The average contribution towards predicted model propensity for each symoblic predictor value.
        iii) Context Specific Numeric Predictor Contributions
            The average contribution towards predicted model propensity for each numeric predictor value decile, grouped by context key partition.
        iv) Overal Symbolic Predictor Contributions
            The average contribution towards predicted model propensity for each symoblic predictor value, grouped by context key partition.

        Each of the aggregates are written to parquet files to a temporary output dirtectory unless specified otherwise.
        """

        self._get_selected_files()
        if len(self.selected_files) == 0:
            logger.warning("No files found to aggregate!")
            return

        try:
            self._run_agg(_PREDICTOR_TYPE.NUMERIC)
        except Exception as e:
            logger.error(f"Failed to aggregate numeric data, err={e}")
            self.conn.close()
            exit(1)

        try:
            self._run_agg(_PREDICTOR_TYPE.SYMBOLIC)
        except Exception as e:
            logger.error(f"Failed to aggregate symbolic data, err={e}")
            self.conn.close()
            exit(1)

        self.conn.close()

    def _populate_selected_files(self):
        last_n_days = [
            (self.start_date + timedelta(days=x)).strftime("%Y%m%d")
            for x in range((self.end_date - self.start_date).days + 1)
        ]
        logger.debug(f'Searching for files for model {self.model_name} from {self.start_date} to {self.end_date}')
        files_ = []
        for day in last_n_days:
            file_pattern = f"{self.data_folder}/{self.model_name}*{day}*.parquet"
            logger.debug(f"Searching for files with pattern: {file_pattern}")
            file_paths = glob(file_pattern)
            logger.debug(f"Found files: {file_paths}")
            for file_path in file_paths:
                if pathlib.Path(file_path).exists():
                    files_.append(file_path)
        logger.info(f"Selected files:= {files_}")
        return files_

    def _get_selected_files(self):
        if len(self.selected_files) == 0:
            self.selected_files = self._populate_selected_files()

        q = ", ".join([f"'{x}'" for x in self.selected_files])
        return q

    def _write_to_parquet(self, df: pl.DataFrame, file_name: str):
        df.write_parquet(f"{self.output_folder}/{file_name}", statistics=False)

    @staticmethod
    def _clean_query(query):
        q = query.replace("\n", " ")

        while "  " in q:
            q = q.replace("  ", " ")

        return q

    def _get_table_name(self, predictor_type) -> _TABLE_NAME:
        return (
            _TABLE_NAME.NUMERIC
            if predictor_type == _PREDICTOR_TYPE.NUMERIC
            else _TABLE_NAME.SYMBOLIC
        )

    def _read_overall_sql_file(self, predictor_type: _PREDICTOR_TYPE):
        sql_file = (
            _TABLE_NAME.NUMERIC_OVERALL
            if predictor_type == _PREDICTOR_TYPE.NUMERIC
            else _TABLE_NAME.SYMBOLIC_OVERALL
        )
        return files(queries_data).joinpath(f"{sql_file.value}.sql").read_text()

    def _read_batch_sql_file(self, predictor_type: _PREDICTOR_TYPE):
        sql_file = (
            _TABLE_NAME.NUMERIC
            if predictor_type == _PREDICTOR_TYPE.NUMERIC
            else _TABLE_NAME.SYMBOLIC
        )

        return files(queries_data).joinpath(f"{sql_file.value}.sql").read_text()

    def _get_create_table_sql_formatted(
        self, tbl_name: _TABLE_NAME, predictor_type: _PREDICTOR_TYPE
    ):
        sql = files(queries_data).joinpath(f"{_TABLE_NAME.CREATE.value}.sql").read_text()

        f_sql = f"""{
            sql.format(
                MEMORY_LIMIT=self.memory_limit,
                ENABLE_PROGRESS_BAR='true' if self.progress_bar else 'false',
                TABLE_NAME=tbl_name.value,
                SELECTED_FILES=self._get_selected_files(),
                PREDICTOR_TYPE=predictor_type.value,
            )
        }"""

        return self._clean_query(f_sql)

    def _get_overall_sql_formatted(self, sql, tbl_name: _TABLE_NAME, where_condition):
        f_sql = f"""{
            sql.format(
                THREAD_COUNT=self.thread_count,
                MEMORY_LIMIT=self.memory_limit,
                LEFT_PREFIX=self.LEFT_PREFIX,
                RIGHT_PREFIX=self.RIGHT_PREFIX,
                ENABLE_PROGRESS_BAR='true' if self.progress_bar else 'false',
                TABLE_NAME=tbl_name.value,
                WHERE_CONDITION=where_condition,
            )
        }"""

        return self._clean_query(f_sql)

    def _get_batch_sql_formatted(
        self, sql, tbl_name: _TABLE_NAME, where_condition="TRUE"
    ):
        f_sql = f"""{
            sql.format(
                THREAD_COUNT=self.thread_count,
                MEMORY_LIMIT=self.memory_limit,
                LEFT_PREFIX=self.LEFT_PREFIX,
                RIGHT_PREFIX=self.RIGHT_PREFIX,
                ENABLE_PROGRESS_BAR='true' if self.progress_bar else 'false',
                TABLE_NAME=tbl_name.value,
                WHERE_CONDITION=where_condition,
            )
        }"""

        return self._clean_query(f_sql)

    def _create_in_mem_table(self, predictor_type: _PREDICTOR_TYPE):
        table_name = self._get_table_name(predictor_type)
        query = self._get_create_table_sql_formatted(table_name, predictor_type)

        self.conn.execute(query)

    def _delete_in_mem_table(self, predictor_type: _PREDICTOR_TYPE):
        table_name = self._get_table_name(predictor_type)

        query = f"""
            DROP TABLE {table_name.value};
        """

        self.conn.execute(query)

    def _get_contexts(self, predictor_type: _PREDICTOR_TYPE):
        table_name = self._get_table_name(predictor_type)

        if self.contexts is None:
            q = f"""
                SELECT {table_name.value}.partition
                FROM {table_name.value}
                GROUP BY {table_name.value}.partition;
            """
            self.contexts = self.conn.execute(q).pl()["partition"].to_list()

        return self.contexts

    def _parquet_in_batches(self, predictor_type: _PREDICTOR_TYPE):
        try:
            table_name = self._get_table_name(predictor_type)

            group_list = self._get_contexts(predictor_type)

            total_groups = len(group_list)
            curr_group = 0

            sql = self._read_batch_sql_file(predictor_type)

            batch_counter = 0

            df_list = []

            while curr_group < total_groups:
                batch = group_list[
                    curr_group : min(curr_group + self.batch_limit, total_groups)
                ]
                where_condition = self._clean_query(f"""
                    {self.LEFT_PREFIX}.partition IN ({self.SEP.join([f"'{row}'" for row in batch])})
                """)

                query = self._get_batch_sql_formatted(sql, table_name, where_condition)

                df = self.conn.execute(query).pl()
                df_list.append(df)

                batch_counter += 1
                curr_group += self.batch_limit
                if batch_counter % 10 == 0:
                    yield {
                        "batch_count": batch_counter,
                        "dataframe": pl.concat(df_list),
                    }
                    df_list = []
                    logger.info(
                        f"Processed {predictor_type} batch {batch_counter}, group: {curr_group}, len: {len(batch)}"
                    )

            if len(df_list) > 0:
                yield {"batch_count": batch_counter, "dataframe": pl.concat(df_list)}
                df_list = []
                logger.info(
                    f"Processed {predictor_type} batch {batch_counter}, group: {curr_group}, len: {len(batch)}"
                )

        except Exception as e:
            logger.error(f"Failed batch for predictor type={predictor_type}, err={e}")
            return

    def _agg_in_batches(self, predictor_type: _PREDICTOR_TYPE):
        for batch in self._parquet_in_batches(predictor_type):
            self._write_to_parquet(
                batch["dataframe"],
                f"{predictor_type.value}_BATCH_{batch['batch_count']}.parquet",
            )

    def _agg_overall(self, predictor_type: _PREDICTOR_TYPE, where_condition="TRUE"):
        df = self._parquet_overall(predictor_type, where_condition)
        self._write_to_parquet(df, f"{predictor_type.value}_OVERALL.parquet")

    def _parquet_overall(self, predictor_type: _PREDICTOR_TYPE, where_condition="TRUE"):
        try:
            sql = self._read_overall_sql_file(predictor_type)

            table_name = self._get_table_name(predictor_type)
            query = self._get_overall_sql_formatted(sql, table_name, where_condition)

            return self.conn.execute(query).pl()

        except Exception as e:
            logger.error(f"Failed for predictor type={predictor_type}, err={e}")
            return

    def _run_agg(self, predictor_type: _PREDICTOR_TYPE):
        try:
            self._create_in_mem_table(predictor_type)

            self._agg_in_batches(predictor_type)

            self._agg_overall(predictor_type)

            self._delete_in_mem_table(predictor_type)
        except Exception as e:
            logger.error(f"Failed for predictor type={predictor_type}, err={e}")
            return

    def _build_where(self, context_keys: dict = None, predictor_names: list = None):
        if context_keys is None:
            context_keys = dict()
        if predictor_names is None:
            predictor_names = []

        where_condition = ""
        for context_key in context_keys.keys():
            if len(where_condition) > 0:
                where_condition += " AND "
            where_condition += f"""
                    {self.LEFT_PREFIX}.{context_key} IN ({self.SEP.join([f"'{context_key_value}'" for context_key_value in context_keys.get(context_key)])})"""
        if len(predictor_names) > 0:
            if len(where_condition) > 0:
                where_condition += " AND "
            where_condition += f"""
                    {self.LEFT_PREFIX}.predictor_name IN ({self.SEP.join([f"'{predictor_name}'" for predictor_name in predictor_names])})"""

        return self._clean_query(where_condition)


class DataLoader:
    _CONTRIBUTION = "contribution"
    _CONTRIBUTION_ABS = "|contribution|"
    _CONTRIBUTION_WEIGHTED = "contribution_weighted"
    _CONTRIBUTION_WEIGHTED_ABS = "|contribution_weighted|"
    _FREQUENCY = "frequency"
    _CONTRIBUTION_MIN = "contribution_min"
    _CONTRIBUTION_MAX = "contribution_max"

    _CONTRIBUTION_CALCULATION = Literal[
        _CONTRIBUTION,
        _CONTRIBUTION_ABS,
        _CONTRIBUTION_WEIGHTED,
        _CONTRIBUTION_WEIGHTED_ABS,
    ]

    _TO_SELECT = [
        "partition",
        "contribution",
        "contribution_abs",
        "frequency",
        "predictor_type",
        "predictor_name",
        "bin_contents",
        "bin_order",
        "contribution_0",
        "contribution_100",
    ]

    _CONTRIBUTIONS_EXP = [
        pl.col("contribution").mean().alias(_CONTRIBUTION),
        pl.col("contribution_abs").mean().alias(_CONTRIBUTION_ABS),
        (
            (pl.col("contribution") * pl.col("frequency")).mean()
            / pl.col("frequency").sum()
        ).alias(_CONTRIBUTION_WEIGHTED),
        (
            (pl.col("contribution_abs") * pl.col("frequency")).mean()
            / pl.col("frequency").sum()
        ).alias(_CONTRIBUTION_WEIGHTED_ABS),
        pl.col("frequency").sum().alias(_FREQUENCY),
        pl.col("contribution_0").min().alias(_CONTRIBUTION_MIN),
        pl.col("contribution_100").max().alias(_CONTRIBUTION_MAX),
    ]

    def __init__(self, data_location):
        self.data_location = data_location
        self._scan_data()

        self.context_df = pl.from_dicts(
            [
                {**json.loads(ck)["partition"], "filter_string": ck}
                for ck in self.df_contextual.collect()["partition"].unique().to_list()
            ]
        )

    def _scan_data(self):
        self.df_contextual = pl.concat(
            [
                pl.scan_parquet(f"{self.data_location}/NUMERIC_BATCH_*.parquet").select(
                    self._TO_SELECT
                ),
                pl.scan_parquet(
                    f"{self.data_location}/SYMBOLIC_BATCH_*.parquet"
                ).select(self._TO_SELECT),
            ]
        ).sort(by="predictor_name")

        self.df_overall = pl.concat(
            [
                pl.scan_parquet(f"{self.data_location}/NUMERIC_OVERALL.parquet").select(
                    self._TO_SELECT
                ),
                pl.scan_parquet(
                    f"{self.data_location}/SYMBOLIC_OVERALL.parquet"
                ).select(self._TO_SELECT),
            ]
        ).sort(by="predictor_name")

    def get_predictor_contributions(
        self,
        contexts: ContextInfo | List[ContextInfo] = None,
        predictors: str | List[str] = None,
        limit: int = 10,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_calculation: _CONTRIBUTION_CALCULATION = _CONTRIBUTION,
    ) -> pl.DataFrame:
        if not isinstance(contexts, List) and contexts is not None:
            contexts = [contexts]
        if isinstance(contexts, List):
            contexts = self._get_context_filters(contexts)

        if isinstance(predictors, str):
            predictors = [predictors]
        elif predictors is None:
            predictors = []

        base_df = self._get_base_df(contexts)

        sort_columns = {
            predictor: self._get_sort_column(predictor, contribution_calculation)
            for predictor in predictors
        }
        df = self._get_aggregates(base_df, predictors).with_columns(
            (
                pl.col("predictor_name").replace(
                    sort_columns, default=contribution_calculation
                )
            ).alias("sort_column"),
        )

        df = df.with_columns(
            pl.coalesce(
                pl.when(pl.col("sort_column") == col).then(pl.col(col))
                for col in df.collect_schema().names()
            )
            .cast(pl.Float32)
            .abs()
            .alias("sort_value")
        )

        df_top_k = df.select(
            pl.all()
            .top_k_by(contribution_calculation, k=limit, reverse=descending)
            .over(self._get_sort_over_columns(predictors), mapping_strategy="explode")
        )

        if missing:
            df_top_k = pl.concat(
                [
                    df_top_k,
                    df.filter(pl.col("predictor_name") == "MISSING").select(pl.all()),
                ]
            ).unique()

        if remaining:
            remaining_base_df = base_df.join(
                df_top_k, on=self._get_group_by_columns(predictors), how="anti"
            )

            if len(predictors) == 0:
                remaining_base_df = remaining_base_df.with_columns(
                    pl.lit("REMAINING").alias("predictor_name"),
                    pl.lit(_PREDICTOR_TYPE.SYMBOLIC.value).alias("predictor_type"),
                )
            else:
                remaining_base_df = remaining_base_df.with_columns(
                    pl.lit("REMAINING").alias("bin_contents"),
                    pl.lit(0).cast(pl.Int64).alias("bin_order"),
                )

            remaining_df = self._get_aggregates(
                remaining_base_df, predictors
            ).with_columns(
                (
                    pl.col("predictor_name").replace(
                        sort_columns, default=contribution_calculation
                    )
                ).alias("sort_column"),
            )

            remaining_df = remaining_df.with_columns(
                pl.coalesce(
                    pl.when(pl.col("sort_column") == col).then(pl.col(col))
                    for col in remaining_df.collect_schema().names()
                )
                .cast(pl.Float32)
                .abs()
                .alias("sort_value")
            )

            df_top_k = pl.concat([df_top_k, remaining_df])

        df_sorted = df_top_k.sort(
            by=[*self._get_sort_over_columns(predictors), "sort_value"]
        )

        columns = [
            contribution_calculation,
            "predictor_name",
            "bin_contents",
            "sort_column",
            "sort_value",
            self._FREQUENCY,
            self._CONTRIBUTION_MIN,
            self._CONTRIBUTION_MAX,
            "partition",
        ]

        columns = [
            column for column in columns if column in df_sorted.collect_schema().names()
        ]

        return df_sorted.collect().select(columns)

    def get_context_keys(self) -> list:
        """Get the context keys for the current data filter.

        Returns
        -------
        list
            A list of context keys.
        """
        return self.context_df.select(pl.exclude("filter_string")).columns

    def _get_filtered_context_df(
        self, context_infos: List[ContextInfo] | ContextInfo = []
    ) -> pl.DataFrame:
        if not isinstance(context_infos, list):
            context_infos = [context_infos]

        if len(context_infos) == 0:
            df = self.context_df
        else:
            df = pl.DataFrame()
            for context_info in context_infos:
                # print(context_info)
                expr = [
                    pl.col(column_name) == column_value
                    for column_name, column_value in context_info.items()
                ]

                df = pl.concat([df, self.context_df.filter(expr)])
        return df

    def _get_context_filters(
        self, context_infos: List[ContextInfo] | ContextInfo = []
    ) -> list[str]:
        df = self._get_filtered_context_df(context_infos)
        return df.select("filter_string").unique().to_series().to_list()

    def get_context_infos(
        self, context_infos: List[ContextInfo] | ContextInfo = []
    ) -> List[ContextInfo]:
        """Get the possible context key filters for the provided context info.

        Parameters
        ----------
        context_keys : ContextInfo, optional
            A dictionary of context keys and their values.

        Returns
        -------
        list[str]
            A list of context key partitions.
        """
        df = self._get_filtered_context_df(context_infos)
        return df.select(pl.exclude("filter_string")).unique().to_dicts()

    def _get_base_df(self, contexts: List[str]) -> pl.LazyFrame:
        if contexts is None:
            # print("No contexts provided, returning overall data")
            return self.df_overall.with_columns(pl.lit("").alias("partition"))
        else:
            # print(f"returning for {contexts}")
            return self.df_contextual.filter(
                pl.col("partition").str.contains_any(contexts)
            )

    def _get_group_by_columns(self, predictors: List[str]) -> List[str]:
        columns = (
            ["predictor_name", "bin_contents", "bin_order"]
            if len(predictors) > 0
            else ["predictor_name", "predictor_type"]
        )
        columns.append("partition")
        return columns

    def _get_sort_over_columns(self, predictors: List[str]) -> List[str]:
        columns = ["predictor_name"] if len(predictors) > 0 else []
        columns.append("partition")
        return columns

    def _get_aggregates(self, df: pl.LazyFrame, predictors: List[str]) -> pl.LazyFrame:
        if len(predictors) > 0:
            return (
                df.filter(pl.col("predictor_name").is_in(predictors))
                .group_by(self._get_group_by_columns(predictors))
                .agg(self._CONTRIBUTIONS_EXP)
            )
        else:
            return df.group_by(self._get_group_by_columns(predictors)).agg(
                self._CONTRIBUTIONS_EXP
            )

    def _get_sort_column(
        self, predictor, contribution_calculation: _CONTRIBUTION_CALCULATION
    ) -> str:
        if predictor is None:
            return contribution_calculation
        else:
            predictor_type = (
                self.df_overall.filter(pl.col("predictor_name") == predictor)
                .select("predictor_type")
                .first()
                .collect()
                .item()
            )
            return (
                "bin_order"
                if predictor_type == _PREDICTOR_TYPE.NUMERIC.value
                else contribution_calculation
            )

    def _get_sort_expression(
        self, predictor: str, contribution_calculation: _CONTRIBUTION_CALCULATION
    ) -> List[pl.Expr]:
        sort_column = self._get_sort_column(predictor, contribution_calculation)
        return (
            pl.col(sort_column).abs()
            if sort_column in list(get_args(self._CONTRIBUTION_CALCULATION))
            else pl.col(sort_column)
        )


class Plotter:
    """A class to plot the contributions of predictors or predictor values."""

    @classmethod
    def plot_overall_contributions(
        cls,
        data_loader: DataLoader,
        context: ContextInfo = None,
        top_n: int = 20,
        top_k: int = 20,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_calculation: DataLoader._CONTRIBUTION_CALCULATION = DataLoader._CONTRIBUTION,
    ) -> List[go.Figure]:
        df = data_loader.get_predictor_contributions(
            context,
            None,
            top_n,
            descending,
            missing,
            remaining,
            contribution_calculation,
        )

        predictors = (
            df.filter(pl.col("predictor_name") != "REMAINING")
            .select("predictor_name")
            .unique()
            .to_series()
            .to_list()
        )

        df_predictors = data_loader.get_predictor_contributions(
            context,
            predictors,
            top_k,
            descending,
            missing,
            remaining,
            contribution_calculation,
        )
        return [cls._plot_overall_contributions(df, context), *cls._plot_predictor_contributions(df_predictors)]

    @staticmethod
    def _plot_overall_contributions(
        df: pl.DataFrame, context: ContextInfo
    ) -> go.Figure:
        title = "Overall average predictor contributions for "
        if context is None:
            title += "the whole model"
        else:
            title += f"{' '.join(context.values())}"

        fig = px.bar(df, x=df.columns[0], y=df.columns[1], orientation="h", title=title)
        fig.data[0].marker = {
            "color": df[df.columns[0]].to_list(),
            "colorscale": "RdBu_r",
            "cmid": 0.0,
        }
        fig.update_layout(
            xaxis_title=df.columns[0],
            yaxis_title='Predictor',
            height=600
        )
        return fig

    @classmethod
    def plot_predictor_contributions(
        cls,
        data_loader: DataLoader,
        context: ContextInfo,
        top_n: int = 20,
        top_k: int = 20,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_calculation: DataLoader._CONTRIBUTION_CALCULATION = DataLoader._CONTRIBUTION,
    ) -> List[go.Figure]:
        df_context = data_loader.get_predictor_contributions(
            [context],
            [],
            top_n,
            descending,
            missing,
            remaining,
            contribution_calculation,
        )
        predictors = (
            df_context.filter(pl.col("predictor_name") != "REMAINING")
            .select("predictor_name")
            .unique()
            .to_series()
            .to_list()
        )
        df = data_loader.get_predictor_contributions(
            context,
            predictors,
            top_k,
            descending,
            missing,
            remaining,
            contribution_calculation,
        )
        return [
            cls._plot_context_table(context),
            cls._plot_overall_contributions(df_context, context),
            *cls._plot_predictor_contributions(df),
        ]

    @staticmethod
    def _plot_predictor_contributions(df: pl.DataFrame) -> go.Figure:
        predictors = df.select("predictor_name").unique().to_series().to_list()
        plots = []
        for predictor in predictors:
            predictor_df = df.filter(pl.col("predictor_name") == predictor)

            fig = px.bar(
                predictor_df,
                x=predictor_df.columns[0],
                y=predictor_df.columns[2],
                orientation="h",
                title=f"Contributions for {predictor}",
            )
            fig.data[0].marker = {
                "color": predictor_df[predictor_df.columns[0]].to_list(),
                "colorscale": "RdBu_r",
                "cmid": 0.0,
            }
            fig.update_layout(
                yaxis_title=predictor,
            )
            plots.append(fig)
        return plots

    @staticmethod
    def _plot_context_table(context_info: ContextInfo) -> go.Figure:
        fig = go.Figure(
            go.Table(
                header=dict(values=["Context key", "Context value"], align="left"),
                cells=dict(
                    values=[list(context_info.keys()), list(context_info.values())],
                    align="left",
                ),
            )
        )
        fig.update_layout(title="Context Information", height=200)
        return fig
