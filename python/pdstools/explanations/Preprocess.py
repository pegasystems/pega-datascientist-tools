__all__ = ["Preprocess"]

import logging
import os
import pathlib
from datetime import timedelta
from glob import glob
from importlib.resources import files as resources_files
from typing import TYPE_CHECKING, Optional

import duckdb
import polars as pl

from ..utils.namespaces import LazyNamespace
from .ExplanationsUtils import _COL, _PREDICTOR_TYPE, _TABLE_NAME
from .resources import queries as queries_data

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .Explanations import Explanations


class Preprocess(LazyNamespace):
    dependencies = ["duckdb", "polars"]
    dependency_group = "explanations"

    SEP = ", "
    LEFT_PREFIX = "l"
    RIGHT_PREFIX = "r"

    def __init__(self, explanations: "Explanations"):
        self.explanations = explanations

        self.explanations_folder = pathlib.Path(self.explanations.data_folder)
        self.data_foldername = "aggregated_data"
        self.data_folderpath = pathlib.Path(
            os.path.join(self.explanations.root_dir, self.data_foldername)
        )

        self.from_date = explanations.from_date
        self.to_date = explanations.to_date
        self.model_name = explanations.model_name

        self.batch_limit = int(os.getenv("BATCH_LIMIT", "10"))
        self.memory_limit = int(os.getenv("MEMORY_LIMIT", "2"))
        self.thread_count = int(os.getenv("THREAD_COUNT", "4"))
        self.progress_bar = os.getenv("PROGRESS_BAR", "0") == "1"

        self._conn = None

        self.selected_files: list[str] = []
        self.contexts: Optional[list[str]] = None
        self.unique_contexts_filename = f"{self.data_folderpath}/unique_contexts.csv"

        super().__init__()

        self.generate()

    def generate(self):
        """Process explanation parquet files and save calculated aggregates.

        This method reads the explanation data from the provided location and creates
        aggregates for multiple contexts which are used to create global explanation plots.

        The different context aggregates are as follows:
        i) Overall Numeric Predictor Contributions
            The average contribution towards predicted model propensity
            for each numeric predictor value decile.
        ii) Overal Symbolic Predictor Contributions
            The average contribution towards predicted model propensity
            for each symoblic predictor value.
        iii) Context Specific Numeric Predictor Contributions
            The average contribution towards predicted model propensity
            for each numeric predictor value decile, grouped by context key partition.
        iv) Overal Symbolic Predictor Contributions
            The average contribution towards predicted model propensity
            for each symoblic predictor value, grouped by context key partition.

        Each of the aggregates are written to parquet files to a temporary output dirtectory
        """

        if self._is_cached():
            logger.debug("Using cached data for preprocessing.")
            return

        try:
            self._populate_selected_files()
            if len(self.selected_files) == 0:
                logger.error("No files found to aggregate!")
                raise ValueError("No files found to aggregate!")
        except ValueError as e:
            logger.error("Failed to populate selected files: %s", e)
            raise

        self._conn = duckdb.connect(database=":memory:")

        try:
            self._run_agg(_PREDICTOR_TYPE.NUMERIC)
        except Exception as e:
            logger.error("Failed to aggregate numeric data, err=%s", e)
            self._conn.close()
            return

        try:
            self._run_agg(_PREDICTOR_TYPE.SYMBOLIC)
        except Exception as e:
            logger.error("Failed to aggregate symbolic data, err=%s", e)
            self._conn.close()
            return

        self._conn.close()

    @staticmethod
    def _clean_query(query):
        q = query.replace("\n", " ")

        while "  " in q:
            q = q.replace("  ", " ")

        return q

    def _is_cached(self):
        if self.data_folderpath.exists() and self.data_folderpath.is_dir():
            if any(self.data_folderpath.iterdir()):
                return True
        self.data_folderpath.mkdir(parents=True, exist_ok=True)
        return False
    
    def _validate_explanations_folder(self):
        if self.explanations_folder.exists() and self.explanations_folder.is_dir():
            if any(self.explanations_folder.iterdir()):
                return True
        return False

    def _run_agg(self, predictor_type: _PREDICTOR_TYPE):
        try:
            self._create_in_mem_table(predictor_type)

            self._agg_in_batches(predictor_type)

            self._agg_overall(predictor_type)

            self._delete_in_mem_table(predictor_type)
        except Exception as e:
            logger.error("Failed for predictor type=%s, err=%s", predictor_type, e)
            return

    def _create_in_mem_table(self, predictor_type: _PREDICTOR_TYPE):
        table_name = self._get_table_name(predictor_type)
        query = self._get_create_table_sql_formatted(table_name, predictor_type)

        self._execute_query(query)

    def _create_unique_contexts_file(self, predictor_type: _PREDICTOR_TYPE):
        if self.contexts is None:
            self._get_contexts(predictor_type=predictor_type)

        if pathlib.Path(self.unique_contexts_filename).exists():
            return

        df = pl.DataFrame(self.contexts, schema={_COL.PARTITON.value: pl.Utf8()})
        df.write_csv(
            self.unique_contexts_filename, include_header=False, quote_style="never"
        )

    def _get_contexts(self, predictor_type: _PREDICTOR_TYPE):
        if self.contexts is None:
            table_name = self._get_table_name(predictor_type)

            q = f"""
                SELECT {table_name.value}.{_COL.PARTITON.value}
                FROM {table_name.value}
                GROUP BY {table_name.value}.{_COL.PARTITON.value};
            """
            self.contexts = self._execute_query(q).pl()[_COL.PARTITON.value].to_list()

        return self.contexts

    def _agg_in_batches(self, predictor_type: _PREDICTOR_TYPE):
        self._create_unique_contexts_file(predictor_type)

        for batch in self._parquet_in_batches(predictor_type):
            self._write_to_parquet(
                batch["dataframe"],
                f"{predictor_type.value}_BATCH_{batch['batch_count']}.parquet",
            )

    def _agg_overall(self, predictor_type: _PREDICTOR_TYPE, where_condition="TRUE"):
        df = self._parquet_overall(predictor_type, where_condition)
        if df is None:
            logger.error("No data found for predictor type=%s", predictor_type)
            return
        self._write_to_parquet(df, f"{predictor_type.value}_OVERALL.parquet")

    def _delete_in_mem_table(self, predictor_type: _PREDICTOR_TYPE):
        table_name = self._get_table_name(predictor_type)

        query = f"""
            DROP TABLE {table_name.value};
        """

        self._execute_query(query)

    @staticmethod
    def _get_table_name(predictor_type) -> _TABLE_NAME:
        return (
            _TABLE_NAME.NUMERIC
            if predictor_type == _PREDICTOR_TYPE.NUMERIC
            else _TABLE_NAME.SYMBOLIC
        )

    def _get_create_table_sql_formatted(
        self, tbl_name: _TABLE_NAME, predictor_type: _PREDICTOR_TYPE
    ):
        sql = self._read_resource_file(
            package_name=queries_data,
            filename_w_ext=f"{_TABLE_NAME.CREATE.value}.sql",
        )

        f_sql = f"""{
            sql.format(
                MEMORY_LIMIT=self.memory_limit,
                ENABLE_PROGRESS_BAR="true" if self.progress_bar else "false",
                TABLE_NAME=tbl_name.value,
                SELECTED_FILES=self._get_selected_files(),
                PREDICTOR_TYPE=predictor_type.value,
            )
        }"""

        return self._clean_query(f_sql)

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
                    {self.LEFT_PREFIX}.{_COL.PARTITON.value} IN ({self.SEP.join([f"'{row}'" for row in batch])})
                """)

                query = self._get_batch_sql_formatted(sql, table_name, where_condition)

                df = self._execute_query(query).pl()
                df_list.append(df)

                batch_counter += 1
                curr_group += self.batch_limit
                if batch_counter % 10 == 0 or curr_group >= total_groups:
                    yield {
                        "batch_count": batch_counter,
                        "dataframe": pl.concat(df_list),
                    }
                    df_list = []
                    logger.info(
                        "Processed %s batch %s, group: %s, len: %s",
                        predictor_type,
                        batch_counter,
                        curr_group,
                        len(batch),
                    )

        except Exception as e:
            logger.error(
                "Failed batch for predictor type=%s, err=%s", predictor_type, e
            )
            raise

    def _parquet_overall(self, predictor_type: _PREDICTOR_TYPE, where_condition="TRUE"):
        try:
            sql = self._read_overall_sql_file(predictor_type)

            table_name = self._get_table_name(predictor_type)
            query = self._get_overall_sql_formatted(sql, table_name, where_condition)

            return self._execute_query(query).pl()

        except Exception as e:
            logger.error("Failed for predictor type=%s, err=%s", predictor_type, e)
            raise e

    def _write_to_parquet(self, df: pl.DataFrame, file_name: str):
        df.write_parquet(f"{self.data_folderpath}/{file_name}", statistics=False)

    def _read_overall_sql_file(self, predictor_type: _PREDICTOR_TYPE):
        sql_file = (
            _TABLE_NAME.NUMERIC_OVERALL
            if predictor_type == _PREDICTOR_TYPE.NUMERIC
            else _TABLE_NAME.SYMBOLIC_OVERALL
        )
        return self._read_resource_file(
            package_name=queries_data, filename_w_ext=f"{sql_file.value}.sql"
        )

    def _read_batch_sql_file(self, predictor_type: _PREDICTOR_TYPE):
        sql_file = (
            _TABLE_NAME.NUMERIC
            if predictor_type == _PREDICTOR_TYPE.NUMERIC
            else _TABLE_NAME.SYMBOLIC
        )

        return self._read_resource_file(
            package_name=queries_data, filename_w_ext=f"{sql_file.value}.sql"
        )

    def _read_resource_file(self, package_name, filename_w_ext):
        return (
            resources_files(package_name)
            .joinpath(filename_w_ext)
            .read_text(encoding="utf-8")
        )

    def _get_overall_sql_formatted(self, sql, tbl_name: _TABLE_NAME, where_condition):
        f_sql = f"""{
            sql.format(
                THREAD_COUNT=self.thread_count,
                MEMORY_LIMIT=self.memory_limit,
                LEFT_PREFIX=self.LEFT_PREFIX,
                RIGHT_PREFIX=self.RIGHT_PREFIX,
                ENABLE_PROGRESS_BAR="true" if self.progress_bar else "false",
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
                ENABLE_PROGRESS_BAR="true" if self.progress_bar else "false",
                TABLE_NAME=tbl_name.value,
                WHERE_CONDITION=where_condition,
            )
        }"""

        return self._clean_query(f_sql)

    def _get_selected_files(self):
        if len(self.selected_files) == 0:
            self._populate_selected_files()

        q = ", ".join([f"'{x}'" for x in self.selected_files])
        return q

    def _populate_selected_files(self):
        if self.from_date is None or self.to_date is None:
            raise ValueError(
                "Either from_date or to_date must be passed before populating selected files."
            )
   
        if not self._validate_explanations_folder():
            raise ValueError(f"Explanations folder {self.explanations_folder} does not exist or is empty.")

        # get list of dates in the range from `from_date` to `to_date`
        date_range_list = [
            (self.from_date + timedelta(days=x)).strftime("%Y%m%d")
            for x in range((self.to_date - self.from_date).days + 1)
        ]

        logger.debug(
            "Searching for files for model %s from %s to %s",
            self.model_name,
            self.from_date,
            self.to_date,
        )
        files_ = []
        for date in date_range_list:
            file_pattern = (
                f"{self.explanations_folder}/{self.model_name}*{date}*.parquet"
            )
            file_paths = glob(file_pattern)
            if file_paths:
                # Get the latest file based on modification time
                # incase of multiple runs on the same day, we want the latest file on the day
                latest_file = max(
                    file_paths, key=lambda x: pathlib.Path(x).stat().st_mtime
                )
                if pathlib.Path(latest_file).exists():
                    files_.append(latest_file)

        logger.info("Selected files:= \n %s", files_)
        self.selected_files = files_

    def _execute_query(self, query: str):
        """Execute a query on the in-memory DuckDB connection."""
        if self._conn is None:
            raise ValueError("DuckDB connection is not initialized.")

        try:
            execution = self._conn.execute(query)
        except duckdb.CatalogException as e:
            logger.error("DatabaseException: %s", e)
            raise
        except Exception as e:
            logger.error("Failed to execute query: %s", e)
            raise

        return execution
