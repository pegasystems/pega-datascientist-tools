__all__ = ["GradientBoostGlobalExplanations"]

import os
import logging
import pathlib
import shutil
import duckdb
import polars as pl

from datetime import datetime, timedelta
from glob import glob
from importlib_resources import files
from typing import Optional

from .resources import queries as queries_data
from .ExplanationsUtils import _PREDICTOR_TYPE, _TABLE_NAME, _COL
from .ExplanationsReportGenerator import ExplanationsReportGenerator

logger = logging.getLogger(__name__)


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
        root_dir: str = ".tmp",
        data_folder: str = "explanations_data",
        output_folder: str = "aggregated_data",
        model_name: str = "",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        progress_bar: bool = False,
        batch_limit: int = 10,
        memory_limit: int = 2,
        thread_count: int = 4,
    ):
        self.conn = None
        self.model_name = model_name

        self.from_date = None
        self.to_date = None
        self._set_date_range(from_date, to_date)

        self.data_folder = data_folder

        self.batch_limit = batch_limit
        self.memory_limit = memory_limit
        self.thread_count = thread_count
        self.progress_bar = progress_bar

        self.root_dir = root_dir
        self.output_folder = output_folder

        self.selected_files: list[str] = []
        self._populate_selected_files()

        self.contexts = None

    def generate_report(
        self,
        data_folder: Optional[str] = None,
        report_dir: str = "reports",
        report_output_dir: str = "_site",
        report_filename: str = "explanations_report.zip",
        top_n: int = 10,
        top_k: int = 10,
        zip_output: bool = True,
        verbose: bool = False,
    ):
        data_folder = data_folder or self.output_folder

        report = ExplanationsReportGenerator(
            root_dir=self.root_dir,
            data_folder=data_folder,
            report_dir=report_dir,
            report_output_dir=report_output_dir,
            report_filename=report_filename,
            top_n=top_n,
            top_k=top_k,
            zip_output=zip_output,
            verbose=verbose,
        )
        report.process()

    def generate_aggregations(self, overwrite: bool = True):
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

        if not overwrite:
            return

        self._overwrite_folder()
        self.conn = duckdb.connect(database=":memory:")

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

    def _overwrite_folder(self):
        output_path = pathlib.Path(os.path.join(self.root_dir, self.output_folder))
        if output_path.exists() and output_path.is_dir():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    def _set_date_range(
        self, from_date: Optional[datetime], to_date: Optional[datetime], days: int = 7
    ):
        """Set the date range for processing explanation files.

        Parameters
        ----------
        start_date : datetime, optional
            The start date for the date range. If None, defaults to 7 days before end_date.
        end_date : datetime, optional
            The end date for the date range. If None, defaults to today.
        """
        if from_date is None and to_date is None:
            to_date = datetime.today()
            from_date = to_date - timedelta(days=days)

        # if only `to_date` is provided, set `from_date` to 7 days before `to_date`
        if from_date is None and to_date is not None:
            from_date = to_date - timedelta(days=days)

        # if only `from_date` is provided, set `to_date` to today
        # it can process from any date until today. eg: from_date = 2023-01-01, to_date = today
        if from_date is not None and to_date is None:
            to_date = datetime.today()

        # validate date range if both from_date and to_date are provided
        if from_date is not None and to_date is not None:
            if from_date > to_date:
                raise ValueError("from_date cannot be after to_date")
            # if (to_date - from_date).days > 30:
            #     raise ValueError("Date range cannot be more than 30 days")

        self.from_date = from_date
        self.to_date = to_date

    def _populate_selected_files(self):
        if self.from_date is None or self.to_date is None:
            raise ValueError(
                "Either from_date or to_date must be passed before populating selected files."
            )

        # get list of dates in the range from `from_date` to `to_date`
        date_range_list = [
            (self.from_date + timedelta(days=x)).strftime("%Y%m%d")
            for x in range((self.to_date - self.from_date).days + 1)
        ]

        logger.debug(
            f"Searching for files for model {self.model_name} from {self.from_date} to {self.to_date}"
        )
        files_ = []
        for date in date_range_list:
            file_pattern = f"{self.data_folder}/{self.model_name}*{date}*.parquet"
            file_paths = glob(file_pattern)
            if file_paths:
                # Get the latest file based on modification time
                # incase of multiple runs on the same day, we want the latest file on the day
                latest_file = max(
                    file_paths, key=lambda x: pathlib.Path(x).stat().st_mtime
                )
                if pathlib.Path(latest_file).exists():
                    files_.append(latest_file)

        logger.info(f"Selected files:= \n {files_}")
        self.selected_files = files_

    def _get_selected_files(self):
        if len(self.selected_files) == 0:
            self._populate_selected_files()

        q = ", ".join([f"'{x}'" for x in self.selected_files])
        return q

    def _write_to_parquet(self, df: pl.DataFrame, file_name: str):
        df.write_parquet(
            f"{os.path.join(self.root_dir, self.output_folder)}/{file_name}",
            statistics=False,
        )

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
        sql = (
            files(queries_data).joinpath(f"{_TABLE_NAME.CREATE.value}.sql").read_text()
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
                SELECT {table_name.value}.{_COL.PARTITON.value}
                FROM {table_name.value}
                GROUP BY {table_name.value}.{_COL.PARTITON.value};
            """
            self.contexts = self.conn.execute(q).pl()[_COL.PARTITON.value].to_list()

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
                    {self.LEFT_PREFIX}.{_COL.PARTITON.value} IN ({self.SEP.join([f"'{row}'" for row in batch])})
                """)

                query = self._get_batch_sql_formatted(sql, table_name, where_condition)

                df = self.conn.execute(query).pl()
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
        if df is None:
            logger.error(f"No data found for predictor type={predictor_type}")
            return
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

    def _build_where(
        self,
        context_keys: Optional[dict] = None,
        predictor_names: Optional[list] = None,
    ):
        context_keys = context_keys or {}
        predictor_names = predictor_names or []

        where_condition = ""
        for context_key in context_keys.keys():
            if len(where_condition) > 0:
                where_condition += " AND "
            where_condition += f"""
                    {self.LEFT_PREFIX}.{context_key} IN (
                        {self.SEP.join([f"'{v}'" for v in context_keys.get(context_key)])}
                    )"""

        if len(predictor_names) > 0:
            if len(where_condition) > 0:
                where_condition += " AND "
            where_condition += f"""
                    {self.LEFT_PREFIX}.predictor_name IN ({self.SEP.join([f"'{predictor_name}'" for predictor_name in predictor_names])})"""

        return self._clean_query(where_condition)
