__all__ = ["GradientBoostGlobalExplanations"]

import json
import pathlib
import shutil
from datetime import datetime, timedelta
from enum import Enum
from glob import glob

import duckdb
import ipywidgets as widgets
import plotly.graph_objects as go
import polars as pl
from IPython.display import display


class PREDICTOR_TYPE(Enum):
    NUMERIC: str = "NUMERIC"
    SYMBOLIC: str = "SYMBOLIC"

class TABLE_NAME(Enum):
    NUMERIC = 'numeric'
    SYMBOLIC = 'symbolic'
    NUMERIC_OVERALL = 'numeric_overall'
    SYMBOLIC_OVERALL = 'symbolic_overall'
    CREATE = "create"

class GradientBoostGlobalExplanations:
    SEP = ", "
    LEFT_PREFIX = "l"
    RIGHT_PREFIX = "r"


    def __init__(
        self,
        model_name: str = "",
        nb_days: int = 7,
        start_date: datetime = datetime.today(),
        files_location: str = None,
        batch_limit: int = 10,
        memory_limit: int = 2,
        thread_count: int = 4,
        model_level_identifier: str = "model",
        queries_folder: str = "resources/queries/explanations",
        output_folder: str = "out",
        output_file: str = "output.parquet",
        progress_bar: bool = False,
        redo: PREDICTOR_TYPE = None,
    ):
        self.conn = duckdb.connect(database=":memory:")
        self.nb_days = nb_days
        self.model_name = model_name
        self.start_date = start_date

        self.files_location = files_location

        self.batch_limit = batch_limit
        self.memory_limit = memory_limit
        self.thread_count = thread_count

        self.model_level_identifier = model_level_identifier

        basePath = pathlib.Path().resolve().parent.parent

        self.queries_folder = f"{basePath}/{queries_folder}"
        self.output_folder = output_folder
        self.output_file = output_file
        self.progress_bar = progress_bar
        self.redo = redo

        self.selected_files = self._populate_selected_files()

        self.contexts = None

        if redo is None:
            # remove output folder if exists
            output_path = pathlib.Path(self.output_folder)
            if output_path.exists() and output_path.is_dir():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            # remove output file if exists
            output_file_path = pathlib.Path(self.output_file)
            if output_file_path.exists():
                output_file_path.unlink()
        else:
            to_delete = pathlib.Path(self.output_folder).glob(f"{redo}_*.parquet")
            for file_n in to_delete:
                file_n.unlink()

    def _populate_selected_files(self):
        def get_last_n_days(n):
            last_n_days = [
                (self.start_date - timedelta(days=i)).strftime("%Y%m%d")
                for i in range(n)
            ]
            return last_n_days

        files_ = []
        for day in get_last_n_days(self.nb_days):
            file_pattern = f"{self.files_location}/{self.model_name}*{day}*.parquet"
            file_paths = glob(file_pattern)
            for file_path in file_paths:
                if pathlib.Path(file_path).exists():
                    files_.append(file_path)
        print(f"Selected files:= {files_}")
        return files_

    def _get_selected_files(self):
        if len(self.selected_files) == 0:
            self.selected_files = self._populate_selected_files()

        q = ", ".join([f"'{x}'" for x in self.selected_files])
        return q

    def _write_to_parquet(self, df: pl.DataFrame, file_name: str):
        df.write_parquet(
            f'{self.output_folder}/{file_name}',
            statistics=False
        )

    @staticmethod
    def _clean_query(query):
        q = query.replace("\n", " ")

        while "  " in q:
            q = q.replace("  ", " ")

        return q

    def _get_table_name(self, predictor_type) -> TABLE_NAME:
        return (
            TABLE_NAME.NUMERIC
            if predictor_type == PREDICTOR_TYPE.NUMERIC
            else TABLE_NAME.SYMBOLIC
        )

    def _read_overall_sql_file(self, predictor_type: PREDICTOR_TYPE):
        sql_file = (
            TABLE_NAME.NUMERIC_OVERALL
            if predictor_type == PREDICTOR_TYPE.NUMERIC
            else TABLE_NAME.SYMBOLIC_OVERALL
        )
        with open(f"{self.queries_folder}/{sql_file.value}.sql", "r") as fr:
            return fr.read()

    def _read_batch_sql_file(self, predictor_type: PREDICTOR_TYPE):
        sql_file = (    
            TABLE_NAME.NUMERIC
            if predictor_type == PREDICTOR_TYPE.NUMERIC
            else TABLE_NAME.SYMBOLIC
        )
        with open(f"{self.queries_folder}/{sql_file.value}.sql", "r") as fr:
            return fr.read()

    def _get_create_table_sql_formatted(self, tbl_name:TABLE_NAME, predictor_type: PREDICTOR_TYPE):
        with open(
            f"{self.queries_folder}/{TABLE_NAME.CREATE.value}.sql", "r"
        ) as fr:
            sql = fr.read()

        f_sql = f"{
            sql.format(
                MEMORY_LIMIT=self.memory_limit,
                ENABLE_PROGRESS_BAR='true' if self.progress_bar else 'false',
                TABLE_NAME=tbl_name.value,
                SELECTED_FILES=self._get_selected_files(),
                PREDICTOR_TYPE=predictor_type.value,
            )
        }"

        return self._clean_query(f_sql)

    def _get_overall_sql_formatted(self, sql, tbl_name: TABLE_NAME, where_condition):
        f_sql = f"{
            sql.format(
                THREAD_COUNT=self.thread_count,
                MEMORY_LIMIT=self.memory_limit,
                LEFT_PREFIX=self.LEFT_PREFIX,
                RIGHT_PREFIX=self.RIGHT_PREFIX,
                ENABLE_PROGRESS_BAR='true' if self.progress_bar else 'false',
                TABLE_NAME=tbl_name.value,
                MODEL_LEVEL_ID=self.model_level_identifier,
                WHERE_CONDITION=where_condition,
            )
        }"

        return self._clean_query(f_sql)

    def _get_batch_sql_formatted(self, sql, tbl_name: TABLE_NAME, where_condition="TRUE"):
        f_sql = f"{
            sql.format(
                THREAD_COUNT=self.thread_count,
                MEMORY_LIMIT=self.memory_limit,
                LEFT_PREFIX=self.LEFT_PREFIX,
                RIGHT_PREFIX=self.RIGHT_PREFIX,
                ENABLE_PROGRESS_BAR='true' if self.progress_bar else 'false',
                TABLE_NAME=tbl_name.value,
                WHERE_CONDITION=where_condition,
            )
        }"

        return self._clean_query(f_sql)

    def _create_in_mem_table(self, predictor_type: PREDICTOR_TYPE, table_name: TABLE_NAME=None):
        table_name = self._get_table_name(predictor_type)
        query = self._get_create_table_sql_formatted(table_name, predictor_type)

        self.conn.execute(query)

    def _delete_in_mem_table(self, predictor_type: PREDICTOR_TYPE):
        table_name = self._get_table_name(predictor_type)

        query = f"""
            DROP TABLE {table_name.value};
        """

        self.conn.execute(query)

    def _get_contexts(self, predictor_type: PREDICTOR_TYPE):
        table_name = self._get_table_name(predictor_type)

        if self.contexts is None:
            q = f"""
                SELECT {table_name.value}.context_keys
                FROM {table_name.value}
                GROUP BY {table_name.value}.context_keys;
            """
            self.contexts = self.conn.execute(q).pl()["context_keys"].to_list()

        return self.contexts

    def agg_in_batches(self, predictor_type: PREDICTOR_TYPE):
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
                    {self.LEFT_PREFIX}.context_keys IN ({self.SEP.join([f"'{row}'" for row in batch])})
                """)

                query = self._get_batch_sql_formatted(sql, table_name, where_condition)

                df = self.conn.execute(query).pl()
                df_list.append(df)

                batch_counter += 1
                curr_group += self.batch_limit
                if batch_counter % 10 == 0:
                    self._write_to_parquet(
                        pl.concat(df_list),
                        f"{predictor_type.value}_BATCH_{batch_counter}.parquet",
                    )
                    df_list = []
                    print(
                        f"Processed {predictor_type} batch {batch_counter}, group: {curr_group}, len: {len(batch)}"
                    )

            if len(df_list) > 0:
                self._write_to_parquet(
                    pl.concat(df_list),
                    f"{predictor_type.value}_BATCH_{batch_counter}.parquet",
                )
                df_list = []
                print(
                    f"Processed {predictor_type} batch {batch_counter}, group: {curr_group}, len: {len(batch)}"
                )

        except Exception as e:
            print(f"Failed batch for predictor type={predictor_type}, err={e}")
            return

    def agg_overall(self, predictor_type: PREDICTOR_TYPE, where_condition="TRUE"):
        try:
            sql = self._read_overall_sql_file(predictor_type)

            table_name = self._get_table_name(predictor_type)
            query = self._get_overall_sql_formatted(sql, table_name, where_condition)

            df = self.conn.execute(query).pl()

            self._write_to_parquet(df, f"{predictor_type.value}_OVERALL.parquet")

        except Exception as e:
            print(f"Failed overall for predictor type={predictor_type}, err={e}")
            return

    def parquet_overall(self, predictor_type, where_condition="TRUE"):
        try:
            sql = self._read_overall_sql_file(predictor_type)

            table_name = self._get_table_name(predictor_type)
            query = self._get_overall_sql_formatted(sql, table_name, where_condition)

            return self.conn.execute(query).pl()

        except Exception as e:
            print(f"Failed for predictor type={predictor_type}, err={e}")
            return

    def run_agg(self, predictor_type: PREDICTOR_TYPE):
        try:
            self._create_in_mem_table(predictor_type)

            self.agg_in_batches(predictor_type)

            self.agg_overall(predictor_type)

            self._delete_in_mem_table(predictor_type)
        except Exception as e:
            print(f"Failed for predictor type={predictor_type}, err={e}")
            return

    def run(self, predictor_type, where_condition):
        try:
            self._create_in_mem_table(predictor_type)

            df = self.parquet_overall(predictor_type, where_condition)

            self._delete_in_mem_table(predictor_type)

            return df
        except Exception as e:
            print(f"Failed for predictor type={predictor_type}, err={e}")
            return

    def process(self):
        self._get_selected_files()
        if len(self.selected_files) == 0:
            print("No files found to aggregate!")
            return

        try:
            if self.redo is None or self.redo == PREDICTOR_TYPE.NUMERIC:
                self.run_agg(PREDICTOR_TYPE.NUMERIC)
        except Exception as e:
            print(f"Failed to aggregate numeric data, err={e}")
            self.conn.close()
            exit(1)

        try:
            if self.redo is None or self.redo == PREDICTOR_TYPE.SYMBOLIC:
                self.run_agg(PREDICTOR_TYPE.SYMBOLIC)
        except Exception as e:
            print(f"Failed to aggregate symbolic data, err={e}")
            self.conn.close()
            exit(1)

        self.conn.close()

    def process_subset(
        self, context_keys: dict = dict(), predictor_names: list = list()
    ):
        self._get_selected_files()
        if len(self.selected_files) == 0:
            print("No files found to aggregate!")
            return

        where_condition: str = self._build_where(context_keys, predictor_names)

        dfs_numeric = []
        try:
            if self.redo is None or self.redo == PREDICTOR_TYPE.NUMERIC:
                dfs_numeric.append(self.run(PREDICTOR_TYPE.NUMERIC, where_condition))
        except Exception as e:
            print(f"Failed to aggregate numeric data, err={e}")
            self.conn.close()
            exit(1)

        dfs_symbolic = []

        try:
            if self.redo is None or self.redo == PREDICTOR_TYPE.SYMBOLIC:
                dfs_symbolic.append(self.run(PREDICTOR_TYPE.SYMBOLIC, where_condition))
        except Exception as e:
            print(f"Failed to aggregate symbolic data, err={e}")
            self.conn.close()
            exit(1)

        self.conn.close()

        return {"numeric": pl.concat(dfs_numeric), "symbolic": pl.concat(dfs_symbolic)}

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


class DataFilter:
    to_select = [
        "context_keys",
        "contribution",
        "contribution_abs",
        "frequency",
        "predictor_type",
        "predictor_name",
        "bin_contents",
        "bin_order",
    ]
    allowed_contexts = []
    data_updates = []

    def __init__(self, data_location):
        self.df_contextual = pl.concat(
            [
                pl.read_parquet(f"{data_location}/NUMERIC_BATCH_*.parquet")
                .select(self.to_select)
                .with_columns(
                    pl.when(pl.col("bin_order") == "MISSING")
                    .then(pl.lit("-1"))
                    .otherwise(pl.col("bin_order"))
                    .cast(pl.Int64)
                    .alias("bin_order")
                ),
                pl.read_parquet(f"{data_location}/SYMBOLIC_BATCH_*.parquet").select(
                    self.to_select
                ),
            ]
        ).sort(by="predictor_name")
        self.df_overall = pl.concat(
            [
                pl.read_parquet(f"{data_location}/NUMERIC_OVERALL.parquet")
                .select(self.to_select)
                .with_columns(
                    pl.when(pl.col("bin_order") == "MISSING")
                    .then(pl.lit("-1"))
                    .otherwise(pl.col("bin_order"))
                    .cast(pl.Int64)
                    .alias("bin_order")
                ),
                pl.read_parquet(f"{data_location}/SYMBOLIC_OVERALL.parquet").select(
                    self.to_select
                ),
            ]
        ).sort(by="predictor_name")

        self.context_df = pl.from_dicts(
            [
                json.loads(ck)["partition"]
                for ck in self.df_contextual["context_keys"].unique().to_list()
            ]
        )
        self.context_keys = self.context_df.columns

        self.init_widgets()

    def init_widgets(self):
        self.context_filters = []
        for context_key in self.context_keys:
            options = self.context_df[context_key].unique().to_list()
            context_filter = widgets.Combobox(
                placeholder=f"Select {context_key}",
                options=options,
                description=f"{context_key}",
                ensure_option=False,
                disabled=False,
            )
            context_filter.observe(self.results, names="value")
            self.context_filters.append(context_filter)

        self.filter_results = widgets.Text(
            description="Filtering For", disabled=True, value="Whole Model"
        )

        context_filter_container = widgets.VBox(self.context_filters)
        self.widget = widgets.HBox([context_filter_container, self.filter_results])

    def get_filtered_options(self):
        filtered_result = self.context_df
        for f in [item for item in self.context_filters if item.value != ""]:
            filtered_result = filtered_result.filter(pl.col(f.description) == f.value)

        for f in [item for item in self.context_filters]:
            f.options = filtered_result[f.description].unique().to_list()
            if len(f.options) == 1:
                f.value = f.options[0]

        self.allowed_contexts = filtered_result.to_dicts()

        if len(self.allowed_contexts) == 1:
            filter_string = f"{json.dumps(filtered_result.to_dicts()[0])}".replace(
                ", ", ","
            ).replace(": ", ":")
        else:
            filter_string = None
        return filter_string

    def update_filter_results(self, filter_string):
        if filter_string is None:
            self.filter_results.value = "Whole Model"
        else:
            self.filter_results.value = filter_string

    def results(self, value):
        filter_string = self.get_filtered_options()
        self.update_filter_results(filter_string)
        for data_update in self.data_updates:
            data_update(self)

    def add_to_data_update(self, update):
        self.data_updates.append(update)

    def display(self):
        return display(self.widget)

    def get_data(self):
        filter_string = self.filter_results.value
        if filter_string == "Whole Model":
            return self.df_overall, filter_string
        else:
            return self.df_contextual.filter(
                pl.col("context_keys").str.contains(filter_string, literal=True)
            ), filter_string

    def get_data_aggregate(self):
        filter_string = self.filter_results.value
        if filter_string == "Whole Model":
            df = self.df_overall
        else:
            df = self.df_contextual.filter(
                pl.col("context_keys").str.contains(filter_string, literal=True)
            )
        return df.group_by(["predictor_name", "predictor_type"]).agg(
            (
                pl.col("frequency")
                * pl.col("contribution_abs")
                / pl.col("frequency").sum()
            )
            .mean()
            .alias("|contribution_weighted|"),
            (pl.col("frequency") * pl.col("contribution") / pl.col("frequency").sum())
            .mean()
            .alias("contribution_weighted"),
            pl.col("contribution_abs").mean().alias("|contribution|"),
            pl.col("contribution").mean().alias("contribution"),
        ), filter_string


class OverAllPredictorContributions:
    _predictor_options = ["BOTH", "SYMBOLIC", "NUMERIC"]

    _predictor_option_drop_down_w = widgets.Dropdown(
        placeholder="Select predictor types",
        options=_predictor_options,
        description="Predictor types",
        ensure_option=True,
        disabled=False,
        value=_predictor_options[0],
    )

    _weighted_mean_w = widgets.Checkbox(
        value=True, description="Weight average by frequency", disabled=False
    )

    _absolute_mean_w = widgets.Checkbox(
        value=False, description="Use absolute contribtuion", disabled=False
    )

    _limit_w = widgets.IntSlider(
        min=1,
        max=100,
        step=1,
        description="Limit",
        ensure_option=True,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        value=10,
    )

    _title_w = widgets.HTML(
        value="<h3>Context: Whole Model</h3>",
        placeholder="Context Information",
    )

    _fig_w = go.FigureWidget()

    def __init__(self, data, title_context):
        self.title_context = title_context
        self.data = data
        self.init_widgets()

    def update_data(self, data):
        self.data = data
        self.init_widgets()

    def update_dataframe(self):
        if self._predictor_option_drop_down_w.value == "BOTH":
            df_filtered = self.data
        else:
            df_filtered = self.data.filter(
                pl.col("predictor_type") == self._predictor_option_drop_down_w.value
            )
        return df_filtered

    def update_figure(self, df):
        if self._weighted_mean_w.value:
            column = "contribution_weighted"
        else:
            column = "contribution"

        if self._absolute_mean_w.value:
            column = "|" + column + "|"

        df = df.sort([pl.col(column).abs(), "predictor_name"], descending=True).limit(
            self._limit_w.value
        )

        with self._fig_w.batch_update():
            self._fig_w.add_trace(go.Bar(orientation="h"))
            self._fig_w.update_layout(
                title=f"Avg {'absolute ' if self._absolute_mean_w.value else ''}{'' if self._predictor_option_drop_down_w.value == 'BOTH' else self._predictor_option_drop_down_w.value.lower() + ' '}predictor contribution",
                xaxis={"title": {"text": column}},
                yaxis={"title": {"text": "Predictor name"}},
                height=max(400, 20 * df.shape[0]),
            )

            self._title_w.value = f"<h3>Context: {self.title_context}<h3>"

            self._fig_w.data[0].x = df[column]
            self._fig_w.data[0].y = df["predictor_name"]
            self._fig_w.data[0].marker = {
                "color": df[column],
                "colorscale": "RdBu",
                "cmid": 0.0,
            }

    def data_update(self, data_filter: DataFilter):
        new_data, self.title_context = data_filter.get_data_aggregate()
        self.update_data(new_data)

    def response(self, change):
        resulting_df = self.update_dataframe()
        self.update_figure(resulting_df)

    def init_widgets(self):
        self._predictor_option_drop_down_w.observe(self.response, names="value")
        self._weighted_mean_w.observe(self.response, names="value")
        self._absolute_mean_w.observe(self.response, names="value")
        self._limit_w.observe(self.response, names="value")

        selectors = widgets.HBox([self._predictor_option_drop_down_w, self._limit_w])
        check_boxes = widgets.HBox([self._weighted_mean_w, self._absolute_mean_w])
        self.widget = widgets.VBox([self._title_w, selectors, check_boxes, self._fig_w])

        self.response("init")

    def display(self):
        display(self.widget)


class OverAllPredictorContributionsByValue:
    _sort_options = ["contribution", "frequency", "contibution weighted by freq"]

    _sort_by_w = widgets.Dropdown(
        placeholder="Select sort scheme",
        options=_sort_options,
        description="Sort by",
        ensure_option=True,
        disabled=False,
        value=_sort_options[0],
    )

    _limit_w = widgets.IntSlider(
        min=1,
        max=100,
        step=1,
        description="Limit",
        ensure_option=True,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        value=10,
    )

    _show_remaining_w = widgets.Checkbox(
        value=True, description="Aggregate remaining", disabled=False
    )

    _show_missing_w = widgets.Checkbox(
        value=False, description="Force missing", disabled=False
    )

    _title_w = widgets.HTML(
        value="<h3>Context: Whole Model</h3>",
        placeholder="Context Information",
    )

    _fig_w = go.FigureWidget()

    def __init__(self, data, title_context):
        self.title_context = title_context
        self._to_select = [
            "contribution",
            "contribution_abs",
            "frequency",
            "predictor_type",
            "predictor_name",
            "bin_contents",
            "bin_order",
            "context_keys",
        ]
        self.init_data(data)

    def init_data(self, data):
        self.data = data
        self._predictor_names = data["predictor_name"].unique().to_list()
        self.init_widgets()

    def update_data(self, data):
        self.data = data
        self._predictor_names = data["predictor_name"].unique().to_list()
        self.update_widgets()

    def update_dataframe(self, predictor_is_symbolic):
        if predictor_is_symbolic:
            if self._sort_by_w.value == "frequency":
                sorted_df = self.data.filter(
                    pl.col("predictor_name") == self._predictor_w.value
                ).sort(by="frequency", descending=True)
            elif self._sort_by_w.value == "contribution":
                sorted_df = self.data.filter(
                    pl.col("predictor_name") == self._predictor_w.value
                ).sort(
                    by=[pl.col("contribution").abs(), "predictor_name"], descending=True
                )
            elif self._sort_by_w.value == "contibution weighted by freq":
                sorted_df = self.data.filter(
                    pl.col("predictor_name") == self._predictor_w.value
                ).sort(
                    by=[
                        pl.col("frequency") * pl.col("contribution").abs(),
                        "predictor_name",
                    ],
                    descending=True,
                )

            top_n = sorted_df.limit(self._limit_w.value).select(
                ["contribution", "frequency", "bin_contents"]
            )

            if self._show_remaining_w.value:
                remaining = (
                    sorted_df.slice(self._limit_w.value)
                    .filter(
                        (pl.col("bin_contents") != "MISSING")
                        | ~pl.lit(self._show_missing_w.value)
                    )
                    .select(
                        [
                            pl.mean("contribution"),
                            pl.sum("frequency"),
                            pl.lit("REMAINING").alias("bin_contents"),
                        ]
                    )
                )
                resulting_df = pl.concat([top_n, remaining])
            else:
                resulting_df = top_n

            if (
                self._show_missing_w.value
                and resulting_df.filter(pl.col("bin_contents") == "MISSING")
                .select(pl.len())
                .item()
                == 0
            ):
                resulting_df = pl.concat(
                    [
                        resulting_df,
                        sorted_df.filter(pl.col("bin_contents") == "MISSING").select(
                            ["contribution", "frequency", "bin_contents"]
                        ),
                    ]
                )
            return resulting_df
        else:
            return self.data.filter(
                pl.col("predictor_name") == self._predictor_w.value
            ).sort(pl.col("bin_order"), descending=False)

    def is_symbolic(self):
        predictor_type = (
            self.data.filter(pl.col("predictor_name") == self._predictor_w.value)
            .select("predictor_type")
            .unique()
            .item()
        )

        is_symbolic = predictor_type == "SYMBOLIC"
        self._sort_by_w.disabled = not is_symbolic
        self._show_remaining_w.disabled = not is_symbolic
        self._show_missing_w.disabled = not is_symbolic
        self._limit_w.disabled = not is_symbolic

        return is_symbolic

    def update_figure(self, df):
        with self._fig_w.batch_update():
            self._fig_w.add_trace(go.Bar(orientation="h"))
            self._fig_w.update_layout(
                title=f"Avg contribution for {self._predictor_w.value}",
                xaxis={"title": {"text": "contribution"}},
                yaxis={"title": {"text": "Predictor value"}},
                height=max(400, 20 * df.shape[0]),
            )

            self._title_w.value = f"<h3>Context: {self.title_context}</h3>"

            self._fig_w.data[0].x = df["contribution"]
            self._fig_w.data[0].y = df["bin_contents"]
            self._fig_w.data[0].marker = {
                "color": df["contribution"],
                "colorscale": "RdBu",
                "cmid": 0.0,
            }

    def response(self, change):
        predictor_is_symbolic = self.is_symbolic()
        resulting_df = self.update_dataframe(predictor_is_symbolic)
        self.update_figure(resulting_df)

    def data_update(self, data_filter: DataFilter):
        new_data, self.title_context = data_filter.get_data()
        self.update_data(new_data)

    def update_widgets(self):
        self._predictor_w.observe(self.response, names="value")
        self._predictor_w.options = self._predictor_names
        self._sort_by_w.observe(self.response, names="value")
        self._limit_w.observe(self.response, names="value")
        self._show_remaining_w.observe(self.response, names="value")
        self._show_missing_w.observe(self.response, names="value")

        selectors = widgets.HBox([self._predictor_w, self._sort_by_w, self._limit_w])
        check_boxes = widgets.HBox([self._show_remaining_w, self._show_missing_w])
        self.widget = widgets.VBox([self._title_w, selectors, check_boxes, self._fig_w])

        self.response("init")

    def init_widgets(self):
        self._predictor_w = widgets.Combobox(
            placeholder="Select Predictor",
            options=self._predictor_names,
            description="Predictor",
            ensure_option=True,
            disabled=False,
            value=self._predictor_names[0],
        )
        self.update_widgets()

    def display(self):
        display(self.widget)


class PredictorContributionExplorer:
    filter_head = widgets.HTML(
        value="<h2>Context Filter</h2>",
    )

    graph_head = widgets.HTML(
        value="<h2>Contribution Graphs</h2>",
    )

    def __init__(self):
        self.data_filter = DataFilter("out")
        self.over_all_widget = OverAllPredictorContributions(
            *self.data_filter.get_data_aggregate()
        )
        self.by_value_widget = OverAllPredictorContributionsByValue(
            *self.data_filter.get_data()
        )
        self.data_filter.add_to_data_update(self.over_all_widget.data_update)
        self.data_filter.add_to_data_update(self.by_value_widget.data_update)

        self.tabs = widgets.Tab()
        self.tabs.children = [self.over_all_widget.widget, self.by_value_widget.widget]
        self.tabs.titles = ["Overall Contribution", "Contribution by Value"]
        self.widget = widgets.VBox(
            [self.filter_head, self.data_filter.widget, self.graph_head, self.tabs]
        )

    def display(self):
        display(self.widget)
