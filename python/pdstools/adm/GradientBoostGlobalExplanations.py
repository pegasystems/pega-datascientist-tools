__all__  = ['GradientBoostGlobalExplanations']

import pathlib
import duckdb
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import yaml

from datetime import datetime, timedelta
import shutil

class PREDICTOR_TYPE:
    NUMERIC = 'NUMERIC'
    SYMBOLIC = 'SYMBOLIC'
    
class GradientBoostGlobalExplanations:
    SEP = ', '
    LEFT_PREFIX = 'l'
    RIGHT_PREFIX = 'r'
    
    CREATE_TABLE_QUERY_FILE = "create_table"
    NUM_QUERY_PER_TREATMENT = "num_data"
    SYM_QUERY_PER_TREATMENT = "sym_data"
    NUM_QUERY_OVERALL = "num_overview"
    SYM_QUERY_OVERALL = "sym_overview"
    
    NUM_TBL_NAME = "num_data"
    SYM_TBL_NAME = "sym_data"

    def __init__(
         self
        ,nb_days: int = 7
        ,files_location: str = None
        ,batch_limit: int = 10
        ,memory_limit: int = 2
        ,thread_count: int = 4
        ,model_level_identifier: str = 'model'
        ,queries_folder: str = 'resources/queries/explanations'
        ,output_folder: str = 'out'
        ,output_file: str = 'output.parquet'
        ,progress_bar: bool = False
        ,redo: str = None  # NUMERIC or SYMBOLIC
    ):
        self.conn = duckdb.connect(database=':memory:')
        self.nb_days = nb_days
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
        
        self.selected_files = self.populate_selected_files()
        
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
            to_delete = pathlib.Path(self.output_folder).glob(f'{redo}_*.parquet')
            for file_n in to_delete:
                file_n.unlink()
            
    def populate_selected_files(self):
        def get_last_n_days(n):
            today = datetime.today()
            last_n_days = [(today - timedelta(days=i)).strftime('%Y%m%d') for i in range(n)]
            return last_n_days
        
        files_ = []
        for day in get_last_n_days(self.nb_days):
            file_path = f'{self.files_location}/{day}.parquet'
            if pathlib.Path(file_path).exists():
                files_.append(f'{self.files_location}/{day}.parquet')
        print(f'Selected files:= {files_}')
        return files_
        
    def get_selected_files(self):
        if len(self.selected_files) == 0:
            self.selected_files = self.populate_selected_files()
            
        q = ', '.join([f"'{x}'" for x in self.selected_files])
        return q
        
    def write_to_parquet(self, df, file_name=None):
        tbl = pa.Table.from_pandas(df)
        pq.write_to_dataset(tbl, root_path=self.output_folder
                            ,basename_template=file_name
                            ,use_dictionary=False, write_statistics=False)
            
    @staticmethod    
    def clean_query(query):
        q = query.replace("\n", " ")
        
        while "  " in q:
            q = q.replace("  ", " ")
        
        return q
    
    def get_table_name(self, predictor_type):
        return self.NUM_TBL_NAME if predictor_type == PREDICTOR_TYPE.NUMERIC else self.SYM_TBL_NAME
    
    def read_overall_sql_file(self, predictor_type):
        sql_file = self.NUM_QUERY_OVERALL if predictor_type == PREDICTOR_TYPE.NUMERIC else self.SYM_QUERY_OVERALL
        with open(f'{self.queries_folder}/{sql_file}.sql', 'r') as fr:
            return fr.read()
        
    def read_batch_sql_file(self, predictor_type):
        sql_file = self.NUM_QUERY_PER_TREATMENT if predictor_type == PREDICTOR_TYPE.NUMERIC else self.SYM_QUERY_PER_TREATMENT
        with open(f'{self.queries_folder}/{sql_file}.sql', 'r') as fr:
            return fr.read()
        
    def read_create_table_sql_file(self):
        with open(f'{self.queries_folder}/{self.CREATE_TABLE_QUERY_FILE}.sql', 'r') as fr:
            return fr.read()
    
    def get_create_table_sql_formatted(self, sql, tbl_name, predictor_type):
        f_sql = f"{sql.format(
             MEMORY_LIMIT=self.memory_limit
            ,ENABLE_PROGRESS_BAR="true" if self.progress_bar else "false"
            ,TABLE_N=tbl_name
            ,SELECTED_FILES=self.get_selected_files()
            ,PREDICTOR_TYPE=predictor_type
        )}"
        
        return self.clean_query(f_sql)
     
    def get_overall_sql_formatted(self, sql, tbl_name):
        f_sql = f"{sql.format(
             THREAD_COUNT=self.thread_count
            ,MEMORY_LIMIT=self.memory_limit
            ,LEFT_PREFIX=self.LEFT_PREFIX
            ,RIGHT_PREFIX=self.RIGHT_PREFIX
            ,ENABLE_PROGRESS_BAR="true" if self.progress_bar else "false"
            ,TABLE_N=tbl_name
            ,MODEL_LEVEL_ID=self.model_level_identifier
            )}"
        
        return self.clean_query(f_sql)
        
    def get_batch_sql_formatted(self, sql, tbl_name, where_condition):
        f_sql = f"{sql.format(
             THREAD_COUNT=self.thread_count
            ,MEMORY_LIMIT=self.memory_limit
            ,LEFT_PREFIX=self.LEFT_PREFIX
            ,RIGHT_PREFIX=self.RIGHT_PREFIX
            ,ENABLE_PROGRESS_BAR="true" if self.progress_bar else "false"
            ,TABLE_N=tbl_name
            ,WHERE_CONDITION=where_condition
            )}"
        
        return self.clean_query(f_sql)
        
    def create_in_mem_table(self, predictor_type=None):
        tbl_name = self.get_table_name(predictor_type)
        
        sql = self.read_create_table_sql_file()
        query = self.get_create_table_sql_formatted(sql, tbl_name, predictor_type)
        
        self.conn.execute(query)
        
    def delete_in_mem_table(self, predictor_type):
        tbl_name = self.get_table_name(predictor_type)
        
        query = f"""
            DROP TABLE {tbl_name};
        """
        
        self.conn.execute(query)
        
    def get_contexts(self, predictor_type):
        
        tbl_name = self.get_table_name(predictor_type)
        
        if self.contexts is None:
            q = f"""
                SELECT context_keys
                FROM {tbl_name}
                GROUP BY context_keys;
            """
            self.contexts = self.conn.execute(q).df()['context_keys'].to_list()
        
        return self.contexts
    
    def agg_in_batches(self, predictor_type):
        
        try:
            TABLE_N = self.get_table_name(predictor_type)
            
            group_list = self.get_contexts(predictor_type)
            
            total_groups = len(group_list)
            curr_group = 0
            
            sql = self.read_batch_sql_file(predictor_type)
            
            batch_counter = 1
            
            df_list = []
            
            while curr_group < total_groups:
                batch = group_list[curr_group:min(curr_group+self.batch_limit, total_groups)]
                
                WHERE_CONDITION = self.clean_query(f"""
                    {self.LEFT_PREFIX}.context_keys IN ({self.SEP.join([f"'{row}'" for row in batch])})
                """)
                
                query = self.get_batch_sql_formatted(sql, TABLE_N, WHERE_CONDITION)
                
                df = self.conn.execute(query).df()
                df_list.append(df)
                
                if batch_counter % 10 == 0:
                    self.write_to_parquet(pd.concat(df_list), f'{predictor_type}_BATCH_{batch_counter}_{{i}}.parquet')
                    df_list = []
                    print(f"Processed batch {batch_counter}, group: {curr_group}, len: {len(batch)}")
                
                curr_group+=self.batch_limit
                batch_counter+=1
                
        except Exception as e:
            print(f'Failed for predictor type={predictor_type}, err={e}')
            return
        
    def agg_overall(self, predictor_type):
        try:
            sql = self.read_overall_sql_file(predictor_type)
            
            TABLE_N = self.get_table_name(predictor_type)
            query = self.get_overall_sql_formatted(sql, TABLE_N)
            
            df = self.conn.execute(query).df()
            
            self.write_to_parquet(df, f'{predictor_type}_OVERALL_{{i}}.parquet')
            
        except Exception as e:
            print(f'Failed for predictor type={predictor_type}, err={e}')
            return
        
    def run_agg(self, predictor_type):
        
        try:
            self.create_in_mem_table(predictor_type)
            
            self.agg_in_batches(predictor_type)
            
            self.agg_overall(predictor_type)
            
            self.delete_in_mem_table(predictor_type)      
        except Exception as e:
            print(f'Failed for predictor type={predictor_type}, err={e}')
            return
        
    def process(self):
        self.get_selected_files()
        if len(self.selected_files) == 0:
            print('No files found to aggregate!')
            return
            
        try:
            if self.redo is None or self.redo == PREDICTOR_TYPE.NUMERIC:
                self.run_agg(PREDICTOR_TYPE.NUMERIC)
        except Exception as e:
            print(f'Failed to aggregate numeric data, err={e}')
            self.conn.close()
            exit(1)
        
        try:
            if self.redo is None or self.redo == PREDICTOR_TYPE.SYMBOLIC:
                self.run_agg(PREDICTOR_TYPE.SYMBOLIC)
        except Exception as e:
            print(f'Failed to aggregate symbolic data, err={e}')
            self.conn.close()
            exit(1)
            
        self.conn.close()
            