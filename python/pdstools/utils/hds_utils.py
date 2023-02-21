import os
import re
import glob
import yaml

import polars as pl

from typing import Literal
from pdstools import ADMDatamart
from random import randint


class Config:

    def __init__(self, config_file=None):

        self.config_file = "config.yml" if config_file is None else config_file

        def str2bool(inp):
            return True if inp.upper() == "Y" else False

        def remove_(inp):
            if inp is None:
                return inp
            return inp if not inp.endswith("/") else re.sub(r".$", "", inp)

        self.data = self.validate_config_file()

        dat = self.get_key("Predictors")
        self.mask_predictor_names = str2bool(dat.get('maskPredictorNames', "Y"))
        self.mask_predictor_values = str2bool(dat.get('maskPredictorValues', "Y"))

        # no anonymization done on these predictors, output is same as input values
        self.special_predictors = [x.strip() for x in
                                   dat.get('specialPredictors', "Decision_DecisionTime,Decision_OutcomeTime").split(
                                       ',')]

        dat = self.get_key("ContextKeys")
        self.mask_context_key_names = str2bool(dat.get('maskContextKeyNames', "Y"))
        self.mask_context_key_values = str2bool(dat.get('maskContextKeyValues', "Y"))
        self.context_key_predictors = dat.get('contextKeyPredictors', "Context_*")

        dat = self.get_key("IHPredictors")
        self.mask_ih_predictor_names = str2bool(dat.get('maskIHPredictorNames', "Y"))
        self.mask_ih_predictor_values = str2bool(dat.get('maskIHPredictorValues', "Y"))
        self.ih_predictors = dat.get('ihPredictors', "IH_*")

        dat = self.get_key("OutcomeColumn")
        self.mask_outcome_name = str2bool(dat.get('maskOutcomeName', "Y"))
        self.mask_outcome_values = str2bool(dat.get('maskOutcomeValues', "Y"))
        self.outcome_accepts = [x.strip() for x in dat.get('outcomeAccepts', "Accepted").split(',')]
        self.outcome_rejects = [x.strip() for x in dat.get('outcomeRejects', "Rejected").split(',')]
        self.outcome_column = dat.get('outcomeColumn', "Decision_Outcome")

        dat = self.get_key("FilePaths")
        self.output_folder = remove_(dat.get('outputFolder', "output"))
        self.datamart_folder = remove_(dat.get('datamartFolder', "datamart"))
        self.hdr_folder = remove_(dat.get('hdrFolder', None))
        self.hash_mapping_folder = f'{self.output_folder}/hash_map'
        self.numeric_mapping_folder = f'{self.output_folder}/numeric_map'

        self.use_datamart = str2bool(dat.get('useDatamart', "N"))
        self.output_format = dat.get("outputFormat", "ndjson")

        self.mapping_filename = f'{self.output_folder}/hds.map'
        self.output_filename = f'{self.output_folder}/hds_out'

        self.validate_filepaths()

    def get_key(self, in_key):
        for item in self.data:
            if in_key in item:
                return item[in_key]
        return {}

    def validate_config_file(self):
        if not os.path.exists(self.config_file):
            return {}
        else:
            with open(self.config_file, "r") as yml_f:
                return yaml.load(yml_f, Loader=yaml.FullLoader)

    def validate_filepaths(self):
        def make_dir(inp):
            if inp is not None and not os.path.exists(inp):
                os.makedirs(inp)

        make_dir(self.output_folder)
        make_dir(self.hdr_folder)
        make_dir(self.hash_mapping_folder)
        make_dir(self.numeric_mapping_folder)

        if self.use_datamart:
            make_dir(self.datamart_folder)


class DataAnonymization:

    def __init__(self, config=None, datamart: ADMDatamart = None):

        self.config = Config() if config is None else config

        self.df = self.load_hdr_files()
        self.df_out = None

        self.rename_mapping = {}

        if self.config.use_datamart:
            self.predictors_by_type = self.read_predictor_type_from_datamart(config.datamart_folder, datamart)
        else:
            self.predictors_by_type = self.read_predictor_type_from_file(self.df)

        self.symbolic_predictors_to_mask, self.numeric_predictors_to_mask, self.column_mapping = \
            self.get_predictors_mapping()

        self.create_mapping_files()

        self.process_df()

        self.write_value_mapping_for_numeric_predictors()

        self.write_value_mapping_for_symbolic_predictors()

        self.write_to_output()

    def write_value_mapping_for_symbolic_predictors(self):
        for col in self.symbolic_predictors_to_mask:
            self.df_out.select([self.column_mapping[col], col]).unique().write_csv(
                f'{self.config.hash_mapping_folder}/{col}.map')

    def write_value_mapping_for_numeric_predictors(self):
        for col in self.numeric_predictors_to_mask:
            columns = [self.column_mapping[col], col]
            self.df_out.select(
                pl.col(columns).min().append(
                    pl.col(columns).max())
            ).write_csv(f'{self.config.numeric_mapping_folder}/{col}.map')

    def write_to_output(self, ext: Literal["ndjson", "parquet", "arrow", "csv"] = None):
        out_filename = self.config.output_filename
        out_format = self.config.output_format if ext is None else ext

        out = self.df_out.select(self.column_mapping.values())

        if out_format == "ndjson":
            out.write_ndjson(f'{out_filename}.json')
        elif out_format == "parquet":
            out.write_parquet(f'{out_filename}.parquet')
        elif out_format == "arrow":
            out.write_ipc(f'{out_filename}.arrow')
        else:
            out.write_csv(f'{out_filename}.csv')

    def create_mapping_files(self):
        with open(self.config.mapping_filename, 'w') as wr:
            for key, mapped_val in self.column_mapping.items():
                wr.write(f'{key}={mapped_val}')
                wr.write("\n")

    def load_hdr_files(self):
        out = []
        for file in glob.glob(f'{self.config.hdr_folder}/*.json'):
            out.append(
                pl.scan_ndjson(file)
            )

        df = pl.concat(out, how='diagonal')
        return df

    @staticmethod
    def read_predictor_type_from_file(df):
        types = {}
        df_ = df.collect().sample(100)
        for col in df_.columns:
            try:
                df_.get_column(col).cast(pl.Float64)
                types[col] = "numeric"
            except:
                types[col] = "symbolic"

        return types

    @staticmethod
    def read_predictor_type_from_datamart(datamart_folder, datamart=None):
        dm = ADMDatamart(path=datamart_folder) if datamart is None else datamart
        df = pl.DataFrame(dm.predictorData)
        df = df.filter(pl.col('EntryType') != 'Classifier') \
            .unique(subset='PredictorName') \
            .select(['PredictorName', 'Type'])
        df_dict = dict(zip(df.get_column('PredictorName'), df.get_column('Type')))
        df_dict = dict((key.replace(".", "_"), value) for (key, value) in df_dict.items())
        return df_dict

    def get_columns_by_type(self):
        columns = self.df.columns
        columns_ = ",".join(columns)

        exp = "[^,]+"
        context_keys_t = re.findall(f"{self.config.context_key_predictors}{exp}", columns_)
        ih_predictors_t = re.findall(f"{self.config.ih_predictors}{exp}", columns_)

        special_predictors_t = []
        for exc in self.config.special_predictors:
            for e in re.findall(exc, columns_):
                special_predictors_t.append(e)

        outcome_column_t = [self.config.outcome_column]

        predictors_t = list(
            set(columns) - set(context_keys_t + ih_predictors_t + special_predictors_t + outcome_column_t))
        return context_keys_t, ih_predictors_t, special_predictors_t, predictors_t

    def get_predictors_mapping(self):
        symbolic_predictors_to_mask, numeric_predictors_to_mask = [], []

        context_keys_t, ih_predictors_t, special_predictors_t, predictors_t = self.get_columns_by_type()
        outcome_t = self.config.outcome_column

        outcome_column = {}
        context_keys = {}
        ih_predictors = {}
        predictors = {}

        for idx, val in enumerate(context_keys_t):
            if self.config.mask_context_key_values:
                if self.predictors_by_type[val] == "symbolic":
                    symbolic_predictors_to_mask.append(val)
                else:
                    numeric_predictors_to_mask.append(val)
            context_keys[val] = f'CK_PREDICTOR_{idx}' if self.config.mask_context_key_names else val

        for idx, val in enumerate(ih_predictors_t):
            if self.config.mask_ih_predictor_values:
                if self.predictors_by_type[val] == "symbolic":
                    symbolic_predictors_to_mask.append(val)
                else:
                    numeric_predictors_to_mask.append(val)
            ih_predictors[val] = f'IH_PREDICTOR_{idx}' if self.config.mask_ih_predictor_names else val

        for idx, val in enumerate(predictors_t):
            if self.config.mask_predictor_values:
                if self.predictors_by_type[val] == "symbolic":
                    symbolic_predictors_to_mask.append(val)
                else:
                    numeric_predictors_to_mask.append(val)
            predictors[val] = f'PREDICTOR_{idx}' if self.config.mask_predictor_names else val

        special_predictors = {x: x for x in special_predictors_t}

        outcome_column[outcome_t] = "OUTCOME" if self.config.mask_outcome_name else outcome_t

        column_mapping = predictors | context_keys | ih_predictors | special_predictors | outcome_column

        return symbolic_predictors_to_mask, numeric_predictors_to_mask, column_mapping

    def process_df(self):
        self.rename_mapping = self.column_mapping.copy()
        for sym in self.symbolic_predictors_to_mask:
            self.rename_mapping[f'{sym}_hashed'] = self.column_mapping[sym]
            self.rename_mapping[sym] = f'{sym}'

        for sym in self.numeric_predictors_to_mask:
            self.rename_mapping[f'{sym}_norm'] = self.column_mapping[sym]
            self.rename_mapping[sym] = f'{sym}'

        def to_hash(cols):
            return (
                pl.when((pl.col(cols).is_not_null()) & (pl.col(cols) != "") & (
                    pl.col(cols).is_in(["true", "false"]).is_not()))
                .then(pl.col(cols).hash(seed=randint(1, 100)))
                .otherwise(pl.col(cols))
                .map_alias(lambda col_name: col_name + "_hashed")
            )

        def to_normalize(cols):
            return (
                    (pl.col(cols) - pl.col(cols).min()) / (pl.col(cols).max() - pl.col(cols).min())
            ).map_alias(lambda col_name: col_name + "_norm")

        def to_boolean(col, positives):
            return (
                pl.when(pl.col(col).is_in(positives)).then(True).otherwise(False)
            ).alias(col)

        self.df_out = (
            self.df.with_columns(
                pl.col(self.numeric_predictors_to_mask).cast(pl.Float64)
            )
            .with_columns(
                [
                    to_hash(self.symbolic_predictors_to_mask),
                    to_normalize(self.numeric_predictors_to_mask),
                    to_boolean(self.config.outcome_column, self.config.outcome_accepts),
                ]
            )
            .select(self.rename_mapping.keys())
            .rename(self.rename_mapping)
            .collect()
        )
