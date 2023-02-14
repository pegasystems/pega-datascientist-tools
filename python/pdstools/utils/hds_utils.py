import os
import re
import glob
import yaml

import polars as pl

from ..adm import ADMDatamart


class Config:

    def __init__(self):

        self.config_file = "config.yml"

        def str2bool(inp):
            return True if inp.upper() == "Y" else False

        self.data = self.validate_config_file()

        dat = self.get_key("Predictors")
        self.mask_predictor_names = str2bool(dat.get('maskPredictorNames', "Y"))
        self.mask_predictor_values = str2bool(dat.get('maskPredictorValues', "Y"))
        self.exclude_predictors = [x.strip() for x in
                                   dat["ExcludePredictors"].split(',')] if "ExcludePredictors" in dat else []

        dat = self.get_key("ContextKeys")
        self.mask_context_key_names = str2bool(dat.get('maskContextKeyNames', "Y"))
        self.mask_context_key_values = str2bool(dat.get('maskContextKeyValues', "Y"))
        self.context_key_predictors = dat.get('ContextKeyPredictors', "Context_*")

        dat = self.get_key("IHPredictors")
        self.mask_ih_predictor_names = str2bool(dat.get('maskIHPredictorNames', "Y"))
        self.mask_ih_predictor_values = str2bool(dat.get('maskIHPredictorValues', "Y"))
        self.ih_predictors = dat.get('IHPredictors', "IH_*")

        dat = self.get_key("OutcomeColumn")
        self.mask_outcome_name = str2bool(dat.get('maskOutcomeName', "Y"))
        self.mask_outcome_values = str2bool(dat.get('maskOutcomeValues', "Y"))
        self.outcome_accepts = [x.strip() for x in dat["OutcomeAccepts"].split(",")] if "OutcomeAccepts" in dat else [
            "Accepted"]
        self.outcome_rejects = [x.strip() for x in dat["OutcomeRejects"].split(",")] if "OutcomeRejects" in dat else [
            "Rejected"]
        self.outcome_column = dat.get('OutcomeColumn', "Outcome")

        dat = self.get_key("FilePaths")
        self.output_folder = dat.get('outputFolder', "output/")
        self.datamart_folder = dat.get('datamartFolder', "datamart/")
        self.hdr_folder = dat.get('hdrFolder', None)
        self.hdr_file = dat.get('hdrFile', None)

        self.use_datamart = str2bool(dat.get('useDatamart', "N"))
        self.output_format = dat.get("outputFormat", "ndjson")

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
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if self.datamart_folder is not None:
            if not os.path.exists(self.datamart_folder):
                os.makedirs(self.datamart_folder)

        if self.hdr_folder is not None:
            if not os.path.exists(self.hdr_folder):
                os.makedirs(self.hdr_folder)


def create_mapping_files(map_filename, headers):
    with open(map_filename, 'w') as wr:
        for key, mapped_val in headers.items():
            wr.write(f'{key}={mapped_val}')
            wr.write("\n")


def load_hdr_files(config):
    out = []
    for file in glob.glob(f'{config.hdr_folder}/*.json'):
        out.append(pl.scan_ndjson(file))

    df = pl.concat(out, how='diagonal')
    return df


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


def read_predictor_type_from_datamart(datamart_folder):
    dm = ADMDatamart(path=datamart_folder)
    df = pl.DataFrame(dm.predictorData)
    df = df.filter(pl.col('EntryType') != 'Classifier') \
        .unique(subset='PredictorName') \
        .select(['PredictorName', 'Type'])
    df_dict = dict(zip(df.get_column('PredictorName'), df.get_column('Type')))
    df_dict = dict((key.replace(".", "_"), value) for (key, value) in df_dict.items())
    return df_dict


def get_columns_by_type(columns, config):
    columns_ = ",".join(columns)

    exp = "[^,]+"
    context_keys_t = re.findall(f"{config.context_key_predictors}{exp}", columns_)
    ih_predictors_t = re.findall(f"{config.ih_predictors}{exp}", columns_)

    excluded_columns_t = []
    for exc in config.exclude_predictors:
        for e in re.findall(exc, columns_):
            excluded_columns_t.append(e)

    outcome_column_t = [config.outcome_column]

    predictors_t = list(set(columns) - set(context_keys_t + ih_predictors_t + excluded_columns_t + outcome_column_t))
    return context_keys_t, ih_predictors_t, predictors_t


def get_predictors_mapping(columns, predictors_by_type, config):
    symbolic_predictors_to_mask, numeric_predictors_to_mask, predictors_ = [], [], []

    context_keys_t, ih_predictors_t, predictors_t = get_columns_by_type(columns, config)
    outcome_t = config.outcome_column

    outcome_column = {}
    context_keys = {}
    ih_predictors = {}
    predictors = {}

    for idx, val in enumerate(context_keys_t):
        if config.mask_context_key_values:
            if predictors_by_type[val] == "symbolic":
                symbolic_predictors_to_mask.append(val)
            else:
                numeric_predictors_to_mask.append(val)

        context_keys[val] = f'CK_PREDICTOR_{idx}' if config.mask_context_key_names else val

    for idx, val in enumerate(ih_predictors_t):
        if config.mask_ih_predictor_values:
            if predictors_by_type[val] == "symbolic":
                symbolic_predictors_to_mask.append(val)
            else:
                numeric_predictors_to_mask.append(val)

        ih_predictors[val] = f'IH_PREDICTOR_{idx}' if config.mask_ih_predictor_names else val

    for idx, val in enumerate(predictors_t):
        if config.mask_predictor_values:
            if predictors_by_type[val] == "symbolic":
                symbolic_predictors_to_mask.append(val)
            else:
                numeric_predictors_to_mask.append(val)

        predictors[val] = f'PREDICTOR_{idx}' if config.mask_predictor_names else val

    outcome_column[outcome_t] = "OUTCOME" if config.mask_outcome_name else outcome_t

    headers = predictors | context_keys | ih_predictors | outcome_column

    return symbolic_predictors_to_mask, numeric_predictors_to_mask, headers


def write_to_output(df, config):
    out_filename = f'{config.output_folder}/hds_out'

    if config.output_format == "ndjson":
        df.write_ndjson(f'{out_filename}.json')
    elif config.output_format == "parquet":
        df.write_parquet(f'{out_filename}.parquet')
    elif config.output_format == "arrow":
        df.write_ipc(f'{out_filename}.arrow')
    else:
        df.write_csv(f'{out_filename}.csv')


def process_df(df, symbolic_predictors_to_mask, numeric_predictors_to_mask, predictors_, config):
    out = df \
        .with_columns(pl.col(numeric_predictors_to_mask).cast(pl.Float64)) \
        .with_columns(
            [
                pl.col(symbolic_predictors_to_mask).hash(),
                ((pl.col(numeric_predictors_to_mask) - pl.col(numeric_predictors_to_mask).min()) /
                  (pl.col(numeric_predictors_to_mask).max() - pl.col(numeric_predictors_to_mask).min())),
                (pl.when(pl.col(config.outcome_column).is_in(config.outcome_accepts)).then(1).otherwise(0)).alias(config.outcome_column),
            ]
        ).select(predictors_.keys()).rename(predictors_).collect()

    write_to_output(out, config)


def main():
    config = Config()

    df = load_hdr_files(config)

    if config.use_datamart:
        predictors_by_type = read_predictor_type_from_datamart(config.datamart_folder)
    else:
        predictors_by_type = read_predictor_type_from_file(df)

    symbolic_predictors_to_mask, numeric_predictors_to_mask, predictors_ = get_predictors_mapping(df.columns,
                                                                                                  predictors_by_type,
                                                                                                  config)

    map_filename = f'{config.output_folder}/hds.map'
    create_mapping_files(map_filename, predictors_)

    process_df(df, symbolic_predictors_to_mask, numeric_predictors_to_mask, predictors_, config)


if __name__ == "__main__":
    main()
