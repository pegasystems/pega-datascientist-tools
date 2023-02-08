import glob

import fire
import json
import re

import numpy as np
import polars as pl

from config import Config
from pdstools import ADMDatamart


def read_datamart_file(datamart_folder):
    dm = ADMDatamart(path=datamart_folder)
    df = dm.predictorData

    # polars method TODO
    # df_p = pl.DataFrame(dm.predictorData)
    # df_predictors_full = df_p.filter(pl.col("EntryType") != "Classifier").select(["PredictorName", "Type"])
    # df_predictors = df_predictors_full.select("PredictorName").unique()

    df_ = list(df[df["EntryType"] != "Classifier"]["PredictorName"].unique())
    columns = {}
    for col in df_:
        columns[col.replace(".", "_")] = df[df["PredictorName"] == col]["Type"].iloc[0]
    return columns


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

    predictors_t = list(set(columns) - set(context_keys_t + ih_predictors_t +  excluded_columns_t + outcome_column_t))
    return context_keys_t, ih_predictors_t, predictors_t


def get_anon_column_names(hdr_file, config):
    with open(hdr_file, 'r') as hdr_f:
        line = hdr_f.readline().strip('\n')
    columns = list(json.loads(line).keys())

    context_keys_t, ih_predictors_t, predictors_t = get_columns_by_type(columns, config)

    outcome_column = {}
    context_keys = {}
    ih_predictors = {}
    predictors = {}

    for i in range(len(context_keys_t)):
        context_keys[context_keys_t[i]] = {
            "enc": f"CK_PREDICTOR_{i}",
            "mask_val": config.mask_context_key_values,
            "mask_name": config.mask_context_key_names
        }

    for i in range(len(ih_predictors_t)):
        ih_predictors[ih_predictors_t[i]] = {
            "enc": f"IH_PREDICTOR_{i}",
            "mask_val": config.mask_ih_predictor_values,
            "mask_name": config.mask_ih_predictor_names
        }

    for i in range(len(predictors_t)):
        predictors[predictors_t[i]] = {
            "enc": f"PREDICTOR_{i}",
            "mask_val": config.mask_predictor_values,
            "mask_name": config.mask_predictor_names
        }

    outcome_column[config.outcome_column] = {
        "enc": "OUTCOME",
        "mask_val": config.mask_outcome_values,
        "mask_name": config.mask_outcome_name
    }

    headers = [predictors, context_keys, ih_predictors, outcome_column]
    return headers


def process_hdr_file(hdr_file, headers, datamart_columns, output_file):
    count = 0
    with open(hdr_file, 'r') as hdr_f:
        while True:
            count += 1
            line = hdr_f.readline()

            if not line:
                break
            print("Processing record #{}".format(count))

            record = process_record(line, headers, datamart_columns)

            write_to_file(record, output_file)


def process_record(record, headers, datamart_columns):
    record = json.loads(record)

    line = {}

    for item in headers:
        for key, val in item.items():
            if datamart_columns is None:
                new_val = mask_value(record[key], None) if val["mask_val"] else record[key]
            elif key in datamart_columns:
                new_val = mask_value(record[key], datamart_columns[key]) if val["mask_val"] else record[key]
            else:
                new_val = mask_value(record[key], None) if val["mask_val"] else record[key]
            new_key = val["enc"] if val["mask_name"] else key
            line[new_key] = new_val

    return line


def write_to_file(line, output_file):
    with open(output_file, 'a') as wr:
        wr.write(",".join(line.values()))
        wr.write("\n")


def mask_value(inp, predictor_type):
    if inp is None:
        return ""

    if predictor_type is None:
        if inp.isnumeric():
            if isinstance(inp, float):
                noise = np.random.normal(0, 1, 1)
                inp += noise
            return inp
        else:
            return str(hash(inp))
    elif predictor_type == "numeric":
        if isinstance(inp, float):
            noise = np.random.normal(0, 1, 1)
            inp += noise
        return inp
    elif predictor_type == "symbolic":
        return str(hash(inp))


def create_mapping_files(file_name, config, headers):
    map_filename = f'{config.output_folder}{file_name}.map'
    with open(map_filename, 'w') as wr:
        for item in headers:
            for key, val in item.items():
                mapped_val = val["enc"] if val["mask_val"] else key
                wr.write(f'{key}={mapped_val}')
                wr.write("\n")


def run(hdr_file, config, datamart_columns):
    headers = get_anon_column_names(hdr_file, config)

    file_name = hdr_file.split("/")[-1].split(".")[0]

    create_mapping_files(file_name, config, headers)

    # write column header to file
    header_str = []
    for item in headers:
        for key, val in item.items():
            new_key = val["enc"] if val["mask_name"] else key
            header_str.append(new_key)

    output_file = f'{config.output_folder}{file_name}.csv'
    with open(output_file, 'w') as out_w:
        out_w.write(",".join(header_str) + "\n")

    # read hdr file
    process_hdr_file(hdr_file, headers, datamart_columns, output_file)


def main(hdr_file=None, config_file='config.yml', use_datamart='N'):

    config = Config(config_file)

    hdr_batch = False
    if hdr_file is None:
        hdr_batch = True

    datamart_columns = None
    if bool(use_datamart):
        datamart_columns = read_datamart_file(config.datamart_folder)

    if hdr_batch:
        for file in glob.glob(f'{config.hdr_folder}/*'):
            run(file, config, datamart_columns)
    else:
        run(hdr_file, config, datamart_columns)


if __name__ == "__main__":
    fire.Fire(main)
