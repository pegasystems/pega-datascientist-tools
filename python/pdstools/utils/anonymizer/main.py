import fire
import json
import re

import numpy as np
import polars as pl

from config import Config

from pdstools import ADMDatamart


def read_datamart_file(filename):
    dm = ADMDatamart(path="datamart/")
    pd = dm.predictorData
    print(dm.context_keys)
    return None

def get_anon_column_names(hdr_file, config):
    with open(hdr_file, 'r') as hdr_f:
        line = hdr_f.readline().strip('\n')
    columns = list(json.loads(line).keys())

    excluded_columns_t = {}
    outcome_column_t = []
    context_keys_t = []
    ih_predictors_t = []
    predictors_t = []
    for idx, col in enumerate(columns):
        for exc in config.exclude_predictors:
            if re.search(exc, col):
                excluded_columns_t[col] = col

        if col in excluded_columns_t:
            continue

        if re.search(config.outcome_column, col):
            outcome_column_t.append(col)
        elif re.search(config.context_key_predictors, col):
            context_keys_t.append(col)
        elif re.search(config.ih_predictors, col):
            ih_predictors_t.append(col)
        else:
            predictors_t.append(col)

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


def process_hdr_file(hdr_file, headers):
    count = 0
    with open(hdr_file, 'r') as hdr_f:
        while True:
            count += 1
            line = hdr_f.readline()

            if not line:
                break
            print("Processing record #{}".format(count))

            record = process_record(line, headers)

            write_to_file(record)


def process_record(record, headers):
    record = json.loads(record)

    line = {}

    for item in headers:
        for key, val in item.items():
            new_val = mask_value(record[key]) if val["mask_val"] else val
            new_key = val["enc"] if val["mask_name"] else key
            line[new_key] = new_val

    return line


def write_to_file(line):
    with open("output.csv", 'a') as wr:
        wr.write(",".join(line.values()))
        wr.write("\n")


def mask_value(inp):
    if inp is None:
        return ""

    if inp.isnumeric():
        if isinstance(inp, float):
            noise = np.random.normal(0, 1, 1)
            inp += noise
        return inp
    else:
        return str(hash(inp))


def create_mapping_files(headers):
    for item in headers:
        for key, val in item.items():
            with open("mapping.txt", 'a') as wr:
                wr.write(f'{key}={val["enc"]}')
                wr.write("\n")


def main(hdr_file=None, config_file='config.yml', datamart_file=None):

    config = Config(config_file)
    datamart = read_datamart_file(datamart_file)

    headers = get_anon_column_names(hdr_file, config)

    create_mapping_files(headers)

    # write column header to file
    header_str = []
    for item in headers:
        for key, val in item.items():
            new_key = val["enc"] if val["mask_name"] else key
            header_str.append(new_key)

    with open("output.csv", 'w') as out_w:
        out_w.write(",".join(header_str) + "\n")

    # read hdr file
    process_hdr_file(hdr_file, headers)


if __name__ == "__main__":
    fire.Fire(main)
