import os
import pathlib
import sys

import polars as pl
import pytest

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import Anonymization


@pytest.fixture
def anonymizer():
    return Anonymization(
        path_to_files=f"{basePath}/data/SampleHDS.json",
        temporary_path="tmp",
        output_file="anonymised.parquet",
        skip_columns_with_prefix=["Context_", "Decision_"],
        batch_size=1000,
        file_limit=10,
    )


def test_anonymize(anonymizer):
    anonymizer.anonymize()
    assert os.path.exists(anonymizer.output_file)


def test_min_max():
    range = [{"min": 0.0, "max": 100.0}]
    expr = Anonymization.min_max("age", range)
    assert isinstance(expr, pl.Expr)


def test_infer_types():
    df = pl.DataFrame(
        {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [1.0, 2.0, 3.0]}
    )
    types = Anonymization._infer_types(df)
    assert types == {"col1": "numeric", "col2": "symbolic", "col3": "numeric"}


def test_chunker():
    files = ["file1.txt", "file2.txt", "file3.txt", "file4.txt", "file5.txt"]
    chunks = list(Anonymization.chunker(files, 2))
    assert chunks == [
        ["file1.txt", "file2.txt"],
        ["file3.txt", "file4.txt"],
        ["file5.txt"],
    ]


def test_chunk_to_parquet(anonymizer):
    files = [f"{basePath}/data/SampleHDS.json"]
    filename = anonymizer.chunk_to_parquet(files, 0)
    assert os.path.exists(filename)


def test_preprocess(anonymizer):
    chunked_files = anonymizer.preprocess(verbose=False)
    assert len(chunked_files) > 0


def test_process(anonymizer):
    chunked_files = anonymizer.preprocess(verbose=False)
    anonymizer.process(chunked_files, verbose=False)
    assert os.path.exists(anonymizer.output_file)
