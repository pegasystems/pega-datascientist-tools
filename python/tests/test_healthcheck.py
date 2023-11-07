import sys
import os

import pathlib

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import ADMDatamart, datasets
import pytest


@pytest.fixture
def sample():
    return datasets.CDHSample()

def testHealthCheckRunsWithoutErrors(sample):
    sample.generateHealthCheck(verbose=True)


def testAdditionalTables(sample):
    sample.exportTables()
    os.remove("Tables.xlsx")


@pytest.fixture
def sample_without_predictorbinning():
    return ADMDatamart(
        path=f"{basePath}/data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
        predictor_filename=None,
    )


def testHealthCheckModelRunsWithoutErrors(sample_without_predictorbinning):
    sample_without_predictorbinning.generateHealthCheck(
        verbose=True, modelData_only=True
    )


def testAdditionalTablesModel(sample_without_predictorbinning):
    sample_without_predictorbinning.exportTables(file="ModelTables.xlsx")
    os.remove("ModelTables.xlsx")


def remove_healthCheck():
    os.remove("ADM_HealthCheck.html")
