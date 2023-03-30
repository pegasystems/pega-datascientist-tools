import sys
sys.path.append("python")
from pdstools import ADMDatamart, datasets
import pytest

@pytest.fixture
def sample():
    return datasets.CDHSample()

def testHealthCheckRunsWithoutErrors(sample):
    sample.generateHealthCheck(verbose=True)

def testHealthCheckRunsWithoutTables(sample):
    sample.generateHealthCheck(include_tables=False)

def testAdditionalTables(sample):
    sample.exportTables()

@pytest.fixture
def sample_with_predictorbinning():
    return ADMDatamart(
        path="data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip",
    )

def testHealthCheckModelRunsWithoutErrors(sample_with_predictorbinning):
    sample_with_predictorbinning.generateHealthCheck(verbose=True)

def testHealthCheckModelRunsWithoutTables(sample_with_predictorbinning):
    sample_with_predictorbinning.generateHealthCheck(include_tables=False)