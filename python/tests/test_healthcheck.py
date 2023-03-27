import sys
sys.path.append("python")
from pdstools import datasets
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