import sys
import pytest
sys.path.append("python")
from cdhtools import ValueFinder, datasets, cdh_utils


@pytest.fixture
def test():
    """Fixture to serve as class to call functions from."""
    return datasets.SampleValueFinder()

def test_manual():
    ValueFinder(path = 'https://raw.githubusercontent.com/pegasystems/cdh-datascientist-tools/master/data', filename="Data-Insights_pyValueFinder_20210824T112615_GMT.zip")

def test_df():
    df = cdh_utils.readDSExport(filename = "Data-Insights_pyValueFinder_20210824T112615_GMT.zip", path ='https://raw.githubusercontent.com/pegasystems/cdh-datascientist-tools/master/data')
    ValueFinder(df = df)

def test_plotPropensityDistribution(test):
    test.plotPropensityDistribution()

def test_plotPropensityThreshold(test):
    test.plotPropensityThreshold()

def test_plotPieCharts(test):
    test.plotPieCharts()

def test_plotDistributionPerThreshold_quantile(test):
    test.plotDistributionPerThreshold()

def test_plotDistributionPerThreshold_propensity(test):
    test.plotDistributionPerThreshold('Propensity')

def test_plotFunnelChartActions(test):
    test.plotFunnelChart('Action')

def test_plotFunnelChartIssues(test):
    test.plotFunnelChart('Issue')