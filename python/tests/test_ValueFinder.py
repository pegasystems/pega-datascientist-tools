"""
Testing the functionality of the ValueFinder class
"""

import pathlib
import sys

import pytest

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import datasets


@pytest.fixture
def vf():
    return datasets.sample_value_finder()


def test_all_plots(vf):
    vf.plotPropensityDistribution()
    vf.plotPropensityThreshold()
    vf.plotPieCharts()
    vf.plotPieCharts(method="quantile")
    vf.plotPieCharts(0.01)
    vf.plotPieCharts(start=0.01, stop=0.5, step=0.01, method="threshold")
    vf.plotDistributionPerThreshold()
    vf.plotDistributionPerThreshold(start=0.01, stop=0.5, step=0.01, method="quantile")
    vf.plotDistributionPerThreshold(start=0.01, stop=0.5, step=0.01, method="threshold")
    vf.plotFunnelChart()
    vf.plotFunnelChart("Group")
    vf.plotFunnelChart("Issue")
