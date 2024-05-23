"""
Testing the functionality of the limits class
"""

import pathlib
import sys

import pytest

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import CDHLimits

def test_limits():
    limits = CDHLimits()
    assert limits.check_limits(CDHLimits.Metrics.Configurations_per_Channel, 2) == CDHLimits.Status.Good
    assert limits.check_limits(CDHLimits.Metrics.Configurations_per_Channel, 3) == CDHLimits.Status.Warning
    assert limits.check_limits(CDHLimits.Metrics.Number_of_Channels, 1) == CDHLimits.Status.Warning
    assert limits.check_limits(CDHLimits.Metrics.Number_of_Channels, 100) == CDHLimits.Status.Good
    assert limits.check_limits(CDHLimits.Metrics.Model_Performance, 54) == CDHLimits.Status.Bad
    assert limits.check_limits(CDHLimits.Metrics.Engagement_Lift, None) == CDHLimits.Status.Bad
    assert limits.check_limits(CDHLimits.Metrics.Engagement_Lift, float('nan')) == CDHLimits.Status.Bad
    assert limits.check_limits(CDHLimits.Metrics.Engagement_Lift, float("inf")) == CDHLimits.Status.Warning
    assert limits.check_limits(CDHLimits.Metrics.Engagement_Lift, -float("inf")) == CDHLimits.Status.Bad

def test_sym_limits():
    limits = CDHLimits()
    assert limits.check_limits(CDHLimits.Metrics.Standard_NBAD_Configuration_Names, "OmniAdaptiveModel") == CDHLimits.Status.Good
    assert limits.check_limits(CDHLimits.Metrics.Standard_NBAD_Configuration_Names, "MyVerySpecialModel") == CDHLimits.Status.Warning
    assert limits.check_limits(CDHLimits.Metrics.Standard_NBAD_Direction_Names, "Inbound") == CDHLimits.Status.Good
    assert limits.check_limits(CDHLimits.Metrics.Standard_NBAD_Direction_Names, None) == CDHLimits.Status.Warning
    assert limits.check_limits(CDHLimits.Metrics.Standard_NBAD_Channel_Names, "Direct Mail") == CDHLimits.Status.Warning
    assert limits.check_limits(CDHLimits.Metrics.Standard_NBAD_Channel_Names, "Web") == CDHLimits.Status.Good
