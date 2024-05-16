from enum import Enum
from collections import namedtuple
from typing import List, Union

from pdstools import ADMDatamart

# TODO ADM DM currently not using the namedtuple types...
NBAD_ModelConfiguration = namedtuple(
    "Configurations",
    ["model_name", "channel", "direction", "standard", "multi_channel"],
)
standardNBADModelConfigurationList: List[NBAD_ModelConfiguration] = [
    NBAD_ModelConfiguration("Web_Click_Through_Rate", "Web", "Inbound", True, False),
    NBAD_ModelConfiguration("WebTreatmentClickModel", "Web", "Inbound", True, False),
    NBAD_ModelConfiguration(
        "Mobile_Click_Through_Rate", "Mobile", "Inbound", True, False
    ),
    NBAD_ModelConfiguration(
        "Email_Click_Through_Rate", "E-mail", "Outbound", True, False
    ),
    NBAD_ModelConfiguration("Push_Click_Through_Rate", "Push", "Outbound", True, False),
    NBAD_ModelConfiguration("SMS_Click_Through_Rate", "SMS", "Outbound", True, False),
    NBAD_ModelConfiguration(
        "Retail_Click_Through_Rate", "Retail", "Inbound", True, False
    ),
    NBAD_ModelConfiguration(
        "Retail_Click_Through_Rate_Outbound", "Retail", "Outbound", True, False
    ),
    NBAD_ModelConfiguration(
        "CallCenter_Click_Through_Rate", "Call Center", "Inbound", True, False
    ),
    NBAD_ModelConfiguration(
        "CallCenterAcceptRateOutbound", "Call Center", "Outbound", True, False
    ),
    NBAD_ModelConfiguration(
        "Assisted_Click_Through_Rate",
        "Assisted",
        "Inbound",
        True,
        False,
    ),  # withdrawn record
    NBAD_ModelConfiguration(
        "Assisted_Click_Through_Rate_Outbound",
        "Assisted",
        "Outbound",
        True,
        False,
    ),  # withdrawn record
    NBAD_ModelConfiguration("Default_Inbound_Model", "Default", "Inbound", True, False),
    NBAD_ModelConfiguration(
        "Default_Outbound_Model", "Default", "Outbound", True, False
    ),
    NBAD_ModelConfiguration(
        "Default_Click_Through_Rate", "Other", "Inbound", True, False
    ),
    NBAD_ModelConfiguration(
        "Other_Inbound_Click_Through_Rate", "Other", "Inbound", True, False
    ),
    NBAD_ModelConfiguration(
        "OmniAdaptiveModel", "Multi-channel", "Multi-channel", True, True
    ),
]

LimitStatus = Enum("LimitStatus", ["Good", "Bad", "Info", "Warning", "Unknown"])
Metrics = Enum(
    "Metrics",
    [
        # cross ref service lims and perhaps notifications
        "Configurations_per_Channel",
        "Number_of_Actions",
        "Number_of_Actions_per_Group",
        "Number_of_Treatments",
        "Number_of_Treatments_per_Channel",
        "Number_of_Treatments_per_Channel_per_Action",
        "Number_of_Issues",
        "Number_of_Groups_per_Issue",
        "Number_of_Channels",
        "Standard_NBAD_Channel_Names",
        "Standard_NBAD_Direction_Names",
        "Standard_NBAD_Configuration_Names",
        "Number_of_Predictors",
        "Number_of_Active_Predictors_per_Model",
        "Model_Performance",
    ],
)

class CDHLimits(object):
    """
    A singleton container for best practice limits for CDH.
    """

    _instance = None

    num_limit_type = namedtuple(
        "NumLimits",
        ["min", "best_practice_min", "best_practice_max", "max", "is_warning"],
    )
    lims = {
        Metrics.Configurations_per_Channel: num_limit_type(1, 1, 2, None, False),
        Metrics.Number_of_Actions: num_limit_type(1, 1000, 2500, None, False),
        Metrics.Number_of_Treatments_per_Channel_per_Action: num_limit_type(
            1, 3, 10, None, True
        ),
        Metrics.Number_of_Predictors: num_limit_type(10, 100, 500, 2000, True),
        Metrics.Standard_NBAD_Direction_Names: set(
            [x.direction for x in standardNBADModelConfigurationList]  # ADMDatamart..
        ),
        Metrics.Standard_NBAD_Channel_Names: set(
            [x.channel for x in standardNBADModelConfigurationList]  # ADMDatamart..
        ),
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CDHLimits, cls).__new__(cls)
        return cls._instance

    def get_limits(self, metric: Metrics) -> Union[num_limit_type, None]:
        if metric not in self.lims:
            return None
        return self.lims[metric]

    def check_limits(self, metric: Metrics, value: object) -> LimitStatus:
        lims = self.get_limits(metric)
        # print(type(lims))
        if isinstance(lims, self.num_limit_type):
            if (lims.min is not None and value < lims.min) or (
                lims.max is not None and value > lims.max
            ):
                return LimitStatus.Bad
            elif (
                lims.best_practice_min is not None and value < lims.best_practice_min
            ) or (
                lims.best_practice_max is not None and value > lims.best_practice_max
            ):
                return LimitStatus.Warning if lims.is_warning else LimitStatus.Info
            else:
                return LimitStatus.Good
        elif isinstance(lims, list) or isinstance(lims, set):
            return LimitStatus.Good if value in lims else LimitStatus.Bad

        return LimitStatus.Unknown


if __name__ == "__main__":
    limits = CDHLimits()
    print(limits.get_limits(Metrics.Configurations_per_Channel))
    print(limits.check_limits(Metrics.Configurations_per_Channel, 2))
    print(limits.check_limits(Metrics.Number_of_Treatments_per_Channel_per_Action, 1))
    print(limits.check_limits(Metrics.Standard_NBAD_Direction_Names, "Anybound"))
    print(limits.check_limits(Metrics.Standard_NBAD_Channel_Names, "Web"))
    print(limits.check_limits(Metrics.Number_of_Predictors, 7))
    print(limits.check_limits(Metrics.Number_of_Predictors, 700))
    print(limits.check_limits(Metrics.Number_of_Predictors, 7000))
    print(limits.check_limits(Metrics.Number_of_Actions, 1))
    print(limits.check_limits(Metrics.Number_of_Actions, 100))
    print(limits.check_limits(Metrics.Number_of_Actions, 1000))
    print(limits.check_limits(Metrics.Number_of_Actions, 10000))
    print(limits.check_limits(Metrics.Number_of_Actions_per_Group, 50))
