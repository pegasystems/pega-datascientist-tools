from enum import Enum
from collections import namedtuple
from typing import Union
from . import NBAD

class CDHLimits(object):
    """
    A singleton container for best practice limits for CDH.
    """
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

    _instance = None

    num_limit = namedtuple(
        "NumLimits",
        ["min", "best_practice_min", "best_practice_max", "max", "is_warning"],
    )
    lims = {
        Metrics.Configurations_per_Channel: num_limit(1, 1, 2, None, False),
        Metrics.Number_of_Actions: num_limit(1, 1000, 2500, None, False),
        Metrics.Number_of_Treatments_per_Channel_per_Action: num_limit(
            1, 3, 10, None, True
        ),
        Metrics.Number_of_Predictors: num_limit(10, 100, 500, 2000, True),
        Metrics.Standard_NBAD_Direction_Names: set(
            [x.direction for x in NBAD.standardNBADModelConfigurations]
        ),
        Metrics.Standard_NBAD_Channel_Names: set(
            [x.channel for x in NBAD.standardNBADModelConfigurations]
        ),
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CDHLimits, cls).__new__(cls)
        return cls._instance

    def get_limits(self, metric: Metrics) -> Union[num_limit, None]:
        if metric not in self.lims:
            return None
        return self.lims[metric]

    def check_limits(self, metric: Metrics, value: object) -> LimitStatus:
        lims = self.get_limits(metric)
        # print(type(lims))
        if isinstance(lims, self.num_limit):
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
            return LimitStatus.Good if value in lims else LimitStatus.Info

        return LimitStatus.Unknown


if __name__ == "__main__":
    limits = CDHLimits()
    print(limits.get_limits(limits.Configurations_per_Channel))

    print(limits.check_limits(limits.Configurations_per_Channel, 2))
    print(limits.check_limits(limits.Number_of_Treatments_per_Channel_per_Action, 1))
    print(limits.check_limits(limits.Standard_NBAD_Direction_Names, "Anybound"))
    print(limits.check_limits(limits.Standard_NBAD_Channel_Names, "Web"))
    print(limits.check_limits(limits.Number_of_Predictors, 7))
    print(limits.check_limits(limits.Number_of_Predictors, 700))
    print(limits.check_limits(limits.Number_of_Predictors, 7000))
    print(limits.check_limits(limits.Number_of_Actions, 1))
    print(limits.check_limits(limits.Number_of_Actions, 100))
    print(limits.check_limits(limits.Number_of_Actions, 1000))
    print(limits.check_limits(limits.Number_of_Actions, 10000))
    print(limits.check_limits(limits.Number_of_Actions_per_Group, 50))
