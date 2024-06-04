from enum import Enum
from collections import namedtuple
import math
from typing import Union
from . import NBAD


class CDHLimits(object):
    """
    A singleton container for best practice limits for CDH.

    Limits taken from https://docs-previous.pega.com/pega-customer-decision-hub-user-guide/87/service-and-data-health-limits-pega-customer-decision-hub-pega-cloud?
    but note these apply to Pega Cloud only.
    """

    Status = Enum("LimitStatus", ["Good", "Bad", "Warning", "Unknown"])
    Metrics = Enum(
        "Metrics",
        [
            # cross ref service lims and perhaps notifications
            
            "Number_of_Issues",
            "Number_of_Groups_per_Issue",
            
            "Number_of_Treatments",
            "Number_of_Treatments_per_Channel",
            "Number_of_Treatments_per_Channel_per_Action",
            
            "Number_of_Actions",
            "Number_of_Actions_per_Group",
            
            "Number_of_Channels",
            "Configurations_per_Channel",
            "Standard_NBAD_Channel_Names",
            "Standard_NBAD_Direction_Names",
            "Standard_NBAD_Configuration_Names",

            "Number_of_Predictors",
            "Number_of_Active_Predictors_per_Model",
            "Model_Performance",
            "Engagement_Lift",

            "Number_of_Responses",
            "Number_of_Positive_Responses",

        ],
    )

    _instance = None

    num_limit = namedtuple(
        "NumLimits",
        ["min", "best_practice_min", "best_practice_max", "max"],
    )
    lims = {
        Metrics.Number_of_Issues: num_limit(1, 5, 25, None),
        Metrics.Number_of_Groups_per_Issue: num_limit(
            1, 5, 25, None
        ),

        Metrics.Number_of_Treatments: num_limit(1, 2500, 5000, 5000),
        Metrics.Number_of_Treatments_per_Channel: num_limit(1, 1000, 2500, 2500),
        Metrics.Number_of_Treatments_per_Channel_per_Action: num_limit(
            1, 1, 5, None
        ),

        Metrics.Number_of_Actions: num_limit(10, 1000, 2500, 2500),
        Metrics.Number_of_Actions_per_Group: num_limit(
            1, 100, 250, None
        ),

        Metrics.Number_of_Channels: num_limit(
            1, 2, None, None
        ),
        Metrics.Configurations_per_Channel: num_limit(1, 1, 2, None),
        Metrics.Standard_NBAD_Direction_Names: set(
            [x.direction for x in NBAD.standardNBADModelConfigurations]
        ),
        Metrics.Standard_NBAD_Channel_Names: set(
            [x.channel for x in NBAD.standardNBADModelConfigurations]
        ),
        Metrics.Standard_NBAD_Configuration_Names: set(
            [x.model_name for x in NBAD.standardNBADModelConfigurations]
        ),

        Metrics.Number_of_Predictors: num_limit(10, 200, 700, 2000), # official limits are different
        Metrics.Number_of_Active_Predictors_per_Model: num_limit(2, 5, 100, None), # no official limits
        Metrics.Model_Performance: num_limit(52, 55, 80, 90), # no official limits
        Metrics.Engagement_Lift: num_limit(0.0, 0.2, 2.0, None), # no official limits
        
        # predictor/model related

        Metrics.Number_of_Responses: num_limit(1.0, 200, None, None),
        Metrics.Number_of_Positive_Responses: num_limit(1.0, 200, None, None),
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CDHLimits, cls).__new__(cls)
        return cls._instance

    def get_limits(self, metric: Metrics) -> Union[num_limit, None]:
        if metric not in self.lims:
            return None
        return self.lims[metric]

    def check_limits(self, metric: Metrics, value: object) -> Status:
        lims = self.get_limits(metric)
        # print(type(lims))
        if isinstance(lims, self.num_limit):
            if value is None or math.isnan(value):
                return self.Status.Bad
            elif (lims.min is not None and value < lims.min) or (
                lims.max is not None and value > lims.max
            ):
                return self.Status.Bad
            elif (
                lims.best_practice_min is not None and value < lims.best_practice_min
            ) or (
                lims.best_practice_max is not None and value > lims.best_practice_max
            ):
                return self.Status.Warning
            else:
                return self.Status.Good
        elif isinstance(lims, list) or isinstance(lims, set):
            return self.Status.Good if value in lims else self.Status.Warning

        return self.Status.Unknown

