from typing import List, Optional
from functools import cached_property
import polars as pl

data = {
    "Issues": [1, 5, 25, None],
    "Groups per Issue": [1, 5, 25, None],
    "Treatments": [1, 2500, 5000, 5000],
    "Treatments per Channel": [1, 1000, 2500, 2500],
    "Treatments per Channel per Action": [1, 1, 5, None],
    "Actions": [10, 1000, 2500, 2500],
    "Actions per Group": [1, 100, 250, None],
    "Channels": [1, 2, None, None],
    "Configurations per Channel": [1, 1, 2, None],
    "Predictors": [10, 200, 700, 2000],
    "Active Predictors per Model": [2, 5, 100, None],
    "Model Performance": [52, 55, 80, 90],
    "Engagement Lift": [0.0, 0.2, 2.0, None],
    "Responses": [1.0, 200, None, None],
    "Positive Responses": [1.0, 200, None, None],
}

pega_cloud_limits = pl.DataFrame(data=data).transpose(include_header=True)
pega_cloud_limits.columns = [
    "metric",
    "min",
    "best_practice_min",
    "best_practice_max",
    "max",
]

NBAD_ModelConfiguration_header = [
    "model_name",
    "channel",
    "direction",
    "standard",
    "multi_channel",
]
NBAD_ModelConfiguration_data = [
    ["Web_Click_Through_Rate", "Web", "Inbound", True, False],
    ["WebTreatmentClickModel", "Web", "Inbound", True, False],
    ["Mobile_Click_Through_Rate", "Mobile", "Inbound", True, False],
    ["Email_Click_Through_Rate", "E-mail", "Outbound", True, False],
    ["Push_Click_Through_Rate", "Push", "Outbound", True, False],
    ["SMS_Click_Through_Rate", "SMS", "Outbound", True, False],
    ["Retail_Click_Through_Rate", "Retail", "Inbound", True, False],
    ["Retail_Click_Through_Rate_Outbound", "Retail", "Outbound", True, False],
    ["CallCenter_Click_Through_Rate", "Call Center", "Inbound", True, False],
    ["CallCenterAcceptRateOutbound", "Call Center", "Outbound", True, False],
    ["Assisted_Click_Through_Rate", "Assisted", "Inbound", True, False],
    ["Assisted_Click_Through_Rate_Outbound", "Assisted", "Outbound", True, False],
    ["Default_Inbound_Model", "Default", "Inbound", True, False],
    ["Default_Outbound_Model", "Default", "Outbound", True, False],
    ["Default_Click_Through_Rate", "Other", "Inbound", True, False],
    ["Other_Inbound_Click_Through_Rate", "Other", "Inbound", True, False],
    ["OmniAdaptiveModel", "Multi-channel", "Multi-channel", True, True],
]

standard_nbad_adm_configurations = pl.DataFrame(
    data=NBAD_ModelConfiguration_data, orient="row"
)
standard_nbad_adm_configurations.columns = NBAD_ModelConfiguration_header

NBAD_Prediction_header = [
    "Prediction",
    "Channel",
    "Direction",
    "isStandardNBADPrediction",
    "isMultiChannelPrediction",
]

NBAD_Prediction_data = [
    ["PredictWebPropensity", "Web", "Inbound", True, False],
    ["PredictMobilePropensity", "Mobile", "Inbound", True, False],
    ["PredictOutboundEmailPropensity", "E-mail", "Outbound", True, False],
    ["PredictOutboundPushPropensity", "Push", "Outbound", True, False],
    ["PredictOutboundSMSPropensity", "SMS", "Outbound", True, False],
    ["PredictInboundRetailPropensity", "Retail", "Inbound", True, False],
    ["PredictOutboundRetailPropensity", "Retail", "Outbound", True, False],
    ["PredictInboundCallCenterPropensity", "Call Center", "Inbound", True, False],
    ["PredictOutboundCallCenterPropensity", "Call Center", "Outbound", True, False],
    ["PredictInboundDefaultPropensity", "Default", "Inbound", True, False],
    ["PredictOutboundDefaultPropensity", "Default", "Outbound", True, False],
    ["PredictInboundOtherPropensity", "Other", "Inbound", True, False],
    ["PredictActionPropensity", "Multi-channel", "Multi-channel", True, True],
    ["PredictTreatmentPropensity", "Multi-channel", "Multi-channel", True, True],
]

colorscales = {
    "Performance": [
        (0, "#d91c29"),
        (0.01, "#F76923"),
        (0.3, "#20aa50"),
        (0.8, "#20aa50"),
        (1, "#0000FF"),
    ],
    "SuccessRate": [
        (0, "#d91c29"),
        (0.01, "#F76923"),
        (0.5, "#F76923"),
        (1, "#20aa50"),
    ],
    "other": ["#d91c29", "#F76923", "#20aa50"],
}

class CDHGuidelines:
    def __init__(self):
        # self.limits = pega_cloud_limits
        # self.configurations = standard_nbad_adm_configurations
        # self.predictions = NBAD_Prediction_data
        self.colorscales = colorscales

    # @cached_property
    # def standard_configurations(self) -> List[str]:
    #     return self.configurations["model_name"].unique().to_list()

    # @cached_property
    # def standard_directions(self) -> List[str]:
    #     return self.configurations["direction"].unique().to_list()

    # @cached_property
    # def standard_channels(self) -> List[str]:
    #     return self.configurations["channel"].unique().to_list()

    def _metric_value(self, series):
        if series.shape[0] != 1:
            return None
        return series.item()

    def min(self, metric: str):
        return self._metric_value(
            pega_cloud_limits.filter(pl.col("metric") == metric)["min"]
        )

    def best_practice_min(self, metric: str):
        return self._metric_value(
            pega_cloud_limits.filter(pl.col("metric") == metric)["best_practice_min"]
        )

    def best_practice_max(self, metric: str):
        return self._metric_value(
            pega_cloud_limits.filter(pl.col("metric") == metric)["best_practice_max"]
        )

    def max(self, metric: str):
        return self._metric_value(
            pega_cloud_limits.filter(pl.col("metric") == metric)["max"]
        )
    
    def get_predictions_channel_mapping(
        self, custom_predictions: Optional[List[List]] = None
    ) -> pl.DataFrame:
        if not custom_predictions:
            custom_predictions = []
        all_predictions = NBAD_Prediction_data + custom_predictions

        df = pl.DataFrame(
            data=all_predictions, orient="row"
        ).with_columns(
            pl.col("column_0").str.to_uppercase()
        ).unique()

        df.columns = NBAD_Prediction_header

        return df
