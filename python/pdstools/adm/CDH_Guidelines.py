__all__ = ["CDHGuidelines"]
from functools import cached_property
from typing import List, Optional

import polars as pl

# Pega Cloud limits:
# https://docs.pega.com/bundle/customer-decision-hub-241/page/customer-decision-hub/cdh-portal/cloud-service-health-limits.html

# Current Excel sheet with KPI limits (internal use)
# boundaries and descriptions should be in sync with that one
# https://pegasystems.sharepoint.com/:x:/s/SP-1-1CustomerEngagement/EZRI5aEOxMlMiieFQL6HysEBheX1S96zmy-t4HROK3tG7w?e=NaLYUD

# TODO: future proof the list of models/predictions with the upcoming support for AGB
# https://git.pega.io/projects/PEGAMKTNG/repos/pega-marketing/pull-requests/23314/overview
# and also with multi-level models with certain suffices. Double check with AB. For example,
# Mobile_Click_Through_Rate_Customer should be ok. For some customers, this is currently flagged.

_data = {
    "Issues": [1, 5, 25, None],
    "Groups per Issue": [1, 5, 25, None],
    "Treatments": [2, 2500, 5000, 5000],
    "Treatments per Channel": [2, 1000, 2500, 2500],
    "Treatments per Channel per Action": [1, 1, 5, None],
    "Actions": [50, 100, 2000, 2500],
    "Actions per Group": [1, 100, 250, None],
    "Channels": [1, 2, None, None],
    "Configurations per Channel": [1, 1, 2, None],
    "Predictors": [50, 200, 700, 2000],
    "Active Predictors per Model": [2, 5, 100, None],
    # below are not part of the standard cloud limits but used in the reports
    "Model Performance": [52, 55, 80, 90],
    "Responses": [1.0, 200, None, None],
    "Positive Responses": [1.0, 200, None, None],
    "Engagement Lift": [0.0, 1.0, None, None],
    "CTR": [0.0, 0.000001, 0.999999, 1.0],
    "OmniChannel": [0.0, 0.5, 1.0, 1.0],
    "Action Optionality": [5.0, 7.0, None, None],
    "Inbound No Action Ratio": [None, 0.0, 0.15, 0.25],
}

_pega_cloud_limits = pl.DataFrame(data=_data).transpose(include_header=True)
_pega_cloud_limits.columns = [
    "metric",
    "min",
    "best_practice_min",
    "best_practice_max",
    "max",
]

_NBAD_ModelConfiguration_header = [
    "Configuration",  # "model_name",
    "Channel",  # channel
    "Direction",  # direction
    "isMultiChannel",  # "multi_channel",
]
_NBAD_ModelConfiguration_data = [
    ["Web_Click_Through_Rate", "Web", "Inbound", False],
    ["WebTreatmentClickModel", "Web", "Inbound", False],
    ["Mobile_Click_Through_Rate", "Mobile", "Inbound", False],
    ["Email_Click_Through_Rate", "E-mail", "Outbound", False],
    ["Push_Click_Through_Rate", "Push", "Outbound", False],
    ["SMS_Click_Through_Rate", "SMS", "Outbound", False],
    ["Retail_Click_Through_Rate", "Retail", "Inbound", False],
    ["Retail_Click_Through_Rate_Outbound", "Retail", "Outbound", False],
    ["CallCenter_Click_Through_Rate", "Call Center", "Inbound", False],
    ["CallCenterAcceptRateOutbound", "Call Center", "Outbound", False],
    ["Assisted_Click_Through_Rate", "Assisted", "Inbound", False],
    ["Assisted_Click_Through_Rate_Outbound", "Assisted", "Outbound", False],
    ["Default_Inbound_Model", "Default", "Inbound", False],
    ["Default_Outbound_Model", "Default", "Outbound", False],
    ["Default_Click_Through_Rate", "Other", "Inbound", False],
    ["Other_Inbound_Click_Through_Rate", "Other", "Inbound", False],
    ["OmniAdaptiveModel", "Multi-channel", "Multi-channel", True],
]

# _standard_nbad_adm_configurations = pl.DataFrame(
#     data=_NBAD_ModelConfiguration_data, orient="row"
# )
# _standard_nbad_adm_configurations.columns = _NBAD_ModelConfiguration_header

_NBAD_Prediction_header = [
    "Prediction",
    "Channel",
    "Direction",
    "isMultiChannel",
]

_NBAD_Prediction_data = [
    ["PredictWebPropensity", "Web", "Inbound", False],
    ["PredictMobilePropensity", "Mobile", "Inbound", False],
    ["PredictOutboundEmailPropensity", "E-mail", "Outbound", False],
    ["PredictOutboundPushPropensity", "Push", "Outbound", False],
    ["PredictOutboundSMSPropensity", "SMS", "Outbound", False],
    ["PredictInboundRetailPropensity", "Retail", "Inbound", False],
    ["PredictOutboundRetailPropensity", "Retail", "Outbound", False],
    ["PredictInboundCallCenterPropensity", "Call Center", "Inbound", False],
    ["PredictOutboundCallCenterPropensity", "Call Center", "Outbound", False],
    ["PredictInboundDefaultPropensity", "Default", "Inbound", False],
    ["PredictOutboundDefaultPropensity", "Default", "Outbound", False],
    ["PredictInboundOtherPropensity", "Other", "Inbound", False],
    ["PredictActionPropensity", "Multi-channel", "Multi-channel", True],
    ["PredictTreatmentPropensity", "Multi-channel", "Multi-channel", True],
]

_colorscales = {
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
        self.colorscales = _colorscales

    @cached_property
    def standard_configurations(self) -> List[str]:
        return sorted(list(set([str(x[0]) for x in _NBAD_ModelConfiguration_data])))

    @cached_property
    def standard_predictions(self) -> List[str]:
        return sorted(list(set([str(x[0]) for x in _NBAD_Prediction_data])))

    def is_standard_configuration(self, field: str = "Configuration") -> pl.Expr:
        # simple approach, concatenating the case-insensitive regexp matches. If the 
        # matches would be more complex we could consider joining together a lot of contains expressions
        # but that seems overkill at the moment.
        standard_names = "|".join([f"(?i){x}" for x in self.standard_configurations])
        return pl.col(field).cast(pl.String).str.contains(standard_names)

    def is_standard_prediction(self, field: str = "Prediction") -> pl.Expr:
        return (
            pl.col(field)
            .cast(pl.String)
            .str.contains_any(self.standard_predictions, ascii_case_insensitive=True)
        )

    @cached_property
    def standard_channels(self) -> List[str]:
        return sorted(
            list(set([str(x[1]) for x in _NBAD_ModelConfiguration_data if not x[3]]))
        )

    @cached_property
    def standard_directions(self) -> List[str]:
        return sorted(
            list(set([str(x[2]) for x in _NBAD_ModelConfiguration_data if not x[3]]))
        )

    def _metric_value(self, series):
        if series.shape[0] != 1:
            return None
        return series.item()

    def min(self, metric: str):
        return self._metric_value(
            _pega_cloud_limits.filter(pl.col("metric") == metric)["min"]
        )

    def best_practice_min(self, metric: str):
        return self._metric_value(
            _pega_cloud_limits.filter(pl.col("metric") == metric)["best_practice_min"]
        )

    def best_practice_max(self, metric: str):
        return self._metric_value(
            _pega_cloud_limits.filter(pl.col("metric") == metric)["best_practice_max"]
        )

    def max(self, metric: str):
        return self._metric_value(
            _pega_cloud_limits.filter(pl.col("metric") == metric)["max"]
        )

    def get_predictions_channel_mapping(
        self, custom_predictions: Optional[List[List]] = None
    ) -> pl.DataFrame:
        if not custom_predictions:
            custom_predictions = []
        all_predictions = _NBAD_Prediction_data + [
            prediction
            for prediction in custom_predictions
            if prediction[0].upper()
            not in {x[0].upper() for x in _NBAD_Prediction_data}
        ]

        df = (
            pl.DataFrame(data=all_predictions, orient="row")
            .with_columns(pl.col("column_0").str.to_uppercase())
            .unique()
        )

        df.columns = _NBAD_Prediction_header

        return df

    def get_all_limits(self) -> pl.DataFrame:
        return _pega_cloud_limits
