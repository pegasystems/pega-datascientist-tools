import yaml

PREDICTORS = "Predictors"
CONTEXT_KEY = "ContextKeys"
IH_PREDICTORS = "IHPredictors"
OUTCOME_COLUMN = "OutcomeColumn"


class Config:

    def __init__(self, config_file):

        # TODO:: validate config file exists
        with open(config_file, "r") as yml_f:
            self.data = yaml.load(yml_f, Loader=yaml.FullLoader)

        dat = self.get_key(PREDICTORS)
        self.mask_predictor_names = bool(dat["maskPredictorNames"])
        self.mask_predictor_values = bool(dat["maskPredictorValues"])
        self.exclude_predictors = [x.strip() for x in dat["ExcludePredictors"].split(',')]

        dat = self.get_key(CONTEXT_KEY)
        self.mask_context_key_names = bool(dat["maskContextKeyNames"])
        self.mask_context_key_values = bool(dat["maskContextKeyValues"])
        self.context_key_predictors = dat["ContextKeyPredictors"]

        dat = self.get_key(IH_PREDICTORS)
        self.mask_ih_predictor_names = bool(dat["maskIHPredictorNames"])
        self.mask_ih_predictor_values = bool(dat["maskIHPredictorValues"])
        self.ih_predictors = dat["IHPredictors"]

        dat = self.get_key(OUTCOME_COLUMN)
        self.mask_outcome_name = bool(dat["maskOutcomeName"])
        self.mask_outcome_values = bool(dat["maskOutcomeValues"])
        self.outcome_column = dat["OutcomeColumn"]

    def get_key(self, in_key):
        for item in self.data:
            if in_key in item:
                return item[in_key]

