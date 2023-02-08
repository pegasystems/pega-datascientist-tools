import yaml
import os

PREDICTORS = "Predictors"
CONTEXT_KEY = "ContextKeys"
IH_PREDICTORS = "IHPredictors"
OUTCOME_COLUMN = "OutcomeColumn"
FILE_PATHS = "FilePaths"


class Config:

    def __init__(self, config_file):

        self.config_file = config_file

        def str2bool(inp):
            return True if inp.upper() == "Y" else False

        self.data = self.validate_config_file()

        dat = self.get_key(PREDICTORS)
        self.mask_predictor_names = str2bool(dat["maskPredictorNames"]) if "maskPredictorNames" in dat else True
        self.mask_predictor_values = str2bool(dat["maskPredictorValues"]) if "maskPredictorValues" in dat else True
        self.exclude_predictors = [x.strip() for x in dat["ExcludePredictors"].split(',')] if "ExcludePredictors" in dat else []

        dat = self.get_key(CONTEXT_KEY)
        self.mask_context_key_names = str2bool(dat["maskContextKeyNames"]) if "maskContextKeyNames" in dat else True
        self.mask_context_key_values = str2bool(dat["maskContextKeyValues"]) if "maskContextKeyValues" in dat else True
        self.context_key_predictors = dat["ContextKeyPredictors"] if "ContextKeyPredictors" in dat else "Context_*"

        dat = self.get_key(IH_PREDICTORS)
        self.mask_ih_predictor_names = str2bool(dat["maskIHPredictorNames"]) if "maskIHPredictorNames" in dat else True
        self.mask_ih_predictor_values = str2bool(dat["maskIHPredictorValues"]) if "maskIHPredictorValues" in dat else True
        self.ih_predictors = dat["IHPredictors"] if "IHPredictors" in dat else "IH_*"

        dat = self.get_key(OUTCOME_COLUMN)
        self.mask_outcome_name = str2bool(dat["maskOutcomeName"]) if "maskOutcomeName" in dat else True
        self.mask_outcome_values = str2bool(dat["maskOutcomeValues"]) if "maskOutcomeValues" in dat else True
        self.outcome_accepts = [x.strip() for x in dat["OutcomeAccepts"].split(",")] if "OutcomeAccepts" in dat else ["Accepted"]
        self.outcome_rejects = [x.strip() for x in dat["OutcomeRejects"].split(",")] if "OutcomeRejects" in dat else ["Rejected"]
        self.outcome_column = dat["OutcomeColumn"] if "OutcomeColumn" in dat else "Outcome"

        dat = self.get_key(FILE_PATHS)
        self.output_folder = dat["outputFolder"] if "outputFolder" in dat else "output/"
        self.datamart_folder = dat["datamartFolder"] if "datamartFolder" in dat else "datamart/"
        self.hdr_folder = dat["hdrDataFolder"] if "hdrDataFolder" in dat else "data/"

        self.validate_filepaths()

    def get_key(self, in_key):
        for item in self.data:
            if in_key in item:
                return item[in_key]
        return {}

    def validate_config_file(self):
        if not os.path.exists(self.config_file):
            return {}
        else:
            with open(self.config_file, "r") as yml_f:
                return yaml.load(yml_f, Loader=yaml.FullLoader)

    def validate_filepaths(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if not os.path.exists(self.datamart_folder):
            os.makedirs(self.datamart_folder)

        if not os.path.exists(self.hdr_folder):
            os.makedirs(self.hdr_folder)




