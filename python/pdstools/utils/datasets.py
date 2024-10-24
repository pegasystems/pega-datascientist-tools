from ..adm.ADMDatamart import ADMDatamart
from ..adm.ADMTrees import ADMTrees
from ..valuefinder.ValueFinder import ValueFinder
from typing import Optional


def cdh_sample(query=None):
    path = "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data"
    models = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip"
    predictors = "Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip"
    return ADMDatamart.from_ds_export(
        model_filename=models,
        predictor_filename=predictors,
        base_path=path,
        query=query,
    )


def sample_trees():
    return ADMTrees(
        "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data/agb/_974a7f9c-66a6-4f00-bf3e-3acf5f188b1d.txt"
    )


def sample_value_finder(
    n_customers: Optional[float] = None, threshold: Optional[float] = None
):
    return ValueFinder.from_ds_export(
        base_path="https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data",
        filename="Data-Insights_pyValueFinder_20210824T112615_GMT.zip",
        n_customers=n_customers,
        threshold=threshold,
    )
