from ..adm.ADMDatamart import ADMDatamart
from ..adm.ADMTrees import ADMTrees
from ..valuefinder.ValueFinder import ValueFinder


def CDHSample(plotting_engine="plotly", query=None, **kwargs):
    path = "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data"
    models = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip"
    predictors = "Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip"
    return ADMDatamart(
        path=path,
        model_filename=models,
        predictor_filename=predictors,
        plotting_engine=plotting_engine,
        query=query,
        **kwargs,
    )


def SampleTrees():
    return ADMTrees(
        "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data/agb/_974a7f9c-66a6-4f00-bf3e-3acf5f188b1d.txt"
    )


def SampleValueFinder(verbose=True):
    return ValueFinder(
        path="https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data",
        filename="Data-Insights_pyValueFinder_20210824T112615_GMT.zip",
        verbose=verbose,
    )
