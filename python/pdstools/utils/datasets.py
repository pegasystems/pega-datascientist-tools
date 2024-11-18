from typing import Optional

from ..adm.ADMDatamart import ADMDatamart
from ..adm.ADMTrees import ADMTrees
from ..utils.types import QUERY
from ..valuefinder.ValueFinder import ValueFinder


def cdh_sample(query: Optional[QUERY] = None) -> ADMDatamart:
    """Import a sample dataset from the CDH Sample application

    Parameters
    ----------
    query : Optional[QUERY], optional
        An optional query to apply to the data, by default None

    Returns
    -------
    ADMDatamart
        The ADM Datamart class populated with CDH Sample data
    """
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


def sample_value_finder(threshold: Optional[float] = None) -> ValueFinder:
    """Import a sample dataset of a Value Finder simulation

    This simulation was ran on a stock CDH Sample system.

    Parameters
    ----------
    threshold : Optional[float], optional
        Optional override of the propensity threshold in the system, by default None

    Returns
    -------
    ValueFinder
        The Value Finder class populated with the Value Finder simulation data
    """
    return ValueFinder.from_ds_export(
        base_path="https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data",
        filename="Data-Insights_pyValueFinder_20210824T112615_GMT.zip",
        n_customers=10000,
        threshold=threshold,
    )
