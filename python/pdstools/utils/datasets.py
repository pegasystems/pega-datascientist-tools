from __future__ import annotations

import pathlib
import warnings

from ..adm.ADMDatamart import ADMDatamart
from ..adm.trees import ADMTreesModel
from ..valuefinder.ValueFinder import ValueFinder
from typing import TYPE_CHECKING

_REPO_DATA_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "data" / "agb"
_SAMPLE_TREES_URL = "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data/agb/ModelExportWithSampleCount.json"

if TYPE_CHECKING:
    from ..utils.types import QUERY


def cdh_sample(query: QUERY | None = None) -> ADMDatamart:
    """Import a sample dataset from the CDH Sample application

    Parameters
    ----------
    query : QUERY | None, optional
        An optional query to apply to the data, by default None

    Returns
    -------
    ADMDatamart
        The ADM Datamart class populated with CDH Sample data

    """
    path = "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data"
    models = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip"
    predictors = "Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip"
    with warnings.catch_warnings(record=True) as w:
        try:
            return ADMDatamart.from_ds_export(
                model_filename=models,
                predictor_filename=predictors,
                base_path=path,
                query=query,
            )
        except Exception as e:
            raise RuntimeError(
                f"Error importing CDH Sample. Warnings: {[str(i) for i in w] if len(w) > 0 else 'None'}, exceptions: {e}",
            ) from e


def sample_trees():
    """Load the anonymized AGB sample model (100 trees, with sampleCount).

    Returns
    -------
    ADMTreesModel
        An :class:`~pdstools.adm.trees.ADMTreesModel` loaded from the
        bundled ``data/agb/ModelExportWithSampleCount.json`` file (dev
        environment) or from the canonical GitHub raw URL (installed
        package).
    """
    local = _REPO_DATA_DIR / "ModelExportWithSampleCount.json"
    source = local if local.exists() else _SAMPLE_TREES_URL
    with warnings.catch_warnings(record=True) as w:
        try:
            return ADMTreesModel.from_file(source)
        except Exception as e:
            raise RuntimeError(
                f"Error importing the Sample Trees dataset. Warnings: {[str(i) for i in w] if len(w) > 0 else 'None'}, exceptions: {e}",
            ) from e


def sample_value_finder(threshold: float | None = None) -> ValueFinder:
    """Import a sample dataset of a Value Finder simulation

    This simulation was ran on a stock CDH Sample system.

    Parameters
    ----------
    threshold : float | None, optional
        Optional override of the propensity threshold in the system, by default None

    Returns
    -------
    ValueFinder
        The Value Finder class populated with the Value Finder simulation data

    """
    with warnings.catch_warnings(record=True) as w:
        try:
            return ValueFinder.from_ds_export(
                base_path="https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data",
                filename="Data-Insights_pyValueFinder_20210824T112615_GMT.zip",
                n_customers=10000,
                threshold=threshold,
            )
        except Exception as e:
            raise RuntimeError(
                f"Error importing the Value Finder dataset. Warnings: {[str(i) for i in w] if len(w) > 0 else 'None'}, exceptions: {e}",
            ) from e
