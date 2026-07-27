from __future__ import annotations

import pathlib
import tempfile
import warnings
from typing import TYPE_CHECKING
from urllib.request import urlretrieve

from ..adm.ADMDatamart import ADMDatamart
from ..valuefinder.ValueFinder import ValueFinder

_REPO_DATA_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "data" / "agb"
_SAMPLE_TREES_URL = "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data/agb/ModelExportWithSampleCount.json"
_SAMPLE_EXPLANATIONS_BASE_URL = (
    "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data/explanations/aggregated_data"
)
_SAMPLE_EXPLANATIONS_FILES = ("OVERVIEW.parquet", "BY_CONTEXT.parquet")
_PARQUET_MAGIC = b"PAR1"

if TYPE_CHECKING:
    from ..adm.trees import ADMTreesModel
    from datetime import datetime

    from ..data_quality._topic_data_quality import TopicDataQuality
    from ..explanations.Explanations import Explanations
    from ..utils.types import QUERY
else:
    ADMTreesModel = None


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
            global ADMTreesModel
            if ADMTreesModel is None:
                from ..adm.trees import ADMTreesModel as _ADMTreesModel

                ADMTreesModel = _ADMTreesModel

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


def _is_valid_parquet(path: pathlib.Path) -> bool:
    """Check whether a file looks like a well-formed parquet file.

    A parquet file starts and ends with the 4-byte magic string ``PAR1``. This
    is a cheap sanity check used to detect a cached file that is missing,
    empty, or a truncated/corrupted download (e.g. an HTML error page saved
    by mistake), so a stale cache doesn't silently break downstream reads.
    """
    magic_len = len(_PARQUET_MAGIC)
    if not path.exists() or path.stat().st_size < magic_len * 2:
        return False
    with path.open("rb") as f:
        head = f.read(magic_len)
        f.seek(-magic_len, 2)
        tail = f.read(magic_len)
    return head == _PARQUET_MAGIC and tail == _PARQUET_MAGIC


def sample_explanations(
    *,
    target_dir: str | pathlib.Path | None = None,
    model_name: str | None = "AdaptiveBoostCT",
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    refresh: bool = False,
) -> "Explanations":
    """Load sample global-explanations aggregates into an Explanations instance.

    Parameters
    ----------
    target_dir : str | pathlib.Path | None, default None
        Local folder where sample aggregate files are stored. If ``None``,
        files are stored in ``<tempdir>/pdstools/aggregated_data``.
    model_name : str | None, default "AdaptiveBoostCT"
        Optional model name propagated to :class:`pdstools.explanations.Explanations`.
    from_date : datetime | None, optional
        Optional lower date bound for the explanations context.
    to_date : datetime | None, optional
        Optional upper date bound for the explanations context.
    refresh : bool, default False
        If ``True``, always re-download sample files even if already present.

    Returns
    -------
    Explanations
        An initialized :class:`pdstools.explanations.Explanations` instance.
    """
    target = (
        pathlib.Path(target_dir)
        if target_dir is not None
        else pathlib.Path(tempfile.gettempdir()) / "pdstools" / "aggregated_data"
    )
    target.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings(record=True) as w:
        try:
            for filename in _SAMPLE_EXPLANATIONS_FILES:
                destination = target / filename
                if refresh or not _is_valid_parquet(destination):
                    urlretrieve(f"{_SAMPLE_EXPLANATIONS_BASE_URL}/{filename}", destination)

            from ..explanations.Explanations import Explanations

            return Explanations.from_aggregates(
                data_folder=target,
                model_name=model_name,
                from_date=from_date,
                to_date=to_date,
            )
        except Exception as e:
            raise RuntimeError(
                f"Error importing the Sample Explanations dataset. Warnings: {[str(i) for i in w] if len(w) > 0 else 'None'}, exceptions: {e}",
            ) from e


def dq_sample(
    *,
    similarity_threshold: float = 0.8,
) -> "TopicDataQuality":
    """Load the built-in smalltalk sample dataset for Topic Data Quality.

    Returns a ready-to-use ``TopicDataQuality`` instance with embeddings,
    UMAP, and similarity already computed.

    Parameters
    ----------
    similarity_threshold : float, default 0.8
        Topic pairs above this TF-IDF cosine similarity are flagged.

    Returns
    -------
    TopicDataQuality
        A fully-initialized instance with precomputed results.
    """
    import polars as pl

    from ..data_quality import TopicDataQuality

    url = "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data/dq_nlp/smalltalk.csv"
    df = pl.read_csv(url)
    dq = TopicDataQuality.from_dataframe(
        df=df,
        text_col="content",
        topic_col="result",
        similarity_threshold=similarity_threshold,
    )
    dq.compute.embeddings()
    dq.compute.umap()
    dq.compute.topic_similarity()
    return dq
