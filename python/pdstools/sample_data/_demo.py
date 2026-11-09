"""Public demo bundle over one shared sample world."""

from __future__ import annotations

import zipfile
from functools import cache, cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from ._world import DEFAULT_DAYS, DEFAULT_SEED, EndDate, SampleWorld, sample_world

if TYPE_CHECKING:
    import os

    import polars as pl

    from ..adm.ADMDatamart import ADMDatamart
    from ..ih.IH import IH
    from ..prediction.Prediction import Prediction

DEFAULT_INTERACTIONS = 100_000

_EXPORT_NAMES = {
    "model_data": "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots",
    "predictor_data": "Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots",
    "prediction_data": "Data-DM-Snapshot_pyGetSnapshot",
}


class DemoData:
    """Datamart, prediction and Interaction History views of one sample world.

    Views are built on first access, so asking only for the prediction does
    not pay for the datamart or the Interaction History. Create instances
    with :func:`demo_data`.
    """

    def __init__(self, world: SampleWorld, interactions: int = DEFAULT_INTERACTIONS):
        self.world = world
        """The immutable world all views derive from."""
        self.interactions = interactions
        """Number of decisions sampled for the Interaction History view."""

    @cached_property
    def datamart(self) -> ADMDatamart:
        """ADM model snapshots and predictor binning of the world."""
        from ..adm.ADMDatamart import ADMDatamart
        from ._adm import model_snapshots
        from ._binning import predictor_binning

        return ADMDatamart(
            model_df=model_snapshots(self.world),
            predictor_df=predictor_binning(self.world),
        )

    @cached_property
    def prediction(self) -> Prediction:
        """Prediction Studio snapshots of the world."""
        from ..prediction.Prediction import Prediction
        from ._prediction import prediction_snapshots

        return Prediction(prediction_snapshots(self.world))

    @cached_property
    def ih(self) -> IH:
        """Interaction History sampled from the world."""
        from ..ih.IH import IH
        from ._ih import interaction_history, outcome_labels

        ih = IH(interaction_history(self.world, self.interactions))
        ih.outcome_labels_used = outcome_labels(self.world)
        return ih

    def write_ds_export(self, path: str | os.PathLike = ".") -> dict[str, Path]:
        """Write the model, predictor and prediction views as Pega exports.

        Each file is a zip holding ``data.json`` in NDJSON format with
        datamart-style field names, named like a Pega dataset export with
        the last day of the window as timestamp. The files can be read back
        with :meth:`pdstools.ADMDatamart.from_ds_export`,
        :meth:`pdstools.Prediction.from_ds_export` or uploaded in the apps.

        Parameters
        ----------
        path : str or os.PathLike, default "."
            Existing or new directory to write into.

        Returns
        -------
        dict[str, Path]
            Written file per view: ``model_data``, ``predictor_data`` and
            ``prediction_data``.

        Examples
        --------
        >>> from pdstools import demo_data
        >>> files = demo_data(end="canonical").write_ds_export("exports")
        >>> ADMDatamart.from_ds_export(files["model_data"].name, files["predictor_data"].name, "exports")
        """
        from ._adm import model_snapshots
        from ._binning import predictor_binning
        from ._prediction import prediction_snapshots

        frames = {
            "model_data": model_snapshots(self.world),
            "predictor_data": predictor_binning(self.world),
            "prediction_data": prediction_snapshots(self.world),
        }
        stamp = f"{self.world.dates[-1]:%Y%m%d}T235959_GMT"
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        written = {}
        for view, frame in frames.items():
            file = target / f"{_EXPORT_NAMES[view]}_{stamp}.zip"
            with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("data.json", _ndjson(frame))
            written[view] = file
        return written


def _ndjson(frame: pl.LazyFrame) -> bytes:
    lines = frame.collect().write_ndjson()
    return lines.encode()


@cache
def _cached_demo(world: SampleWorld, interactions: int) -> DemoData:
    return DemoData(world, interactions)


def demo_data(
    days: int = DEFAULT_DAYS,
    *,
    seed: int | None = DEFAULT_SEED,
    end: EndDate = None,
    scale: float = 1.0,
    interactions: int = DEFAULT_INTERACTIONS,
) -> DemoData:
    """Synthetic CDH demo data: datamart, prediction and Interaction History.

    All views come from one seeded synthetic world (no customer data and no
    stored files), so models, predictions and interactions tell the same
    story. Results are cached per process: repeated calls with the same
    arguments return the same object, so treat its views as read-only.

    Parameters
    ----------
    days : int, default 182
        Window length in days; at least 56.
    seed : int or None, default 42
        Seed for reproducible data; ``None`` draws a random seed (not cached).
    end : date, "canonical" or None, default None
        Last day of the window, resolved to the Sunday on or before it.
        ``None`` uses the most recent completed Sunday, ``"canonical"`` pins
        Sunday 2000-07-02. IH experiment assignments follow calendar months,
        so changing the end date can also change the sampled interactions.
    scale : float, default 1.0
        Multiplier on response volumes.
    interactions : int, default 100000
        Number of decisions sampled for the Interaction History view.

    Returns
    -------
    DemoData
        Bundle with ``datamart``, ``prediction`` and ``ih`` views, built
        lazily on first access.

    Examples
    --------
    >>> from pdstools import demo_data
    >>> demo = demo_data()
    >>> demo.datamart.plot.bubble_chart()
    >>> demo.prediction.plot.performance_trend()
    """
    world = sample_world(seed=seed, days=days, end=end, scale=scale)
    if seed is None:
        return DemoData(world, int(interactions))
    return _cached_demo(world, int(interactions))
