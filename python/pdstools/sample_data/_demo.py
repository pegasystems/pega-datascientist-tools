"""Public demo bundle over one shared sample world."""

from __future__ import annotations

import zipfile
from datetime import date
from functools import cache, cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
    "ih_data": "Data-pxStrategyResult_pxInteractionHistory",
}
_SNAPSHOT_SUFFIX = "GeneratedDemoData"
_SNAPSHOT_END = date(2026, 11, 15)
_SNAPSHOT_NAMES = {
    view: f"{prefix}_{_SNAPSHOT_END:%Y%m%d}T235959_GMT_{_SNAPSHOT_SUFFIX}.zip" for view, prefix in _EXPORT_NAMES.items()
}
_REPO_DATA_DIR = Path(__file__).parents[3] / "data"
_RAW_GITHUB_DATA_URL = "https://raw.githubusercontent.com/pegasystems/pega-datascientist-tools/master/data"


class DemoData:
    """Datamart, prediction and Interaction History views of one sample world.

    Views are built on first access, so asking only for the prediction does
    not pay for the datamart or the Interaction History. Create instances
    with :func:`demo_data`.
    """

    def __init__(
        self,
        world: SampleWorld | None,
        interactions: int = DEFAULT_INTERACTIONS,
        *,
        source: Literal["generated", "snapshot"] = "generated",
    ):
        self.world = world
        """The immutable world all generated views derive from; ``None`` for snapshots."""
        self.interactions = interactions
        """Number of decisions sampled for the Interaction History view."""
        self.source = source
        """Whether the views are generated or loaded from the versioned snapshot."""

    @cached_property
    def datamart(self) -> ADMDatamart:
        """ADM model snapshots and predictor binning of the world."""
        from ..adm.ADMDatamart import ADMDatamart
        from ._adm import model_snapshots
        from ._binning import predictor_binning

        if self.source == "snapshot":
            return ADMDatamart.from_ds_export(
                model_filename=_SNAPSHOT_NAMES["model_data"],
                predictor_filename=_SNAPSHOT_NAMES["predictor_data"],
                base_path=_snapshot_base_path(),
            )
        if self.world is None:
            raise RuntimeError("Generated demo data requires a sample world.")
        return ADMDatamart(
            model_df=model_snapshots(self.world),
            predictor_df=predictor_binning(self.world),
        )

    @cached_property
    def prediction(self) -> Prediction:
        """Prediction Studio snapshots of the world."""
        from ..prediction.Prediction import Prediction
        from ._prediction import prediction_snapshots

        if self.source == "snapshot":
            return Prediction.from_ds_export(
                _SNAPSHOT_NAMES["prediction_data"],
                _snapshot_base_path(),
            )
        if self.world is None:
            raise RuntimeError("Generated demo data requires a sample world.")
        return Prediction(prediction_snapshots(self.world))

    @cached_property
    def ih(self) -> IH:
        """Interaction History sampled from the world."""
        from ..ih.IH import IH
        from ._ih import interaction_history, outcome_labels

        if self.source == "snapshot":
            return IH.from_ds_export(_snapshot_file("ih_data"))
        if self.world is None:
            raise RuntimeError("Generated demo data requires a sample world.")
        ih = IH(interaction_history(self.world, self.interactions))
        ih.outcome_labels_used = outcome_labels(self.world)
        return ih

    def write_ds_export(
        self,
        path: str | os.PathLike = ".",
        *,
        suffix: str | None = None,
    ) -> dict[str, Path]:
        """Write the demo views as Pega dataset export zips.

        Each file is a zip holding ``data.json`` in NDJSON format with
        datamart-style field names, named like a Pega dataset export with
        the last day of the window as timestamp. The files can be read back
        with :meth:`pdstools.ADMDatamart.from_ds_export`,
        :meth:`pdstools.Prediction.from_ds_export`, and
        :meth:`pdstools.IH.from_ds_export`.

        Parameters
        ----------
        path : str or os.PathLike, default "."
            Existing or new directory to write into.
        suffix : str or None, keyword-only, default None
            Optional suffix appended after the timestamp in each filename.

        Returns
        -------
        dict[str, Path]
            Written file per view: ``model_data``, ``predictor_data``,
            ``prediction_data`` and ``ih_data``.

        Examples
        --------
        >>> from pdstools import demo_data
        >>> files = demo_data(end="canonical").write_ds_export("exports")
        >>> ADMDatamart.from_ds_export(files["model_data"].name, files["predictor_data"].name, "exports")
        """
        if self.world is None:
            raise ValueError("Only generated demo data can be written as a Pega export.")

        import polars as pl

        from ._adm import model_snapshots
        from ._binning import predictor_binning
        from ._ih import interaction_history
        from ._prediction import prediction_snapshots

        ih_frame = interaction_history(self.world, self.interactions).with_columns(
            pxOutcomeTime=pl.col("pxOutcomeTime").dt.strftime("%Y%m%dT%H%M%S%.3f GMT"),
        )
        frames = {
            "model_data": model_snapshots(self.world),
            "predictor_data": predictor_binning(self.world),
            "prediction_data": prediction_snapshots(self.world),
            "ih_data": ih_frame,
        }
        stamp = f"{self.world.dates[-1]:%Y%m%d}T235959_GMT"
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        written = {}
        for view, frame in frames.items():
            file_suffix = f"_{suffix}" if suffix else ""
            file = target / f"{_EXPORT_NAMES[view]}_{stamp}{file_suffix}.zip"
            with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("data.json", _ndjson(frame))
            written[view] = file
        return written


def _snapshot_base_path() -> Path | str:
    if all((_REPO_DATA_DIR / filename).is_file() for filename in _SNAPSHOT_NAMES.values()):
        return _REPO_DATA_DIR
    return _RAW_GITHUB_DATA_URL


def _snapshot_file(view: str) -> Path | str:
    base_path = _snapshot_base_path()
    if isinstance(base_path, Path):
        return base_path / _SNAPSHOT_NAMES[view]
    return f"{base_path}/{_SNAPSHOT_NAMES[view]}"


def _ndjson(frame: pl.LazyFrame) -> bytes:
    lines = frame.collect().write_ndjson()
    return lines.encode()


@cache
def _cached_demo(world: SampleWorld, interactions: int) -> DemoData:
    return DemoData(world, interactions)


@cache
def _cached_snapshot_demo() -> DemoData:
    return DemoData(None, source="snapshot")


def demo_data(
    days: int = DEFAULT_DAYS,
    *,
    seed: int | None = DEFAULT_SEED,
    end: EndDate = None,
    scale: float = 1.0,
    interactions: int = DEFAULT_INTERACTIONS,
    source: Literal["generated", "snapshot"] = "generated",
) -> DemoData:
    """Synthetic CDH demo data: datamart, prediction and Interaction History.

    By default, all views come from one seeded synthetic world generated on
    demand, so models, predictions and interactions tell the same story.
    ``source="snapshot"`` instead loads the fixed, versioned Pega export zips
    stored in the repository or from its GitHub raw files. Snapshot mode is
    intended for application startup and only supports the fixed snapshot settings.
    Results are cached per process: repeated calls with the same arguments
    return the same object, so treat its views as read-only.

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
    source : {"generated", "snapshot"}, default "generated"
        Generate data on demand, or load the fixed versioned snapshot. Snapshot
        mode uses 182 days ending 2026-11-15, seed 42, scale 1 and 100000
        interactions; custom generation options are not supported.

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
    >>> app_demo = demo_data(source="snapshot")
    >>> app_demo.datamart.plot.bubble_chart()
    """
    if source not in ("generated", "snapshot"):
        raise ValueError("source must be either 'generated' or 'snapshot'.")
    if source == "snapshot":
        if (
            days != DEFAULT_DAYS
            or seed != DEFAULT_SEED
            or end is not None
            or scale != 1.0
            or interactions != DEFAULT_INTERACTIONS
        ):
            raise ValueError(
                "The stored demo snapshot has fixed settings: 182 days, seed 42, "
                "end date 2026-11-15, scale 1 and 100000 interactions.",
            )
        return _cached_snapshot_demo()

    world = sample_world(seed=seed, days=days, end=end, scale=scale)
    if seed is None:
        return DemoData(world, int(interactions))
    return _cached_demo(world, int(interactions))
