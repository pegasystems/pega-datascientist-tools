"""Shared fixtures for the ADM test package."""

import polars as pl
import pytest
from pdstools import ADMDatamart, demo_data
from pdstools.sample_data._adm import model_snapshots
from pdstools.sample_data._binning import predictor_binning
from pdstools.sample_data._world import sample_world


@pytest.fixture(scope="session")
def demo_bundle():
    """Canonical demo-world frames shared by ADM tests."""
    return demo_data(end="canonical")


@pytest.fixture
def sample(demo_bundle):
    """Fresh datamart wrapper so test mutations do not leak across tests."""
    return ADMDatamart(
        model_df=demo_bundle.datamart.model_data,
        predictor_df=demo_bundle.datamart.predictor_data,
    )


@pytest.fixture(scope="session")
def sample_binaggregator():
    """BinAggregator over the Sales / Web / OmniAdaptiveModel slice of the demo world."""
    world = sample_world(end="canonical")
    sample = ADMDatamart(
        model_snapshots(world),
        predictor_binning(world),
        query=(
            (pl.col("Issue") == "Sales")
            & (pl.col("Channel") == "Web")
            & (pl.col("Configuration") == "OmniAdaptiveModel")
        ),
    )
    return sample.bin_aggregator
