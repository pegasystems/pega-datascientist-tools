"""Regression tests for ``BinAggregator.accumulate_num_binnings``.

The method was refactored from a per-model Python loop calling
``combine_two_numbinnings`` into a single batched, vectorised cross-join +
group_by aggregation. These tests pin the contract:

* The original implementation is order-independent (verified empirically).
* The new batched implementation produces bit-equal (within 1e-9) results
  versus a verbatim copy of the old one across many synthetic inputs.
* Edge cases (no models, disjoint bins, identical bins, immature models).
* End-to-end equivalence via ``roll_up`` against the cdh_sample fixture.
"""

from __future__ import annotations

import logging
import random

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pdstools import datasets
from pdstools.adm.BinAggregator import BinAggregator


@pytest.fixture(scope="module")
def cdhsample_binaggregator():
    sample = datasets.cdh_sample(
        query=((pl.col("Issue").is_in(["Sales"])) & (pl.col("Direction") == "Inbound")),
    )
    return sample.bin_aggregator


def _accumulate_num_binnings_reference(
    ba: BinAggregator,
    predictor: str,
    modelids: list,
    target_binning: pl.DataFrame,
) -> pl.DataFrame:
    """Verbatim copy of the OLD ``accumulate_num_binnings`` body.

    Kept here so we can lock the new batched implementation against the
    historical sequential one within float tolerance.
    """
    for id in modelids:
        source_binning = ba.get_source_numbinning(predictor, id)
        target_binning = ba.combine_two_numbinnings(source_binning, target_binning)
    return target_binning


def _make_target(low: float, high: float, n: int) -> pl.DataFrame:
    """Build an empty target binning with ``n`` evenly-spaced bins."""
    import numpy as np

    edges = np.linspace(low, high, n + 1)
    return pl.DataFrame(
        {
            "PredictorName": ["P"] * n,
            "BinIndex": list(range(1, n + 1)),
            "BinLowerBound": edges[:-1].tolist(),
            "BinUpperBound": edges[1:].tolist(),
            "BinSymbol": [f"<{edges[i + 1]:.2f}" for i in range(n)],
            "Lift": [0.0] * n,
            "BinResponses": [0.0] * n,
            "BinCoverage": [0.0] * n,
            "Models": [0] * n,
        },
    )


def _make_source(
    model_id: str,
    edges: list[float],
    lifts: list[float],
    responses: list[float],
) -> pl.DataFrame:
    n = len(edges) - 1
    assert len(lifts) == n and len(responses) == n
    return pl.DataFrame(
        {
            "ModelID": [model_id] * n,
            "PredictorName": ["P"] * n,
            "BinIndex": list(range(1, n + 1)),
            "BinType": ["INTERVAL"] * n,
            "BinLowerBound": [float(x) for x in edges[:-1]],
            "BinUpperBound": [float(x) for x in edges[1:]],
            "BinSymbol": [f"<{edges[i + 1]:.2f}" for i in range(n)],
            "Lift": [float(x) for x in lifts],
            "BinResponses": [float(x) for x in responses],
        },
    )


# ---------------------------------------------------------------------------
# (a) Order-independence of the OLD implementation.
# ---------------------------------------------------------------------------


def test_old_accumulate_is_order_independent(cdhsample_binaggregator):
    """Old sequential accumulate must be order-independent within ~1e-9.

    This is the invariant that justifies the batched refactor. If this ever
    fails we have a hidden ordering bug — STOP and investigate before
    trusting the batched implementation.
    """
    ba = cdhsample_binaggregator
    predictor = "Customer.AnnualIncome"
    ids = ba.all_predictorbinning.select(pl.col("ModelID").unique().sort()).collect()["ModelID"].to_list()
    assert len(ids) >= 2

    target_template = ba.create_empty_numbinning(
        predictor=predictor,
        n=10,
        distribution="lin",
        boundaries=[],
    )

    runs = []
    for seed in [0, 1, 2]:
        rng = random.Random(seed)
        order = ids[:]
        rng.shuffle(order)
        runs.append(
            _accumulate_num_binnings_reference(
                ba,
                predictor,
                order,
                target_template.clone(),
            ),
        )

    base = runs[0]
    for other in runs[1:]:
        assert_frame_equal(
            base.select("Lift", "BinResponses", "BinCoverage", "Models"),
            other.select("Lift", "BinResponses", "BinCoverage", "Models"),
            check_exact=False,
            abs_tol=1e-9,
            check_dtypes=False,
            rel_tol=1e-9,
        )


# ---------------------------------------------------------------------------
# (b) Bit-exact (within float tolerance) equivalence: new vs reference.
# ---------------------------------------------------------------------------


def _make_synthetic_case(seed: int, n_models: int, n_bins: int, *, disjoint: bool = False, immature: bool = False):
    """Build a (target, [sources], modelids) tuple from ``seed``."""
    rng = random.Random(seed)
    target = _make_target(0.0, 100.0, n_bins)

    sources = []
    modelids = []
    for m in range(n_models):
        # Each model has its own bin layout, possibly outside [0,100].
        if disjoint:
            base = 200.0 + 50 * m  # entirely above target range
            edges = [base + i * 5.0 for i in range(n_bins + 1)]
        else:
            n_sb = rng.randint(2, max(3, n_bins))
            raw = sorted(rng.uniform(-10, 110) for _ in range(n_sb + 1))
            # ensure strictly increasing
            edges = []
            prev = raw[0]
            edges.append(prev)
            for x in raw[1:]:
                if x <= prev:
                    x = prev + 1e-3
                edges.append(x)
                prev = x

        nb = len(edges) - 1
        if immature and m % 2 == 0:
            responses = [0.0] * nb
        else:
            responses = [rng.uniform(0, 5000) for _ in range(nb)]
        lifts = [rng.uniform(-1.5, 2.0) for _ in range(nb)]
        mid = f"model_{seed}_{m}"
        sources.append(_make_source(mid, edges, lifts, responses))
        modelids.append(mid)

    return target, sources, modelids


@pytest.mark.parametrize(
    "seed,n_models,n_bins,kw",
    [
        (1, 1, 5, {}),
        (2, 5, 10, {}),
        (3, 20, 8, {}),
        (4, 5, 5, {"disjoint": True}),
        (5, 6, 12, {"immature": True}),
    ],
)
def test_batched_matches_reference(
    cdhsample_binaggregator,
    seed,
    n_models,
    n_bins,
    kw,
):
    """Batched ``_combine_many_numbinnings`` ≡ verbatim sequential reference."""
    ba = cdhsample_binaggregator
    target, sources, modelids = _make_synthetic_case(
        seed,
        n_models,
        n_bins,
        **kw,
    )

    ref = target.clone()
    for src in sources:
        ref = ba.combine_two_numbinnings(src, ref)

    new = ba._combine_many_numbinnings(
        pl.concat(sources, how="vertical_relaxed"),
        target.clone(),
        n_models=len(modelids),
    )

    cols = ["Lift", "BinResponses", "BinCoverage", "Models"]
    assert_frame_equal(
        ref.select(cols),
        new.select(cols),
        check_exact=False,
        abs_tol=1e-9,
        check_dtypes=False,
        rel_tol=1e-9,
    )


# ---------------------------------------------------------------------------
# (c) Edge cases.
# ---------------------------------------------------------------------------


def test_no_sources_leaves_target_unchanged(cdhsample_binaggregator):
    """Empty modelids: target untouched, Models incremented by 0."""
    ba = cdhsample_binaggregator
    target = _make_target(0.0, 100.0, 5)
    out = ba._combine_many_numbinnings(
        pl.concat([_make_source("ignored", [0, 1], [0.0], [0.0])], how="vertical_relaxed").head(0),
        target.clone(),
        n_models=0,
    )
    assert_frame_equal(out, target)


def test_disjoint_bins_only_increment_models(cdhsample_binaggregator):
    """Source entirely outside target range: only Models changes."""
    ba = cdhsample_binaggregator
    target = _make_target(0.0, 100.0, 4)
    src = _make_source("m", [200.0, 210.0, 220.0], [1.0, -0.5], [100.0, 200.0])
    out = ba._combine_many_numbinnings(src, target.clone(), n_models=1)

    assert out["Lift"].to_list() == target["Lift"].to_list()
    assert out["BinResponses"].to_list() == target["BinResponses"].to_list()
    assert out["BinCoverage"].to_list() == target["BinCoverage"].to_list()
    assert out["Models"].to_list() == [1] * 4


def test_source_identical_to_target_passes_through(cdhsample_binaggregator):
    """Source bins == target bins: target adopts source lift/responses exactly."""
    ba = cdhsample_binaggregator
    target = _make_target(0.0, 100.0, 4)
    edges = [0.0, 25.0, 50.0, 75.0, 100.0]
    lifts = [0.5, -0.25, 1.5, -1.0]
    resps = [100.0, 200.0, 300.0, 400.0]
    src = _make_source("m", edges, lifts, resps)

    out = ba._combine_many_numbinnings(src, target.clone(), n_models=1)

    assert out["Lift"].to_list() == pytest.approx(lifts, abs=1e-12)
    assert out["BinResponses"].to_list() == pytest.approx(resps, abs=1e-9)
    # full coverage = 1.0 per bin
    assert out["BinCoverage"].to_list() == pytest.approx([1.0] * 4, abs=1e-12)
    assert out["Models"].to_list() == [1] * 4


def test_all_zero_responses_immature_model(cdhsample_binaggregator):
    """Immature model (BinResponses = 0): contributes lift but no responses."""
    ba = cdhsample_binaggregator
    target = _make_target(0.0, 100.0, 4)
    src = _make_source("m", [0.0, 50.0, 100.0], [1.0, -1.0], [0.0, 0.0])
    out = ba._combine_many_numbinnings(src, target.clone(), n_models=1)

    assert out["BinResponses"].to_list() == [0.0] * 4
    # Each target bin is fully covered by exactly one source bin.
    assert out["BinCoverage"].to_list() == pytest.approx([1.0] * 4, abs=1e-12)
    assert out["Lift"].to_list() == pytest.approx([1.0, 1.0, -1.0, -1.0], abs=1e-12)


# ---------------------------------------------------------------------------
# (d) End-to-end via ``roll_up`` against the public cdh_sample fixture.
# ---------------------------------------------------------------------------


def test_roll_up_matches_reference_implementation(cdhsample_binaggregator):
    """``roll_up`` (which calls accumulate_num_binnings) matches old impl."""
    ba = cdhsample_binaggregator
    predictor = "Customer.AnnualIncome"

    new_result = ba.roll_up(predictor, n=10, return_df=True)

    ids = ba.all_predictorbinning.select(pl.col("ModelID").unique().sort()).collect()["ModelID"].to_list()
    target_template = ba.create_empty_numbinning(
        predictor=predictor,
        n=10,
        distribution="lin",
        boundaries=[],
    )
    ref_result = _accumulate_num_binnings_reference(
        ba,
        predictor,
        ids,
        target_template.clone(),
    )

    cols = ["Lift", "BinResponses", "BinCoverage", "Models"]
    assert_frame_equal(
        ref_result.select(cols),
        new_result.select(cols),
        check_exact=False,
        abs_tol=1e-9,
        check_dtypes=False,
        rel_tol=1e-9,
    )


# ---------------------------------------------------------------------------
# Benchmark (manual: ``python -m python.tests.adm.test_BinAggregator_accumulate``)
# ---------------------------------------------------------------------------


def _benchmark():
    import time

    logging.basicConfig(level=logging.WARNING)

    sample = datasets.cdh_sample()
    ba = sample.bin_aggregator
    predictor = "Customer.AnnualIncome"

    # Synthesise 50 source binnings × 20 bins each by replicating the few real ones.
    base_ids = ba.all_predictorbinning.select(pl.col("ModelID").unique().sort()).collect()["ModelID"].to_list()
    real_sources = [ba.get_source_numbinning(predictor, mid) for mid in base_ids]

    rng = random.Random(0)
    sources = []
    for i in range(50):
        s = real_sources[i % len(real_sources)].clone()
        s = s.with_columns(pl.lit(f"synth_{i}").alias("ModelID"))
        # jitter the bin bounds a touch
        jitter = rng.uniform(-100.0, 100.0)
        s = s.with_columns(
            (pl.col("BinLowerBound") + jitter).alias("BinLowerBound"),
            (pl.col("BinUpperBound") + jitter).alias("BinUpperBound"),
        )
        sources.append(s)

    target_template = ba.create_empty_numbinning(
        predictor=predictor,
        n=20,
        distribution="lin",
        boundaries=[],
    )

    # ---- old (sequential) ----
    t0 = time.perf_counter()
    ref = target_template.clone()
    for src in sources:
        ref = ba.combine_two_numbinnings(src, ref)
    old_dt = time.perf_counter() - t0

    # ---- new (batched) ----
    t0 = time.perf_counter()
    new = ba._combine_many_numbinnings(
        pl.concat(sources, how="vertical_relaxed"),
        target_template.clone(),
        n_models=len(sources),
    )
    new_dt = time.perf_counter() - t0

    print(f"sequential combine_two_numbinnings x{len(sources)}: {old_dt:.3f}s")
    print(f"batched _combine_many_numbinnings           : {new_dt:.3f}s")
    print(f"speedup: {old_dt / new_dt:.1f}x")

    cols = ["Lift", "BinResponses", "BinCoverage", "Models"]
    assert_frame_equal(
        ref.select(cols),
        new.select(cols),
        check_exact=False,
        abs_tol=1e-9,
        rel_tol=1e-9,
    )
    print("✓ outputs equivalent within 1e-9")


if __name__ == "__main__":
    _benchmark()
