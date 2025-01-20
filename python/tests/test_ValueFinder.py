"""
Testing the functionality of the ValueFinder class
"""

import pathlib

import polars as pl
import pytest
from pdstools import ValueFinder, datasets, read_ds_export

base_path = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture
def vf():
    return ValueFinder.from_ds_export(
        filename="Data-Insights_pyValueFinder_20210824T112615_GMT.zip",
        base_path=base_path / "data",
    )


def test_sample():
    datasets.sample_value_finder()


def test_from_ds():
    ValueFinder.from_ds_export(
        filename="Data-Insights_pyValueFinder_20210824T112615_GMT.zip",
        base_path=base_path / "data",
    )


def test_with_direct_df():
    df = read_ds_export(
        filename="Data-Insights_pyValueFinder_20210824T112615_GMT.zip",
        path=base_path / "data",
    )
    ValueFinder(df=df)


def test_from_dataflow(): ...


def test_with_n_customers():
    assert (
        ValueFinder.from_ds_export(
            filename="Data-Insights_pyValueFinder_20210824T112615_GMT.zip",
            base_path=base_path / "data",
            n_customers=10_000,
        ).n_customers
        == 10_000
    )


def test_custom_threshold():
    assert (
        ValueFinder.from_ds_export(
            filename="Data-Insights_pyValueFinder_20210824T112615_GMT.zip",
            base_path=base_path / "data",
            threshold=0.01,
        ).threshold
        == 0.01
    )


def test_query(vf: ValueFinder):
    _vf = ValueFinder.from_ds_export(
        filename="Data-Insights_pyValueFinder_20210824T112615_GMT.zip",
        base_path=base_path / "data",
        query=pl.col("Stage") != "Arbitration",
    )
    assert (
        _vf.df.select(pl.first().len()).collect().item()
        != vf.df.select(pl.first().len()).collect().item()
    )


def test_empty_data_ds():
    with pytest.raises(ValueError):
        ValueFinder.from_ds_export("NO_FILE")


def test_empty_data_df():
    with pytest.raises(FileNotFoundError):
        ValueFinder.from_dataflow_export(["UNKNOWN"])


def test_save_data(vf: ValueFinder):
    data = vf.save_data("cache")
    print(data)
    ValueFinder(pl.scan_ipc(data))


def test_customer_summary(vf: ValueFinder):
    summary = vf.aggregates.get_customer_summary().collect()
    assert summary.shape == (27_133, 8)
    assert summary.filter(CustomerID="Customer-1", Stage="Suitability").row(0) == (
        "Customer-1",
        "Suitability",
        0.2692307692307692,
        0.2465484951862448,
        0.2692307692307692,
        1,
        True,
        False,
    )


def test_counts_per_stage(vf: ValueFinder):
    counts = vf.aggregates.get_counts_per_stage().collect()
    assert counts.shape == (len(vf.nbad_stages), 4)
    assert vf.aggregates.get_counts_per_stage().filter(
        Stage="Eligibility"
    ).collect().row(0) == ("Eligibility", 6901, 357, 66)


def test_max_propensity_per_customer(vf: ValueFinder):
    assert (
        vf.aggregates.max_propensity_per_customer.filter(CustomerID="Customer-1").row(
            0
        )[2]
        == 0.2692307692307692
    )


def test_get_threshold_from_quantile(vf: ValueFinder):
    quantile = 0.05
    _threshold = 0.012096774193548388
    assert _threshold not in vf.aggregates._quantile_from_threshold

    threshold = vf.aggregates.get_threshold_from_quantile(quantile)
    assert threshold == _threshold
    assert vf.aggregates._quantile_from_threshold.get(threshold) == quantile


def test_get_counts_for_threshold(vf: ValueFinder):
    assert vf.aggregates.get_counts_for_threshold(0.05).filter(Stage="Arbitration").row(
        0
    ) == ("Arbitration", 5332, 1177, 815)


def test_plot_funnel_chart(vf: ValueFinder):
    assert (
        vf.plot.funnel_chart(return_df=True)
        .select(pl.col("Count").top_k(1))
        .collect()
        .item()
        == 2365
    )
    vf.plot.funnel_chart()


def test_propensity_distribution(vf: ValueFinder):
    vf.plot.propensity_distribution()


def test_propensity_threshold(vf: ValueFinder):
    vf.plot.propensity_threshold()


def test_get_thresholds(vf: ValueFinder):
    assert vf.plot._get_thresholds() == [0.012096774193548388]

    with pytest.raises(ValueError):
        vf.plot._get_thresholds([0.01], [0.01])

    assert vf.plot._get_thresholds([0.01]) == [0.01]

    thresholds = vf.plot._get_thresholds(None, [0.01])
    assert isinstance(thresholds, map)
    assert list(thresholds) == [0.005376344086021506]


def test_pie_charts(vf: ValueFinder):
    vf.plot.pie_charts()


def test_distribution_per_threshold(vf: ValueFinder):
    vf.plot.distribution_per_threshold()
