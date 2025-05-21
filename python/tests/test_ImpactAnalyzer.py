"""
Testing the functionality of the ImpactAnalyzer class
"""

import os
import json
import pathlib

import polars as pl
import pytest
from pdstools import ImpactAnalyzer
from pathlib import Path


def test_from_pdc():
    """Test creating an ImpactAnalyzer instance from JSON data"""
    # Path to the sample data - use absolute path from project root
    sample_path = (
        pathlib.Path(__file__).parents[2] / "data/ia/CDH_Metrics_ImpactAnalyzer.json"
    )

    # Create an ImpactAnalyzer instance
    analyzer = ImpactAnalyzer.from_pdc(sample_path)

    # Verify the instance was created correctly
    assert isinstance(analyzer, ImpactAnalyzer)
    assert isinstance(analyzer.ia_data, pl.LazyFrame)

    # Verify the data was loaded correctly
    collected_data = analyzer.ia_data.collect()
    assert collected_data.height > 0
    assert "Experiment" in collected_data.columns

    # Verify summarizations
    agg = analyzer._summarize(by=[]).collect()
    assert "CTR" in agg.columns
    assert "CTR_Lift" in agg.columns
    assert "Value_Lift" in agg.columns

    # Verify against the numbers from Pega in the data
    original_pdc_data = (
        ImpactAnalyzer.from_pdc(
            sample_path,
            return_df=True,
        )
        .filter(pl.col("IsActive") & (pl.col("ChannelName") == "SMS"))
        .collect()
    )

    recreated_summary_data = (
        analyzer.plot.overview(by=["Channel"], return_df=True)
        .collect()
        .filter(Channel="SMS")
    )
    assert [round(x, 8) for x in original_pdc_data["EngagementLift"].to_list()] == [round(x, 8) for x in recreated_summary_data["CTR_Lift"].to_list()[1:]]


def test_overall_summary():
    """Test creating an ImpactAnalyzer instance from JSON data"""
    # Path to the sample data - use absolute path from project root
    sample_path = (
        pathlib.Path(__file__).parents[2] / "data/ia/CDH_Metrics_ImpactAnalyzer.json"
    )

    # Create an ImpactAnalyzer instance
    analyzer = ImpactAnalyzer.from_pdc(sample_path)
    summary = analyzer.overall_summary().collect()
    
    assert summary.width == 4
    assert [round(x,6) for x in summary['CTR_Lift'].to_list()] == [0.0, 0.002653, 0.009563, 0.002215, 0.003784]

def test_summary_by_channel():
    """Test creating an ImpactAnalyzer instance from JSON data"""
    # Path to the sample data - use absolute path from project root
    sample_path = (
        pathlib.Path(__file__).parents[2] / "data/ia/CDH_Metrics_ImpactAnalyzer.json"
    )

    # Create an ImpactAnalyzer instance
    analyzer = ImpactAnalyzer.from_pdc(sample_path)
    summary = analyzer.summary_by_channel().collect()
    
    assert summary.width == 5
    assert summary.height == 25
