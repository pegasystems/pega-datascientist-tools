"""
Testing the functionality of the ImpactAnalyzer class
"""

import os
import json
import pathlib

import polars as pl
import pytest
from pdstools import ImpactAnalyzer


def test_from_pdc():
    """Test creating an ImpactAnalyzer instance from JSON data"""
    # Path to the sample data - use absolute path from project root
    sample_path = pathlib.Path(__file__).parents[2] / "data/ia/CDH_Metrics_ImpactAnalyzer.json"
    
    # Create an ImpactAnalyzer instance
    analyzer = ImpactAnalyzer.from_pdc(sample_path)
    
    # Verify the instance was created correctly
    assert isinstance(analyzer, ImpactAnalyzer)
    assert isinstance(analyzer.ia_data, pl.LazyFrame)
    
    # Verify the data was loaded correctly
    collected_data = analyzer.ia_data.collect()
    assert collected_data.height > 0
    assert "ExperimentName" in collected_data.columns
    assert "CTR" in collected_data.columns
    # assert "CTR_Lift" in collected_data.columns
    # assert "Value_Lift" in collected_data.columns

