"""
Testing the functionality of the pega_template module
"""

import plotly.io as pio
import plotly.graph_objects as go
import pytest
from pdstools.utils import pega_template


def test_colorway_defined():
    """Test that the colorway is defined with the expected colors"""
    assert isinstance(pega_template.colorway, list)
    assert len(pega_template.colorway) > 0
    assert pega_template.colorway[0] == "#001F5F"  # dark blue
    assert pega_template.colorway[2] == "#F76923"  # orange


def test_color_scales_defined():
    """Test that the color scales are defined"""
    assert isinstance(pega_template.neutral_positive, list)
    assert isinstance(pega_template.negative_positive, list)
    assert isinstance(pega_template.positive_negative, list)
    assert isinstance(pega_template.performance, list)
    assert isinstance(pega_template.success, list)


def test_pega_template_registered():
    """Test that the pega template is registered with plotly"""
    assert "pega" in pio.templates
    
    # Check that the template has the expected properties
    template = pio.templates["pega"]
    assert isinstance(template, go.layout.Template)
    # The colorway in the template is a tuple, but our reference is a list
    # Convert both to sets for comparison
    assert set(template.layout.colorway) == set(pega_template.colorway)
    assert template.layout.hovermode == "closest"


def test_color_templates_registered():
    """Test that the color templates are registered with plotly"""
    color_templates = [
        "neutral_positive",
        "negative_positive",
        "performance",
        "success"
    ]
    
    for template_name in color_templates:
        assert template_name in pio.templates
        template = pio.templates[template_name]
        assert isinstance(template, go.layout.Template)
        # The colorway in the template is a tuple, but our reference is a list
        # Convert both to sets for comparison
        assert set(template.layout.colorway) == set(pega_template.colorway)


def test_template_usage():
    """Test using the template to create a figure"""
    # Create a simple figure with the pega template
    fig = go.Figure(
        data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])],
        layout=go.Layout(template="pega")
    )
    
    # Check that the template was applied
    # The colorway in the template is a tuple, but our reference is a list
    # Convert both to sets for comparison
    assert set(fig.layout.template.layout.colorway) == set(pega_template.colorway)
    
    # The hovermode is set in the template but not automatically applied to the figure
    # We would need to update the figure or check the template's hovermode instead
    assert fig.layout.template.layout.hovermode == "closest"
