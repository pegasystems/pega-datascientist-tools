"""Plot utilities for pdstools visualizations."""

from typing import Any, Dict, List, Tuple, Union

# Colorscales for metric visualizations in Plotly charts
# These define continuous color gradients based on metric values

COLORSCALES: Dict[str, Any] = {
    "Performance": [
        (0, "#d91c29"),  # Red - poor performance
        (0.01, "#F76923"),  # Orange - below threshold
        (0.3, "#20aa50"),  # Green - acceptable
        (0.8, "#20aa50"),  # Green - good
        (1, "#0000FF"),  # Blue - exceptional (overfit?)
    ],
    "SuccessRate": [
        (0, "#d91c29"),  # Red - no success
        (0.01, "#F76923"),  # Orange - low success
        (0.5, "#F76923"),  # Orange - moderate
        (1, "#20aa50"),  # Green - high success
    ],
    "other": ["#d91c29", "#F76923", "#20aa50"],  # Default: Red -> Orange -> Green
}


def get_colorscale(
    metric: str, default: str = "other"
) -> Union[List[Tuple[float, str]], List[str]]:
    """Get the colorscale for a metric.

    Parameters
    ----------
    metric : str
        The metric name to look up (e.g., "Performance", "SuccessRate").
    default : str, optional
        The default colorscale key to use if metric not found, by default "other".

    Returns
    -------
    Union[List[Tuple[float, str]], List[str]]
        A Plotly-compatible colorscale (list of (position, color) tuples or list of colors).

    Examples
    --------
    >>> get_colorscale("Performance")
    [(0, '#d91c29'), (0.01, '#F76923'), (0.3, '#20aa50'), (0.8, '#20aa50'), (1, '#0000FF')]
    >>> get_colorscale("UnknownMetric")
    ['#d91c29', '#F76923', '#20aa50']
    """
    return COLORSCALES.get(metric, COLORSCALES.get(default, COLORSCALES["other"]))
