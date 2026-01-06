"""Resources module for pdstools configuration data."""

from pathlib import Path

_RESOURCES_DIR = Path(__file__).parent


def get_metric_limits_path() -> Path:
    """Get the path to the MetricLimits.csv file.

    Returns
    -------
    Path
        Path to the MetricLimits.csv resource file containing best practice
        limits for various metrics used in pdstools.
    """
    return _RESOURCES_DIR / "MetricLimits.csv"


__all__ = ["get_metric_limits_path"]
