"""Shared constants and helpers for decision-analyzer plot submodules."""

DEFAULT_BOXPLOT_POINT_CAP = 20000


def _boxplot_point_cap(self) -> int:
    sample_size = getattr(self._decision_data, "sample_size", None)
    if isinstance(sample_size, int) and sample_size > 0:
        return sample_size
    return DEFAULT_BOXPLOT_POINT_CAP
