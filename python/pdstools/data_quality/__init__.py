from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdstools.data_quality._topic_data_quality import TopicDataQuality


def __getattr__(name: str):
    if name == "TopicDataQuality":
        from pdstools.data_quality._topic_data_quality import TopicDataQuality

        return TopicDataQuality
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TopicDataQuality"]
