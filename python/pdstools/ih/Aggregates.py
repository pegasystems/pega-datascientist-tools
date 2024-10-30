from typing import TYPE_CHECKING

from ..utils.namespaces import LazyNamespace

if TYPE_CHECKING:
    from .IH import IH as IH_Class


class Aggregates(LazyNamespace):
    def __init__(self, ih: "IH_Class"):
        self.ih = ih
