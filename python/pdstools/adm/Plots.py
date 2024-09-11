from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .new_ADMDatamart import ADMDatamart


class Plots:
    def __init__(self, datamart: "ADMDatamart"):
        self.datamart = datamart

    def plot_bubble_chart(self): ...
