__all__ = ["ExplanationsExplorer"]

from .ExplanationsDataLoader import ExplanationsDataLoader
from .ExplanationsUtils import ContextInfo, _CONTRIBUTION_TYPE
from .ExplanationsDataPlotter import ExplanationsDataPlotter
from ipywidgets import widgets
from IPython.display import display
from typing import List, Optional


class ExplanationsExplorer:
    _ANY_CONTEXT = "Any"
    _context_filter: ContextInfo = {}
    _context_infos: List[ContextInfo] = []
    _selected_context: Optional[ContextInfo] = None

    def __init__(self,
                 data_location: str = ".tmp/aggregated_data"):
        self.data_loader = ExplanationsDataLoader(
            data_location=data_location
        )

        self._init_context_info()

    def _init_context_info(self):
        self._context_infos = self.data_loader.get_context_infos()
        if not self._context_infos:
            raise ValueError("No context information available. Please load data first.")

        self._context_filter = {key: self._ANY_CONTEXT for key in self._context_infos[0].keys()}

    
    def context_selector(self):
        filtered_context_infos = self._filter_context_infos()
        context_filter_widgets = {}

        def on_filter_change(change):
            nonlocal context_filter_widgets, context_select_widget, filtered_context_infos
            if change['new'] in context_filter_widgets[change['owner'].description].options:
                self._context_filter[change['owner'].description] = change['new']
                filtered_context_infos = self._filter_context_infos()

            for name, widget in [item for item in context_filter_widgets.items() if item[0] != change['owner'].description or self._context_filter[item[0]] == self._ANY_CONTEXT]:
                widget.options = [self._ANY_CONTEXT] + sorted(set(context_info[name] for context_info in filtered_context_infos))

            context_select_widget.options = [self._ANY_CONTEXT] + list(self._context_infos_to_dictionary(filtered_context_infos).keys())

        def on_context_select(change):
            nonlocal filtered_context_infos
            if change['new'] == self._ANY_CONTEXT:
                self.set_selected_context(None)
            else:
                context_info = self._context_infos_to_dictionary(filtered_context_infos).get(change['new'])
                if context_info:
                    self.set_selected_context(context_info)

        context_filter_widgets = {name: widgets.Combobox(
            options=[self._ANY_CONTEXT] + sorted(set(context_info[name] for context_info in filtered_context_infos)),
            value=self._ANY_CONTEXT,
            description=name,
            layout=widgets.Layout(width='auto'),
            debounce=2000,
        ) for name in self._context_filter.keys()}

        for context_filter_widget in context_filter_widgets.values():
            context_filter_widget.observe(on_filter_change, names='value')

        context_filter_widgets_container = widgets.VBox(list(context_filter_widgets.values()))
        
        context_select_widget = widgets.Select(
            options=[self._ANY_CONTEXT] + list(self._context_infos_to_dictionary(filtered_context_infos).keys()),
            description='Context Info',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='auto'),
            rows=self._context_filter.keys().__len__() + 1
        )
        context_select_widget.observe(on_context_select, names='value')

        filters_header = widgets.HTML(description="", value="<h3>Select Context Filters</>", layout=widgets.Layout(width='auto'))
        context_header = widgets.HTML(description="", value="<h3>Select from Filtered Contexts</>",  layout=widgets.Layout(width='auto'))

        context_widget = widgets.GridBox(
            [filters_header, context_header, context_filter_widgets_container, context_select_widget],
            layout=widgets.Layout(
                width='100%',
                grid_template_columns='auto auto',
                grid_template_rows='auto auto',
                grid_gap='20px',
                align_items='center',
            )
        )

        display(context_widget)

    def plot_contributions(self, 
                           top_n: int = 10, 
                           top_k: int = 10,
                           descending: bool = True,
                           missing: bool = True,
                           remaining: bool = True,
                           contricontribution_calculation: _CONTRIBUTION_TYPE = _CONTRIBUTION_TYPE.CONTRIBUTION):
        if self._selected_context is None:
            overall_plot, predictor_plots = ExplanationsDataPlotter.plot_contributions_for_overall(self.data_loader, top_n, top_k, descending, missing, remaining, contricontribution_calculation)
            for plot in [overall_plot] + predictor_plots:
                display(plot)
        else:
            context_plot, overall_plot, predictor_plots = ExplanationsDataPlotter.plot_contributions_by_context(self.data_loader, self._selected_context, top_n, top_k, descending, missing, remaining, contricontribution_calculation)  
            for plot in [context_plot, overall_plot] + predictor_plots:
                display(plot)


    def set_selected_context(self, context_info: Optional[ContextInfo]):
        self._selected_context = context_info or None

    def _context_info_to_string(self, context_info: ContextInfo) -> str:
        return ' - '.join(f"{value}".strip() for value in context_info.values())

    def _filter_context_infos(self) -> List[ContextInfo]:
        filtered_context_infos = []
        for context_info in self._context_infos:
            if all(value == self._ANY_CONTEXT or context_info[key] == value for key, value in self._context_filter.items()):
                filtered_context_infos.append(context_info)

        return filtered_context_infos 

    def _context_infos_to_dictionary(self, context_infos: ContextInfo) -> dict[str, ContextInfo]:
        return {(self._context_info_to_string(context_info)): context_info for context_info in context_infos}
