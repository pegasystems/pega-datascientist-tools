__all__ = ["ExplanationsExplorer"]

from .ExplanationsDataLoader import ExplanationsDataLoader
from .ExplanationsUtils import ContextInfo, ContextOperations, _CONTRIBUTION_TYPE
from .ExplanationsDataPlotter import ExplanationsDataPlotter
from ipywidgets import widgets
from IPython.display import display
from typing import List, Optional, cast


class ExplanationsExplorer:
    _ANY_CONTEXT = "Any"
    _CHANGED_VALUE = "new"
    _CHANGED_WIDGET = "owner"

    _context_selector_widget: Optional[widgets.Select]

    def __init__(self, data_location: str = ".tmp/aggregated_data"):
        self.data_loader = ExplanationsDataLoader(data_location=data_location)
        self._context_operations = ContextOperations(self.data_loader.df_contextual)

        self._raw_context_info_list: List[ContextInfo] = []
        self._filtered_context_info_list: List[ContextInfo] = []

        self._selected_context_key_values = {
            key: self._ANY_CONTEXT
            for key in self._context_operations.get_context_keys()
        }

        self._init_context_info()

        # init widget objects
        self._context_combobox_widgets: dict[str, widgets.Combobox] = {}
        self._context_selector_widget = None

    def _init_context_info(self):
        self._raw_context_info_list = self._context_operations.get_list()
        if not self._raw_context_info_list:
            raise ValueError(
                "No context information available. Please load data first."
            )

    def display_context_selector(self):
        self._filtered_context_info_list = self._filter_context_infos()

        # create widgets for each context key
        self._context_combobox_widgets = self._get_context_combobox_widgets()
        context_picker_container = widgets.VBox(
            list(self._context_combobox_widgets.values())
        )

        # create widget for list of filtered context selector
        self._context_selector_widget = self._get_context_selector_widget()

        context_picker_header = widgets.HTML(
            description="",
            value="<h3>Select Context Filters</>",
            layout=widgets.Layout(width="auto"),
        )
        context_list_header = widgets.HTML(
            description="",
            value="<h3>Contexts available from selection, pick one</>",
            layout=widgets.Layout(width="auto"),
        )

        grid = widgets.GridBox(
            [
                context_picker_header,
                context_list_header,
                context_picker_container,
                self._context_selector_widget,
            ],
            layout=widgets.Layout(
                width="100%",
                grid_template_columns="auto auto",
                grid_template_rows="auto auto",
                grid_gap="20px",
                align_items="center",
                margin="0 0 30px 0",  # top, right, bottom, left
            ),
        )

        display(grid)

    def plot_contributions(
        self,
        top_n: int = 10,
        top_k: int = 10,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_calculation: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value,
    ):
        contribution_type = _CONTRIBUTION_TYPE.validate_and_get_type(
            contribution_calculation
        )

        selected_context = self._get_selected_context(with_any_option=False)

        if selected_context is None:
            print("No context selected, plotting overall contributions.")
            overall_plot, predictor_plots = (
                ExplanationsDataPlotter.plot_contributions_for_overall(
                    data_loader=self.data_loader,
                    top_n=top_n,
                    top_k=top_k,
                    descending=descending,
                    missing=missing,
                    remaining=remaining,
                    contribution_calculation=contribution_type.value,
                )
            )

            for plot in [overall_plot] + predictor_plots:
                display(plot)
        else:
            context_plot, overall_plot, predictor_plots = (
                ExplanationsDataPlotter.plot_contributions_by_context(
                    data_loader=self.data_loader,
                    context=selected_context,
                    top_n=top_n,
                    top_k=top_k,
                    descending=descending,
                    missing=missing,
                    remaining=remaining,
                    contribution_calculation=contribution_type.value,
                )
            )

            for plot in [context_plot, overall_plot] + predictor_plots:
                display(plot)

    def set_selected_context(self, context_info: Optional[ContextInfo]):
        self._selected_context_key_values = cast(dict[str, str], context_info)

    def _get_selected_context(self, with_any_option: bool = True) -> ContextInfo | None:
        # return None if with_any_option is False and any context key value is set to ANY_CONTEXT
        if not with_any_option and any(
            x == self._ANY_CONTEXT for x in self._selected_context_key_values.values()
        ):
            return None

        return cast(ContextInfo, self._selected_context_key_values)

    def _get_context_combobox_widgets(self) -> dict[str, widgets.Combobox]:
        ctx_options = {
            ctx_key: [self._ANY_CONTEXT]
            + sorted(
                set(
                    context_info[ctx_key]
                    for context_info in self._filtered_context_info_list
                )
            )
            for ctx_key in self._selected_context_key_values.keys()
        }

        widget_dict = {}
        for name, options in ctx_options.items():
            widget_dict[name] = widgets.Combobox(
                placeholder="Any",
                options=options,
                description=name,
                ensure_option=False,
                layout=widgets.Layout(width="auto"),
                debounce=2000,
            )

        for context_filter_widget in widget_dict.values():
            context_filter_widget.observe(self._on_combobox_change, names="value")

        return widget_dict

    def _get_context_selector_widget(self) -> widgets.Select:
        options = self._get_options_for_context_selector()

        widget = widgets.Select(
            options=options,
            description="Context Info",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="auto"),
            rows=self._selected_context_key_values.keys().__len__() + 1,
        )
        widget.observe(self._on_selector_change, names="value")

        return widget

    def _on_combobox_change(self, change):
        changed_value = self._get_changed_value(change)
        changed_widget_id = self._get_changed_widget(change).description
        changed_widget_options = self._context_combobox_widgets[
            changed_widget_id
        ].options

        # set the selected context-key value to the changed value
        if (
            changed_value in changed_widget_options
            or changed_value == self._ANY_CONTEXT
        ):
            self._selected_context_key_values[changed_widget_id] = changed_value
            self._filtered_context_info_list = self._filter_context_infos()
        else:
            raise ValueError(
                f"Invalid value '{changed_value}' for context key '{changed_widget_id}'. "
                f"Available options are: {changed_widget_options}"
            )

        # update all widgets options according to the changed value, if changed value is ANY_CONTEXT, reset the options
        for widget_id, widget in self._context_combobox_widgets.items():
            widget.options = self._get_options_for_context_widget(widget_id)

        # update the context selector widget options
        if self._context_selector_widget:
            self._context_selector_widget.options = (
                self._get_options_for_context_selector()
            )

    def _on_selector_change(self, change):
        changed_value = self._get_changed_value(change)
        if changed_value == self._ANY_CONTEXT:
            context_info = None
        else:
            context_info = self._context_infos_to_dictionary(
                self._filtered_context_info_list
            ).get(changed_value, None)

        self.set_selected_context(context_info)

    def _get_changed_value(self, change):
        ret = change.get(self._CHANGED_VALUE, None)
        return self._ANY_CONTEXT if ret == "" else ret

    def _get_changed_widget(self, change):
        return change.get(self._CHANGED_WIDGET, None)

    def _get_options_for_context_widget(
        self, name: str, with_any_option: bool = True
    ) -> list[str]:
        options = [self._ANY_CONTEXT] if with_any_option else []

        options_list_sorted = sorted(
            set(context_info[name] for context_info in self._filtered_context_info_list)
        )
        options += cast(List[str], options_list_sorted)
        return options

    def _get_options_for_context_selector(
        self, with_any_option: bool = True
    ) -> list[str]:
        options = [self._ANY_CONTEXT] if with_any_option else []
        options += cast(
            list[str],
            (
                self._context_infos_to_dictionary(
                    self._filtered_context_info_list
                ).keys()
            ),
        )
        return options

    def _filter_context_infos(self) -> List[ContextInfo]:
        filtered_context_infos = []
        for context_info in self._raw_context_info_list:
            if all(
                value == self._ANY_CONTEXT or context_info[key] == value
                for key, value in self._selected_context_key_values.items()
            ):
                filtered_context_infos.append(context_info)
        return filtered_context_infos

    def _context_infos_to_dictionary(
        self, context_infos: List[ContextInfo]
    ) -> dict[str, ContextInfo]:
        return {
            (
                self._context_operations.get_context_info_str(context_info, " - ")
            ): context_info
            for context_info in context_infos
        }
