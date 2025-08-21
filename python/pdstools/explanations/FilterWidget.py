__all__ = ["FilterWidget"]

from collections import OrderedDict
from typing import TYPE_CHECKING, List, Optional, cast

from IPython.display import display
from ipywidgets import widgets

from ..utils.namespaces import LazyNamespace
from .ExplanationsUtils import ContextInfo

if TYPE_CHECKING:
    from .Explanations import Explanations


class FilterWidget(LazyNamespace):
    dependencies = ["ipywidgets"]
    dependency_group = "explanations"

    _ANY_CONTEXT = "Any"
    _CHANGED_VALUE = "new"
    _CHANGED_WIDGET = "owner"

    _selector_widget: Optional[widgets.Select]

    def __init__(self, explanations: "Explanations"):
        self.explanations = explanations

        # init context lists
        self._raw_list: List[ContextInfo] = []
        self._filtered_list: List[ContextInfo] = []

        # init widget objects
        self._combobox_widgets: dict[str, widgets.Combobox] = {}
        self._selector_widget = None

        self._context_operations = None
        self._selected_context_key = None

        super().__init__()

    def interactive(self):
        """Initializes the interactive filter widget and displays it.
        This is used in combination with explanations.plot.contributions() to allow users to
        filter by context if required.
        Select the context from the list of contexts for plotting contributions for selected context else
        the overall contributions will be plotted.
        Alternatively, the context can be set using `set_selected_context()` method.
        """
        try:
            self.explanations.aggregate.validate_folder()
        except Exception as e:
            raise e

        self._context_operations = self.explanations.aggregate.context_operations

        self._raw_list = self._context_operations.get_list()
        self._init_selected_context()

        self._display_context_selector()

    def set_selected_context(self, context_info: Optional[ContextInfo] = None):
        """Set the selected context information.
        Args:
            context_info (Optional[ContextInfo]):
                If None, initializes the selected context with 'Any' for all keys. 
                i.e overall model contributions
                If provided, sets the selected context to the given context information.
                Context is passed as a dictionary
                Eg. context_info =
                    {
                        "pyChannel": "channel1",
                        "pyDirection": "direction1",
                        ...
                    }
        """
        if context_info is None:
            self._init_selected_context()
        else:
            self._selected_context_key = cast(dict[str, str], context_info)

    def is_context_selected(self) -> bool:
        """Method returns True only if all context keys 
        are selected with a value other than 'Any'.
        """
        if self._selected_context_key is None:
            return False
        return all(
            value != self._ANY_CONTEXT for value in self._selected_context_key.values()
        )

    def get_selected_context(self):
        """Get the currently selected context information."""
        return self._get_selected_context(with_any_option=False)

    def _init_selected_context(self):
        self._selected_context_key = {
            key: self._ANY_CONTEXT
            for key in self._context_operations.get_context_keys()
        }

    def _display_context_selector(self):
        self._filtered_list = self._filter_contexts_by_selected()

        # create widgets for each context key
        self._combobox_widgets = self._get_context_combobox_widgets()
        context_picker_container = widgets.VBox(list(self._combobox_widgets.values()))

        # create widget for list of filtered context selector
        self._selector_widget = self._get_context_selector_widget()

        clear_widget = widgets.Button(
            description="Clear",
            button_style="danger",
            layout=widgets.Layout(grid_column="1 / span 2", width="100%"),
        )
        clear_widget.on_click(self._on_button_clicked)

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
                self._selector_widget,
                clear_widget,
            ],
            layout=widgets.Layout(
                # width="100%",
                grid_template_columns="repeat(2, 1fr)",
                grid_template_rows="auto auto auto",
                grid_gap="20px",
                align_items="center",
                margin="0 0 30px 0",  # top, right, bottom, left
            ),
        )

        display(grid)

    def _get_selected_context(
        self, with_any_option: bool = True
    ) -> Optional[ContextInfo]:
        # return None if with_any_option is False and any context key value is set to ANY_CONTEXT
        if not with_any_option and any(
            x == self._ANY_CONTEXT for x in self._selected_context_key.values()
        ):
            return None

        return cast(ContextInfo, self._selected_context_key)

    def _get_context_combobox_widgets(self) -> dict[str, widgets.Combobox]:
        ctx_options = {
            ctx_key_name: [self._ANY_CONTEXT]
            + sorted(
                set(context_info[ctx_key_name] for context_info in self._filtered_list)
            )
            for ctx_key_name in self._selected_context_key.keys()
        }

        widget_dict = OrderedDict()
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
            rows=len(self._selected_context_key.keys()) + 1,
        )
        widget.observe(self._on_selector_change, names="value")

        return widget

    def _on_button_clicked(self, _):
        """Clear the filter widget and reset the context selection."""
        self._init_selected_context()
        self._filtered_list = self._filter_contexts_by_selected()

        # reset all combobox widgets
        for _, widget in reversed(list(self._combobox_widgets.items())):
            widget.value = ""

        # reset the context selector widget
        self._selector_widget.value = self._ANY_CONTEXT

    def _on_combobox_change(self, change):
        changed_value = self._get_changed_value(change)
        changed_widget_id = self._get_changed_widget(change).description
        changed_widget_options = self._combobox_widgets[changed_widget_id].options

        # set the selected context-key value to the changed value
        if (
            changed_value in changed_widget_options
            or changed_value == self._ANY_CONTEXT
        ):
            self._selected_context_key[changed_widget_id] = changed_value
            self._filtered_list = self._filter_contexts_by_selected()
        else:
            raise ValueError(
                f"Invalid value '{changed_value}' for context key '{changed_widget_id}'. "
                f"Available options are: {changed_widget_options}"
            )

        # update all widgets options according to the changed value
        # if changed value is ANY_CONTEXT, reset the options
        for widget_id, widget in self._combobox_widgets.items():
            widget.options = self._get_options_for_context_widget(widget_id)

        # update the context selector widget options
        if self._selector_widget:
            self._selector_widget.options = self._get_options_for_context_selector()
            self._selector_widget.value = self._ANY_CONTEXT

    def _on_selector_change(self, change):
        changed_value = self._get_changed_value(change)
        if changed_value == self._ANY_CONTEXT:
            context_info = None
        else:
            context_info = self._context_infos_to_dictionary(self._filtered_list).get(
                changed_value, None
            )

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
            set(context_info[name] for context_info in self._filtered_list)
        )
        options += cast(List[str], options_list_sorted)
        return options

    def _get_options_for_context_selector(
        self, with_any_option: bool = True
    ) -> list[str]:
        options = [self._ANY_CONTEXT] if with_any_option else []
        options += cast(
            list[str],
            (self._context_infos_to_dictionary(self._filtered_list).keys()),
        )
        return options

    def _filter_contexts_by_selected(self) -> List[ContextInfo]:
        filtered_context_infos = []
        for context_info in self._raw_list:
            if all(
                value == self._ANY_CONTEXT or context_info[key] == value
                for key, value in self._selected_context_key.items()
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
