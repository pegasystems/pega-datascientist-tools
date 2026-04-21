"""Win/loss distribution plots."""

from typing import cast

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


def global_winloss_distribution(self, level, win_rank, return_df=False, additional_filters=None):
    df = self._decision_data.scoring.get_win_loss_distribution_data(
        level, win_rank, additional_filters=additional_filters
    )
    if return_df:
        return df

    color_discrete_map = self._decision_data.color_mappings.get(level, {})

    df_collected = df.collect()
    wins_df = df_collected.filter(pl.col("Status") == "Wins")
    losses_df = df_collected.filter(pl.col("Status") == "Losses")

    mandatory = self._decision_data.mandatory_actions if level == "Action" else set()

    def _label(value: str) -> str:
        return f"★ {value}" if value in mandatory else value

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=["Wins", "Losses"],
    )

    if wins_df.height > 0:
        wins_values = wins_df[level].to_list()
        colors_wins = [color_discrete_map.get(val, "#cccccc") for val in wins_values]
        fig.add_trace(
            go.Pie(
                labels=[_label(v) for v in wins_values],
                values=wins_df["Percentage"],
                marker=dict(colors=colors_wins),
                textposition="auto",
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if losses_df.height > 0:
        losses_values = losses_df[level].to_list()
        colors_losses = [color_discrete_map.get(val, "#cccccc") for val in losses_values]
        fig.add_trace(
            go.Pie(
                labels=[_label(v) for v in losses_values],
                values=losses_df["Percentage"],
                marker=dict(colors=colors_losses),
                textposition="auto",
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        font_size=12,
        showlegend=False,
    )

    if mandatory:
        fig.add_annotation(
            text="★ marks auto-detected mandatory actions (priority ≥ 5M); they bypass arbitration.",
            xref="paper",
            yref="paper",
            x=0,
            y=-0.12,
            showarrow=False,
            font=dict(size=11),
            xanchor="left",
        )

    return fig


def create_win_distribution_plot(
    data: pl.DataFrame,
    win_count_col: str,
    scope_config: dict[str, str | list[str]],
    title_suffix: str,
    y_axis_title: str,
) -> tuple[go.Figure, pl.DataFrame]:
    """
    Create a win distribution bar chart with highlighted selected items.

    This function creates a bar chart showing win counts across actions, groups, or issues
    based on the scope configuration. It automatically aggregates data appropriately and
    highlights the selected item in red while showing others in grey.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing win distribution data with action identifiers and win counts
    win_count_col : str
        Column name containing win counts to plot (e.g., "original_win_count", "new_win_count")
    scope_config : dict[str, str | list[str]]
        Configuration dictionary from get_scope_config() containing:
        - level: "Action", "Group", or "Issue"
        - group_cols: List of columns for grouping
        - x_col: Column name for x-axis
        - selected_value: Value to highlight in red
        - plot_title_prefix: Prefix for plot title
    title_suffix : str
        Suffix to add to plot title (e.g., "Current Performance", "After Lever Adjustment")
    y_axis_title : str
        Title for y-axis (e.g., "Current Win Count", "New Win Count")

    Returns
    -------
    tuple[go.Figure, pl.DataFrame]
        - Plotly figure with bar chart
        - Processed plot data (aggregated if needed)

    Notes
    -----
    - For Action level: Shows individual actions
    - For Group/Issue level: Automatically aggregates data by summing win counts
    - Selected item is highlighted in red (#FF0000), others in grey
    - "No Winner" bar (if present in data) is shown in orange (#FFA500) to highlight interactions without winners
    - If selected item not found, uses light blue as fallback color
    - X-axis labels are hidden to avoid clutter, scope level shown as x-axis title
    - "No Winner" data is calculated and added by get_win_distribution_data() when all_interactions parameter is provided

    Examples
    --------
    >>> scope_config = get_scope_config("Service", "Cards", "MyAction")
    >>> fig, plot_data = create_win_distribution_plot(
    ...     distribution_data,
    ...     "new_win_count",
    ...     scope_config,
    ...     "After Lever Adjustment",
    ...     "New Win Count"
    ... )
    """
    if scope_config["level"] == "Action":
        plot_data = data
    else:
        no_winner_data = data.filter(pl.col("Action") == "No Winner")
        regular_data = data.filter(pl.col("Action") != "No Winner")

        if regular_data.height > 0:
            aggregated_regular = (
                regular_data.group_by(scope_config["group_cols"])
                .agg(pl.sum(win_count_col))
                .sort(win_count_col, descending=True)
            )

            if no_winner_data.height > 0:
                group_cols = list(cast(list[str], scope_config["group_cols"]))
                columns_to_keep = group_cols + [win_count_col]
                no_winner_data_selected = no_winner_data.select(columns_to_keep)
                plot_data = pl.concat([aggregated_regular, no_winner_data_selected])
            else:
                plot_data = aggregated_regular
        else:
            if no_winner_data.height > 0:
                group_cols = list(cast(list[str], scope_config["group_cols"]))
                columns_to_keep = group_cols + [win_count_col]
                plot_data = no_winner_data.select(columns_to_keep)
            else:
                plot_data = pl.DataFrame()

    fig = go.Figure()

    if scope_config["x_col"] == "Group" and "Issue" in plot_data.columns:
        hover_template = "<b>%{text}</b><br>Issue: %{customdata}<br>Win Count: %{y}<extra></extra>"
        customdata = plot_data["Issue"]
    elif scope_config["x_col"] == "Action" and "Group" in plot_data.columns and "Issue" in plot_data.columns:
        hover_template = (
            "<b>%{text}</b><br>Group: %{customdata[0]}<br>Issue: %{customdata[1]}<br>Win Count: %{y}<extra></extra>"
        )
        customdata = list(zip(plot_data["Group"], plot_data["Issue"], strict=False))
    else:
        hover_template = "<b>%{text}</b><br>Win Count: %{y}<extra></extra>"
        customdata = None

    fig.add_trace(
        go.Bar(
            x=plot_data[scope_config["x_col"]],
            y=plot_data[win_count_col],
            text=plot_data[scope_config["x_col"]],
            textposition="auto",
            hovertemplate=hover_template,
            customdata=customdata,
        )
    )

    colors = ["grey"] * plot_data.shape[0]
    x_values = list(plot_data[scope_config["x_col"]])

    try:
        selected_index = x_values.index(scope_config["selected_value"])
        colors[selected_index] = "#FF0000"
    except ValueError:
        pass

    try:
        no_winner_index = x_values.index("No Winner")
        colors[no_winner_index] = "#FFA500"
    except ValueError:
        pass

    if all(color == "grey" for color in colors):
        fig.data[0]["marker_color"] = "lightblue"
    else:
        fig.data[0]["marker_color"] = colors

    fig.update_yaxes(title=y_axis_title)
    fig.update_xaxes(showticklabels=False, title=scope_config["level"])
    fig.update_layout(
        title=f"{scope_config['plot_title_prefix']} - {title_suffix} (Selected: {scope_config['selected_value']})",
        showlegend=False,
    )

    return fig, plot_data
