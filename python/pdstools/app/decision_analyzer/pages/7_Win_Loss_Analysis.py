import polars as pl
import streamlit as st
from da_streamlit_utils import (
    channel_direction_selector,
    ensure_data,
    get_current_index,
    get_data_filters,
    show_filtered_counts,
)

from pdstools.decision_analyzer.utils import (
    get_first_level_stats,
)

MAX_COMPARISON_LABEL_LEN = 80


def _describe_comparison_group() -> str:
    """Build a human-readable label from the active comparison filter selections.

    Examples:
        Issue: Service
        Issue: Service, Sales
        Issue: Service and Group: Cards
        (falls back to "comparison group" when too long or nothing selected)
    """
    columns = st.session_state.get("localmultiselect", [])
    if not columns:
        return "comparison group"

    parts: list[str] = []
    for col in columns:
        values = st.session_state.get(f"localselected_{col}", [])
        if not values:
            continue
        parts.append(f"{col}: {', '.join(values)}")

    if not parts:
        return "comparison group"

    label = " and ".join(parts)
    if len(label) > MAX_COMPARISON_LABEL_LEN:
        return "comparison group"
    return label


# TODO: the rank of winning may not be used or not properly in the analyses shown
# TODO: double check the numbers - I sometimes can't intuitively relate the bar charts to the box plots
# TODO: generalize and relabel the arbitration properties - they're repeated all over the place and may not even be the actual property names (just from my mock data)
# TODO: instead of sampling here, use the aggregated data and the sampling done inside of that now - perhaps with a larger n if need be
# TODO: the two bar charts as we show them in Global Sensitivity may be preferable over the streamlit 2-column view?
# TODO: also because now the colors are inconsistent - separate for both plots
# TODO: colors get too pale if the counts become really small, becomes invisible

"# Win/Loss Analysis"

"""
Understand competitive dynamics between your offers. When one offer wins the customer
interaction, which other offers did it beat? Define a comparison group (e.g., upsell offers)
to see what they typically win against and lose to.

**Use this to:** Identify which offers compete most directly and understand why certain
offers consistently win or lose.
"""
ensure_data()
st.session_state["sidebar"] = st.sidebar

# TODO see if this works when we have many channels
facetting = "pyChannel/pyDirection"

# st.session_state.df = st.session_state.df.with_columns(
#     pl.col(pl.Categorical).cast(pl.Utf8)
# )
with st.session_state["sidebar"]:
    scope_options = st.session_state.decision_data.getPossibleScopeValues()
    filtered_data = st.session_state.decision_data.filtered_sample
    comparison_filter_columns = [
        c
        for c in st.session_state.decision_data.getAvailableFieldsForFiltering(
            categoricalOnly=True,
        )
        if c not in {"Stage", "Channel", "Direction"}
    ]

    "### Define a Comparison Group"

    st.session_state["local_filters"] = get_data_filters(
        filtered_data,
        columns=comparison_filter_columns,
        queries=[],
        filter_type="local",
        default_select_all_categories=False,
        selector_label="Compare offers by",
        sort_columns=False,
    )
    if st.session_state["local_filters"] != []:
        statsBeforeExtraFilter = get_first_level_stats(
            filtered_data,
        )
        statsAfterExtraFilter = get_first_level_stats(
            filtered_data,
            st.session_state["local_filters"],
        )
        show_filtered_counts(statsBeforeExtraFilter, statsAfterExtraFilter)
    else:
        st.warning("No comparison group defined")

    scope_index = get_current_index(scope_options, "scope")
    st.selectbox(
        "Select Scope",
        options=scope_options,
        # column names are already friendly
        index=scope_index,
        key="scope",
    )

    top_k = None
    if st.session_state.scope == "Action":
        top_k = st.number_input(
            "Top N elements to show",
            min_value=1,
            max_value=30,  # TODO this is a generic session, make common across many pages
            value=10,
        )

    channel_direction_selector()

# Check for empty results when a specific channel is selected
if st.session_state.get("page_channel_filter", "Any") != "Any":
    filtered_count = filtered_data.select(pl.len()).collect().item()
    if filtered_count == 0:
        st.warning(
            f"No data available for {st.session_state.page_channel_filter}. "
            "Try selecting 'Any' or adjusting global filters."
        )
        st.stop()

channel_filter = st.session_state.get("page_channel_expr")


def get_groupby_columns(scope_options, current_scope_key):
    """Get the columns to group by for win/loss analysis.
    Returns [current_scope] + [next_scope_level] if next level exists.
    This creates the y-axis (current scope) and color grouping (next scope level).
    """
    current_index = get_current_index(scope_options, current_scope_key)
    groupby_cols = [st.session_state[current_scope_key]]

    # Add the next scope level for color grouping if it exists
    next_index = current_index + 1
    if next_index < len(scope_options):
        groupby_cols.append(scope_options[next_index])

    return groupby_cols


if st.session_state.local_filters != []:
    groupby_cols = get_groupby_columns(scope_options, "scope")
    winning_from = st.session_state.decision_data.get_win_loss_distribution_data(
        level=groupby_cols,
        group_filter=st.session_state["local_filters"],
        status="Wins",
        top_k=top_k,
        additional_filters=channel_filter,
    )
    losing_to = st.session_state.decision_data.get_win_loss_distribution_data(
        level=groupby_cols,
        group_filter=st.session_state["local_filters"],
        status="Losses",
        top_k=top_k,
        additional_filters=channel_filter,
    )

    comparison_label = _describe_comparison_group()
    scope_label = " and ".join(groupby_cols)

    with st.container(border=True):
        win_rank = st.number_input(
            "Define winning: rank in top N",
            min_value=1,
            value=1,
            help="A win means at least one offer from the comparison group ranks N or better.",
        )

        counts = st.session_state.decision_data.get_win_loss_counts(
            group_filter=st.session_state["local_filters"],
            win_rank=win_rank,
            additional_filters=channel_filter,
        )
        win_count = counts["wins"]
        loss_count = counts["losses"]
        total = counts["total"]
        win_pct = (win_count / total * 100) if total > 0 else 0
        loss_pct = (loss_count / total * 100) if total > 0 else 0

        col1, col2 = st.columns(2)
        with col1:
            """## Win Analysis"""
            st.info(
                f"**{comparison_label}** wins **{win_count}** out of **{total}** decisions (**{win_pct:.1f}%**)",
            )
            f"""Distribution of the {scope_label} the comparison group wins from"""

            st.plotly_chart(
                st.session_state.decision_data.plot.distribution(
                    winning_from,
                    st.session_state.scope,
                    groupby_cols[1] if len(groupby_cols) > 1 else None,
                    "Avg per Decision",
                    horizontal=True,
                ),
                key="win_distribution_chart",
            )

        with col2:
            """## Loss Analysis"""
            st.info(
                f"**{comparison_label}** loses **{loss_count}** out of **{total}** decisions (**{loss_pct:.1f}%**)",
            )
            f"""Distribution of the {scope_label} the comparison group loses to"""

            st.plotly_chart(
                st.session_state.decision_data.plot.distribution(
                    losing_to,
                    st.session_state.scope,
                    groupby_cols[1] if len(groupby_cols) > 1 else None,
                    "Avg per Decision",
                    horizontal=True,
                ),
                key="loss_distribution_chart",
            )

    "## Why Do These Offers Win?"

    f"""
    See which factors drive **{comparison_label}** to the top. The chart shows how many
    additional wins each factor contributes. If an offer wins 600 times now but only
    200 times without considering value, then value is adding 400 wins — pushing this
    offer ahead of others.
    """
    if win_count == 0:
        st.warning(f"**{comparison_label}** never wins in the arbitration")
    else:
        st.plotly_chart(
            st.session_state.decision_data.plot.sensitivity(
                reference_group=st.session_state["local_filters"],
                additional_filters=channel_filter,
            ),
            key="sensitivity_chart",
        )
    "## Comparison: Your Group vs Others"

    """
    Compare how your selected offers score on key factors (value, priority, propensity)
    against competing offers in the same customer interactions. This reveals whether
    your group wins through higher scores or other strategic factors.
    """

    fig, warning_message = st.session_state.decision_data.plot.prio_factor_boxplots(
        reference=st.session_state["local_filters"],
        additional_filters=channel_filter,
    )
    sample_warning = "Showing a representative sample of"
    if warning_message and not warning_message.startswith(sample_warning):
        st.warning(warning_message)
    if fig is not None:
        st.plotly_chart(
            fig,
            key="prio_factor_boxplots_chart",
        )

    "## How Often Do These Offers Rank First?"

    """
    Shows where your selected offers typically rank in customer interactions. Lower
    ranks (closer to 1) mean these offers frequently win. Higher ranks indicate
    they're often beat by other offers.
    """
    st.plotly_chart(
        st.session_state.decision_data.plot.rank_boxplot(
            reference=st.session_state["local_filters"],
            additional_filters=channel_filter,
        ),
        key="rank_boxplot_chart",
    )
else:
    st.warning("Please Define a Group to compare against the rest of the actions")
