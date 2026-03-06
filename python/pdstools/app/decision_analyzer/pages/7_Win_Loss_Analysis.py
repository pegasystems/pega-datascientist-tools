import streamlit as st
from da_streamlit_utils import (
    ensure_data,
    get_current_index,
    get_data_filters,
    show_filtered_counts,
)

from pdstools.decision_analyzer.utils import (
    get_first_level_stats,
)

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

    top_k = st.number_input(
        "Top N elements to show",
        min_value=1,
        max_value=30,  # TODO this is a generic session, make common across many pages
        value=10,
    )
    st.number_input(
        "Top-N actions that define Winning",
        min_value=1,
        max_value=10,  # TODO why restrict to 10, lets use the upper bound from the data.
        value=st.session_state.win_rank if "win_rank" in st.session_state else 1,
        key="win_rank",
    )
    scope_index = get_current_index(scope_options, "scope")
    st.selectbox(
        "Select Scope",
        options=scope_options,
        # column names are already friendly
        index=scope_index,
        key="scope",
    )

    "### Define a Comparison Group"

    st.session_state["local_filters"] = get_data_filters(
        st.session_state.decision_data.sample,
        columns=st.session_state.decision_data.getAvailableFieldsForFiltering(
            categoricalOnly=True,
        ),
        queries=[],
        filter_type="local",
    )
    if st.session_state["local_filters"] != []:
        statsBeforeExtraFilter = get_first_level_stats(
            st.session_state.decision_data.sample,
        )
        statsAfterExtraFilter = get_first_level_stats(
            st.session_state.decision_data.sample,
            st.session_state["local_filters"],
        )
        show_filtered_counts(statsBeforeExtraFilter, statsAfterExtraFilter)
    else:
        st.warning("No comparison group defined")

    # I've removed the additional model selection / listing for now
    # models = show_filtered_counts(
    #     st.session_state.df, st.session_state.local_filters
    # )
    # with st.expander("Selected actions"):
    #     st.write(models)

# st.write(st.session_state.to_dict().keys())


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

    interactions_where_comparison_group_wins = st.session_state.decision_data.get_winning_or_losing_interactions(
        win_rank=st.session_state.win_rank,
        group_filter=st.session_state["local_filters"],
        win=True,
    )
    winning_from = st.session_state.decision_data.winning_from(
        interactions=interactions_where_comparison_group_wins,
        win_rank=st.session_state.win_rank,
        groupby_cols=groupby_cols,
        top_k=top_k,
    )
    interactions_where_comparison_group_loses = st.session_state.decision_data.get_winning_or_losing_interactions(
        win_rank=st.session_state.win_rank,
        group_filter=st.session_state["local_filters"],
        win=False,
    )
    losing_to = st.session_state.decision_data.losing_to(
        interactions=interactions_where_comparison_group_loses,
        win_rank=st.session_state.win_rank,
        groupby_cols=groupby_cols,
        top_k=top_k,
    )

    col1, col2 = st.columns(2)
    with col1:
        """## Win Analysis"""
        win_count = interactions_where_comparison_group_wins.collect().shape[0]

        st.info(
            # TODO these numbers may not be correct
            f"The action(s) in the comparison group win {win_count} times",
        )
        f"""Distribution of the {st.session_state.scope}s that the comparison group wins from in Arbitration"""

        st.plotly_chart(
            st.session_state.decision_data.plot.distribution(
                winning_from,
                st.session_state.scope,
                groupby_cols[1] if len(groupby_cols) > 1 else None,
                "Decisions",
                horizontal=True,
                # models=models,
            ),
            width="stretch",
            key="win_distribution_chart",
        )

    with col2:
        """## Loss Analysis"""
        st.info(
            f"The action(s) in the comparison group loses {interactions_where_comparison_group_loses.collect().shape[0]} times",
        )
        f"""Distribution of the {st.session_state.scope}s that the comparison group loses to in Arbitration"""

        st.plotly_chart(
            st.session_state.decision_data.plot.distribution(
                losing_to,
                st.session_state.scope,
                groupby_cols[1] if len(groupby_cols) > 1 else None,
                "Decisions",
                horizontal=True,
            ),
            width="stretch",
            key="loss_distribution_chart",
        )

    "## Why Do These Offers Win?"

    """
    See which factors drive your comparison group to the top. The chart shows how many
    additional wins each factor contributes. If an offer wins 600 times now but only
    200 times without considering value, then value is adding 400 wins — pushing this
    offer ahead of others.
    """
    if win_count == 0:
        st.warning("The selected comparison Group never wins in the arbitration")
    else:
        st.plotly_chart(
            st.session_state.decision_data.plot.sensitivity(
                reference_group=st.session_state["local_filters"],
            ),
            width="stretch",
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
    )
    if warning_message:
        st.warning(warning_message)
    if fig is not None:
        st.plotly_chart(
            fig,
            width="stretch",
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
        ),
        width="stretch",
        key="rank_boxplot_chart",
    )
else:
    st.warning("Please Define a Group to compare against the rest of the actions")
