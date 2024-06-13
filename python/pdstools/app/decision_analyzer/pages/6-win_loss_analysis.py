import streamlit as st


from da_streamlit_utils import (
    get_current_scope_index,
    get_data_filters,
    show_filtered_counts,
    ensure_data,
)
from pdstools.decision_analyzer.utils import (
    NBADScope_Mapping,
    get_first_level_stats,
)

# TODO: the rank of winning may not be used or not properly in the analyses shown
# TODO: double check the numbers - I sometimes can't intuitively relate the bar charts to the box plots
# TODO: generalize and relabel the arbitration properties - they're repeated all over the place and may not even be the actual property names (just from my mock data)
# TODO: instead of sampling here, use the aggregated data and the sampling done inside of that now - perhaps with a larger n if need be
# TODO: the two bar charts as we show them in Global Sensitivity may be preferable over the streamlit 2-column view?
# TODO: also because now the colors are inconsistent - separate for both plots
# TODO: colors get too pale if the counts become really small, becomes invisible

"# Win Loss Analysis"

"""
This analysis shows that if a (group of) action(s) is "winning", what
it is winning from. A comparison, or reference, group needs to be
defined. You can, for example, investigate that when *upsell actions* are 
winning, what are they generally winning from. This analysis only applies
to Arbitration.
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
    scope_index = get_current_scope_index(scope_options)
    st.selectbox(
        "Select Scope",
        options=scope_options,
        format_func=lambda option: NBADScope_Mapping[option],
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
            st.session_state.decision_data.sample
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

if st.session_state.local_filters != []:

    groupby_cols = [st.session_state.scope] + (
        [scope_options[get_current_scope_index(scope_options) + 1]]
        if (get_current_scope_index(scope_options) + 1) < len(scope_options)
        else []
    )

    interactions_where_comparison_group_wins = (
        st.session_state.decision_data.get_winning_or_losing_interactions(
            win_rank=st.session_state.win_rank,
            group_filter=st.session_state["local_filters"],
            win=True,
        )
    )
    winning_from = st.session_state.decision_data.winning_from(
        interactions=interactions_where_comparison_group_wins,
        win_rank=st.session_state.win_rank,
        groupby_cols=groupby_cols,
        top_k=top_k,
    )
    interactions_where_comparison_group_loses = (
        st.session_state.decision_data.get_winning_or_losing_interactions(
            win_rank=st.session_state.win_rank,
            group_filter=st.session_state["local_filters"],
            win=False,
        )
    )
    losing_to = st.session_state.decision_data.winning_from(
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
            f"The action(s) in the comparison group win {win_count} times"
        )
        f"""Distribution of the {NBADScope_Mapping[st.session_state.scope]}s that the comparison group wins from in Arbitration"""

        st.plotly_chart(
            st.session_state.decision_data.plot.plot_distribution(
                winning_from,
                st.session_state.scope,
                (
                    scope_options[get_current_scope_index(scope_options) + 1]
                    if (get_current_scope_index(scope_options) + 1) < len(scope_options)
                    else None
                ),
                "Decisions",
                horizontal=True,
                # models=models,
            ),
            use_container_width=True,
        )

    with col2:
        """## Loss Analysis"""
        st.info(
            f"The action(s) in the comparison group loses {interactions_where_comparison_group_loses.collect().shape[0]} times"
        )
        f"""Distribution of the {NBADScope_Mapping[st.session_state.scope]}s that the comparison group loses to in Arbitration"""

        st.plotly_chart(
            st.session_state.decision_data.plot.plot_distribution(
                losing_to,
                st.session_state.scope,
                (
                    scope_options[get_current_scope_index(scope_options) + 1]
                    if (get_current_scope_index(scope_options) + 1) < len(scope_options)
                    else None
                ),
                "Decisions",
                horizontal=True,
            ),
            use_container_width=True,
        )

    if win_count > 0:
        "## What are the Prioritization Factors that make these actions win?"

        """
        We simply count the number of times the selected offer(s) are in the top-1 when dropping one of the prioritization factors from the priortization formula.

        So if it wins 600 times right now, but when leaving out value it only wins 200 times, that means value pushes the selected offer(s) up. The difference of +400 is shown in the bar chart below.

        """
        st.plotly_chart(
            st.session_state.decision_data.plot.plot_sensitivity(
                limit_xaxis_range=False,
            ),
            use_container_width=True,
        )
    "## Why are the actions winning"

    """
    Here we show the distribution of the various arbitration factors of the
    comparison group vs the other actions that make it to arbitration for the
    same interactions.
    """

    fig, warning_message = (
        st.session_state.decision_data.plot.plot_prio_factor_boxplots(
            reference=st.session_state["local_filters"],
            sample_size=10000,
        )
    )
    if warning_message:
        st.warning(warning_message)
    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    "## Rank Distribution of Comparison Group"

    """
    Showing the distribution of the prioritization rank of the selected actions.

    If the rank is low, the selected actions are not (often) winning.
    """
    st.plotly_chart(
        st.session_state.decision_data.plot.plot_rank_boxplot(
            reference=st.session_state["local_filters"],
        ),
        use_container_width=True,
    )
else:
    st.warning("Please Define a Group to compare against the rest of the actions")
