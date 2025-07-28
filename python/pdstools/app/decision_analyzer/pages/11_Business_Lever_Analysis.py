import polars as pl
import streamlit as st

from da_streamlit_utils import (
    ensure_data,
)
from pdstools.decision_analyzer.utils import (
    create_hierarchical_selectors,
    get_scope_config,
)
from pdstools.decision_analyzer.plots import (
    create_win_distribution_plot,
    create_parameter_distribution_boxplots,
)

# TODO not so sure what to do with this tool - maybe generalize to work across a selection not just a single action and figure out a multiplier
# TODO but do show the effect of levering right away (distributions side to side) just like we should do in the thresholding analysis (share code)
# TODO Start the target win ratio at > 0. Terminology could be clearer.
# TODO Code clean up, align session state usage and caching with the rest of the pages, move plots to plots module
# TODO The distributions themselves seem useful - see about moving / copying to thresholding page or even separate analysis
# TODO instead of sampling here, use the aggregated data and the sampling done inside of that now - perhaps with a larger n if need be

# st.set_page_config(page_title="Lever", layout="wide")
"# What-If Analysis for Business Levers"

"""
This interactive tool helps to discover the levers to make some
actions be presented more. By experimenting with different levers, you can see how the volume distribution would differ according to past data.
Keep in mind that if you start boosting the volume of an action with levers, it will get shown to more "uninterested people" making the propensity and therefore click through rate lower. So
presentation volume does not 100% correlate with click/accept count.

💡 **Definition of Win**: An action is considered winning when it is Rank 1 in Arbitration, therefore considered that it will be presented to the customers.
"""
ensure_data()
# TODO figure out how to move the actual code into the data class, avoid using st.session_state.decision_data.decision_data directly

arbitration_data = st.session_state.decision_data.arbitration_stage

with st.sidebar:
    st.session_state.win_rank = st.number_input(
        "Min Rank for Win",
        min_value=1,
        max_value=st.session_state.decision_data.max_win_rank,
        value=st.session_state.win_rank if "win_rank" in st.session_state else 1,
    )

    # Create hierarchical selectors using utility function
    selectors = create_hierarchical_selectors(
        arbitration_data,
        st.session_state.get("selected_issue"),
        st.session_state.get("selected_group"),
        st.session_state.get("selected_action"),
    )

    st.selectbox("Select Issue", key="selected_issue", **selectors["issues"])
    st.selectbox("Select Group", key="selected_group", **selectors["groups"])
    st.selectbox("Select Action", key="selected_action", **selectors["actions"])

    # Apply button to run analysis
    if st.button("Apply Selection", type="primary"):
        st.session_state.analysis_applied = True

scope_config = get_scope_config(
    st.session_state.selected_issue,
    st.session_state.selected_group,
    st.session_state.selected_action,
)
lever_condition = scope_config["lever_condition"]

# Only run analysis when Apply button is clicked
if st.session_state.get("analysis_applied", False):
    relevant_interactions = st.session_state.decision_data.arbitration_stage.filter(
        lever_condition
    )
    interactions_survived_till_arbitration = (
        relevant_interactions.select("pxInteractionID").collect().n_unique()
    )
    current_number_of_wins = (
        relevant_interactions.filter(pl.col("pxRank") == 1)
        .select("pxInteractionID")
        .collect()
        .n_unique()
    )
    # Calculate key metrics
    funnel_loss = (
        st.session_state.decision_data.sample_size
        - interactions_survived_till_arbitration
    )
    funnel_loss_pct = (funnel_loss / st.session_state.decision_data.sample_size) * 100
    current_win_rate = (
        current_number_of_wins / st.session_state.decision_data.sample_size
    ) * 100
    max_possible_win_rate = (
        interactions_survived_till_arbitration
        / st.session_state.decision_data.sample_size
    ) * 100
    current_win_rate_at_arbitration = (
        current_number_of_wins / interactions_survived_till_arbitration
    ) * 100
    st.markdown("#### 📊 How Often Selected Actions Survive till Arbitration?")
    st.markdown(
        "Selected actions might get filtered out in the funnel before ever reaching to arbitration stage."
    )
    st.markdown(
        "We can only increase the volume in the decisions where the selected actions reach to arbitration."
    )

    st.markdown(f"""
    **Your selected actions' journey:**
    - Selected actions are filtered before arbitration in **{funnel_loss:,} out of {st.session_state.decision_data.sample_size:,} interactions ({funnel_loss_pct:.2f}%)**.
    - **In {interactions_survived_till_arbitration:,} interactions, selected actions survive untill arbitration**, these are the decisions where you can make your actions win by boosting levers
    - Currently winning **{current_number_of_wins:,} out of {interactions_survived_till_arbitration:,} arbitrations** ({current_win_rate_at_arbitration:.2f}% win rate at arbitration)
    """)

    # Get baseline distribution data
    original_distribution = st.session_state.decision_data.get_win_distribution_data(
        lever_condition,
        all_interactions=st.session_state.decision_data.sample_size,
    )

    # Show original distribution
    st.markdown("### 📊 Current Win Distribution")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Win Rate",
            f"{current_win_rate:.2f}%",
            help="Decisions where the winner is a selected action divided by all decisions",
        )

    with col2:
        st.metric(
            "Funnel Loss",
            f"{funnel_loss_pct:.2f}%",
            f"-{funnel_loss:,} interactions",
            delta_color="off",
            help="Decisions where none of the selected actions survive until arbitration",
        )

    with col3:
        st.metric(
            "Max Possible Win Rate",
            f"{max_possible_win_rate:.2f}%",
            f"if won all {interactions_survived_till_arbitration:,} arbitrations",
            help="Win rate if we max out the lever and selected actions win every time they survive until arbitration.",
        )
    original_fig, original_plot_data = create_win_distribution_plot(
        original_distribution,
        "original_win_count",
        scope_config,
        "In Arbitration",
        "Current Win Count",
    )
    st.plotly_chart(original_fig, use_container_width=True)
    # Parameter Distribution Analysis
    show_distributions = st.checkbox(
        "Show distribution of arbitration components",
        help="Compare parameter distributions between your selected actions and competitors in interactions where your actions survived to arbitration",
    )

    if show_distributions:
        if interactions_survived_till_arbitration == 0:
            st.warning(
                "⚠️ Your selected actions never survive until arbitration. No head-to-head comparisons available."
            )
        else:
            with st.spinner("Plotting arbitration components..."):
                # Get the actual interaction IDs where selected actions survived
                relevant_interactions = (
                    st.session_state.decision_data.arbitration_stage.filter(
                        lever_condition
                    )
                    .select("pxInteractionID")
                    .unique()
                )

                # Filter sample to only those interactions (all actions in head-to-head battles)
                segmented_df = (
                    st.session_state.decision_data.sample.filter(
                        pl.col("StageGroup").is_in(
                            st.session_state.decision_data.stages_from_arbitration_down
                        )
                    )
                    .join(relevant_interactions, on="pxInteractionID", how="inner")
                    .with_columns(
                        segment=pl.when(lever_condition)
                        .then(pl.lit("Selected Actions"))
                        .otherwise(pl.lit("Others"))
                    )
                    .select(
                        ["Propensity", "Value", "Context Weight", "Levers", "segment"]
                    )
                    .collect()
                )

                if segmented_df.height == 0:
                    st.warning("No data available for parameter distribution analysis.")
                else:
                    st.markdown(
                        "### 📊 Parameter Distributions in Head-to-Head Battles"
                    )
                    st.markdown(
                        f"*Comparing your selected actions vs competitors in {interactions_survived_till_arbitration:,} interactions where your actions survived to arbitration*"
                    )

                    fig = create_parameter_distribution_boxplots(segmented_df)
                    st.plotly_chart(fig, use_container_width=True)
    with st.expander("🎯 Boosting Strategies", expanded=False):
        st.markdown(f"""
        **1. Address funnel losses:** If {funnel_loss_pct:.2f}% filter-out rate is too high, investigate earlier decision stages to understand why your actions are eliminated.

        **2. Increase levers:** Use the lever slider below to simulate different values. You have **{interactions_survived_till_arbitration:,}** decisions where the selected actions survive till arbitration.

        💡 *Note: This analysis focuses on your selected actions. Total arbitration activity across all actions is shown in the charts.*

        ⚠️ **Important:** Boosting your selected actions will suppress other actions in the same arbitration decisions - this is a zero-sum redistribution, not an increase in total wins.
        """)

        # Lever controls
        slider_max = st.selectbox(
            "Slider Precision", options=[1.0, 10.0, 100, 1000], index=1
        )
        slider_min = 0 if isinstance(slider_max, int) else 0.0
        value = 1 if isinstance(slider_max, int) else 1.0

        lever = st.slider(
            "Select Lever", min_value=slider_min, max_value=slider_max, value=value
        )

        # Calculate new distribution with lever changes
        distribution = st.session_state.decision_data.get_win_distribution_data(
            lever_condition,
            lever,
            all_interactions=st.session_state.decision_data.sample_size,
        )

        # Show new distribution
        st.markdown("### 🚀 Win Distribution With Selected Lever")
        new_fig, new_plot_data = create_win_distribution_plot(
            distribution,
            "new_win_count",
            scope_config,
            "After Lever Adjustment",
            "New Win Count",
        )
        st.plotly_chart(new_fig, use_container_width=True)

        # Show summary statistics
        total_new_wins = new_plot_data["new_win_count"].sum()
        selected_data = new_plot_data.filter(
            pl.col(scope_config["x_col"]) == scope_config["selected_value"]
        )
        selected_wins = (
            selected_data["new_win_count"].sum() if selected_data.shape[0] > 0 else 0
        )

        # Calculate deltas
        selected_wins_delta = selected_wins - current_number_of_wins
        new_win_rate_at_arbitration = (
            selected_wins / interactions_survived_till_arbitration
        ) * 100
        new_overall_win_rate = (
            selected_wins / st.session_state.decision_data.sample_size
        ) * 100
        win_rate_delta_arbitration = (
            new_win_rate_at_arbitration - current_win_rate_at_arbitration
        )
        win_rate_delta_overall = new_overall_win_rate - current_win_rate

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Number Of Interactions With A Winner",
                f"{total_new_wins:,}",
                help="In some interactions, there may be no action left at all for arbitration",
            )
        with col2:
            st.metric(
                f"Selected {scope_config['level']} Wins",
                f"{selected_wins:,}",
                delta=f"{selected_wins_delta:+,}",
            )
        with col3:
            st.metric(
                f"Selected {scope_config['level']} Win Rate at Arbitration",
                f"{new_win_rate_at_arbitration:.2f}%",
                delta=f"{win_rate_delta_arbitration:+.2f}%",
            )
        with col4:
            st.metric(
                f"Selected {scope_config['level']} Overall Win Rate",
                f"{new_overall_win_rate:.2f}%",
                delta=f"{win_rate_delta_overall:+.2f}%",
            )

    with st.expander(":green[Lever Finder]:male-detective:", expanded=False):
        # Only show lever finder for specific action selection
        st.session_state.target_win_percentage = st.slider(
            "Target Win Ratio", min_value=0, max_value=100
        )

        calculate_lever = st.button("Calculate lever")
        if calculate_lever:
            with st.spinner("Calculating..."):
                # TODO refactor this into the DecisionData class
                lever_for_desired_ratio = (
                    st.session_state.decision_data.find_lever_value(
                        lever_condition=lever_condition,
                        target_win_percentage=st.session_state.target_win_percentage,
                        win_rank=st.session_state.win_rank,
                        high=100,
                    )
                )
                if isinstance(lever_for_desired_ratio, float):
                    st.metric(
                        f"""Lever you need to win in
                        {st.session_state.target_win_percentage}% of the interactions""",
                        lever_for_desired_ratio,
                    )
