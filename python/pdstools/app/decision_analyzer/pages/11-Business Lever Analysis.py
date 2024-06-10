import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl
import streamlit as st

from utils import ensure_data, find_lever_value

# TODO not so sure what to do with this tool - maybe generalize to work across a selection not just a single action and figure out a multiplier
# TODO but do show the effect of levering right away (distributions side to side) just like we should do in the thresholding analysis (share code)
# TODO Start the target win ratio at > 0. Terminology could be clearer.
# TODO Code clean up, align session state usage and caching with the rest of the pages, move plots to plots module
# TODO The distributions themselves seem useful - see about moving / copying to thresholding page or even separate analysis
# TODO instead of sampling here, use the aggregated data and the sampling done inside of that now - perhaps with a larger n if need be

# st.set_page_config(page_title="Lever", layout="wide")
"# What-If Analysis for Business Levers"

"""
This interactive tool helps to discover the levers to make an 
action win. There is both a slider to do this manually
and an automatic way to find the minimum lever value.
"""
ensure_data()
# TODO lets not put everything in session_state
# TODO figure out how to move the actual code into the data class, avoid using st.session_state.decision_data.decision_data directly
if "list_of_all_actions" not in st.session_state:
    st.session_state["list_of_all_actions"] = (
        st.session_state.decision_data.sample.filter(
            pl.col("pxEngagementStage").is_in(["Arbitration", "Final"])
        )
        .select("pyName")
        .unique()
        .collect()
        .get_column("pyName")
        .to_list()
    )
with st.sidebar:
    st.session_state.win_rank = st.number_input(
        "Min Rank for Win",
        min_value=1,
        max_value=st.session_state.decision_data.max_win_rank,
        value=st.session_state.win_rank if "win_rank" in st.session_state else 1,
    )

    st.session_state.action = st.selectbox(
        "Select Action",
        options=st.session_state.list_of_all_actions,
        index=0,
        help="If you can't see an action in this list. They are either not in the data or they never reached to the Arbitration stage.",
    )
    st.session_state.max_search_range = st.selectbox(
        "Search Range Max", [10, 100, 1000], index=1
    )

slider_max = st.selectbox("Slider Precision", options=[1.0, 10.0, 100, 1000], index=1)
slider_min = 0 if isinstance(slider_max, int) else 0.0
value = 1 if isinstance(slider_max, int) else 1.0

lever = st.slider(
    "Select Lever", min_value=slider_min, max_value=slider_max, value=value
)

ranked_df = st.session_state.decision_data.reRank(
    overrides=[
        (
            pl.when(pl.col("pyName") == st.session_state.action)
            .then(pl.lit(lever))
            .otherwise(pl.col("Weight"))
        ).alias("Weight")
    ]
)
rank_1_df = (
    (
        ranked_df.filter(pl.col("rank_PVCL") == 1)
        .filter(pl.col("pxEngagementStage").is_in(["Arbitration", "Final"]))
        .collect()
    )
    .group_by("pyName")
    .agg(pl.count("pxInteractionID").alias("Win Count"))
    .sort("Win Count", descending=True)
)
# TODO lets put into utils, this list is in many places
parameters = ["FinalPropensity", "Value", "ContextWeight", "Weight"]

segmented_df = (
    # TODO refactor this to work with the DecisionData class
    st.session_state.decision_data.sample.filter(
        pl.col("pxEngagementStage").is_in(["Arbitration", "Final"])
    )
    .with_columns(
        Weight=pl.when(pl.col("pyName") == st.session_state.action)
        .then(pl.lit(lever))
        .otherwise(pl.col("Weight"))
    )
    .with_columns(
        segment=pl.when(pl.col("pyName") == st.session_state.action)
        .then(pl.col("pyName"))
        .otherwise(pl.lit("Others"))
    )
    .select(parameters + ["segment"])
    .collect()
)

## Distribution Container
if st.checkbox("Show the distribution of parameters", False):
    with st.container(border=True):
        segments = segmented_df["segment"].unique()
        colors = ["blue", "red"]
        fig = sp.make_subplots(rows=4, cols=1, subplot_titles=parameters)

        for i, metric in enumerate(parameters, start=1):
            for j, segment in enumerate(segments):
                fig.add_trace(
                    go.Histogram(
                        x=segmented_df.filter(segment=segment)[metric],
                        name=segment,
                        nbinsx=50,
                        histnorm="probability density",
                        marker_color=colors[j],  # use consistent color for each segment
                        showlegend=i == 1,  # show legend only for the first plot
                    ),
                    row=i,
                    col=1,
                )

        fig.update_layout(height=800, width=600)
        fig.update_yaxes(automargin=True)

        st.plotly_chart(fig, use_container_width=True)

##

if rank_1_df.filter(pl.col("pyName") == st.session_state.action).shape[0] == 0:
    st.warning(f"{st.session_state.action} never wins with the lever: **{lever}** ")
else:
    win_count = rank_1_df.filter(
        pl.col("pyName") == st.session_state.action
    ).get_column("Win Count")[0]
    total_count = (
        st.session_state.decision_data.sample.filter(
            pl.col("pyName") == st.session_state.action
        )
        .select("pxInteractionID")
        .unique()
        .collect()
        .shape[0]
    )
    st.write(f"Win Count: {win_count}")
    st.write(f"Win Ratio: {round(((win_count/total_count)*100), 5)}%")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=rank_1_df["pyName"],
            y=rank_1_df["Win Count"],
            text=rank_1_df["pyName"],
            textposition="auto",
            hovertemplate="<b>%{text}</b><br>Win Count: %{y}<extra></extra>",
        )
    )
    fig.update_yaxes(title="Win Count")
    fig.update_xaxes(showticklabels=False)  # hide x-axis labels
    bin_index = list(fig.data[0]["x"]).index(st.session_state.action)
    fig.data[0]["marker_color"] = (
        ["grey"] * bin_index
        + ["#FF0000"]
        + ["grey"] * (rank_1_df.shape[0] - bin_index - 1)
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    ranked_df = st.session_state.decision_data.reRank(
        overrides=[
            (
                pl.when(pl.col("pyName") == st.session_state.action)
                .then(pl.lit(lever))
                .otherwise(pl.col("Weight"))
            ).alias("Weight")
        ]
    )

st.subheader(":green[Lever Finder]:male-detective:")

st.session_state.target_win_percentage = st.slider(
    "Target Win Ratio", min_value=0, max_value=100
)

calculate_lever = st.button("Calculate lever")
if calculate_lever:
    with st.spinner("Calculating..."):
        # TODO refactor this into the DecisionData class
        lever_for_desired_ratio = find_lever_value(
            st.session_state.decision_data,
            st.session_state.action,
            target_win_percentage=st.session_state.target_win_percentage,
            win_rank=st.session_state.win_rank,
            high=st.session_state.max_search_range,
            ranking_stages=st.session_state.decision_data.stages_from_arbitration_down,
        )
        if isinstance(lever_for_desired_ratio, float):
            st.metric(
                f"""Lever you need for **{st.session_state.action}** to win in
                {st.session_state.target_win_percentage}% of the interactions""",
                lever_for_desired_ratio,
            )
