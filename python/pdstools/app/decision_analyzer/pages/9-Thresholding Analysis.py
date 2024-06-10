import plotly.express as px
import polars as pl
import streamlit as st

from plots import plot_distribution, plot_threshold_deciles
from utils import ensure_data

# TODO Interactive Thresholding isn't working properly yet. Also show the total numbers.
# TODO Instead of priority/propensity side to side have a drop-down to select which property to show
# TODO show volume change and distribution change, experiment with showing delta with a grouped bar chart; underlying data may be one DF with a column for the "condition"
# TODO can then also show expected sum of propensities for the top-1 ranked items. Win loss looks similar.

"# Analysis for Propensity and Priority thresholding"

"""
Analysis of the propensity and other arbitration factors.

There will be an easy way to see the effects of thresholding on volume and distribution of
the actions.

* What is the effect of new offers (new models with propensity 0.5, showing a peak there)?
* What should my priority / propensity threshold be and how does this effect the volumes and distributions? You will want to see this for a specific channel, which can easily be accomplished by globally filtering on that channel only.
* Is the random control group working? (propensity spreading 0-1)

"""
ensure_data()


st.session_state["sidebar"] = st.sidebar
thresholding_mapping = {
    # TODO generalize, move to one of the utils
    "FinalPropensity": "Propensity",
}

with st.session_state["sidebar"]:
    # TODO: work in progress
    thresholding_on = st.radio(
        "Thresholding",
        options=list(thresholding_mapping.keys()),
        format_func=lambda option: thresholding_mapping[option],
        horizontal=True,
    )
    value_range = st.session_state.decision_data.getThresholdingData(
        thresholding_on, quantile_range=[0, 100]
    )["Threshold"].to_list()

    current_threshold = st.slider(
        "Threshold :sunglasses:", value_range[0], value_range[1]
    )

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(
        px.histogram(
            # Note this is not overly expensive as we have sampled the values into the pre-agg views
            st.session_state.decision_data.getPreaggregatedFilterView
            # TODO this breaks when the list is size > 1, figure out how to solve elegantly
            .select(pl.col("FinalPropensity").explode(), pl.col("Decisions")).collect(),
            x="FinalPropensity",
            y="Decisions",
        ),
        use_container_width=True,
    )
with col2:
    st.plotly_chart(
        px.histogram(
            st.session_state.decision_data.getPreaggregatedFilterView
            # TODO this breaks when the list is size > 1, figure out how to solve elegantly
            .select(pl.col("Priority").explode(), pl.col("Decisions")).collect(),
            x="Priority",
            y="Decisions",
            log_y=True,  # TODO maybe make this a UI control
        ),
        use_container_width=True,
    )

threshold_deciles_data = st.session_state.decision_data.getThresholdingData(
    thresholding_on
)
# st.dataframe(plotData)

st.plotly_chart(
    plot_threshold_deciles(
        threshold_deciles_data, thresholding_mapping[thresholding_on]
    ),
    use_container_width=True,
)

# TODO fix this, not working properly. Filtering isn't working, should probably be on the
# sampled values not the min/max. Bars perhaps as facets rather than separate plots.
"""Below plot is not working yet"""

# st.write(current_threshold)
xxx = st.session_state.decision_data.getDistributionData(
    "Final",
    ["pyIssue", "pyGroup"],
    trend=False,
    additional_filters=(
        pl.col(f"{thresholding_on}_min") > current_threshold
    ),  # Hmm, probalby not the right way
    # additional_filters=((pl.col(thresholding_on).list.eval(pl.element() > current_threshold)).list.any()),
)
# st.write(xxx.head().collect())
st.write(
    plot_distribution(
        xxx,
        scope="pyIssue",
        breakdown="pyGroup",
        title="Effect of Thresholding",
        horizontal=True,
    )
)
