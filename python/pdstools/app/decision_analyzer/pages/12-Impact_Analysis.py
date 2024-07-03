import polars as pl
import streamlit as st


# TODO: Implement - or drop, find some use for this first, see once we get better data incl impact analyzer data

"# Impact Analyzer like analysis"

"""
Analysis of the effect of using different elements of the framework.

Although we do not have the actual responses/outcomes in the DA dataset we
can probably get far by using the propensities as proxies and analyse the
effect of the different experiments.

The Prediction experiments of model control group

The Impact Analyzer experiments with turning on/off different elements of the framework

* Is the random control group working? (reasonable test percentages)

"""

st.session_state["sidebar"] = st.sidebar

st.warning(
    """
We currently don't get the experiment properties that Impact Analyzer uses, we only
have (manipulated) data for the Prediction Studio control groups.
"""
)

st.dataframe(
    st.session_state.decision_data.getABTestResults().with_columns(
        pl.col("Control Percentage") * 100,
        # TODO perhaps the re-mapping of stage names can be done in plotly as well
        # instead of changing the data like we do here
        pl.col("pxEngagementStage")
        .replace(
            st.session_state.decision_data.NBADStages_Mapping
        )  # Replacing with "remaining" view labels
        .cast(
            pl.Enum(list(st.session_state.decision_data.NBADStages_Mapping.values()))
        ),
    ),
    hide_index=True,
    column_config={
        "pxEngagementStage": "Stage",
        "Control Percentage": st.column_config.NumberColumn(format="%.2f %%"),
    },
)
