from __future__ import annotations

import polars as pl
import streamlit as st

from pdstools.app.decision_analyzer.da_streamlit_utils import (
    collect_page_filters,
    contextual_filters,
    ensure_data,
)
from pdstools.utils.streamlit_utils import standard_page_config

standard_page_config(page_title="What-if: Business Levers · Decision Analysis")
ensure_data()


def _persist_selection(key: str) -> None:
    st.session_state[key] = st.session_state[f"_{key}"]
    children = {
        "lever_issue": ["lever_group", "lever_action"],
        "lever_group": ["lever_action"],
    }
    for child in children.get(key, []):
        st.session_state.pop(child, None)
        st.session_state.pop(f"_{child}", None)


def _selectbox(label: str, key: str, options: list[str]) -> str:
    if not options:
        st.warning(f"No {label.lower()} values are available for the selected scope.")
        st.stop()
    current = st.session_state.get(key, options[0])
    if current not in options:
        st.info(f"Previous {label.lower()} selection is no longer available.")
        current = options[0]
        st.session_state.lever_override_active = False
    st.session_state[key] = current
    st.session_state[f"_{key}"] = current
    selection = st.selectbox(label, options, key=f"_{key}", on_change=_persist_selection, args=(key,))
    if selection is None:
        st.warning(f"Select a {label.lower()} to continue.")
        st.stop()
    return selection


def _persist_percentage() -> None:
    st.session_state.lever_percentage = st.session_state._lever_percentage
    st.session_state.lever_override_active = True


def _reset() -> None:
    st.session_state.lever_percentage = 0
    st.session_state._lever_percentage = 0
    st.session_state.lever_override_active = False


st.title("What-if: Business Levers")
st.caption(
    "Illustrate how replacing business levers changes the winning-offer mix. "
    "This simulation uses recorded candidates; it does not update product configuration "
    "or predict future customer responses."
)

da = st.session_state.decision_data
if st.session_state.get("lever_data_id") != id(da):
    for key in ("lever_level", "lever_issue", "lever_group", "lever_action"):
        st.session_state.pop(key, None)
        st.session_state.pop(f"_{key}", None)
    _reset()
    st.session_state.lever_data_id = id(da)

controls = st.sidebar.container()
with st.sidebar:
    contextual_filters()

filters = collect_page_filters()
identities = (
    da.filtered(filters)
    .filter(pl.col(da.level).is_in(da.stages_from_arbitration_down) & pl.col("Priority").is_not_null())
    .select("Issue", "Group", "Action")
    .unique()
    .collect()
)
if identities.is_empty():
    st.warning("No arbitration candidates are available. Adjust the population filters or load different data.")
    st.stop()

with controls:
    st.subheader("Lever adjustment")
    level = _selectbox("Adjust at", "lever_level", ["Issue", "Group", "Action"])
    issue = _selectbox("Issue", "lever_issue", identities["Issue"].drop_nulls().unique().sort().to_list())
    condition = pl.col("Issue") == issue
    target = [issue]
    if level in ("Group", "Action"):
        groups = identities.filter(condition)["Group"].drop_nulls().unique().sort().to_list()
        group = _selectbox("Group", "lever_group", groups)
        condition &= pl.col("Group") == group
        target.append(group)
    if level == "Action":
        actions = identities.filter(condition)["Action"].drop_nulls().unique().sort().to_list()
        action = _selectbox("Action", "lever_action", actions)
        condition &= pl.col("Action") == action
        target.append(action)

    st.session_state._lever_percentage = st.session_state.lever_percentage
    st.slider(
        "Lever adjustment relative to neutral (1.0)",
        min_value=-50,
        max_value=50,
        step=1,
        format="%d%%",
        key="_lever_percentage",
        on_change=_persist_percentage,
        help="Replace selected levers with 1 + percentage / 100; do not multiply their existing values.",
    )
    st.button("Reset", on_click=_reset, help="Remove the override and restore the original levers.")

lever = 1 + st.session_state.lever_percentage / 100 if st.session_state.lever_override_active else None
target_label = " / ".join(target)
if lever is None:
    st.info("No override active. Move the slider to replace the selected levers.")
else:
    st.info(
        f"Set the lever to **{lever:.2f}** for **{target_label}**. "
        "Existing lever values are replaced; competitors retain their original levers."
    )
st.caption(
    "Before and After both reconstruct priority as Propensity × Value × Context Weight × Levers. "
    "They may differ from recorded output. Missing component values use the neutral value 1.0. "
    "Mandatory-action precedence is preserved. 0% sets the lever to 1.0; Reset removes the override."
)

distribution, summary_frame = da.scoring.lever_simulation(condition, lever, additional_filters=filters)
summary = summary_frame.row(0, named=True)
st.caption(f"Analyzed population: {summary['Interactions']:,} sampled interactions, including those without a winner.")

before_metric, after_metric, change_metric = st.columns(3)
before_metric.metric("Selected scope: Before", f"{summary['Before share (%)']:.1f}%")
after_metric.metric("Selected scope: After", f"{summary['After share (%)']:.1f}%")
change_metric.metric("Change", f"{summary['Change (pp)']:+.1f} pp")

st.subheader("Winning-offer distribution")
st.caption(
    "Tile sizes represent rank-1 wins, not candidate appearances. Explore Issue → Group → Action. "
    "The table below shows shares of the entire analyzed population, rather than shares within a parent tile."
)
for column, scenario in zip(st.columns(2), ("Before", "After"), strict=True):
    with column:
        st.markdown(f"### {scenario}")
        chart_data = distribution.select("Issue", "Group", "Action", pl.col(scenario).alias("Decisions")).filter(
            pl.col("Decisions") > 0
        )
        st.plotly_chart(
            da.plot.distribution_as_treemap(
                df=chart_data.lazy(),
                stage=scenario,
                scope_options=["Issue", "Group", "Action"],
            ),
            key=f"lever_{scenario.lower()}",
        )

st.subheader("Offer share changes")
st.dataframe(distribution, hide_index=True)
st.caption(
    f"Re-evaluated **{summary['Re-evaluated decisions']:,} decisions** at arbitration "
    f"in **{summary['Elapsed (s)']:.3f} s** "
    "(sampled, filtered population; scoring and aggregation only)."
)
