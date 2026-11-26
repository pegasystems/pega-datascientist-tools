from __future__ import annotations

import polars as pl
import streamlit as st

from pdstools.app.decision_analyzer.da_streamlit_utils import (
    collect_page_filters,
    contextual_filters,
    ensure_data,
    win_rank_selector,
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
st.caption("Replace business levers and compare the winning-offer mix.")

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
    win_rank_selector()
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
    st.caption("Original levers retained. Move the slider to apply a replacement.")
else:
    st.caption(f"Replacement lever: **{lever:.2f}** for **{target_label}**. Other offers retain their original levers.")

win_rank = st.session_state.win_rank
distribution, summary_frame = da.scoring.lever_simulation(
    condition, lever, additional_filters=filters, win_rank=win_rank
)
summary = summary_frame.row(0, named=True)

st.subheader(f"Winning share for selected {level.lower()}: {target_label}")
st.caption(
    f"Share of all **{summary['Interactions']:,} analyzed decisions** with "
    f"**at least one selected offer in the Top-{win_rank}**. Competitors remain in the comparison."
)
before_metric, after_metric, change_metric = st.columns(3, gap="small")
before_metric.metric("Before", f"{summary['Before share (%)']:.1f}%")
after_metric.metric("After", f"{summary['After share (%)']:.1f}%")
change_metric.metric("Change", f"{summary['Change (pp)']:+.1f} pp")

for column, scenario in zip(st.columns(2, gap="small"), ("Before", "After"), strict=True):
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
            ).update_layout(height=320, margin=dict(t=24, b=0, l=0, r=0)),
            key=f"lever_{scenario.lower()}",
        )

st.subheader("Offer share changes")
st.dataframe(distribution, hide_index=True, height=240)
decision_label = "decision" if summary["Re-evaluated decisions"] == 1 else "decisions"
st.caption(
    f"Re-evaluated **{summary['Re-evaluated decisions']:,} {decision_label}** at arbitration "
    f"in **{summary['Elapsed (s)']:.3f} s** "
    "(sampled, filtered population; scoring and aggregation only)."
)

with st.expander("Simulation assumptions"):
    st.markdown(
        "Before and After reconstruct priority as **Propensity × Value × Context Weight × Levers**, "
        "so they may differ from recorded output. Missing component values use 1.0. "
        "Mandatory-action precedence is preserved. **0% sets the lever to 1.0; Reset removes the override.**\n\n"
        f"Treemap tile sizes count Top-{win_rank} offer appearances. Explore Issue → Group → Action. "
        "The table shows each action's coverage of all analyzed decisions, rather than shares within a parent tile. "
        "For Top-N, action coverage percentages can overlap and need not sum to 100%. "
        "Selected-issue/group coverage counts each decision only once.\n\n"
        "Results use the sampled, filtered population of recorded candidates. This illustration does not "
        "update product configuration or predict future customer responses."
    )
