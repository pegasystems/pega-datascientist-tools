# python/pdstools/app/decision_analyzer/pages/12_Single_Decision.py
import html as html_mod
from itertools import groupby

import polars as pl
import streamlit as st
from pdstools.app.decision_analyzer.da_streamlit_utils import (
    ensure_data,
    stage_level_selector,
)

from pdstools.utils.streamlit_utils import standard_page_config

standard_page_config(page_title="Single Decision · Decision Analysis")

"# Single Decision"

"""
Debug a single decision by inspecting its full pipeline. The grid shows
every action (or issue / group) as a row and every stage as a column.

* **✅ N** — N actions passed this stage.
* **❌ N** — N actions were filtered out, with the responsible
  component(s) shown.
* **∅** — no actions remaining (all filtered at earlier stages).
"""

ensure_data()

da = st.session_state.decision_data
available_cols = set(da.decision_data.collect_schema().names())

# ── Sidebar ────────────────────────────────────────────────────────────────
st.session_state["sidebar"] = st.sidebar
with st.session_state["sidebar"]:
    stage_level_selector()

# ── Interaction selector ──────────────────────────────────────────────────
sample_ids = (
    da.decision_data.select("Interaction ID").unique().head(50).collect().get_column("Interaction ID").sort().to_list()
)

col_select, col_text = st.columns([1, 1])
with col_select:
    dropdown_id = st.selectbox(
        "Select Interaction ID",
        options=sample_ids,
        index=0,
        help="First 50 unique Interaction IDs from the dataset.",
    )
with col_text:
    typed_id = st.text_input(
        "Or type an Interaction ID",
        value="",
        help="Overrides the dropdown when non-empty.",
    )

selected_id = typed_id.strip() if typed_id.strip() else dropdown_id

if not selected_id:
    st.warning("Please select or enter an Interaction ID.")
    st.stop()

# ── Query data for selected interaction ───────────────────────────────────
desired_cols = [
    "Interaction ID",
    "Issue",
    "Group",
    "Action",
    "Stage",
    "Stage Group",
    "Stage Order",
    "Record Type",
    "Component Name",
    "Rank",
    "Propensity",
    "Value",
    "Context Weight",
    "Levers",
    "Priority",
    "Channel",
    "Direction",
    "Subject ID",
]
query_cols = [c for c in desired_cols if c in available_cols]

interaction_df = (
    da.decision_data.filter(pl.col("Interaction ID") == selected_id)
    .select(query_cols)
    .sort("Stage Order", "Rank")
    .collect()
)

if interaction_df.is_empty():
    st.warning(f'No data found for Interaction ID "{selected_id}".')
    st.stop()

# ── Context: channel / direction ──────────────────────────────────────────
context_parts = []
if "Channel" in interaction_df.columns:
    channel = interaction_df.get_column("Channel")[0]
    if channel is not None:
        context_parts.append(f"**Channel:** {channel}")
if "Direction" in interaction_df.columns:
    direction = interaction_df.get_column("Direction")[0]
    if direction is not None:
        context_parts.append(f"**Direction:** {direction}")
if "Subject ID" in interaction_df.columns:
    subject_id = interaction_df.get_column("Subject ID")[0]
    if subject_id is not None:
        subject_id = str(subject_id).split("^")[0]
        context_parts.append(f"**Subject:** {subject_id}")
if context_parts:
    st.info(" · ".join(context_parts))

# V1 hint
if da.extract_type == "explainability_extract":
    st.caption("ℹ️ V1 (Explainability Extract) data has limited stage detail and no filter-component information.")

# ── Build the grid ────────────────────────────────────────────────────────
level_col = da.level  # "Stage Group" or "Stage"

# Determine which hierarchy levels are available
scope_hierarchy = ["Issue", "Group", "Action"]
available_scopes = [c for c in scope_hierarchy if c in available_cols]

# Ordered stages: use the full pipeline from the DA so that stages where
# actions silently pass through (e.g. Arbitration) still appear as columns.
stages_ordered = da.AvailableNBADStages

# When viewing at "Stage" level, build a Stage → Stage Group mapping for
# grouped column headers.
stage_to_group: dict[str, str] = {}
show_group_header = False
if level_col == "Stage" and "Stage Group" in available_cols:
    stage_to_group = dict(da.stage_to_group_mapping)  # {Stage: StageGroup}
    # Placeholder stages (group name used as stage name) map to themselves
    for s in stages_ordered:
        if s not in stage_to_group:
            stage_to_group[s] = s
    show_group_header = bool(stage_to_group)

pvcl_factors = [f for f in ["Propensity", "Value", "Context Weight", "Levers", "Priority"] if f in available_cols]


def _cell_label(filtered_df: pl.DataFrame, n_surviving: int, *, show_components: bool = True) -> str:
    """Summarise what happened at one stage for a scope-group.

    Parameters
    ----------
    filtered_df
        Rows with Record Type == FILTERED_OUT at this stage (may be empty).
    n_surviving
        Number of actions still alive *entering* this stage.
    show_components
        Whether to list component names in the cell.
    """
    has_action = "Action" in filtered_df.columns if not filtered_df.is_empty() else False
    n_filtered = (
        filtered_df.select("Action").n_unique() if has_action and not filtered_df.is_empty() else len(filtered_df)
    )
    n_passing = n_surviving - n_filtered

    parts = []
    if n_filtered > 0:
        if show_components and "Component Name" in filtered_df.columns:
            components = filtered_df.get_column("Component Name").drop_nulls()
            if len(components) > 0:
                unique_components = (
                    components.value_counts().sort("count", descending=True).get_column("Component Name").to_list()
                )
                if len(unique_components) <= 5:
                    comp_text = "\n".join(unique_components)
                else:
                    comp_text = "\n".join(unique_components[:5]) + f"\n+{len(unique_components) - 5} more"
            else:
                comp_text = ""
        else:
            comp_text = ""
        label = f"❌\u00a0{n_filtered}"
        if comp_text:
            label += f"\n{comp_text}"
        parts.append(label)
    if n_passing > 0:
        parts.append(f"✅\u00a0{n_passing}")
    return "\n".join(parts) if parts else "—"


def _pvcl_values(subset: pl.DataFrame, is_aggregate: bool) -> dict:
    """Compute PVCL display values for a scope-group."""
    result: dict = {}
    if not pvcl_factors:
        return result
    # Deduplicate to one row per action (take highest Stage Order)
    if "Action" in subset.columns:
        per_action = subset.sort("Stage Order", descending=True).group_by("Action", maintain_order=True).first()
    else:
        per_action = subset
    for factor in pvcl_factors:
        if factor not in per_action.columns:
            continue
        vals = per_action.get_column(factor).drop_nulls()
        if len(vals) == 0:
            result[factor] = ""
        elif not is_aggregate or len(vals) == 1:
            result[factor] = f"{vals[0]:g}"
        else:
            mn, md, mx = vals.min(), vals.median(), vals.max()
            if mn == mx:
                result[factor] = f"{mn:g}"
            else:
                result[factor] = f"{mn:.3g}|{md:.3g}|{mx:.3g}"
    return result


def _stage_cells(subset: pl.DataFrame, *, show_components: bool = True) -> dict:
    """Compute per-stage cell labels, tracking surviving actions cumulatively.

    The data only records events (FILTERED_OUT or OUTPUT rows).  Actions
    that silently pass through a stage have no row there, so we track the
    surviving count by starting with the total number of unique actions and
    subtracting those filtered at each successive stage.
    """
    result: dict = {}
    has_action = "Action" in subset.columns
    has_record_type = "Record Type" in subset.columns

    # Total unique actions in this scope-group
    total_actions = subset.select("Action").n_unique() if has_action else len(subset)
    surviving = total_actions

    for stage in stages_ordered:
        stage_data = subset.filter(pl.col(level_col) == stage)

        if surviving <= 0:
            result[stage] = "∅"
            continue

        # Get filtered rows at this stage
        if has_record_type and not stage_data.is_empty():
            filtered_at_stage = stage_data.filter(pl.col("Record Type") == "FILTERED_OUT")
        else:
            filtered_at_stage = stage_data.clear() if not stage_data.is_empty() else pl.DataFrame()

        n_filtered = (
            filtered_at_stage.select("Action").n_unique()
            if has_action and not filtered_at_stage.is_empty()
            else len(filtered_at_stage)
        )

        if n_filtered > 0:
            label = _cell_label(filtered_at_stage, surviving, show_components=show_components)
            surviving -= n_filtered
        elif not stage_data.is_empty():
            # Stage has rows (e.g. OUTPUT) but no filtering — all survive
            label = f"✅\u00a0{surviving}"
        else:
            # No rows at this stage — actions silently pass through
            label = f"✅\u00a0{surviving}"

        result[stage] = label
    return result


# Build hierarchical rows: Issue → Group → Action
# Each row carries a "depth" (0=Issue, 1=Group, 2=Action) and a "key" for
# parent-child toggling.
grid_rows: list[dict] = []

for issue, issue_df in interaction_df.group_by("Issue", maintain_order=True):
    if not isinstance(issue, str):
        issue = issue[0] if isinstance(issue, tuple) else str(issue)

    # Issue-level row (depth 0)
    n_actions_issue = issue_df.select("Action").n_unique() if "Action" in issue_df.columns else 0
    issue_label = f"{issue} ({n_actions_issue})" if n_actions_issue else issue
    issue_row = {"_depth": 0, "_key": issue, "_label": issue_label}
    issue_row.update(_pvcl_values(issue_df, is_aggregate=True))
    issue_row.update(_stage_cells(issue_df))
    grid_rows.append(issue_row)

    if "Group" not in available_cols:
        continue

    for group, group_df in issue_df.group_by("Group", maintain_order=True):
        if not isinstance(group, str):
            group = group[0] if isinstance(group, tuple) else str(group)

        # Group-level row (depth 1)
        group_key = f"{issue}||{group}"
        n_actions_group = group_df.select("Action").n_unique() if "Action" in group_df.columns else 0
        group_label = f"{group} ({n_actions_group})" if n_actions_group else group
        group_row = {"_depth": 1, "_key": group_key, "_parent": issue, "_label": group_label}
        group_row.update(_pvcl_values(group_df, is_aggregate=True))
        group_row.update(_stage_cells(group_df))
        grid_rows.append(group_row)

        if "Action" not in available_cols:
            continue

        for action, action_df in group_df.group_by("Action", maintain_order=True):
            if not isinstance(action, str):
                action = action[0] if isinstance(action, tuple) else str(action)

            # Action-level row (depth 2)
            action_key = f"{issue}||{group}||{action}"
            action_row = {"_depth": 2, "_key": action_key, "_parent": group_key, "_label": action}
            action_row.update(_pvcl_values(action_df, is_aggregate=False))
            action_row.update(_stage_cells(action_df))
            grid_rows.append(action_row)

if not grid_rows:
    st.warning("No actions found for the selected interaction.")
    st.stop()

# ── Grand totals row ──────────────────────────────────────────────────────
total_actions = (
    interaction_df.select("Action").n_unique() if "Action" in interaction_df.columns else len(interaction_df)
)
totals_row = {"_depth": -1, "_key": "__totals__", "_label": f"Total ({total_actions})"}
for f in pvcl_factors:
    totals_row[f] = ""
totals_row.update(_stage_cells(interaction_df, show_components=False))
grid_rows.append(totals_row)

# ── Render hierarchical grid using <details> for native expand/collapse ───
# We render the tree with HTML5 <details>/<summary> elements: each parent
# row becomes a <summary> that toggles its child <div> on click, with zero
# JavaScript. This lets us use st.html() (which strips <script> tags) and
# avoids the soon-to-be-removed components.html API.
#
# Layout uses CSS Grid (not <table>) so that nested <details> blocks can
# share a single column template — every row, at every depth, declares the
# same grid-template-columns and therefore aligns vertically with the
# header. <details> inside <tbody> is not valid HTML and breaks table
# layout, so we trade table semantics for a stacked-card layout.
#
# Trade-off vs. the previous JS-based version: the global "Expand all" and
# "Collapse all" buttons are gone (<details> is per-element; bringing them
# back would require a custom Streamlit component). Each parent row still
# expands/collapses by clicking its summary header.
# TODO: bringing back expand-all would require a custom Streamlit component.

display_pvcl = [f for f in pvcl_factors if any(f in r for r in grid_rows)]
n_pvcl = len(display_pvcl)
n_stages = len(stages_ordered)

# Shared grid template: one wide name column, then PVCL columns, then stage
# columns. Every row reuses this so nested cards line up with the header.
grid_template = (
    "grid-template-columns: minmax(180px, 2fr)"
    + (" minmax(70px, 1fr)" * n_pvcl)
    + (" minmax(60px, 1fr)" * n_stages)
    + ";"
)

# Build parent → children map for recursive rendering.
children_map: dict[str, list[dict]] = {}
roots: list[dict] = []
totals_row: dict | None = None
for r in grid_rows:
    d = r["_depth"]
    if d == -1:
        totals_row = r
    elif d == 0:
        roots.append(r)
    else:
        children_map.setdefault(r.get("_parent", ""), []).append(r)


def _row_cells(row: dict) -> str:
    """Build the inner grid cells (name + PVCL + stages) for a single row."""
    depth = row["_depth"]
    is_totals = depth == -1
    label = html_mod.escape(row["_label"])
    indent = 0 if is_totals else depth * 20
    # Empty .toggle span reserves width on every row so leaf labels align
    # horizontally with parent labels (the ::before triangle is only added
    # to summary rows via CSS).
    name_cell = (
        f'<div class="cell name-cell" style="padding-left:{indent + 6}px">'
        f'<span class="toggle"></span><b>{label}</b></div>'
    )

    pvcl_cells_html = "".join(f'<div class="cell">{html_mod.escape(str(row.get(f, "")))}</div>' for f in display_pvcl)

    stage_cells = []
    for stage in stages_ordered:
        val = html_mod.escape(str(row.get(stage, "—"))).replace("\n", "<br>")
        if "❌" in val:
            cls = "cell cell-filtered"
        elif "✅" in val:
            cls = "cell cell-pass"
        elif "∅" in val:
            cls = "cell cell-empty"
            val = "∅"
        else:
            cls = "cell cell-skip"
        stage_cells.append(f'<div class="{cls}">{val}</div>')

    return name_cell + pvcl_cells_html + "".join(stage_cells)


def _render_node(row: dict) -> str:
    """Recursively render a row + its descendants as nested <details>."""
    depth = row["_depth"]
    cls = f"depth-{depth}"
    cells = _row_cells(row)
    children = children_map.get(row["_key"], [])
    if not children:
        return f'<div class="grid-row {cls}" style="{grid_template}">{cells}</div>'
    body = "".join(_render_node(child) for child in children)
    return (
        f'<details class="row-details">'
        f'<summary class="grid-row {cls}" style="{grid_template}">{cells}</summary>'
        f'<div class="grid-body">{body}</div>'
        f"</details>"
    )


# ── Header ─────────────────────────────────────────────────────────────────
if show_group_header:
    # Two-row header in a single grid: name + PVCL columns span both rows;
    # stage-group cells occupy row 1 (with grid-column: span N); individual
    # stage names occupy row 2.
    header_cells = ['<div class="cell hdr name-hdr">&nbsp;</div>']
    for f in display_pvcl:
        header_cells.append(f'<div class="cell hdr pvcl-hdr">{html_mod.escape(f)}</div>')

    grouped_stages = [
        (grp, list(members)) for grp, members in groupby(stages_ordered, key=lambda s: stage_to_group.get(s, s))
    ]
    for grp, members in grouped_stages:
        escaped = html_mod.escape(grp)
        header_cells.append(f'<div class="cell group-header" style="grid-column: span {len(members)};">{escaped}</div>')
    for col in stages_ordered:
        label = html_mod.escape(col).replace(" ", "<br>")
        header_cells.append(f'<div class="cell stage-col">{label}</div>')

    header_html = (
        f'<div class="grid-row header-row two-row" style="{grid_template} grid-template-rows: auto auto;">'
        + "".join(header_cells)
        + "</div>"
    )
else:
    header_cells = ['<div class="cell hdr name-hdr">&nbsp;</div>']
    for f in display_pvcl:
        header_cells.append(f'<div class="cell hdr pvcl-hdr">{html_mod.escape(f)}</div>')
    for col in stages_ordered:
        label = html_mod.escape(col).replace(" ", "<br>")
        header_cells.append(f'<div class="cell stage-col">{label}</div>')
    header_html = f'<div class="grid-row header-row" style="{grid_template}">' + "".join(header_cells) + "</div>"

# ── Body ──────────────────────────────────────────────────────────────────
body_html = "".join(_render_node(r) for r in roots)
if totals_row is not None:
    body_html += f'<div class="grid-row totals-row" style="{grid_template}">' + _row_cells(totals_row) + "</div>"

if pvcl_factors:
    st.caption("PVCL values at Issue/Group level show **min|median|max** across actions.")

st.html(
    f"""
    <style>
    .decision-grid {{
        font-family: "Source Sans Pro", sans-serif;
        font-size: 0.85rem;
        overflow-x: auto;
    }}
    .decision-grid .grid-row {{
        display: grid;
        border-bottom: 1px solid #ddd;
    }}
    .decision-grid .grid-row > div {{
        padding: 6px 8px;
        border-right: 1px solid #eee;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .decision-grid .header-row {{
        background: #f8f9fa;
        font-weight: 600;
        border-bottom: 2px solid #adb5bd;
    }}
    .decision-grid .header-row.two-row .name-hdr,
    .decision-grid .header-row.two-row .pvcl-hdr {{
        grid-row: 1 / span 2;
        align-self: stretch;
    }}
    .decision-grid .group-header {{
        text-align: center;
        background: #e9ecef;
        font-size: 0.80rem;
    }}
    .decision-grid .stage-col {{
        text-align: center;
        font-size: 0.78rem;
        line-height: 1.2;
        white-space: normal;
    }}
    .decision-grid .cell-filtered {{ background: #fff0f0; }}
    .decision-grid .cell-pass {{ background: #f0fff0; }}
    .decision-grid .cell-skip {{ color: #bbb; }}
    .decision-grid .cell-empty {{
        color: #999;
        text-align: center;
        background: #f5f5f5;
    }}
    .decision-grid .depth-0 .name-cell {{ font-size: 0.95rem; }}
    .decision-grid .depth-1 {{ background: #fafafa; }}
    .decision-grid .depth-2 {{ background: #f5f5f5; }}
    .decision-grid .totals-row {{
        border-top: 2px solid #333;
        background: #f8f9fa;
        font-weight: 600;
    }}
    .decision-grid .toggle {{
        display: inline-block;
        width: 1em;
        margin-right: 4px;
        font-size: 0.7rem;
        color: #666;
    }}
    .decision-grid summary.grid-row {{
        cursor: pointer;
        list-style: none;
    }}
    .decision-grid summary.grid-row::-webkit-details-marker {{
        display: none;
    }}
    .decision-grid summary.grid-row .toggle::before {{
        content: '▶';
        display: inline-block;
        transition: transform 0.15s;
    }}
    .decision-grid details[open] > summary.grid-row .toggle::before {{
        transform: rotate(90deg);
    }}
    </style>
    <div class="decision-grid">
    {header_html}
    {body_html}
    </div>
    """
)
