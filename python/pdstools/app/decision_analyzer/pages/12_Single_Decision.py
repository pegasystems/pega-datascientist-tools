# python/pdstools/app/decision_analyzer/pages/12_Single_Decision.py
import html as html_mod
from itertools import groupby

import polars as pl
import streamlit as st
import streamlit.components.v1 as components
from da_streamlit_utils import (
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
        label = f"❌ {n_filtered}"
        if comp_text:
            label += f"\n{comp_text}"
        parts.append(label)
    if n_passing > 0:
        parts.append(f"✅ {n_passing}")
    return " · ".join(parts) if parts else "—"


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
            label = f"✅ {surviving}"
        else:
            # No rows at this stage — actions silently pass through
            label = f"✅ {surviving}"

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

# ── Render hierarchical HTML table ────────────────────────────────────────
# Columns for display
display_pvcl = [f for f in pvcl_factors if any(f in r for r in grid_rows)]
header_cols = [""] + display_pvcl + stages_ordered  # "" = name/label column

# Check if there are children at each level
has_groups = any(r["_depth"] == 1 for r in grid_rows)
has_actions = any(r["_depth"] == 2 for r in grid_rows)

# Build header — two-row when Stage-level with group header
if show_group_header:
    # Row 1: Stage Group spanning cells + empty cells for name/PVCL columns
    n_fixed = 1 + len(display_pvcl)  # name col + PVCL cols
    group_header_cells = ['<th rowspan="2"></th>']  # name col spans both rows
    for f in display_pvcl:
        group_header_cells.append(f'<th rowspan="2">{html_mod.escape(f)}</th>')

    # Group consecutive stages by their Stage Group
    grouped_stages = []
    for grp, members in groupby(stages_ordered, key=lambda s: stage_to_group.get(s, s)):
        member_list = list(members)
        grouped_stages.append((grp, member_list))

    for grp, members in grouped_stages:
        escaped = html_mod.escape(grp)
        group_header_cells.append(f'<th class="group-header" colspan="{len(members)}">{escaped}</th>')
    group_header_html = "<tr>" + "".join(group_header_cells) + "</tr>"

    # Row 2: individual Stage names (multi-line)
    stage_header_cells = []
    for col in stages_ordered:
        label = html_mod.escape(col).replace(" ", "<br>")
        stage_header_cells.append(f'<th class="stage-col">{label}</th>')
    stage_header_html = "<tr>" + "".join(stage_header_cells) + "</tr>"

    header_html = group_header_html + stage_header_html
else:
    # Single-row header (Stage Group level or no mapping)
    header_cells = []
    for col in header_cols:
        escaped = html_mod.escape(col)
        if col in stages_ordered:
            label = escaped.replace(" ", "<br>")
            header_cells.append(f'<th class="stage-col">{label}</th>')
        else:
            header_cells.append(f"<th>{escaped}</th>")
    header_html = "<tr>" + "".join(header_cells) + "</tr>"

# Build body rows
body_html_parts = []
for r in grid_rows:
    depth = r["_depth"]
    key = html_mod.escape(r["_key"])
    parent = html_mod.escape(r.get("_parent", ""))
    label = html_mod.escape(r["_label"])

    is_totals = depth == -1

    # Determine if this row has children
    has_children = (depth == 0 and has_groups) or (depth == 1 and has_actions)

    # Row attributes
    cls = "totals-row" if is_totals else f"depth-{depth}"
    hidden = ' style="display:none"' if depth > 0 else ""
    data_attrs = f'data-key="{key}" data-depth="{depth}"'
    if parent:
        data_attrs += f' data-parent="{parent}"'

    # Label cell with indent and toggle icon
    if is_totals:
        name_cell = f"<td><b>{label}</b></td>"
    else:
        indent = depth * 20
        if has_children:
            toggle = '<span class="toggle">▶</span> '
        else:
            toggle = '<span style="display:inline-block;width:1em"></span> '
        name_cell = f'<td style="padding-left:{indent + 6}px">{toggle}<b>{label}</b></td>'

    # PVCL cells
    pvcl_cells = []
    for f in display_pvcl:
        val = html_mod.escape(str(r.get(f, "")))
        pvcl_cells.append(f"<td>{val}</td>")

    # Stage cells
    stage_cells = []
    for stage in stages_ordered:
        val = html_mod.escape(str(r.get(stage, "—"))).replace("\n", "<br>")
        if "❌" in val:
            stage_cells.append(f'<td class="cell-filtered">{val}</td>')
        elif "✅" in val:
            stage_cells.append(f'<td class="cell-pass">{val}</td>')
        elif "∅" in val:
            stage_cells.append('<td class="cell-empty">∅</td>')
        else:
            stage_cells.append(f'<td class="cell-skip">{val}</td>')

    all_cells = name_cell + "".join(pvcl_cells) + "".join(stage_cells)
    body_html_parts.append(f'<tr class="{cls}" {data_attrs}{hidden}>{all_cells}</tr>')

if pvcl_factors:
    st.caption("PVCL values at Issue/Group level show **min|median|max** across actions.")

# Estimate height: generous to avoid cutting off rows.
n_all_rows = len(grid_rows)
estimated_height = 300 + n_all_rows * 45

components.html(
    f"""
    <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: "Source Sans Pro", sans-serif; }}
    .decision-grid {{
        overflow-x: auto;
    }}
    .decision-grid table {{
        border-collapse: collapse;
        font-size: 0.85rem;
    }}
    .decision-grid th, .decision-grid td {{
        border: 1px solid #ddd;
        padding: 6px 8px;
        text-align: left;
    }}
    .decision-grid th {{
        background: #f8f9fa;
        font-weight: 600;
    }}
    .decision-grid th.group-header {{
        text-align: center;
        background: #e9ecef;
        font-size: 0.80rem;
        border-bottom: 2px solid #adb5bd;
    }}
    .decision-grid th.stage-col {{
        vertical-align: bottom;
        text-align: center;
        padding: 4px 6px;
        font-size: 0.78rem;
        line-height: 1.2;
        min-width: 60px;
    }}
    .decision-grid .cell-filtered {{
        background: #fff0f0;
    }}
    .decision-grid .cell-pass {{
        background: #f0fff0;
    }}
    .decision-grid .cell-skip {{
        color: #bbb;
    }}
    .decision-grid .cell-empty {{
        color: #999;
        text-align: center;
        background: #f5f5f5;
    }}
    .decision-grid .depth-0 td:first-child {{
        font-size: 0.95rem;
    }}
    .decision-grid .depth-1 {{
        background: #fafafa;
    }}
    .decision-grid .depth-2 {{
        background: #f5f5f5;
    }}
    .decision-grid .totals-row {{
        border-top: 2px solid #333;
        background: #f8f9fa;
        font-weight: 600;
    }}
    .decision-grid .toggle {{
        font-size: 0.7rem;
        display: inline-block;
        width: 1em;
        cursor: pointer;
        user-select: none;
        transition: transform 0.15s;
    }}
    .decision-grid .toggle.expanded {{
        transform: rotate(90deg);
    }}
    .expand-controls {{
        margin-bottom: 6px;
        display: flex;
        gap: 8px;
    }}
    .expand-controls button {{
        background: #f8f9fa;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 0.8rem;
        cursor: pointer;
        font-family: inherit;
    }}
    .expand-controls button:hover {{
        background: #e9ecef;
    }}
    </style>
    <div class="decision-grid">
    <div class="expand-controls">
      <button onclick="expandAll()">Expand all</button>
      <button onclick="collapseAll()">Collapse all</button>
    </div>
    <table>
    <thead>{header_html}</thead>
    <tbody>{"".join(body_html_parts)}</tbody>
    </table>
    </div>
    <script>
    function toggleChildren(el) {{
        const row = el.closest('tr');
        const key = row.dataset.key;
        const depth = parseInt(row.dataset.depth);
        const isExpanding = !el.classList.contains('expanded');
        el.classList.toggle('expanded');
        el.textContent = isExpanding ? '\\u25BC' : '\\u25B6';

        let sibling = row.nextElementSibling;
        while (sibling) {{
            const sibDepth = parseInt(sibling.dataset.depth);
            if (sibDepth <= depth) break;

            if (isExpanding) {{
                if (sibDepth === depth + 1) {{
                    sibling.style.display = '';
                    const childToggle = sibling.querySelector('.toggle');
                    if (childToggle && childToggle.classList.contains('expanded')) {{
                        childToggle.classList.remove('expanded');
                        childToggle.textContent = '\\u25B6';
                    }}
                }}
                if (sibDepth > depth + 1) {{
                    sibling.style.display = 'none';
                }}
            }} else {{
                sibling.style.display = 'none';
                const childToggle = sibling.querySelector('.toggle');
                if (childToggle && childToggle.classList.contains('expanded')) {{
                    childToggle.classList.remove('expanded');
                    childToggle.textContent = '\\u25B6';
                }}
            }}
            sibling = sibling.nextElementSibling;
        }}
    }}

    // Attach click handlers
    document.querySelectorAll('.toggle').forEach(function(el) {{
        el.addEventListener('click', function() {{ toggleChildren(this); }});
    }});

    function expandAll() {{
        document.querySelectorAll('tbody tr').forEach(function(row) {{
            if (parseInt(row.dataset.depth) >= 0) {{
                row.style.display = '';
            }}
            const toggle = row.querySelector('.toggle');
            if (toggle) {{
                toggle.classList.add('expanded');
                toggle.textContent = '\\u25BC';
            }}
        }});
    }}

    function collapseAll() {{
        document.querySelectorAll('tbody tr').forEach(function(row) {{
            const depth = parseInt(row.dataset.depth);
            if (depth > 0) {{
                row.style.display = 'none';
            }}
            const toggle = row.querySelector('.toggle');
            if (toggle) {{
                toggle.classList.remove('expanded');
                toggle.textContent = '\\u25B6';
            }}
        }});
    }}
    </script>
    """,
    height=estimated_height,
    scrolling=True,
)
