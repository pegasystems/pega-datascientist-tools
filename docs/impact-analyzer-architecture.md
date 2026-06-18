# Impact Analyzer Architecture

**Updated:** May 13, 2026 — Guide for making changes to the Impact Analyzer
tool. Captures non-obvious design decisions and the experiment model.

For method signatures, column schemas, and page docstrings, read the source
directly.

---

## Main Class

`python/pdstools/impactanalyzer/ImpactAnalyzer.py`

- `ia_data: pl.LazyFrame` — long-format aggregate frame, one row per
  (SnapshotTime, Channel/Direction, Issue/Group/Action/Treatment,
  ControlGroup, Outcome) with `Impressions`, `Accepts`, `Value`.
- `outcome_labels_used: dict | None` — resolved outcome-label config.
- `__init__` is **pure**: takes the already-loaded LazyFrame + config.
  No file/network IO.

### Constructors (all IO lives here)

- `from_pdc(path, ...)` — Pega Decision Cloud `CDH_Metrics_*.json` snapshots.
- `from_vbd(path, ...)` — Visual Business Director export. Accepts a zip
  with `data.json` (NDJSON of VBD rows) or a parquet/CSV.

### Namespace facade

- `.plot` → `Plots` (channel/issue/group bar charts, lift waterfalls)
- `.statistics` → significance tests (Z-test for proportions)

---

## The 5 Experiments

Both the Pega Infinity product and pdstools define **5** experiments
(not 6 — the `ModelControl_1`/`_2` naming has historically caused this
miscount). Each compares one ControlGroup arm against another.

Names and ordering match the Pega Infinity Impact Analyzer product UI;
`ImpactAnalyzer.default_ia_experiments` is the single source of truth
and its dict insertion order is the canonical display order. Anything
that lists experiments (the lift chart, plot legends, the experiment
card grid, the per-experiment colour map) must follow it. Inside
`summarize_experiments` the `Experiment` column is cast to a
`pl.Enum` over those keys so a final sort produces the canonical
order regardless of by-column.

| # | Display name | Test arm | Control arm |
|---|---|---|---|
| 1 | NBA vs Random Relevant Action | `NBA` | `NBAPrioritization` |
| 2 | NBA vs Arbitrating by Propensity-only | `NBA` | `PropensityPriority` |
| 3 | NBA vs Arbitrating with No Levers | `NBA` | `LeverPriority` |
| 4 | NBA vs Only Eligibility Criteria | `NBA` | `EngagementPolicy` |
| 5 | Adaptive Model Propensity vs Random Propensity | `ModelControl_2` | `ModelControl_1` |

Defined in `ImpactAnalyzer.default_ia_experiments`.

### What each ControlGroup arm represents

From `ImpactAnalyzer.default_ia_controlgroups`:

| ControlGroup | Meaning |
|---|---|
| `NBA` | Full NBA arbitration as configured |
| `NBAPrioritization` | Random eligible action |
| `PropensityPriority` | Model propensity only (no V/C/L) |
| `LeverPriority` | No levers (p, V, C only) |
| `EngagementPolicy` | Only eligibility policies |
| `ModelControl_1` | Random propensity (= same concept as source `ModelControl`) |
| `ModelControl_2` | Model propensity only (= same concept as `PropensityPriority`) |

### Avoiding the `ModelControl_1/_2` confusion

`ModelControl_1` and `ModelControl_2` are **not** a 50/50 split of any
single source arm. Conceptually:

- `ModelControl_1` is the same arm as the source `NBAHealth_ModelControl`
  (random-propensity).
- `ModelControl_2` is the same arm as `NBAHealth_PropensityPriority`
  (model-propensity), which also serves as the control of experiment #2.

So a source dataset with 5 distinct arms is sufficient — experiment #5
re-uses two of them under aliased names. Any converter that "splits"
ModelControl 50/50 to populate `_1`/`_2` is fabricating data and will
produce zero lift on experiment #5.

---

## Source Data Shapes

### PDC (`from_pdc`)

JSON with `pyMktValue` strings encoding the ControlGroup
(`"NBAHealth_<arm>"`). Pre-aggregated counts per
day/channel/scope/outcome.

### VBD (`from_vbd`)

NDJSON of event-style rows (`pyOutcome`, `pxAggregateCount`, `pxMktValue`,
`pyOutcomeTime`, scope keys). Aggregated on load.

### Bundled sample: `data/ia/ImpactAnalyzer_InfinityDemo.zip`

Built one-off from the Infinity demo dataset
`embed-pegamkt-nbaexperiment-metrics_demo_nbahealthfile_*.json` (gzip+base64
CSV embedded in `pyConfiguration.pyDefinition.pyFileContent`).

The source CSV has 5 `ExperimentName` values matching the 5 source arms.
The converter (a one-off local script, not committed) emits VBD rows
under canonical pdstools control-group names. `NBAHealth_PropensityPriority`
and `NBAHealth_ModelControl` rows are also re-emitted under the aliases
`NBAHealth_ModelControl_2` and `NBAHealth_ModelControl_1` respectively
so all 5 pdstools experiments resolve. **No counts are fabricated**;
the alias rows carry the same impressions/accepts as their source arm.

> **Caveat — experiment #5 in the bundled sample.** `ModelControl_2`
> and `PropensityPriority` describe the same policy (model-p only,
> no V/C/L) but in real deployments they are independent
> randomization slices and observed counts can differ substantially.
> In the Infinity demo source, `PropensityPriority` shows a 8.5 %
> accept rate vs `ModelControl`'s 1.5 % on identical impression
> populations — same effect was observed at T-Mobile. By aliasing
> `PropensityPriority → ModelControl_2`, the bundled sample's
> experiment #5 ("Adaptive Model Propensity vs Random Propensity") shows a
> conceptually-correct but numerically-optimistic lift. Treat it as
> a demonstration of the experiment shape, not as a calibration
> benchmark.

### Test fixture: `data/ia/CDH_Metrics_ImpactAnalyzer.json`

Kept for `from_pdc` unit tests. Do not delete.

---

## Streamlit App

`python/pdstools/app/impact_analyzer/`

- `Home.py` / `_home_page.py`: data-source selector (File upload / Sample data),
  delegates loading to `ia_streamlit_utils.load_sample()` which calls
  `ImpactAnalyzer.from_vbd(...)` on the bundled sample.
- Pages are thin presentation layers — all calculations live in the library
  (`ImpactAnalyzer`, `Plots`, `statistics`).

---

## Tests

- Core: `python/tests/impact_analyzer/test_ImpactAnalyzer.py` — exact-value
  assertions on impressions, lift, propensities; uses small synthetic VBD
  parquets + the PDC fixture.
- Streamlit AppTests: `python/tests/streamlit_apps/impact_analyzer/` —
  exercise loader → page render path against the bundled sample.
