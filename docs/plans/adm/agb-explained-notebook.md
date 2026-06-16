# AGBExplained: new article notebook + 13 library plot methods

**Priority:** P1

**Files touched:**
- `python/pdstools/adm/trees/_plots.py` — 13 plot methods in `Trees.plot.*` namespace (9 new + 4 original)
- `python/pdstools/adm/trees/_model.py` — wires `self.plot = Plots(self)`; all `plot_*` methods removed
- `python/pdstools/adm/trees/__init__.py` — exports `Plots`
- `python/pdstools/adm/trees/_nodes.py` — add `sampleCount` to `Node`
- `python/pdstools/utils/datasets.py` — swap `sample_trees()` to new sample file
- `data/agb/ModelExportWithSampleCount.json` — new anonymized 100-tree sample (produced from one-off script)
- `examples/articles/AGBExplained.ipynb` — create new article notebook
- `examples/adm/AGBModelVisualisation.ipynb` — **delete** (all content ported to AGBExplained)
- `python/tests/adm/test_ADMTrees.py` — tests for all 13 plot methods

## Problem

`ADMTreesModel` has no library methods for analysing training dynamics,
feature importance by gain, or predictor role (depth vs coverage).  The
only AGB tutorial notebook (`AGBModelVisualisation.ipynb`) is a demo, not
a mathematical explainer, and it duplicates patterns better served by a
companion to `examples/articles/ADMExplained.ipynb`.

The current toy sample (`_974a7f9c-…txt` via `from_url`) has no
`sampleCount` data on tree nodes — the replacement 100-tree export does,
enabling three streaming-timeline plot methods that reveal how the model
was trained over time.

## Proposed approach

### ✅ Phase 0 — Anonymized sample file (DONE)

Produce `data/agb/ModelExportWithSampleCount.json` from the richer
100-tree export using a **one-off anonymization script** (not checked in).

Action/name set values, client-specific `Customer.*` attribute names, and
proprietary score predictor names were anonymized to generic tokens
(`Action_01`, `Name_01`, `Customer.Attr01`, `Customer.Scores.ExtModelScore1`,
etc.).  Standard Pega namespaces (`IH.*`, `py*`) and top-level metadata
(AUC, training stats, model version) were kept as-is.
The one-off anonymization script was **not** checked in.

Update `datasets.sample_trees()` to load from `data/agb/ModelExportWithSampleCount.json`
via `ADMTreesModel.from_file`.  The old remote `.txt` URL is replaced.
The original un-anonymized export is **never checked in**.

### ✅ Phase 0.5 — Add `sampleCount` to `Node` (DONE)

In `python/pdstools/adm/trees/_nodes.py`:
- Add `sampleCount: int | None = None` to the `Node` dataclass
- Update `_iter_nodes` to pass `tree.get("sampleCount")` when yielding each node

Backward-compatible: the old sample and any user-supplied model without
`sampleCount` keys → all `Node.sampleCount` are `None`; no existing code
is affected.

### ✅ Phase 1 — 13 plot methods in `_plots.py` namespace (DONE)

All methods live in `python/pdstools/adm/trees/_plots.py` as the `Plots(LazyNamespace)`
class and are accessed via `Trees.plot.<method>()`.  The 4 original methods
(`splits_per_variable`, `tree`, `contribution_per_tree`, `splits_per_variable_type`)
were migrated from `_model.py`; 9 new methods were added.

All 9 new methods follow the `show: bool = True` / `return_df: bool = False`
keyword-only pattern (note: keyword-only — `*` before `show` and `return_df`)
with `@overload` stubs.  Plotly is lazy-imported inside each method body;
`MissingDependenciesException(["plotly"], "AGB", deps_group="adm")` is raised
when it is absent.

Node labels in `plot.tree()` are NFD-normalized and stripped of non-ASCII
characters before being passed to pydot, preventing `UnicodeEncodeError` on
Windows (cp1252 encoding).

**Methods 1–6 — no `sampleCount` required:**

1. **`Trees.plot.gain_per_tree()`** (Fig 2a)
   - Data: `tree_stats` — `gains.list.sum()` as `total_gain`
   - Chart: bar of `total_gain` per `treeID`; secondary y-axis line for root `score`
   - `return_df` columns: `treeID`, `total_gain`, `score`

2. **`Trees.plot.cumulative_gain_share()`** (Fig 2c)
   - Data: cumsum of per-tree total gain divided by grand total
   - Chart: line (S-curve); annotate 50 % crossover tree
   - `return_df` columns: `treeID`, `cumulative_gain_share`

3. **`Trees.plot.feature_importance_by_gain(top_n=15)`** (Fig 3)
   - Data: `gains_per_split.group_by("predictor").agg(pl.sum("gains"))`, `head(top_n)`
   - Namespace: first dot-segment of predictor name
   - Chart: horizontal bar, color by namespace
   - `return_df` columns: `predictor`, `total_gain`, `namespace`

4. **`Trees.plot.early_vs_late_gain()`** (Fig 3b)
   - Data: fresh tree walk — per predictor sum gains in first-quartile trees vs last-quartile trees
   - Chart: scatter, log-log axes, x=early_gain, y=late_gain, size=total_gain, color=namespace; y=x diagonal reference line
   - `return_df` columns: `predictor`, `early_gain`, `late_gain`, `total_gain`, `namespace`

5. **`Trees.plot.gain_by_namespace()`** (Fig 6)
   - Data: `gains_per_split` with namespace from first dot-segment; `group_by` namespace, sum, normalise
   - Chart: horizontal bar sorted descending
   - `return_df` columns: `namespace`, `total_gain`, `gain_share`

6. **`Trees.plot.feature_role_map()`** (Fig 4)
   - Data: fresh tree walk via `_iter_nodes` — per predictor: `mean_depth`, `tree_coverage` (distinct trees used), `total_gain`, `namespace`
   - Chart: bubble scatter — x=mean_depth, y=tree_coverage, size=total_gain, color=namespace; hover shows predictor name
   - `return_df` columns: `predictor`, `mean_depth`, `tree_coverage`, `total_gain`, `namespace`

**Methods 7–9 — require `sampleCount`:**

Methods 7–9 raise `ValueError("sampleCount data not available …")` when
all `Node.sampleCount` values are `None` (i.e. the model was loaded from a
file without this field).

7. **`Trees.plot.training_stream_timeline()`** (Fig 2d)
   - Data: root `sampleCount` per tree (= cumulative responses seen when tree was built)
   - Chart: line, x=treeID, y=root_sample_count
   - `return_df` columns: `treeID`, `root_sample_count`

8. **`Trees.plot.inter_tree_gaps()`** (Fig 2e)
   - Data: `root_sampleCount[i] − root_sampleCount[i−1]` (responses between consecutive tree additions)
   - Chart: bar showing training cadence / activity bursts
   - `return_df` columns: `treeID`, `sample_gap`

9. **`Trees.plot.gain_decay_dual_lens()`** (Fig 2b)
   - Data: per tree — `total_gain`, `treeID`, `node_age = totalCount − root_sampleCount`
   - Chart: dual x-axis line: left axis = treeID, right axis = node_age; y = total_gain
   - `return_df` columns: `treeID`, `node_age`, `total_gain`

**Namespace extraction** (methods 3–6, 8, 9): first dot-segment of the
predictor name, e.g. `IH.Mobile.Inbound.Clicked.pxLastOutcomeTime.DaysSince`
→ `IH`.  Predictors with no dot (e.g. `pyTreatment`) → use the full name.

### ✅ Phase 2 — Update `datasets.sample_trees()` (DONE)

Change `sample_trees()` in `python/pdstools/utils/datasets.py` from `from_url(remote_url)` to
`ADMTreesModel.from_file(_DATA_DIR / "ModelExportWithSampleCount.json")`.
Keep the `warnings.catch_warnings` + `RuntimeError` wrapper pattern.

### ⏳ Phase 3 — `examples/articles/AGBExplained.ipynb` (TODO)

Style: same as `ADMExplained.ipynb` — markdown-heavy, LaTeX formulas,
code verifies the math, `great_tables.GT` tables, cells **not pre-run**.
`datasets.sample_trees()` is the single data source.

#### Required section: Model AUC

Include a dedicated **Model AUC** section (mirror the one added to `ADMExplained.ipynb`).
Key points to cover:

- AGB uses PAVA to produce a calibrated score distribution, exactly like Naïve Bayes. The
  ensemble raw score is mapped through this distribution to return a propensity.
- The validated AUC is computed from those PAVA bins and stored as the top-level `auc`
  field in the model export (accessed via `Trees.metrics["auc"]`). The
  field name changed from `performance` to `auc` in more recent export versions;
  `_model.py` reads both (`props.get("auc") or props.get("performance")`).
- AUC is validated via **test-then-train**: each incoming response is first scored by
  the current ensemble (propensity via PAVA), the outcome is compared, and that
  validated prediction contributes to the running AUC *before* the weights are updated.
  This makes it a live, unbiased performance estimate.
- Because each AGB model corresponds to one action / treatment context key combination,
  the `auc` is inherently per-action. There is no per-action breakdown within a single
  model export.
- An overall portfolio AUC = response-count-weighted average of the individual model
  AUCs. This is not stored anywhere; it must be derived from the datamart
  (`Performance` × `ResponseCount` weighted average across all models).
- Code cell: show `Trees.metrics["auc"]` and `Trees.metrics["response_positive_count"]` /
  `Trees.metrics["response_negative_count"]`; contrast with the NB case where AUC
  can be re-derived from Classifier bins (not possible here — bins not in the export).

Contrast note to include: for NB the PAVA Classifier bins are exported in the datamart,
so `auc_from_bincounts()` can re-derive the AUC from scratch. For AGB the PAVA bins are
internal — only the pre-computed scalar AUC is exported. Both are validated estimates;
the difference is just observability.

Imports block (mirrors `ADMExplained.ipynb` style):
```python
import polars as pl
from math import exp
from great_tables import GT
from pdstools import datasets

Trees = datasets.sample_trees()
```

The `pio.renderers.default = "notebook_connected"` cell (with a Jupyter tag
to hide it in docs renders) must appear before the imports cell — same
pattern as `ADMExplained.ipynb`.

---
## Terminology, formula accuracy, and alignment with official Pega position

### Sources consulted (in priority order)

1. **Pega DAI internal documentation** (`docs/features/machine-learning-ai/adaptive-models/`, `docs/features/telemetry-metrics/events/PEGA_ADM05.md`, `docs/epics/epic-94657/`, `docs/epics/epic-106851/`)
2. **Pega AGB Whitepaper** (Adaptive Gradient Boosting — A Pega Whitepaper, ~2021; accurate on algorithm; *outdated on product defaults*)
3. **`python/pdstools/adm/trees/_model.py`** — reference implementation; `score()` and `get_visited_nodes()` are authoritative for the actual scoring formula

### A. Required terminology

| Term to use | Do NOT use | Source |
|---|---|---|
| **Adaptive Gradient Boosting (AGB)** | "AGB algorithm", "gradient boost algorithm" | Whitepaper, DAI docs |
| **gradient boosting technique** (user-facing) | "gradient boosting algorithm" | DAI FAQ: "using the gradient boosting technique" |
| **Naïve Bayes** (the classic algorithm) | "classic algorithm", "classic ADM", "old algorithm" | DAI docs use "Naïve Bayes model" |
| **Prediction Studio** | "prediction studio", "ADM UI" | Product name |
| **Pega Adaptive Decision Manager (ADM)** | "ADM" without expansion on first use | Official product name |
| **propensity** | "score", "probability" (use "probability" only in math notation) | Standard Pega terminology |
| **responses** (positive / negative) | "samples", "examples", "data points" | DAI docs: "positive training samples", "negative training samples" |
| **feature importance** | "predictor importance" (both are used, but "feature importance" is more prominent in UI) | DAI FAQ heading |
| **Prediction Studio feature importance** is scaled 0–100, summing to 100 | Raw gain ≠ Prediction Studio importance | DAI FAQ: "scaled from 0 to 100" |
| **concept drift** | "behavioral drift" | Whitepaper, DAI docs |
| **ADWIN** (Adaptive Sliding Window) | "sliding window algorithm" | Whitepaper: "adaptive sliding window algorithm (ADWIN)" |
| **complexity threshold (Gamma, γ)** | "split threshold" | Whitepaper: "complexity threshold (Gamma)" |
| **learning rate (eta, η)** | "shrinkage", "step size" | Whitepaper: "learning rate eta" |
| **pruning** | "trimming", "cutting" | DAI docs: "tree_splits_pruned_count", "pruning event" |
| **tree depth** | "tree depth" ✅ (correct — used in telemetry) | PEGA_ADM05 metrics |
| **context key predictors** | "context keys" as predictor type | PEGA_ADM05 metrics |
| **symbolic predictors** | "categorical predictors" | PEGA_ADM05 metrics |
| **numeric predictors** | "continuous predictors" | PEGA_ADM05 metrics |
| **interaction history predictors (IH)** | "history features" | PEGA_ADM05 metrics |
| **predictor saturation** | "encoding overflow" | PEGA_ADM05: "number_of_saturated_symbolic_predictors" |
| **AUC** (Area Under the Curve) | "accuracy", "ROC AUC" | Whitepaper: "the accuracy of the adaptive models is expressed in AUC" |
| **test-then-train** | "train-test" | Whitepaper: "test-then-train, meaning the predictive power of the model … is a validated metric" |
| **SHAP / Shapley value** | "attributions" | DAI docs: epic-94657 for single-case explanation |

### B. IMPORTANT: AGB is now the DEFAULT algorithm

The whitepaper (2021) stated: *"The classic algorithm is still the default algorithm in Pega Prediction Studio."*
This is **outdated**. Per DAI docs (PEGA_ADM05 purpose):
> *"Gradient Boosting models are the default predictive models used for NBA recommendations as of 25…"*

The notebook must **not** position AGB as an advanced optional feature. It is now standard.
The correct framing is: "AGB is the default algorithm powering adaptive models in Pega."

### C. Score formula — authoritative reference

The scoring formula is implemented in `ADMTreesModel.score()`:
```python
total = sum(get_visited_nodes(tree_id, x)[1] for tree_id in range(len(self.model)))
return 1 / (1 + exp(-total))
```

**Key insight**: the leaf scores stored in the model JSON **already incorporate the learning rate (eta)** from training — they are `η × f_k(x)` not raw `f_k(x)`. The `learningRateEta` field in the model config is for reference only (not applied again at scoring time). This is consistent with standard XGBoost behaviour.

**LaTeX to use in notebook:**

The raw score (log-odds space):
$$s = \sum_{k=1}^{K} \text{score}_k(x)$$
where $\text{score}_k(x)$ is the leaf score of tree $k$ for input $x$.
The score already includes the learning rate applied during training.

The propensity (probability space):
$$p = \sigma(s) = \frac{1}{1 + e^{-s}}$$

**Cold-start behaviour**: A model with zero trees scores everything as $\sigma(0) = 0.5$. The initial propensity is exactly 0.5 regardless of input.

**DO NOT write**: `p = σ(η × Σ f_k(x))` separately — η is already inside each term. Either show it as the sum of stored scores (which is what the code does), or note parenthetically that training bakes η in.

### D. Gain formula — authoritative reference

The whitepaper shows the gain formula as a figure (image, not extractable text).
Based on the whitepaper description and the XGBoost heritage, the gain formula used is:

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

Where for binary log-loss:
- $G = \sum g_i$ where $g_i = p_i - y_i$ (residual / first-order gradient)
- $H = \sum h_i$ where $h_i = p_i(1 - p_i)$ (Hessian / second-order gradient)
- $\lambda$ = L2 regularisation on leaf scores
- $\gamma$ = complexity threshold (Gamma) — minimum gain required to create a split

A split is only created when $\text{Gain} > 0$ after subtracting γ. The gain values stored in the JSON node objects are large positive floats (e.g. 1597.4, 448.8); these are the pre-γ-subtracted gain values.

**In the notebook**: Show this formula in LaTeX in Section 1 when explaining node anatomy. Describe g_i as the prediction error (residual) and h_i as the Hessian. Mention γ as the guard that prevents over-splitting.

**DO NOT say**: "information gain" (that's the entropy-based ID3/C4.5 formula, not XGBoost's gradient-based gain). Use "split gain" or "gradient-based gain".

### E. Feature importance — what Prediction Studio shows vs what the notebook shows

**Prediction Studio** shows feature importance scaled 0–100 summing to 100:
> *"Feature importance is the representation of the relative weight scaled from 0 to 100 that a predictor has on the overall model."* — DAI FAQ

**The notebook plots** show raw total gain (sum of `node.gain` across all splits for a predictor). This is more useful for analysis (absolute scale shows which predictors dominate).

The notebook **must clarify** this distinction:
- Raw gain from the model JSON: used in all `Trees.plot.*` methods — absolute, not normalized
- Prediction Studio feature importance: normalized to 0–100, summing to 100 across all active predictors

Also: Prediction Studio offers **treatment-level feature importance** (per-treatment), not just global. Mention this briefly in Section 4 when discussing feature importance.

### F. ADWIN and drift detection — missing from plan

The plan currently jumps from training stream plots (Section 2) to scoring (Section 3) without explaining the mechanism behind tree growth/pruning. Add in **Section 2 or Section 5**:

- ADWIN monitors each node's prediction error. When error rises (concept drift detected), the ADWIN window shrinks.
- If the gain at a node falls below γ (complexity threshold), the branch is **pruned** — all nodes below are forgotten.
- `tree_splits_pruned_count` (PEGA_ADM05 telemetry) tracks pruning events over the model lifetime.
- A sharp rise in pruning events indicates concept drift or a data quality issue.
- This is how AGB automatically adapts — no manual "adaptiveness" parameter like the classic Naïve Bayes model.

In Section 5 (Model Health), `Trees.metrics` output like `number_of_stump_trees` (trees with no splits) and high pruning rates are health signals to discuss here.

### G. Warm-start / knowledge transfer — from whitepaper, missing from plan

Add a brief note (new sub-bullet in Section 2 or a sidebar) from the whitepaper:
> *"When a new treatment is introduced and associated to the same action, the model will start learning with an advantage because it has already discovered a profile for similar treatments in the same action. The same benefit applies when a new action is introduced within an existing group."*

This is a commercially important differentiator worth mentioning — a product owner would expect to see it.

### H. Predictor pre-processing — from whitepaper, missing from plan

Section 1 or Section 4 should briefly mention:
- **Symbolic predictors**: pre-processed via custom encoding (more granular than Naïve Bayes bins)
- **Numeric predictors**: pre-processed using **percentile streaming** — robust against extreme outliers
- Both pre-processing steps can contribute to higher predictive power vs the classic algorithm

### I. Context keys as predictors — missing from plan

The sample model includes `pyTreatment` and group-level predictors as context keys (visible in the sample data: `pyTreatment in { Action_01 }`). Section 4 namespace explanation should include:

- `py*` = Pega system context keys (e.g. `pyTreatment`, `pyGroup`)
- `IH.*` = Interaction History predictors (e.g. `IH.Mobile.Inbound.Clicked.pxLastOutcomeTime.DaysSince`)
- `Customer.*` = Customer attribute predictors

The `Trees.predictor_categorization()` method in the code maps these namespaces. The notebook's namespace explanation should match this mapping exactly.

### J. "node_age" terminology in plan — note

`node_age = totalCount - root_sampleCount` in the `gain_decay_dual_lens` plot is a computed analysis term, not an official Pega term. In the notebook narrative use descriptive language: "number of additional responses seen since this tree was added" rather than implying "node_age" is a named concept in the product.

---

## Sections (revised per above)

1. **The Building Block: A Single Decision Tree**
   - Node anatomy: `split`, `gain`, `score`, left/right children
   - Show raw dict for tree 0; explain JSON fields
   - `Trees.predictors` — name→type mapping (symbolic vs numeric predictor types)
   - Split gain formula in LaTeX: $\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$
     - Explain: $g_i = p_i - y_i$ (prediction error), $h_i = p_i(1-p_i)$ (Hessian), $\gamma$ = complexity threshold
     - A split is created only when Gain > 0; γ prevents over-splitting
   - Mention ADWIN: each node tracks its prediction error with an Adaptive Sliding Window; shrinking window → lower gain → potential pruning
   - `Trees.plot.tree(0)` — visualise tree 0

2. **The Ensemble: How Trees Build on Each Other**
   - Gradient boosting intuition: each new tree corrects the residuals of the previous trees
   - Additive score formula: $s = \sum_{k=1}^{K} \text{score}_k(x)$; propensity $p = \sigma(s)$
   - **Cold start**: model starts with zero trees → $s=0$ → $p = 0.5$; propensity converges as responses accumulate
   - Note: leaf scores stored in the model JSON already incorporate the learning rate (η); no separate η factor at scoring time
   - **Warm start**: when new treatments/actions are added with similar attributes, the model starts with an advantage — already learned predictors carry over
   - `Trees.tree_stats` — GT table (per-tree: total gain, root score, node count)
   - `Trees.plot.gain_per_tree()` — gain contribution per tree; early trees dominate
   - `Trees.plot.cumulative_gain_share()` — S-curve: what fraction of total gain is captured by tree K
   - Sub-section: *Training stream* (requires `sampleCount` data)
     - Whitepaper: AGB is an online learner — each new tree is built after processing additional responses from the live stream
     - `Trees.plot.training_stream_timeline()` — total responses seen when each tree was added
     - `Trees.plot.inter_tree_gaps()` — response volume between consecutive tree additions; reveals training cadence, bursts of activity
     - `Trees.plot.gain_decay_dual_lens()` — relates tree gain to both tree index and responses-since-that-tree-was-added (cumulative responses as the x-axis alternative)

3. **Scoring a Customer**
   - Define a concrete `x` dict from predictor values present in the sample model
   - `Trees.plot.tree(0, highlighted=x)` — visualise traversal path through tree 0
   - `Trees.get_all_visited_nodes(x)` — DataFrame of per-tree leaf scores
   - Raw sum + sigmoid: $s = \sum_k \text{score}_k(x)$, $p = \frac{1}{1+e^{-s}}$
   - Show inline: `1 / (1 + exp(-sum_of_scores))` matches `Trees.score(x)` exactly
   - `Trees.plot.contribution_per_tree(x)` — running propensity as each tree is added; shows convergence

4. **Feature Importance: Which Predictors Drive the Model?**
   - **Split gain** as importance metric: a predictor's total gain = sum of gains at all split nodes using that predictor across all trees
   - Clarify: this is *raw total gain* from the model JSON — useful for analysis. **Prediction Studio** shows this normalized to 0–100 summing to 100 (active predictors only)
   - Mention: Prediction Studio also provides **treatment-level feature importance** (per-treatment view)
   - `Trees.grouped_gains_per_split` — summary table
   - `Trees.plot.splits_per_variable(subset=[…])` — box-plot of per-split gain distribution per predictor
   - `Trees.plot.feature_importance_by_gain()` — ranked horizontal bar by total gain, colored by namespace
   - `Trees.plot.gain_by_namespace()` — gain share by predictor namespace
   - Namespace explanation (matches `Trees.predictor_categorization()`):
     - `py*` = Pega context keys (e.g. `pyTreatment`, `pyGroup`)
     - `IH.*` = Interaction History predictors
     - `Customer.*` = Customer attribute predictors
   - Predictor pre-processing: symbolic predictors use custom encoding (more granular bins); numeric predictors use **percentile streaming** (robust against outliers)
   - `Trees.plot.early_vs_late_gain()` — "early learner vs late refiner": predictors dominant in first-quartile trees vs last-quartile trees
   - `Trees.plot.feature_role_map()` — bubble chart: x=mean depth, y=tree coverage, size=total gain; "routers" (shallow, many trees) vs "refiners" (deep, narrow coverage)
   - Optional inline cell: gain distribution by split depth (not a library method)
   - Note on SHAP: for single-case propensity attribution, Prediction Studio uses **Shapley values** (SHAP); the notebook explains split-gain importance at the model level, which is complementary

5. **Model Health at a Glance**
   - **Pruning** as the key health signal: AGB automatically prunes branches when ADWIN detects rising prediction error (concept drift or data quality issues). `Trees.metrics` captures this via `number_of_stump_trees` (fully pruned trees).
   - `Trees.metrics` — GT table grouped by category
   - Key signals explained:
     - `score_decay_ratio`: ratio of mean|root score| in last 10 vs first 10 trees. `< 1` = converging (good), `>> 1` = instability
     - `top_predictor_gain_share`: fraction of total gain from the single most important predictor. High = over-reliance (risk if that predictor degrades)
     - `predictor_gain_entropy`: normalized Shannon entropy of the gain distribution across predictors. Low = concentrated gain (few predictors dominate). High = spread out.
     - `mean_gain_first_half` vs `mean_gain_last_half`: lower in the second half = model converging (normal); equal or higher = model still learning aggressively (may need more trees)
     - `number_of_stump_trees`: trees with no splits (root-only leaves). High count = heavy pruning, possible concept drift episode
   - `Trees.plot.splits_per_variable_type()` — usage counts for symbolic vs numeric predictor splits
   - Link to official monitoring: in production, these patterns are tracked via PEGA_ADM05 telemetry and visible in Grafana dashboards and Prediction Studio model reports

### ⏳ Phase 4 — Delete `AGBModelVisualisation.ipynb` (TODO)

`git rm examples/adm/AGBModelVisualisation.ipynb`

All content from the old notebook is covered in `AGBExplained.ipynb`:

| Old notebook section | Where in AGBExplained |
|---|---|
| `Trees.metrics` / `metric_descriptions()` | Section 5 |
| `Trees.model[k]` raw dict / node anatomy | Section 1 |
| `Trees.predictors` | Section 1 |
| `Trees.tree_stats` | Section 2 |
| `Trees.splits_per_tree[k]` / `gains_per_tree[k]` | Subsumed by `tree_stats` GT table (Section 2) |
| `Trees.grouped_gains_per_split` | Section 4 |
| `Trees.plot.splits_per_variable(subset=[…])` | Section 4 |
| `Trees.plot.tree(k)` | Section 1 |
| `Trees.plot.tree(k, highlighted=x)` | Section 3 |
| `Trees.get_all_visited_nodes(x)` | Section 3 |
| Sigmoid derivation + `Trees.score(x)` | Section 3 |
| `Trees.plot.contribution_per_tree(x)` | Section 3 |
| `sampleX` random helper | Replaced by concrete `x` dict (better for didactics) |
| PNG/PDF export note | Deliberately omitted — low value one-liner |

## Verification

```bash
uv run pytest python/tests/adm/test_ADMTrees.py -q
uv run ruff check python/pdstools/adm/trees/
uv run ruff format --check python/pdstools/adm/trees/
```

For each of the 9 new methods verify:
- `return_df=True` → `pl.DataFrame` with documented columns
- `show=False` → `plotly.graph_objects.Figure`
- Methods 7–9: `ValueError` when `sampleCount` absent (loaded from old-style file)

## Scope

- **In**: 9 new library plot methods (plus 4 migrated originals), 1 new article notebook, 1 anonymized sample, `Node.sampleCount`
- **Out**: Fig 5 (parent→child interaction graph; needs graph-layout library, low-confidence interpretation), Fig 4b gain-by-depth as a library method (3-line inline cell in the notebook suffices)
- Anonymization script is one-off / local only — never checked in
