# Decision Analyzer architecture

The Decision Analysis Tool is a Streamlit interface over `DecisionAnalyzer`.
Analysis, scoring, and plotting live in the library so they can also be used
from notebooks and scripts. The app handles loading, filters, widget state,
and presentation.

## Data flow

The CLI passes launch options to the Streamlit subprocess through environment
variables. Home loads the input and stores the analyzer in session state;
analysis pages share it and pass their filter expressions to library methods.
See the [app navigation](../python/pdstools/app/decision_analyzer/_navigation.py),
[shared app helpers](../python/pdstools/app/decision_analyzer/da_streamlit_utils.py),
and [DecisionAnalyzer](../python/pdstools/decision_analyzer/DecisionAnalyzer.py).

Input columns are normalized to the display names in the
[column schema](../python/pdstools/decision_analyzer/column_schema.py). The
analyzer detects Explainability Extract (v1) versus Action Analysis (v2) from
the available columns. V1 has synthetic Arbitration and Output stages; v2
provides the pipeline stages. Analyses requiring earlier stages therefore
depend on v2 data.

## Data, sampling, and caching

`decision_data` is the cleaned full input; aggregate views generally use this
full data, while scoring and re-ranking use `sample`. The sample selects
whole interactions by a deterministic hash, targeting 10,000 interactions
by default. The actual count can differ from the target and a sample may still
contain many rows per interaction. CLI ingestion sampling is separate and can
reduce the input before the analyzer's sample is taken.

Full-data pre-aggregations and the sample are materialized and cached; they can
be expensive on large inputs. Changing stage granularity with `set_level()`
invalidates cached properties and parameterized analysis caches. The app also
caches selected computations and persists widget values separately from widget
keys so selections can survive page navigation.

## Ranking and simulations

Ranking gives mandatory actions precedence, then orders by priority, stage
order, and the Issue / Group / Action hierarchy. By default, finite recorded
priorities of at least 4,999,999 mark mandatory actions; callers can override
that detection.

Lever and threshold what-if analyses share the library's sample-backed
scoring and compare the same population of rankable arbitration decisions.
They reconstruct priority from exported factors, rather than replaying the
decision engine. Mandatory precedence is retained; missing factors and value
weighting require approximations. Thresholding keeps decisions with no
surviving offers in the comparison denominator. Top-N coverage can overlap
across categories.

The lever scenario replaces selected lever values, leaving competitors in the
population; thresholding removes offers below global thresholds before
re-ranking. See [`Scoring`](../python/pdstools/decision_analyzer/_scoring.py)
for the assumptions and API details. Shared colors come from the full
dataset, although the simulation distribution uses its own baseline ordering
for Action colors.

## Aggregate summaries and row cohorts

Aggregate methods return compact chart or table data. When a result needs to
be handed off downstream, expose or reuse a row-producing method on
`da.aggregates` that retains `Interaction ID`, using full data for exact
cohorts. `da.get_interaction_ids("aggregates.method_name", ...)` projects the
unique IDs; the method's arguments and filters define the cohort.

`Interaction ID` is the handoff boundary. The downstream application owns
identity and profile enrichment. For an auditable handoff, retain the cohort
method, arguments, filters, count, and generation timestamp alongside the
IDs. See the [aggregate methods](../python/pdstools/decision_analyzer/_aggregates.py)
and the `get_interaction_ids()` docstring for available cohorts.

Open work is tracked in the [Decision Analysis roadmap](decision-analyzer-roadmap.md).
