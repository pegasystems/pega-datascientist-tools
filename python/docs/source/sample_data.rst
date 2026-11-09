Shared synthetic sample data
============================

:func:`pdstools.demo_data` returns the datamart, prediction and Interaction
History views of one seeded synthetic world, generated on demand without
customer data or stored files:

.. code-block:: python

    from pdstools import demo_data

    demo = demo_data()
    demo.datamart.plot.bubble_chart()
    demo.prediction.plot.performance_trend()
    demo.ih.plot.success_rate()

Views are built on first access and cached per process for the same
arguments, so treat them as read-only. :meth:`pdstools.ADMDatamart.from_mock_data`,
:meth:`pdstools.Prediction.from_mock_data`, :meth:`pdstools.IH.from_mock_data`
and :meth:`pdstools.ImpactAnalyzer.from_mock_data` return a single view of the
same world.

``demo.write_ds_export(path)`` writes the model, predictor and prediction views
as Pega dataset export zips, for example to try the command line or the
Streamlit upload flows with real files. The reproducibility foundations,
catalogs and world object are internal.

To include the matching prediction view in a report generated from the bundle,
pass it explicitly:

.. code-block:: python

    demo.datamart.generate.health_check(prediction=demo.prediction)

Rendering the Health Check template without input paths uses all three demo
views. Reports generated from supplied model data do not substitute synthetic
predictions when prediction data is absent.

Maintained specification
------------------------

The package-owned specification contains the fictional action hierarchy,
treatment baselines, channel volumes, model configurations, predictor policies,
distributions and timed stories. Frozen records and tuples make the specification
immutable and hashable. Construction validates references, probability and volume
constraints, exclusive Web Loans routing and its deny-by-default predictor policy.

Catalog model contexts distinguish primary, challenger and derived Omni views.
AGB configuration-summary rows will be generated separately and are not extra
action-level contexts. The internal specification derives a fresh catalog manifest
and Markdown rationale tables from these records; neither is a second maintained
copy of the catalog.

These are synthetic metadata tables, not generated model snapshots or private
source-document audit records. Predictor distributions describe marginal shapes;
they are not instructions to sample correlated fields independently.

Calendar contract
-----------------

All views will use a canonical window ending on Sunday, July 2, 2000. The default
window contains 182 dates, starting on Monday, January 3. Windows contain exactly
the requested number of dates, with a minimum of 56; a 60-day window consequently
has a partial first week.

An explicit end date is normalized to the Sunday on or before that date. Without
an explicit end date, the most recent completed Sunday in UTC is used. Every
timestamp shift is a whole number of weeks, preserving weekdays and weekly
stories. The canonical calendar is available for reproducible checks.

Shared world
------------

One immutable world holds the expected, noise-free behaviour of every channel
on every window day: response volume, model-chosen success rate, lift over
random selection, model AUC and delayed-feedback state. Every view samples
noise around the same world, so Prediction Studio, ADM and Interaction History
tell consistent stories: Mobile learning and its AGB challenger, the Email feed
dip with negative lift, Web and Email weekend dips, and positive-only weeks
followed by a burst of negatives. Story days rescale to shorter windows by whole
weeks, keeping weekdays intact.

The prediction view emits the real snapshot shape: a ``Daily`` AUC row per
prediction at noon with full totals, and ``Test``, ``Control`` and ``NBA`` rows at
midnight. The model-control experiment uses 1 percent of traffic. Same-day arms
share part of their outcome noise, as they see the same customers and context, so
the small arms still show the lift story. Call Center has no ``NBA`` row and its
tiny arms are deliberately invalid.

ADM views
---------

The model snapshot view has one row per model per day for the 864 action-level
models, plus AGB configuration-summary rows that the loader drops after using
their importance. Actions launch and retire during the window; Mobile models
mature from about 1.2 to 2 percent success rate. A few models are seeded as
stuck (no active predictors), weak, or overfitting.

The predictor binning view describes the last snapshot. Its bin totals
reconcile exactly with the model snapshots, and each stored Performance is the
AUC of the emitted bins. Naive Bayes predictors have about five bins and a
median univariate AUC of about 0.56; numeric bins use one cut grid per
predictor, so aggregation can align bins across models. Interaction History and
Param predictors rank highest, and three shared predictors lead both Mobile
configurations. AGB models carry feature importance only; their context keys
are inactive at model level but ranked in the configuration summary.

Standard configurations also include ``Customer.ChurnRiskModelScore``, a
synthetic, precomputed customer churn-risk score. By default, a naming-pattern
match categorizes names ending in ``ModelScore`` as ``External Model`` in both
Naive Bayes and AGB plots; ordinary fields such as ``Customer.CreditScore`` keep
their ``Customer`` category. Custom categorization can override the default.
The score is not included in the stricter Web Loans configuration; no external
model is downloaded or run.

Interaction History and Impact Analyzer views
---------------------------------------------

The Interaction History view samples decisions from the same world: Web and
Mobile show one or two offers, Email one, and the Call Center a small share of
outbound contacts. Each offered action is an exposure with one impression
(Impression inbound, Pending outbound) followed by its responses: Clicked on
Web and Mobile, Opened then Accepted on Email, or Accepted on Call Center.
Conversion events are also generated for the conversion experiment. Accepts
carry the action's synthetic value.
Channel success rates track the world, and Mobile mixes Naive Bayes and AGB
decisions after its challenger starts.

Every customer belongs to one Impact Analyzer control group per month. The
five stock experiments each hold about 1 percent of customers, sharing the NBA
group as their test arm, and each arm changes the arbitration: random
relevance, propensity only, no business levers, no engagement policies, or a
random instead of a model propensity in the model-control arm. The Impact Analyzer view
aggregates these exposures per day, control group, channel and action, so it
reconciles exactly with the Interaction History. Arms on small channels or
short windows can be empty or without accepts; they are never filled in.

Randomness contract
-------------------

Each entity has a local random stream derived from the root seed and its ordered
identity components using SHA-256. Adding or reordering unrelated catalog entries
does not perturb existing streams. Python's process-randomized ``hash()`` and the
global random state are not used.

The generator will use Polars and the Python standard library. Generated sample
rows are not bundled assets, and the published legacy sample archives remain
unchanged.
