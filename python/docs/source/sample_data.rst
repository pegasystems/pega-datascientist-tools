Shared synthetic sample data
============================

The shared sample generator is being introduced in layers. Its reproducibility
foundations, immutable catalogs and shared world are internal.
:meth:`pdstools.Prediction.from_mock_data` and
:meth:`pdstools.ADMDatamart.from_mock_data` already sample the shared world; the
other sample loaders and mock methods have not yet switched to the generator.

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

Randomness contract
-------------------

Each entity has a local random stream derived from the root seed and its ordered
identity components using SHA-256. Adding or reordering unrelated catalog entries
does not perturb existing streams. Python's process-randomized ``hash()`` and the
global random state are not used.

The generator will use Polars and the Python standard library. Generated sample
rows are not bundled assets, and the published legacy sample archives remain
unchanged.
