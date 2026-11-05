Shared synthetic sample data
============================

The shared sample generator is being introduced in layers. Its reproducibility
foundations and immutable catalogs are internal; the existing sample loaders and
mock methods have not yet switched to the new generator.

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

Randomness contract
-------------------

Each entity has a local random stream derived from the root seed and its ordered
identity components using SHA-256. Adding or reordering unrelated catalog entries
does not perturb existing streams. Python's process-randomized ``hash()`` and the
global random state are not used.

The generator will use Polars and the Python standard library. Generated sample
rows are not bundled assets, and the published legacy sample archives remain
unchanged.
