Shared synthetic sample data
============================

The shared sample generator is being introduced in layers. Its reproducibility
foundations are internal; the existing sample loaders and mock methods have not
yet switched to the new generator.

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
