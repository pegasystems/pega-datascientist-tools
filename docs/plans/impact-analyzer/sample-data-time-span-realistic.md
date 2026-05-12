# Sample data — widen default time span

**Priority:** P3
**Touches:** `python/pdstools/app/impact_analyzer/sample_data.py`

The static sample data generator defaults to 7 snapshots × 1 day = 7
days. Real-world Pega monitoring data often spans weeks or months. The
trend charts on the Overall Summary page consequently feel cramped.

The live generator advances the synthetic clock by 1 day per tick, so
several minutes of live generation can simulate days/weeks — but that's a
workaround, not a default.

## Approach

- Bump default `n_snapshots` (e.g. 30) and / or default interval so the
  shipped sample shows a multi-week trend.
- Expose interval and span as parameters on `generate_sample_data()` and
  pipe them through to the Streamlit "Sample data" tab (or just pick a
  better default — no need for a UI knob).
