# Channels — overview should always primary-axis on channel

**Priority:** P1
**Touches:** `python/pdstools/app/impact_analyzer/pages/2_Channels.py`

The Channels "overview" section (top of the page) renders charts and
tables that aren't consistently faceted by channel. Since this whole
page is about cross-channel comparison, the overview should always lead
with channel as the primary axis (rows or columns), regardless of which
experiment is selected.

## Approach

- Audit each chart and table in the overview section for whether channel
  is the primary axis.
- For any that aren't, re-render with channel on the y-axis (or as
  facet_col), even when the data has only one channel (in which case
  the chart degenerates gracefully to a single row).
- Update the section header / caption to make the channel-first framing
  explicit.
