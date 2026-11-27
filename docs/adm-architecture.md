# ADM architecture

`ADMDatamart` is the shared data and analysis layer for ADM notebooks,
the ADM Health Check app, and reports. The Streamlit app is a presentation
layer; calculations and report generation belong in the library.

## Data flow

The datamart accepts model snapshots and predictor-binning snapshots as
Polars LazyFrames. Constructors such as `from_ds_export()` and `from_s3()`
handle loading before passing the data to `ADMDatamart`.

The constructor normalizes each input and exposes `model_data` and
`predictor_data`. If both are present, `combined_data` joins their latest
snapshots on `ModelID`. “Latest” here is the latest snapshot time in each
table, not the latest snapshot per model. The input `query` filters model
data; it does not directly filter predictor data.

Known AGB configuration-summary rows with empty context are removed from
model data to avoid counting totals as models. When the configuration mapping
is unambiguous, predictor importance can first be inherited from the summary
model. Rows with an unknown model technique are not filtered as AGB totals.

## Analysis and reports

Analysis is organized through namespaces on the datamart: `aggregates` for
data summaries, `plot` for figures, `analysis` for health findings, `agb` for
tree models, `bin_aggregator` for bin comparisons, and `generate` for reports.
The same library APIs are available to notebooks and the Health Check app.
Optional plotting and report dependencies are checked when their namespace
methods are used.

## Active ranges

`active_ranges()` documents the detailed output and confidence intervals.
The important distinction is that Naive Bayes uses predictor log odds to
derive a reachable score range, while AGB uses observed occupied classifier
bins: predictor log odds do not describe the tree ensemble.

AGB's segment AUC and its configuration-level confidence interval answer
different questions. The interval pools the selected segments of a shared
configuration and is conditional on the fitted model; it does not include
model-fit uncertainty. Do not count the repeated configuration interval once
per segment in portfolio summaries.

See the [`active_ranges()` docstring](../python/pdstools/adm/ADMDatamart.py)
for the fields and scale conventions, and the [ADM roadmap](adm-roadmap.md)
for open work.
