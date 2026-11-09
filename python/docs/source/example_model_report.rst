Naive Bayes Model Report example
================================

`Open the example Naive Bayes Model Report <_static/reports/naive-bayes-model-report.html>`_.
The standalone report examines the synthetic Fixed Rate Loan model for Web,
including its predictor bins and contributions.

This report is regenerated during each HTML documentation build using only
:func:`pdstools.demo_data`. All data is synthetic; the canonical end date and
model ID keep the example reproducible.
JavaScript libraries are loaded from CDNs, so viewing the report requires an
internet connection.

To generate the same report locally:

.. code-block:: python

    from pdstools import demo_data

    sample = demo_data(end="canonical")
    sample.datamart.generate.model_reports(
        "05085857-c52c-5daa-b51b-0af576a038af",
        output_dir="reports",
        full_embed=False,
    )
