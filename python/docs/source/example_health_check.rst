ADM Health Check example
========================

`Open the example ADM Health Check <_static/reports/health-check.html>`_.
The standalone report includes model, predictor, and Prediction Studio analyses.

This report is regenerated during each HTML documentation build using only
:func:`pdstools.demo_data`. All data is synthetic; the canonical end date keeps
the example reproducible.
JavaScript libraries are loaded from CDNs, so viewing the report requires an
internet connection.

To generate the same report locally:

.. code-block:: python

    from pdstools import demo_data

    sample = demo_data(end="canonical")
    sample.datamart.generate.health_check(
        prediction=sample.prediction,
        output_dir="reports",
        full_embed=False,
    )
