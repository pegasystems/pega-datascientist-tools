.. _logging:

Logging and Debugging
=====================

PDS Tools uses Python's built-in ``logging`` module for debugging and diagnostics. By default, logging is disabled to avoid cluttering output. You can enable logging to troubleshoot issues or understand what the library is doing.

Enabling Logging
----------------

Basic Configuration
^^^^^^^^^^^^^^^^^^^

To see debug messages from pdstools in your Python scripts or notebooks:

.. code-block:: python

    import logging

    # Enable debug logging for all pdstools modules
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Now use pdstools as normal
    from pdstools.adm import ADMDatamart
    dm = ADMDatamart(...)

Selective Module Logging
^^^^^^^^^^^^^^^^^^^^^^^^^

To enable logging for specific pdstools modules only:

.. code-block:: python

    import logging

    # Configure root logger at INFO level
    logging.basicConfig(level=logging.INFO)

    # Enable DEBUG for specific modules
    logging.getLogger('pdstools.adm').setLevel(logging.DEBUG)
    logging.getLogger('pdstools.decision_analyzer').setLevel(logging.DEBUG)

CLI Applications
^^^^^^^^^^^^^^^^

When using the CLI applications (Decision Analysis Tool, ADM Health Check), you can enable logging by setting the ``PDSTOOLS_LOG_LEVEL`` environment variable:

.. code-block:: bash

    # Enable debug logging
    export PDSTOOLS_LOG_LEVEL=DEBUG
    pdstools decision_analyzer --data-path data.parquet

    # Or inline
    PDSTOOLS_LOG_LEVEL=DEBUG pdstools decision_analyzer --data-path data.parquet

Available log levels: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``

Log Levels
----------

PDS Tools uses the following log levels:

- **DEBUG**: Detailed diagnostic information, iteration progress, data transformations
- **INFO**: Confirmation that things are working as expected (rarely used in library code)
- **WARNING**: Something unexpected happened but the library can continue (e.g., deprecated parameters)
- **ERROR**: A serious problem occurred, but not fatal (e.g., failed to load optional data)
- **CRITICAL**: A serious error that prevents the program from continuing

Logging in Custom Code
-----------------------

If you're extending pdstools or writing custom analysis code, follow the same pattern:

.. code-block:: python

    import logging

    # At module level
    logger = logging.getLogger(__name__)

    # In your functions
    def my_analysis_function(data):
        logger.debug(f"Processing {len(data)} records")
        logger.info("Analysis complete")
        logger.warning("Found unexpected values in column X")
        logger.error("Failed to compute metric Y")
