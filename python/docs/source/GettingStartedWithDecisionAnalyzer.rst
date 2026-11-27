Getting started with the Decision Analysis Tool
===============================================

The Decision Analysis Tool is a browser-based interface for exploring Pega
decision data without writing code. It supports Explainability Extract v1
(arbitration only) and Action Analysis / Explainability Extract v2 (the full
decision pipeline).

Explore the decision pipeline
-----------------------------

- **Overview** summarizes the data and decision outcomes.
- **Action Distribution** and **Action Funnel** show the offer mix at each
  stage and where actions are filtered out.
- **Optionality Analysis** and **Offer Quality Analysis** highlight decisions
  with limited choice or low-quality offers.
- **Single Decision** traces individual actions through the available stages
  and filtering components.

Understand arbitration
----------------------

- **Global Sensitivity** explores the influence of propensity, value, context,
  and levers on ranking.
- **Win / Loss Analysis** compares competing actions and their arbitration
  factors.
- **Arbitration Distribution** shows the distributions of prioritization
  components.

Try what-if scenarios
---------------------

- **Thresholding Analysis** explores the effect of minimum propensity and
  priority thresholds on offer volume and the winning mix.
- **What-if: Business Levers** changes the lever for an issue, group, or action
  and compares the winning distribution before and after the change,
  including effects on competing offers.

These simulations are an initial illustration, not a replay of the decision
engine. They re-rank sampled arbitration candidates using reconstructed
priorities and approximations for missing data. Use a full decision simulation
or a small hold-out experiment to investigate promising changes further.

Installation
------------

Install pdstools with the application dependencies using your preferred Python
package manager. Quarto and Pandoc are not needed for Decision Analysis.

.. tabs::
   .. tab:: uv tool (recommended)

      We have a strong preference for `uv <https://github.com/astral-sh/uv>`_ as it's fast, reliable, and handles Python versions automatically.

      **Step 1:** Install uv

      If you haven't yet, install uv from https://github.com/astral-sh/uv. We recommend using the standalone installer, as it has a ``uv self update`` function.

      **Step 2:** Install the pdstools applications as uv tool

      .. code-block:: bash

         uv tool install 'pdstools[app]'

      This will install the pdstools application globally on your system, making the ``pdstools`` command available from any terminal.

      .. Note:: You do not need to create a virtual environment with this method - uv handles that for you. This assures global access and avoids dependency conflicts.

      .. Note:: If you are a developer and want to contribute to the codebase, consider using the "uv + venv" method instead to install the app into a local virtual environment.


   .. tab:: uv + venv

      We have a strong preference for `uv <https://github.com/astral-sh/uv>`_ as it's fast, reliable, and handles Python versions automatically.

      **Step 1:** Install uv

      If you haven't yet, install uv from https://github.com/astral-sh/uv. We recommend using the standalone installer, as it has a ``uv self update`` function.

      **Step 2:** Create a virtual environment

      Navigate to your desired directory and run:

      .. code-block:: bash

         uv venv

      **Step 3:** Install pdstools with app dependencies

      .. code-block:: bash

         uv pip install --upgrade 'pdstools[app]'

      **Note:** If you don't have Python or no compatible version installed, uv will automatically install a compatible version for you.

   .. tab:: pip + venv

      This is the traditional Python approach using pip with virtual environments.

      **Step 1:** Create a virtual environment

      Navigate to your desired directory and run:

      .. code-block:: bash

         python -m venv .venv

      **Step 2:** Activate the virtual environment

      **On macOS/Linux:**

      .. code-block:: bash

         source .venv/bin/activate

      **On Windows:**

      .. code-block:: bash

         .venv\Scripts\activate

      **Step 3:** Upgrade pip (recommended)

      .. code-block:: bash

         python -m pip install --upgrade pip

      **Step 4:** Install pdstools with app dependencies

      .. code-block:: bash

         pip install --upgrade 'pdstools[app]'

      **Remember:** Always activate your virtual environment before working with the application:

      - ``source .venv/bin/activate`` (macOS/Linux)
      - ``.venv\Scripts\activate`` (Windows)

   .. tab:: pip (global)

      **⚠️ Warning:** Installing packages globally can lead to dependency conflicts. We strongly recommend using virtual environments (see other tabs).

      **Step 1:** Upgrade pip (recommended)

      .. code-block:: bash

         python -m pip install --upgrade pip

      **Step 2:** Install pdstools with app dependencies globally

      .. code-block:: bash

         pip install --upgrade 'pdstools[app]'

      **Consider using virtual environments:** Global installations can cause conflicts with other Python projects. Consider switching to the "uv" or "pip + venv" methods for better project isolation.

Launching the Decision Analysis Tool
------------------------------------

Once everything is installed, you can launch the Decision Analysis Tool:

.. tabs::
   .. tab:: uv tool (recommended)

      .. code-block:: bash

         pdstools decision_analyzer

   .. tab:: uv + venv

      .. code-block:: bash

         uv run pdstools decision_analyzer

   .. tab:: pip + venv

      First activate your virtual environment, then:

      .. code-block:: bash

         pdstools decision_analyzer

   .. tab:: pip (global)

      .. code-block:: bash

         pdstools decision_analyzer

The app opens in your browser. If it does not, copy the Local URL from the
terminal. Stop the app with Ctrl+C in that terminal.

Sharing on a local network
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
   A network-accessible app is not protected by authentication or HTTPS by
   default. The Decision Analysis launcher also disables Streamlit's XSRF
   protection. Anyone who can reach the app may be able to access data loaded
   into it. Use only approved, non-sensitive data on a trusted network; for
   broader access, use an appropriately secured deployment.

To let colleagues on the same network connect, start the app listening on
network interfaces:

.. code-block:: bash

   pdstools decision_analyzer --server.address 0.0.0.0 --server.port 8521

If you installed into a uv-managed virtual environment, prefix the command
with ``uv run``. Find your computer's local IP address in its network settings
and share ``http://<your-IP>:8521``. Your computer must remain on, and your
firewall and network must allow connections to that port.

To restrict access to your own computer instead, use
``--server.address 127.0.0.1``. A URL beginning with ``localhost`` or
``127.0.0.1`` cannot be opened from a colleague's computer.

CLI Options
^^^^^^^^^^^

The ``pdstools decision_analyzer`` command accepts several options to control
data loading and sampling. All options can also be set via environment variables
(useful for containerised or headless deployments).

``--data-path PATH``
   Path to a data file or directory to load on startup.
   Supports parquet, csv, json, arrow, zip, tar (including .tar.gz, .tgz), and
   partitioned folders. ZIP archives can contain supported data files directly
   or nest ZIP archives around Parquet files.
   When provided, the app loads that data automatically instead of falling
   back to the built-in sample dataset. You can still override it by
   uploading a file through the UI.
   *(env var:* ``PDSTOOLS_DATA_PATH`` *)*

``--sample VALUE``
   Pre-ingestion interaction sampling for large datasets.
   Specify an absolute count (e.g. ``100000``, ``100k``, ``1M``) or a percentage
   (e.g. ``10%``). Sampling is done at the interaction level: a random
   subset of interaction IDs is selected and **all rows for each sampled
   interaction are kept**, preserving the complete decision funnel per
   interaction.
   *(env var:* ``PDSTOOLS_SAMPLE_LIMIT`` *)*

``--filter "Column Name=value1,value2,..."``
   Pre-ingestion row filter for extracting specific data from large files.
   Supports three syntax forms:

   - **Categorical:** ``"Column=value1,value2,..."`` — exact match on any listed value
   - **Numeric:** ``"Column>=N"``, ``"Column<=N"``, ``"Column>N"``, ``"Column<N"``
   - **Date range:** ``"Column=YYYY-MM-DD..YYYY-MM-DD"`` — inclusive date range

   Use user-friendly column names (e.g. ``Channel``, ``Decision Time``,
   ``ModelPositives``). Multiple ``--filter`` flags are ANDed together.
   Can be combined with ``--sample`` (filter is applied first, then sampling
   runs on the filtered result). Filtered data is cached as parquet for fast
   reloading.
   *(env var:* ``PDSTOOLS_FILTER`` *)*

``--temp-dir DIR``
   Directory for temporary files such as the sampled-data parquet cache.
   Defaults to the current working directory.
   *(env var:* ``PDSTOOLS_TEMP_DIR`` *)*

.. warning::
   **Concurrent Access:** Running multiple instances of the Decision Analysis Tool
   simultaneously with shared temp directories can cause race conditions when
   reading/writing cached files. If you need to run multiple instances concurrently,
   use separate ``--temp-dir`` paths for each instance.

**Examples:**

.. code-block:: bash

   # Load a parquet file directly
   pdstools decision_analyzer --data-path /path/to/data.parquet

   # Load a partitioned directory
   pdstools decision_analyzer --data-path /path/to/export_folder/

   # Sample 100 000 interactions from a large dataset
   pdstools decision_analyzer --data-path /path/to/data.parquet --sample 100000

   # Sample 10% of interactions, store temp files in /tmp
   pdstools decision_analyzer --data-path /path/to/data.parquet --sample 10% --temp-dir /tmp

   # Sample using shorthand notation (1M = 1 million interactions)
   pdstools decision_analyzer --data-path /path/to/data.parquet --sample 1M

   # Filter to specific interactions
   pdstools decision_analyzer --data-path /path/to/data.zip --filter "Interaction ID=ABC-123,DEF-456"

   # Filter to a specific customer
   pdstools decision_analyzer --data-path /path/to/data.zip --filter "Subject ID=customer-42"

   # Combine multiple filters (ANDed)
   pdstools decision_analyzer --data-path /path/to/data.zip --filter "Interaction ID=ABC-123" --filter "Channel=Web"

   # Filter first, then sample the filtered result
   pdstools decision_analyzer --data-path /path/to/data.zip --filter "Subject ID=customer-42" --sample 100

   # Filter by date range
   pdstools decision_analyzer --data-path /path/to/data.zip --filter "Decision Time=2024-01-01..2024-12-31"

   # Filter by numeric threshold
   pdstools decision_analyzer --data-path /path/to/data.zip --filter "ModelPositives>=100"

   # Combine all filter types
   pdstools decision_analyzer --data-path /path/to/data.zip --filter "Channel=Web" --filter "ModelPositives>=50" --filter "Decision Time=2024-06-01..2024-12-31"

Using the Decision Analysis Tool
--------------------------------

Load data
^^^^^^^^^

The Home page loads sample data automatically when no input is specified.
Upload your own file to replace it, or use ``--data-path`` to load a file or
directory at startup. See your Pega documentation for export instructions.

The app detects v2 when both strategy and stage-group information are present;
otherwise it treats the data as v1. V1 has synthetic Arbitration and Output
stages, so it cannot show filtering at earlier pipeline stages.

Choose a focus
^^^^^^^^^^^^^^

Use the sidebar controls to select a stage or analysis scope. Many pages also
offer a shared **Channel / Direction** filter; the selection persists across
pages that use it. Overview shows global metrics. Use ``--filter`` when you
want to filter data before loading.

The Home page's sample-size setting controls the interaction sample used by
some analyses. It is separate from ``--sample``, which reduces the data before
loading.

Compare scenarios
^^^^^^^^^^^^^^^^^

On **What-if: Business Levers**, select an issue, group, or action and move
the slider to replace its levers. The percentage is relative to the selected
offers' median lever: all selected offers receive the same replacement value,
not a per-offer multiplier. Reset restores their original levers.

On **Thresholding Analysis**, set minimum propensity or recorded-priority
thresholds and compare the resulting winning mix. Both pages support Top-N
winners and Issue, Group, or Action analysis granularity. Counts at different
categories can overlap when Top-N is greater than one. Mandatory offers retain
precedence in re-ranking; changing their levers alone cannot change their order.

Upgrading pdstools
------------------

If you already had an older version of pdstools, make sure to upgrade to the latest version:

.. tabs::
   .. tab:: uv tool (recommended)

      .. code-block:: bash

         uv tool upgrade pdstools

   .. tab:: uv + venv

      .. code-block:: bash

         uv pip install --upgrade 'pdstools[app]'

   .. tab:: pip + venv

      First activate your virtual environment, then:

      .. code-block:: bash

         pip install --upgrade 'pdstools[app]'

   .. tab:: pip (global)

      .. code-block:: bash

         pip install --upgrade 'pdstools[app]'

Troubleshooting
---------------

**Application doesn't start**

- Ensure you have installed pdstools with app dependencies
- Check that you're using a compatible Python version
- If using virtual environments, make sure it's activated

**Analysis fails to run**

- Verify your decision data files are in the correct format
- Check the application logs in the terminal for specific error messages

**Large datasets crash or fail with capacity errors**

Standard polars uses 32-bit indexing and cannot handle datasets with more
than ~2 billion elements. If you are working with very large decision data
exports, install the 64-bit runtime extra:

.. code-block:: bash

   pip install 'polars[rt64]'

This drops in a 64-bit runtime alongside polars, which is selected
automatically at import time — no code changes required. Alternatively,
use the ``--sample`` CLI flag to reduce the data before ingestion, or use
``--filter`` to extract only the rows you need:

.. code-block:: bash

   pdstools decision_analyzer --sample 500000 --data-path /path/to/data
   pdstools decision_analyzer --filter "Subject ID=customer-42" --data-path /path/to/data

**For more help:**

- Review the `DecisionAnalyzer class documentation <autoapi/pdstools/decision_analyzer/DecisionAnalyzer/index.html>`_
- Check the `Decision Analysis Tool example <https://pegasystems.github.io/pega-datascientist-tools/latest/articles/decision_analyzer.html>`_
- Raise an issue on `GitHub <https://github.com/pegasystems/pega-datascientist-tools>`_
