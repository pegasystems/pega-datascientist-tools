Getting Started with the Decision Analyzer Tool
======================================

What is the Decision Analyzer Tool?
-----------------------------------

The Decision Analyzer Tool is a stand-alone tool designed to analyze Explainability Extract datasets from Pega. V1 only includes arbitration stage data, V2 provides data from all stages in the decision funnel, enabling comprehensive analysis of the full decision-making process.

Like the ADM Health Check Tool, you will need to have Python and install pdstools, but you do not need to run a (data science) environment, and there is no need to create a script - it is all configured from a user-friendly interface.

The Decision Analysis Tool provides:

- **Overview**: Key metrics and insights about your offer strategy at a glance
- **Action Funnel**: Visualize how offers flow through the full decision pipeline and identify where they drop off
- **Action Distribution**: Understand the distribution of actions at the arbitration stage using interactive treemaps
- **Optionality Analysis**: Analyze the number of actions available per customer and personalization opportunities
- **Global Sensitivity Analysis**: Understand how arbitration factors (propensity, value, levers, context weights) affect decision-making
- **Win/Loss Analysis**: Examine which actions win or lose in arbitration and the factors behind these outcomes
- **Arbitration Component Distribution**: Analyze the distribution of prioritization components to identify potential issues
- **No coding required**: Everything is configured through a web-based UI
- **Interactive visualizations**: Hover over charts for detailed information and insights

Installation
------------

Before installing the Python components, you need to install these external applications:

- **Quarto**: Download from `Quarto releases <https://github.com/quarto-dev/quarto-cli/releases/tag/v1.3.450>`_ (v1.3.450 or later)
- **Pandoc**: Download from `Pandoc.org <https://pandoc.org>`_

These are standalone applications, not Python libraries, and must be installed separately on your system.

To use the Decision Analyzer Tool, you need to install several Python components. Choose your preferred Python package manager:

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

CLI Options
^^^^^^^^^^^

The ``pdstools decision_analyzer`` command accepts several options to control
data loading and sampling. All options can also be set via environment variables
(useful for containerised or headless deployments).

``--data-path PATH``
   Path to a data file or directory to load on startup.
   Supports parquet, csv, json, arrow, zip, tar (including .tar.gz, .tgz), and partitioned folders.
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

The app should open up in your system browser. On first run, you may get a promotional message from Streamlit asking for your email address - you can leave this empty if you want. If the app does not open automatically, simply copy the Local URL from your terminal and paste it into your browser.

Using the Decision Analysis Tool
--------------------------------

**Step 1: Import Your Decision Data**

Start by upload your data through the data import section in the Home page.

.. note::
   The Decision Analysis Tool works with both **Explainability Extract V1 and V2** datasets from Pega. The application automatically detects which version you have by checking for the presence of the ``pxStrategyName`` column in your data. If this column is not present, the data is treated as V1. V1 only contains data from the arbitration stage.

   **V2 datasets** include data from all decision stages (eligibility, applicability, suitability, and arbitration), enabling comprehensive analysis of the full decision funnel. **V1 datasets** only include arbitration stage data, so some analyses specific to the full decision funnel will be hidden when working with V1 data.

   For information about exporting this data from Pega, refer to your Pega documentation.

**Step 2: Focus Your Analysis**

Use per-page filters in the sidebar (Channel/Direction, Stage, Scope) to focus
on specific segments. For pre-ingestion data selection, use ``--filter`` CLI flags.

**Step 3: Analyze the results**

You can now analyze the results from various angles (optionality, funnel effects, win-loss etc.). There are separate analysis pages for each of the types of analysis.

The analysis provided in this tool are similar but not necessarily identical to the ones that will be delivered in product.

**Channel Filtering**

Many analysis pages include a **Channel / Direction** filter in the sidebar
that lets you focus on a specific channel combination:

- **Any** (default): Shows aggregated data across all channels that pass global filters
- **Specific channel**: Shows data only for that Channel/Direction combination
  (e.g., "Web/Inbound", "Email/Outbound")

The channel filter is available on these pages:

- Action Distribution
- Action Funnel
- Global Sensitivity
- Win/Loss Analysis
- Optionality Analysis
- Offer Quality Analysis
- Thresholding Analysis
- Arbitration Component Distribution

Your selection persists as you navigate between pages, allowing you to maintain
the same channel focus across different analyses.

.. note::
   The Overview page intentionally shows global metrics across all channels and
   does not have a channel filter.

.. note::
   In the Action Funnel page, the Filter Impact table intentionally remains unfiltered
   to show all filter events across channels.

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

- Ensure you have installed all dependencies (Quarto, Pandoc, pdstools with app dependencies)
- Check that you're using a compatible Python version (3.9 or higher)
- If using virtual environments, make sure it's activated

**Analysis fails to run**

- Verify your decision data files are in the correct format
- Ensure Quarto and Pandoc are properly installed and accessible from the command line
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
- Check the `Decision Analysis Tool example <https://pegasystems.github.io/pega-datascientist-tools/articles/decision_analyzer.html>`_
- Raise an issue on `GitHub <https://github.com/pegasystems/pega-datascientist-tools>`_
