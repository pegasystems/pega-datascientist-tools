Getting Started with the Decision Analyzer Tool
======================================

What is the Decision Analyzer Tool?
-----------------------------------

The Decision Analyzer Tool is a stand-alone tool designed to analyze Explainability Extract datasets from Pega. V1 only includes arbitration stage data, V2 provides data from all stages in the decision funnel, enabling comprehensive analysis of the full decision-making process.

Like the ADM Health Check Tool, you will need to have Python and install pdstools, but you do not need to run a (data science) environment, and there is no need to create a script - it is all configured from a user-friendly interface.

The Decision Analyzer provides:

- **Decision Funnel Analysis**: Visualize which actions are filtered out at different stages and by which components
- **Action Distribution Analysis**: Understand the distribution of actions at the arbitration stage using interactive treemaps
- **Global Sensitivity Analysis**: Analyze how arbitration factors (propensity, value, levers, context weights) affect decision-making
- **Win/Loss Analysis**: Examine which actions win or lose in arbitration and understand the factors behind these outcomes
- **Arbitration Component Distribution**: Analyze the distribution of prioritization components to identify potential issues
- **Lever Experimentation**: Test different lever values to understand their impact on action win rates
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

   .. tab:: uv (Recommended)

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

Launching the Decision Analyzer
-------------------------------

Once everything is installed, you can launch the Decision Analyzer application:

.. tabs::

   .. tab:: uv (Recommended)

      .. code-block:: bash

         uv run pdstools run decision_analyzer

   .. tab:: pip + venv

      First activate your virtual environment, then:

      .. code-block:: bash

         pdstools run decision_analyzer

   .. tab:: pip (global)

      .. code-block:: bash

         pdstools run decision_analyzer

The app should open up in your system browser. On first run, you may get a promotional message from Streamlit asking for your email address - you can leave this empty if you want. If the app does not open automatically, simply copy the Local URL from your terminal and paste it into your browser.

Using the Decision Analyzer
---------------------------

**Step 1: Import Your Decision Data**

Start by upload your data through the data import section in the Home page.

.. note::
   The Decision Analyzer works with both **Explainability Extract V1 and V2** datasets from Pega. The application automatically detects which version you have by checking for the presence of the ``pxStrategyName`` column in your data. If this column is not present, the data is treated as V1. V1 only contains data from the arbitration stage.

   **V2 datasets** include data from all decision stages (eligibility, applicability, suitability, and arbitration), enabling comprehensive analysis of the full decision funnel. **V1 datasets** only include arbitration stage data, so some analyses specific to the full decision funnel will be hidden when working with V1 data.

   For information about exporting this data from Pega, refer to your Pega documentation.

**Step 2: Apply Data Filters**

Select only certain Issues, Channels or other dimensions to focus your analysis on. You can choose any data field to filter on in the **Global Filters** page.

**Step 3: Analyze the results**

You can now analyze the results from various angles (optionality, funnel effects, win-loss etc.). There are separate analysis pages for each of the types of analysis.

The analysis provided in this tool are similar but not necessarily identical to the ones that will be delivered in product.

Upgrading pdstools
------------------

If you already had an older version of pdstools, make sure to upgrade to the latest version:

.. tabs::

   .. tab:: uv (Recommended)

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

**For more help:**

- Review the `DecisionAnalyzer class documentation <https://pegasystems.github.io/pega-datascientist-tools/autoapi/pdstools/decision_analyzer/decision_data/index.html>`_
- Check the `Decision Analyzer example <https://pegasystems.github.io/pega-datascientist-tools/articles/decision_analyzer.html>`_
- Raise an issue on `GitHub <https://github.com/pegasystems/pega-datascientist-tools>`_
