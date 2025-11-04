Getting Started With Decision Analyzer
======================================

What is the Decision Analyzer Tool?
-----------------------------------

The Decision Analyzer is a stand-alone tool designed to analyze Explainability Extract V2 datasets from Pega. Unlike V1 which only included arbitration stage data, V2 provides data from all stages in the decision funnel, enabling comprehensive analysis of the full decision-making process. Like the Health Check application, you will need to have Python and install pdstools, but you do not need to run a (data science) environment, and there is no need to create a script - it is all configured from a user-friendly interface.

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

Upload your Explainability Extract V2 dataset files through the data import interface. The Decision Analyzer is specifically designed to work with this dataset format which includes data from all stages in the decision funnel.

.. note:: 
   The Decision Analyzer works with **Explainability Extract V2** datasets from Pega. This version includes data from all decision stages (not just arbitration like V1), enabling comprehensive analysis of the full decision funnel. For information about exporting this data from Pega, refer to your Pega documentation.

**Step 2: Explore Decision Funnel Analysis**

Analyze your decision funnel to understand:

- **Where actions get filtered**: See which actions are dropped at different stages
- **Which components filter actions**: Identify which business rules or components are most restrictive
- **Stage-by-stage breakdown**: Get a complete view from initial eligibility through final arbitration

**Step 3: Examine Action Distribution**

Use the interactive treemap visualizations to:

- **Detect rare survivors**: Identify groups of actions that rarely make it to arbitration
- **Compare issue/group performance**: See how different action categories perform
- **Analyze arbitration patterns**: Understand which actions dominate the final selection

**Step 4: Analyze Prioritization Factors**

Investigate the four key arbitration factors:

- **Propensity**: AI-driven likelihood of customer engagement
- **Value**: Business value of the action
- **Levers**: Manual adjustments to action priority
- **Context Weights**: Situational importance factors

**Step 5: Experiment with Levers** 

Test different lever configurations to:

- **Boost underperforming actions**: Increase win rates for specific action groups
- **Balance action distribution**: Ensure fair representation of different offers
- **Understand trade-offs**: See how boosting one group affects others

.. note::
   Remember that increasing lever values for one action group will decrease win counts for other groups. Careful comparison of before and after distributions is essential.

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
- Visit the main `pdstools documentation <https://pegasystems.github.io/pega-datascientist-tools/>`_
