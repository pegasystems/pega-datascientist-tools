Getting Started with the ADM Health Check Tool
===============

What is the ADM Health Check Tool?
-------------------------------------------------

The ADM Health Check Tool makes it easy to run the ADM Health Check and create individual model reports without coding. You will need to have Python and install pdstools, but you do not need to run a (data science) development environment, and there is no need to create a script - it is all configured from a user-friendly interface.

The application provides:

- **ADM Health Check**: A generic overview of ADM models in your system, including charts like the "Bubble Chart" and many more, with recommendations
- **Individual Model Reports**: Detailed views of individual Bayesian ADM models, including binning details of all predictors
- **No coding required**: Everything is configured through a web-based UI
- **Excel Export**: An Excel file with the model and predictor details for further analysis
- **Automatic report generation**: Downloads ready-to-use HTML and Excel reports

Installation
------------

Before installing the Python components, you need to install these external applications:

- **Quarto**: Download from `Quarto releases <https://github.com/quarto-dev/quarto-cli/releases/tag/v1.3.450>`_ (v1.3.450 or later)
- **Pandoc**: Download from `Pandoc.org <https://pandoc.org>`_

These are standalone applications, not Python libraries, and must be installed separately on your system.


Installation
------------

To use the stand-alone health check application, you need to install several Python components. Choose your preferred Python package manager:

.. tabs::
   .. group-tab:: uv tool (recommended)

      We have a strong preference for `uv <https://github.com/astral-sh/uv>`_ as it's fast, reliable, and handles Python versions automatically.

      **Step 1:** Install uv

      If you haven't yet, install uv from https://github.com/astral-sh/uv. We recommend using the standalone installer, as it has a ``uv self update`` function.

      **Step 2:** Install the pdstools applications as uv tool

      The simplest method of running the Health Check application is by installing it to your system as a tool:

      .. code-block:: bash

         uv tool install 'pdstools[app]'

      This will install the pdstools application globally on your system, making the ``pdstools`` command available from any terminal.

      .. Note:: You do not need to create a virtual environment with this method - uv handles that for you. This assures global access and avoids dependency conflicts.

      .. Note:: If you are a developer and want to contribute to the codebase, consider using the "uv + venv" method instead to install the app into a local virtual environment.


   .. group-tab:: uv + venv

      We have a strong preference for `uv <https://github.com/astral-sh/uv>`_ as it's fast, reliable, and handles Python versions automatically.

      **Step 1:** Install uv

      If you haven't yet, install uv from https://github.com/astral-sh/uv. We recommend using the standalone installer, as it has a ``uv self update`` function.

      **Step 2:** Create a virtual environment

      Navigate to your desired directory and run:

      .. code-block:: bash

         uv venv

      **Step 3:** Install pdstools with app dependencies

      .. code-block:: bash

         uv pip install 'pdstools[app]'

      .. note:: If you don't have Python or no compatible version installed, uv will automatically install a compatible version for you.

   .. group-tab:: pip + venv

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

   .. group-tab:: pip (global)

      **⚠️ Warning:** Installing packages globally can lead to dependency conflicts. We strongly recommend using virtual environments (see other tabs).

      **Step 1:** Upgrade pip (recommended)

      .. code-block:: bash

         python -m pip install --upgrade pip

      **Step 2:** Install pdstools with app dependencies globally

      .. code-block:: bash

         pip install --upgrade 'pdstools[app]'

      **Consider using virtual environments:** Global installations can cause conflicts with other Python projects. Consider switching to the "uv" or "pip + venv" methods for better project isolation.

Launching the Application
-------------------------

Once everything is installed, you can launch the Health Check application:

.. tabs::

   .. group-tab:: uv tool (recommended)

      From any terminal, simply run:

      .. code-block:: bash

         pdstools

      This will prompt you to open either the ADM Health Check or the Decision Analyzer application. Choose the Health Check option.

   .. group-tab:: uv + venv

      .. code-block:: bash

         uv run pdstools

   .. group-tab:: pip + venv

      First activate your virtual environment, then:

      .. code-block:: bash

         pdstools

   .. group-tab:: pip (global)

      .. code-block:: bash

         pdstools

The app should open up in your system browser. On first run, you may get a promotional message from Streamlit asking for your email address - you can leave this empty if you want. If the app does not open automatically, simply copy the Local URL from your terminal and paste it into your browser.

Using the Application
---------------------

**Step 1: Navigate to Health Check**

In the app, navigate to the Health Check tab (in the left pane). This shows instructions.

**Step 2: Import Your Data**

Click the "Data Import" tab in the main screen to load your data. You have several options:

.. note::
   If you haven't downloaded the ADM Datamart yet, see `How to export the ADM Datamart <https://docs.pega.com/bundle/platform/page/platform/decision-management/enabling-monitoring-database-export.html>`_ for instructions.

- **Direct file path**: Provide the folder path where the ADM files are located (e.g., ``/User/Downloads/``). The tool will automatically find the relevant files in that directory.
- **Direct file upload**: Browse and upload your local files through the web interface.
- **CDH Sample**: For testing, you can skip uploading your own data and select "CDH Sample" from the Data Import dropdown.

.. note::
   There is no need to extract ZIP files - the application will handle that automatically.

**Step 3: Configure Report (Optional)**

The "Report Configuration" section has advanced options but can generally be left with default settings.

**Step 4: Generate and Download Reports**

Click "Generate" to create the ADM Health Check report. The download button will appear when generation is finished. The downloaded report will appear in your browser's default download location as an HTML file that you can open in any web browser.

Upgrading pdstools
------------------

If you already had an older version of pdstools, make sure to upgrade to the latest version:

.. tabs::
   .. group-tab:: uv tool (recommended)

      .. code-block:: bash

         uv tool upgrade pdstools

   .. group-tab:: uv + venv

      .. code-block:: bash

         uv pip install --upgrade 'pdstools[app]'

   .. group-tab:: pip + venv

      First activate your virtual environment, then:

      .. code-block:: bash

         pip install --upgrade 'pdstools[app]'

   .. group-tab:: pip (global)

      .. code-block:: bash

         pip install --upgrade 'pdstools[app]'

Troubleshooting
---------------

**Application doesn't start**

- Ensure you have installed all dependencies (Quarto, Pandoc, pdstools with app dependencies)
- Check that you're using a compatible Python version (3.9 or higher)
- If using virtual environments, make sure it's activated

**Reports fail to generate**

- Verify your ADM datamart files are in the correct format
- Ensure Quarto and Pandoc are properly installed and accessible from the command line
- Check the application logs in the terminal for specific error messages

**For more help:**
- Check the `example ADM analysis <https://pegasystems.github.io/pega-datascientist-tools/articles/Example_ADM_Analysis.html>`_
- Raise an issue on `GitHub <https://github.com/pegasystems/pega-datascientist-tools>`_.
