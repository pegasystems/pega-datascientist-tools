Getting Started for Data Scientists
===============

Installation
------------

**Quick Start**

.. code-block:: bash

   uv pip install pdstools

----

**Instructions**

Pega Data Scientist Tools (pdstools) is a public Python library and it is `published on pypi <https://pypi.org/project/pdstools/>`_. As such, you can install it just like any other Python library; using your package manager of choice. 

Choose your preferred installation method:

.. tabs::

   .. tab:: uv (Recommended)

      We have a strong preference for `uv <https://github.com/astral-sh/uv>`_ as it's fast, reliable, and handles Python versions automatically.

      **Why uv?** uv automatically manages Python versions, creates isolated environments, and is significantly faster than traditional pip workflows.

      **Step 1:** Install uv

      If you haven't yet, install uv from https://github.com/astral-sh/uv. We recommend using the standalone installer, as it has a ``uv self update`` function.

      **Step 2:** Create a virtual environment

      Navigate to your desired directory and run:

      .. code-block:: bash

         uv venv

      **Step 3:** Install pdstools

      .. code-block:: bash

         uv pip install pdstools

      **For optional dependencies:**

      .. code-block:: bash

         uv pip install 'pdstools[api]'

      **For Jupyter notebooks:**

      .. code-block:: bash

         uv pip install ipykernel nbformat

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

      **Step 4:** Install pdstools

      .. code-block:: bash

         pip install pdstools

      **For optional dependencies:**

      .. code-block:: bash

         pip install 'pdstools[api]'

      **For Jupyter notebooks:**

      .. code-block:: bash

         pip install ipykernel nbformat

      **Remember:** Always activate your virtual environment before working with pdstools:

      - ``source .venv/bin/activate`` (macOS/Linux) 
      - ``.venv\Scripts\activate`` (Windows)

   .. tab:: pip (global)

      **⚠️ Warning:** Installing packages globally can lead to dependency conflicts. We strongly recommend using virtual environments (see other tabs).

      **Step 1:** Upgrade pip (recommended)

      .. code-block:: bash

         python -m pip install --upgrade pip

      **Step 2:** Install pdstools globally

      .. code-block:: bash

         pip install pdstools

      **For optional dependencies:**

      .. code-block:: bash

         pip install 'pdstools[api]'

      **For Jupyter notebooks:**

      .. code-block:: bash

         pip install ipykernel nbformat

      **Consider using virtual environments:** Global installations can cause conflicts with other Python projects. Consider switching to the "uv" or "pip + venv" methods for better project isolation.

Optional dependencies
---------------------

We intentionally limit the number of big and heavy core dependencies. This means that while initial installation is very fast, you may at some points run into import errors and will be required to install additional dependency groups. 

To install extra dependencies, you can put them in square brackets after a package name. For example, to install the optional dependencies required for using the API features of pdstools:

.. tabs::

   .. tab:: uv (Recommended)

      .. code-block:: bash

         uv pip install 'pdstools[api]'

   .. tab:: pip + venv

      First activate your virtual environment, then:

      .. code-block:: bash

         pip install 'pdstools[api]'

   .. tab:: pip (global)

      .. code-block:: bash

         pip install 'pdstools[api]'

For an overview of all optional dependencies and the dependency groups they will be installed for, run the following code:

.. code-block:: python

   from pdstools.utils.show_versions import dependency_great_table

   dependency_great_table()

Python compatibility
--------------------

Even though *uv* takes care of installing your python version, sometimes you have no choice of available versions. For this reason, we try to be as supportive in Python versions as we can; so our latest supported python version depends on our core dependencies, particularly `Polars <https://github.com/pola-rs/polars>`_. As of 2024, Polars supports Python version 3.9 and higher, hence so do we.

Checking the Installation
---------------

With ``pdstools[adm]`` installed, you can test that it's installed.

If you want to run code from a python notebook, install the additional packages first:

.. tabs::

   .. tab:: uv (Recommended)

      .. code-block:: bash

         uv pip install ipykernel nbformat

   .. tab:: pip + venv

      First activate your virtual environment, then:

      .. code-block:: bash

         pip install ipykernel nbformat

   .. tab:: pip (global)

      .. code-block:: bash

         pip install ipykernel nbformat

To create a 'bubble chart' on sample ADM data that plots action Success Rates vs Model Performance, similar to the one in Prediction Studio:

.. code-block:: python

   from pdstools import cdh_sample

   cdh_sample().plot.bubble_chart()

Next Steps
---------------

To run these analyses over your own data, please refer to the `ADMDatamart class documentation <https://pegasystems.github.io/pega-datascientist-tools/autoapi/pdstools/adm/ADMDatamart/index.html>`_ or for an example of how to use it, refer to the `Example ADM Analysis <https://pegasystems.github.io/pega-datascientist-tools/articles/Example_ADM_Analysis.html>`_.

To run the Stand-Alone Application, please refer to the `Command Line Interface documentation <https://pegasystems.github.io/pega-datascientist-tools/cli.html>`_ or the `ADM Health Check Article </pega-datascientist-tools/python/docs/build/html/GettingStartedWithTheStandAloneApplication.html>`_

For information on how to use the Infinity DX client, please refer to the `Infinity class documentation <https://pegasystems.github.io/pega-datascientist-tools/autoapi/pdstools/infinity/client/index.html>`_ or the `Prediction Studio API Explainer article <https://github.com/pegasystems/pega-datascientist-tools/blob/master/examples/prediction_studio/PredictionStudio.ipynb>`_.

PDSTools supports analysis of serveral other Pega Data. Please see the Examples in the documentation.