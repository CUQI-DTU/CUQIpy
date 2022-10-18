
Getting Started
===============

Prerequisites
-------------

CUQIpy is python package. To install it, you need to have Python installed.
We recommend installing python via the anaconda distribution:

- |Anaconda|.

.. |Anaconda| raw:: html

   <a href="https://www.anaconda.com/products/distribution" target="_blank">Anaconda distribution</a>

Anaconda comes with many useful python libraries pre-installed and makes it easy to run CUQIpy code
via the jupyter notebook app. In addition, CUQIpy plugins often require 3rd party libraries that can most
easily be installed through anaconda.

.. _install:

Installation
------------

Installing CUQIpy is easy. Open your terminal (Linux and Mac) or Anaconda Prompt (Windows) and install it using pip:

.. code-block:: sh

   pip install cuqipy

This will install the latest version of CUQIpy and all its dependencies.

Verification
------------
To ensure that CUQIpy is installed correctly, launch the python interpreter by typing:

.. code-block:: sh

   python

then import CUQIpy into your python session by typing:

.. code-block:: python

   import cuqi

If no error messages are displayed, CUQIpy is likely installed correctly. The python interpreter
can be exited by typing ``exit()``.

Start using CUQIpy
------------------
You are now ready to start using CUQIpy! A good place to start is the |cuqipy-demos| repository
containing a number of jupyter notebooks, which is a great way to interactively learn how to use CUQIpy.

To get started with the demos, first download and extract the zip file.
Then launch the jupyter notebook app (that came pre-installed with anaconda)
either from the Windows start menu or via the terminal by typing:

.. |cuqipy-demos| raw:: html

   <a href="https://github.com/CUQI-DTU/CUQIpy-demos/releases" target="_blank">CUQIpy demos</a>

.. code-block:: sh

   jupyter notebook

Navigate to the folder where you extracted the zip file using the file browser inside the notebook app, and 
open one of the notebooks. If you are new to jupyter notebooks, see the |jupyter-tutorial| for a quick introduction.

.. |jupyter-tutorial| raw:: html

   <a href="https://www.dataquest.io/blog/jupyter-notebook-tutorial/" target="_blank">Jupyter notebook tutorial</a>

More information on CUQIpy can be found in the :doc:`User Guide <index>`.

Plugins (optional)
------------------
CUQIpy can be extended with additional functionality by installing optional plugins. These can be found here:
|plugins|.

.. |plugins| raw:: html

   <a href="https://github.com/CUQI-DTU?q=CUQIpy-" target="_blank">CUQIpy plugins</a>

Often the plugins use 3rd party libraries that are not compatible with each other, so it is always
recommended to install the plugins in a separate environment. We recommended using anaconda to 
|conda-env| and install the plugins in that environment.

.. |conda-env| raw:: html

   <a href="https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html" target="_blank">create a new environment</a>

Running the Tests (optional)
----------------------------

To fully make sure that CUQIpy runs as expected on your machine you can run the automatic tests.
This requires `pytest` to be installed. You can install it using pip:

.. code-block:: sh

   pip install pytest

Then run the tests from the terminal (Linux or Mac) or Anaconda Prompt (Windows) by typing:

.. code-block:: sh

   python -m pytest -v
