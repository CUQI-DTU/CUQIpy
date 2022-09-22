
Getting Started
===============

Prerequisites
-------------

You need to have Python installed on your system. We recommend python via the
anaconda distribution:

- |Anaconda|

.. |Anaconda| raw:: html

   <a href="https://www.anaconda.com/products/distribution" target="_blank">Anaconda distribution</a>

Anaconda provides many useful libraries and makes it easy to run CUQIpy using the pre-installed
jupyter notebook app. In addition, CUQIpy plugins often require 3rd party libraries that can most
easily be installed via anaconda.

.. _install:

Installation
------------

Installing CUQIpy is easy. Open your terminal (Linux and Mac) or Anaconda Prompt (Windows) and install it using pip:

.. code-block:: sh

   pip install cuqipy

Verification
------------
To ensure that CUQIpy is installed correctly, in the terminal (Linux or Mac) or Anaconda Prompt (Windows) type:

.. code-block:: sh

   python

then enter the following code

.. code-block:: python

   import cuqi

If no error messages are displayed, you are ready to go!

Start using CUQIpy
------------------
You are now ready to start using CUQIpy!

A good place to start is the |cuqipy-demos| repository, which contains a number of jupyter notebooks.

Download and extract the zip file and then launch the jupyter notebook app (that came pre-installed with anaconda)
either from the start menu (Windows) or from the terminal (Linux or Mac) by typing:

.. |cuqipy-demos| raw:: html

   <a href="https://github.com/CUQI-DTU/CUQIpy-demos/releases" target="_blank">CUQIpy demos</a>

.. code-block:: sh

   jupyter notebook

and navigate to the folder where you extracted the zip file using the file browser inside the notebook app.

More information can be found in the :doc:`User Guide <index>`.

Plugins
-------
CUQIpy can be extended with additional functionality by installing optional plugins. These can be found at
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

To fully make sure that cuqipy runs as expected on your machine you can run the automatic tests.
This requires `pytest` to be installed. You can install it using pip:

.. code-block:: sh

   pip install pytest

Then run the tests from the terminal (Linux or Mac) or Anaconda Prompt (Windows) by typing:

.. code-block:: sh

   python -m pytest -v
