
Getting Started
===============

Prerequisites
-------------

You need to have Python installed on your system. You can check if you have
Python installed by running the following command in a terminal or powershell:

.. code-block:: sh

   python --version

If you don't have Python installed, the following are some popular ways to install Python:

- |Anaconda| (recommended)
- |Python|

.. |Anaconda| raw:: html

   <a href="https://www.anaconda.com/products/distribution" target="_blank">Anaconda distribution</a>

.. |Python| raw:: html

   <a href="https://www.python.org/downloads/" target="_blank">Python website</a>

It is also recommended to use an IDE to run the code. The following are some popular choices:

- |Visual Studio Code| (recommended)
- |Spyder|
- |PyCharm|

.. |Visual Studio Code| raw:: html

   <a href="https://code.visualstudio.com/" target="_blank">Visual Studio Code</a>

.. |Spyder| raw:: html

   <a href="https://www.spyder-ide.org/" target="_blank">Spyder</a>

.. |PyCharm| raw:: html

   <a href="https://www.jetbrains.com/pycharm/" target="_blank">PyCharm</a>

*Anaconda distribution + Visual Studio Code* is our recommended setup. Once both are installed use the terminal (Linux or Mac) 
or Anaconda Prompt (Windows) to carry out the remaining steps below.


.. _install:

Installation
------------

Installing CUQIpy is easy. You can install it using pip:

.. code-block:: sh

   pip install cuqipy


Plugins
~~~~~~~
CUQIpy can be extended with additional functionality by installing optional plugins. These can be found at
[CUQI-DTU](https://github.com/CUQI-DTU?q=CUQIpy-).

Often the plugins use 3rd party libraries that are not compatible with each other, so it is always
recommended to install the plugins in a seperate environment. We recommended using anaconda to 
[create a new environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
and install the plugins in that environment.

Verification
------------
To ensure that CUQIpy is installed correctly, you can run the following example.

From the terminal (Linux or Mac) or Anaconda Prompt (Windows) type:

.. code-block:: sh

   python

then enter the following code

.. code-block:: python

   import cuqi

If the import succeeds CUQIpy is most likely working correctly.

Start using CUQIpy
------------------
You are now ready to start using CUQIpy!

Follow the resources linked in the :doc:`User Guide <index>` to learn more.

Running the Tests*
------------------

To fully make sure that cuqipy runs as expected on your machine you can run the automatic tests.
This requires `pytest` to be installed. You can install it using pip:

.. code-block:: sh

   pip install pytest

Then run the tests from the terminal (Linux or Mac) or Anaconda Prompt (Windows) by typing:

.. code-block:: sh

   python -m pytest -v
