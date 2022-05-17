
Getting Started
===============

Prerequisites
-------------

You need to have Python installed. The following are some popular ways to install Python:

- |Anaconda| (recommended)
- |Python|

.. |Anaconda| raw:: html

   <a href="https://www.anaconda.com/products/distribution" target="_blank">Anaconda distribution</a>

.. |Python| raw:: html

   <a href="https://www.python.org/downloads/" target="_blank">Python website</a>

It is also recommended to use an IDE. The following are some popular choices:

- |Visual Studio Code| (recommended)
- |Spyder|
- |PyCharm|

.. |Visual Studio Code| raw:: html

   <a href="https://code.visualstudio.com/" target="_blank">Visual Studio Code</a>

.. |Spyder| raw:: html

   <a href="https://www.spyder-ide.org/" target="_blank">Spyder</a>

.. |PyCharm| raw:: html

   <a href="https://www.jetbrains.com/pycharm/" target="_blank">PyCharm</a>

*Anaconda + VS Code* is our recommended setup. Once both are installed use the terminal (Linux or Mac) 
or Anaconda Prompt (Windows) to carry out the remaining steps below.


.. _install:

Installation
------------

To install locally follow the instructions below.

.. note::
    CUQIpy is currently not publicly available and therefore not released through PyPI or Anaconda.
    For this reason installation requires a few extra manual steps.

Download CUQIpy
~~~~~~~~~~~~~~~

If you have git installed you can clone the ``cuqipy`` repository with the following command:

.. code-block:: sh
 
   git clone https://lab.compute.dtu.dk/cuqi/cuqipy.git

Alternatively you can download a zip of the latest versions of CUQIpy from here:

- `CUQIpy (stable) <https://lab.compute.dtu.dk/cuqi/cuqipy/-/archive/master/cuqipy-master.zip>`_ (recommended)
- `CUQIpy (latest) <https://lab.compute.dtu.dk/cuqi/cuqipy/-/archive/dev/cuqipy-dev.zip>`_

Extract the zip file to a directory named ``cuqipy`` removing the ``-master`` or ``-dev`` suffix.

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

The required dependencies of cuqipy are listed in the ``requirements.txt`` file and can be
installed via conda or pip.

First ensure you are in the project directory:

.. code-block:: sh

   cd cuqipy

Then install using pip (recommended):

.. code-block:: sh

   pip install -r requirements.txt

or conda (if you have conda installed):

.. code-block:: sh

   conda install --file requirements.txt



Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

CUQIpy also optionally interfaces with a number of 3rd party libraries.
Please follow the install instructions on the website of the respective libraries.

-  `ASTRA Toolbox <https://github.com/astra-toolbox/astra-toolbox>`_:
   High-performance primitives for 2D and 3D tomography.
-  `CIL <https://github.com/TomographicImaging/CIL>`_: Tomographic imaging library including reconstruction algorithms.
-  `FEniCS <https://fenicsproject.org>`_: For PDE modeling with FEniCS.
-  `findiff <https://github.com/maroba/findiff>`_: For some PDE based demos.
-  `pytest <https://docs.pytest.org>`_: To run the automatic tests on your local
   machine.

In development (find respective branches in the source repository)

- `Matlab <https://mathworks.com/help/matlab/matlab-engine-for-python.html>`_: Using Matlab functions as part of CUQIpy.
- `MUQ <https://mituq.bitbucket.io/>`_: MIT UQ library for defining and solving UQ problems.
- `PyTorch <https://pytorch.org>`_: GPU-accelerated tensor computations with autograd support.
- `Umbridge <https://github.com/UM-Bridge/umbridge>`_: Coupling computational models and statistical methods through HTTP.

Verification
------------
To ensure that CUQIpy is installed correctly, you can run the following example (while in the ``cuqipy`` directory).

From the terminal (Linux or Mac) or Anaconda Prompt (Windows) type :

.. code-block:: sh

   python

then enter the following code

.. code-block:: python

   import cuqi

If the import succeeds cuqipy is most likely working correctly.

If the import fails, you can check the error message. 
Most likely cause of failure is that CUQIpy is not in the interpreter path.
You can add to the current path of the interpreter using (``sys.path``) as shown below.

.. code-block:: python

   import sys
   sys.path.append('/path/to/cuqipy/')

This is also useful if you are writing scripts in another directory and want to import CUQIpy as a part of those scripts.

.. tip:: 

   In VS Code, you can get tab-completion for CUQIpy by adding the following line to your ``.vscode/settings.json`` file of your current project.:

   .. code-block:: json

      "python.analysis.extraPaths": ["/path/to/cuqipy/"]


Running the Tests
-----------------

To fully make sure that cuqipy runs as expected on your machine you should run the automatic tests.
While in the project directory ``cuqipy``, run (requires pytest):

.. code-block:: sh

   python -m pytest

Building Documentation
----------------------

To generate sphinx html documentation in your local machine, make sure
you have working installation of sphinx and all the extensions listed 
in the ``docs/conf.py`` file. 

Then run the following command in ``cuqipy`` directory:

.. code-block:: sh

   sphinx-build -b html docs/. docs/_build

Then open ``docs/_build/index.html`` using your preferred web browser to
browse the documentation.
