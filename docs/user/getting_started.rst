
Getting Started
===============

.. _install:

Installation
------------

Download CUQIpy
~~~~~~~~~~~~~~~ 

To install ``cuqipy`` on your local machine, clone the ``cuqipy``
repository:

.. code-block:: sh
 
   git clone https://lab.compute.dtu.dk/cuqi/cuqipy.git

Then go to the project directory:

.. code-block:: sh

   cd cuqipy

You can run some demos, for example:

.. code-block:: sh

   cd demos 
   python demo00_MinimalExample.py

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

Requirements of cuqipy are listed in ``cuqipy/requirements.txt`` and can be
installed via conda by (while in ``cuqipy`` directory)

.. code-block:: sh

   conda install --file requirements.txt

or using pip by

.. code-block:: sh

   pip install -r requirements.txt

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

-  `pytest <https://docs.pytest.org>`__: To run the tests on your local
   machine
-  `ASTRA Toolbox <https://github.com/astra-toolbox/astra-toolbox>`__:
   For modeling CT problems
-  `CIL <https://github.com/TomographicImaging/CIL>`__: For modeling CT
   problems
-  `FEniCS <https://fenicsproject.org>`__: For modeling with FEniCS
-  `findiff <https://github.com/maroba/findiff>`__: For some if the PDE
   based demos

Running the Tests
-----------------

To make sure that cuqipy runs as expected on your machine and that all
requirements are met, you can run the tests. While in the project
directory ``cuqipy``, run:

.. code-block:: sh

   python -m pytest

Building Documentation
----------------------

To generate sphinx html documentation in your local machine, make sure
you have working installation of sphinx and sphinx-rtd-theme. Then run
the following commands in cuqipy directory:

.. code-block:: sh

   cd docs sphinx-build -b html . _build

Then open docs/_build/index.html using your preferred web browser to
browse cuqipy documentation.


.. todo::
   CUQIpy basics
