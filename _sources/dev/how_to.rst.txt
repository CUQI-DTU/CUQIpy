How-To Guides for contributors
------------------------------

This section contains simple "how-to" guides for contributors.
For a Step-by-step guide to contributing to the project, see
:doc:`Step-by-step contributor guide <step_by_step>`.

All these guides assume that you have cloned the repository and
are working in the root directory of the repository.

Install developer dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install the dependencies for the development of the package, run:

.. code-block:: bash

   pip install -r requirements-dev.txt

Run tests
~~~~~~~~~

To run the tests, run:

.. code-block:: bash

   python -m pytest -v


Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

To generate sphinx html documentation in your local machine run

.. code-block:: sh

   sphinx-build -b html docs/. docs/_build

Then open ``docs/_build/index.html`` using your preferred web browser to
browse the documentation.
