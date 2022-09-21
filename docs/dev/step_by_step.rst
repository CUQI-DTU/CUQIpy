Step-by-step contributor guide
------------------------------

This is a guide for contributors who want to add a new feature to the
project. It explains how to fork the project, how to create a new branch,
how to run the tests, how to add documentation, and how to submit a pull
request.

**1. Ensure git is installed on your system.**

On Linux and Mac git comes pre-installed. On Windows, you can install git from:

- |git|

.. |git| raw:: html

   <a href="https://git-scm.com/download/win" target="_blank">Git for windows</a>

**2. Fork the project**

The first step to contributing is to fork the project on Github. This
creates your own copy of the project that you can edit. To fork the
project, you can click the "Fork" button on the project page or 
`click here <https://github.com/CUQI-DTU/CUQIpy/fork>`_.

**3. Clone the repository**

The next step is to clone the repository to your local machine. To do
this, you can run the following command:

.. code-block:: bash

   git clone https://github.com/username/CUQIpy.git

where ``username`` is your Github username.

Then enter the repository

.. code-block:: bash

   cd cuqipy

**4. Create a virtual environment (optional)**

It is recommended to create a virtual environment for the project. This
allows you to install the project dependencies without affecting other
projects on your system.

Using conda (recommended)

.. code-block:: bash

   conda create -n cuqipy-dev
   conda activate cuqipy-dev

Using virtualenv

.. code-block:: bash

   python -m venv cuqipy-dev
   source cuqipy-dev/bin/activate

**5. Install the package in editable mode**

To install the package in editable mode, run the following command:

.. code-block:: bash

   pip install -e .

This will install the package so that any changes you make to the code will be reflected automatically reflected in the installed package.

**6. Install developer dependencies**

To run tests or build the documentation, you will need to install the
developer dependencies. To do this, run the following command:

.. code-block:: bash

   pip install -r requirements-dev.txt

**7. Run the tests**

To ensure that the package is working correctly, you should run the
tests. To do this, run the following command:

.. code-block:: bash

   python -m pytest -v

After making changes to the code, you should run the tests again to
ensure that the changes have not broken anything.

**8. Create a new branch for your changes.**

Using git, you can create a new branch for your changes. This allows you
to make changes without affecting the main code base. To create a new
branch, run the following command:

.. code-block:: bash

   git checkout -b my-changes

This creates a new branch named `my-changes`` and switches to it.

**9. Make your changes to the code.**

Now you can make your changes to the code.

You can get an overview of the changes by running:

.. code-block:: bash

   git status

You can see the diff of the changes by running:

.. code-block:: bash

   git diff

**10. Commit your changes.**

It is recommended commit your changes regularly. See the resource 
`Git Guides - Git Commit <https://github.com/git-guides/git-commit>`_ for more information.

For example if you fixed a typo in the file ``cuqi/distribution/_distribution.py``, you can commit it by running:

.. code-block:: bash

   git add cuqi/distribution/_distribution.py
   git commit -m "Fixed typo in distribution"

Please provide a meaningful commit message.

Keep repeating the two previous steps until you are happy with your changes.

**11. Push your changes to the repository.**

Once you are satisfied with your changes, you can push the ``my-changes`` branch to the repository.

.. code-block:: bash

   git push origin my-changes

**12. Submit a pull request.**

After pushing your changes to the repository, it is time to submit a pull request.

.. tip::
   A pull request is a request to *merge* your code into the main code-base. After pushing your changes they still live in a separate branch and need to be reviewed before they are merged into the main code-base.

The easiest way to submit a pull request is to use the link provided by git after you have pushed your changes. The message looks like this

.. code-block:: verbatim

   remote: To create a pull request for 'my-changes', visit:
   remote:   https://github.com/CUQI-DTU/username/pull/new/my-changes

where ``username`` is your Github username.
   
Alternatively you can submit a pull request by going to the project page on Github and clicking the "Pull requests" button.

Please provide a description of your changes and a link to the issue you are addressing.

**13. Add reviewer to the pull request.**

After you have submitted a t, you should add one of the core developers as a reviewer.

**14. Wait for your pull request to be reviewed.**

Once you have submitted a pull request, it will be reviewed by one of the project maintainers. If there are any issues with your pull request, you will be notified and asked to make changes.

If your pull request is accepted, it will be merged into the main code-base.

For more information on the git workflow see 
`Git Guide <https://github.com/git-guides>`_.
