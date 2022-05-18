Contributor's Guide
===================

Interested in contributing to the project? Here are some tips and links to get you started:

.. todo::
   Add more tips for contributors

Step-by-step instructions
-------------------------

1. Ensure git is installed on your system.

   On Linux and Mac git comes pre-installed. On Windows, you can install git from:

   - |git|

.. |git| raw:: html

   <a href="https://git-scm.com/download/win" target="_blank">Git for windows</a>


2. Clone the repository

   .. code-block:: bash

      git clone https://lab.compute.dtu.dk/cuqi/cuqipy.git

3. Create a new branch for your changes.

   .. code-block:: bash

      git checkout -b my-changes

   This creates a new branch named my-changes and switches to it.

4. Make your changes to the code.

   Simply edit the files as you normally would. Although see step 5 for how often to *commit*.

   You can get an overview of the changes by running:

   .. code-block:: bash

      git status

   You can see the diff of the changes by running:

   .. code-block:: bash

      git diff

5. Commit your changes.

   It is recommended commit your changes regularly. See the resource 
   `GitHub wiki <https://help.github.com/articles/using-git-commits-and-pull-requests/>`_ for more information.

   For example if fixed a typo in the file ``samplers.py``, you can commit it by running:

   .. code-block:: bash

      git add samplers.py
      git commit -m "Fixed typo in samplers.py"

   Please provide a meaningful commit message.

6. Push your changes to the repository.

   Once you are satisfied with your changes, you can push the ``my-changes`` branch to the repository.

   .. code-block:: bash

      git push origin my-changes

7. Submit a merge request.

   After pushing your changes to the repository, it is time to submit a merge request.

   A merge request is a request to *merge* your code into the main code-base.

   You can submit a merge request by logging into the repository:

   https://lab.compute.dtu.dk/cuqi/cuqipy

   A button will appear in the top right corner of the page. Click it.

   After pushing your changes git will also give you the option to create a merge request directly by clicking the link in the terminal.


.. toctree::
   :glob:
   :maxdepth: 2
   :hidden:

   *