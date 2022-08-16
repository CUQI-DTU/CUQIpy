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

      git clone https://github.com/CUQI-DTU/CUQIpy.git

   Then enter the repository

   .. code-block:: bash

      cd cuqipy

3. Create a new branch for your changes.

   .. code-block:: bash

      git checkout -b my-changes

   This creates a new branch named my-changes and switches to it.

4. Make your changes to the code.

   Simply edit the files as you normally would.
   
   **Note:** Remember to *commit* (step 5) often.

   You can get an overview of the changes by running:

   .. code-block:: bash

      git status

   You can see the diff of the changes by running:

   .. code-block:: bash

      git diff

5. Commit your changes.

   It is recommended commit your changes regularly. See the resource 
   `Git Guides - Git Commit <https://github.com/git-guides/git-commit>`_ for more information.

   For example if fixed a typo in the file ``samplers.py``, you can commit it by running:

   .. code-block:: bash

      git add samplers.py
      git commit -m "Fixed typo in samplers.py"

   Please provide a meaningful commit message.

   Keep repeating steps 4 and 5 until you are happy with your changes.

6. Push your changes to the repository.

   Once you are satisfied with your changes, you can push the ``my-changes`` branch to the repository.

   .. code-block:: bash

      git push origin my-changes

7. Submit a pull request.

   After pushing your changes to the repository, it is time to submit a pull request.

   .. tip::
      A pull request is a request to *merge* your code into the main code-base. After pushing your changes they still live in a separate branch and need to be reviewed before they are merged into the main code-base.

   The easiest way to submit a pull request is to use the link provided by git after you have pushed your changes. The message looks like this

   .. code-block:: verbatim

      remote: To create a pull request for 'my-changes', visit:
      remote:   https://github.com/CUQI-DTU/CUQIpy/pull/new/my-changes
     
   Alternatively you can submit a pull request by logging into the `Source repository <https://github.com/CUQI-DTU/CUQIpy>`_. If you have just pushed some changes, a button should appear in the top right corner of the page. Click it. You can also navigate to the pull requests tab and create a pull request from the interface.

8. Add reviewer to the pull request.

   After you have submitted a merge request, you should add one of the core developers as a reviewer.

For more information on the git workflow see 
`Git Guide <https://github.com/git-guides>`_.


.. toctree::
   :glob:
   :maxdepth: 2
   :hidden:

   *