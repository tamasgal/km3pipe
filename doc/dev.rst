Filing Bugs
-----------

We use the chat.km3net.de #km3pipe channel for communication. Otherwise,
please consider submitting an issue at git.km3net.de/km3py/km3pipe.

Please include your operating system type and version number, as well
as your Python, km3pipe, pandas, numpy, and scipy versions. This
information can be found by running the following code snippet:

.. code-block:: python

  import platform; print(platform.platform())
  import sys; print("Python", sys.version)
  import numpy; print("NumPy", numpy.__version__)
  import scipy; print("SciPy", scipy.__version__)
  import pandas; print("Pandas", pandas.__version__)
  import tables; print("PyTables", tabes.__version__)
  import km3pipe; print("KM3Pipe", km3pipe.__version__)


Improve KM3Pipe
---------------

Here is the recommended workflow if you want to improve KM3Pipe. This is a
standard procedure, nothing exotic! You create a fork (your full own copy of the
repository), change the code and when you are happy with the changes, you create
a merge request, so we can review, discuss and add your contribution.
Merge requests are automatically tested on our CI server, which is running
Jenkins: ``http://pi1155.physik.uni-erlangen.de:8080/job/KM3Pipe/`` and reports
any error back to the Gitlab web interface.

Make a Fork of KM3Pipe
~~~~~~~~~~~~~~~~~~~~~~

Go to ``http://git.km3net.de/km3py/km3pipe`` and click on "Fork".

After that, you will have a full copy of KM3Pipe with write access under an URL
like this: ``http://git.km3net.de/your_git_username/km3pipe``

Clone your Fork to your PC
~~~~~~~~~~~~~~~~~~~~~~~~~~

Get a local copy to work on (use the SSH address, not the HTTP one)::

    git clone git@git.km3net.de:your_git_username/km3pipe.git

Now you need to add a reference to the original repository, so you can sync your
own fork with the KM3Pipe repository::

    git remote add upstream git@git.km3net.de:km3py/km3pipe.git


Keep your Fork Up to Date
~~~~~~~~~~~~~~~~~~~~~~~~~

To get the most recent commits (including all branches), run::

    git fetch upstream

This will download all the missing commits and branches and are now accessible
using the ``upstream/...`` prefix.

If you want to update for example the ``develop`` branch, switch to it first::

    git checkout develop

and then merge the ``upstream/develop`` into it::

    git merge upstream/develop

Push your changes to Gitlab
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To push all your changes to your fork, run::

    git push


Install in Developer Mode
-------------------------

KM3Pipe can be installed in `dev-mode`, which means, it links itself to your
site-packages and you can edit the sources and test them without the need
to reinstall KM3Pipe all the time. Although you will need to restart any
``python``, ``ipython`` or ``jupyter``-notebook (only the kernel!) if you
imported km3pipe before  you made the changes.

Go to your own fork folder (as described above) and check out the branch you
want to work on::

    git checkout develop  # the main development branch (should always be stable)
    make install-dev

Make sure to run the test suite first to see if everything is working
correctly::

    $ make test

Run the test every time you make changes to see if you broke anything!
    
You can also start a script which will watch for file changes and retrigger
a test suite run every time for you. It's a nice practice to have a terminal
open running this script to check your test results continuously::

    make test-loop


We develop new features and fix bugs on separate branches and merge them
back to ``develop`` when they are stable.

You can however stay on your develop branch if you want to, although we
recommend working on a separate branch.

*While on the ``develop`` branch*, create a feature branch::

    git checkout -b feature/my_cool_new_class

Don't forget to push it to your fork regularly. Also keep in mind that the first
time you push a newly created branch, you will be prompted to set the target
branch on your fork. The command is then displayed, but for the sake of
completeness::

    git push --set-upstream origin feature/my_cool_new_class


Create a Merge Request (aka Pull Request)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you are finished with your new feature or bugfix, make sure you have pushed
the latest commits to your fork on the Git server by executing::

    git push

Let's say that the branch with the commits you want to merge into the original
KM3Pipe repository are on ``feature/my_cool_new_class``. Go to the "New Merge
Request" page: (http://git.km3net.de/tgal/km3pipe/merge_requests/new) and select
your ``feature/my_cool_new_class`` branch as the source branch. 
