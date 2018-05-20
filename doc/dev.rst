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

Check out our KanBan board http://git.km3net.de/km3py/km3pipe/boards,
which shows all the open issues in three columns:

- *Backlog*: The issues which are yet to be discussed (e.g. not clear how to proceed)
- *ToDo*: Issues tagged with this label are ready to be tackled
- *In Progress*: These issues are currently "work in progress". They can however be
  put tossed back to *ToDo* column at any time if the development is suspended.

Here is the recommended workflow if you want to improve KM3Pipe. This is a
standard procedure for collaborative software development, nothing exotic!

Feel free to contribute ;)

Make a Fork of KM3Pipe
~~~~~~~~~~~~~~~~~~~~~~

You create a fork (your full own copy of the
repository), change the code and when you are happy with the changes, you create
a merge request, so we can review, discuss and add your contribution.
Merge requests are automatically tested on our CI server, which is running
Jenkins: http://pi1155.physik.uni-erlangen.de:8080/job/KM3Pipe and reports
any error back to the Gitlab web interface.

Go to http://git.km3net.de/km3py/km3pipe and click on "Fork".

After that, you will have a full copy of KM3Pipe with write access under an URL
like this: ``http://git.km3net.de/your_git_username/km3pipe``

Clone your Fork to your PC
~~~~~~~~~~~~~~~~~~~~~~~~~~

Get a local copy to work on (use the SSH address, not the HTTP one)::

    git clone git@git.km3net.de:your_git_username/km3pipe.git

Now you need to add a reference to the original repository, so you can sync your
own fork with the KM3Pipe repository::

    cd km3pipe
    git remote add upstream git@git.km3net.de:km3py/km3pipe.git


Keep your Fork Up to Date
~~~~~~~~~~~~~~~~~~~~~~~~~

To get the most recent commits (including all branches), run::

    git fetch upstream

This will download all the missing commits and branches which are now accessible
using the ``upstream/...`` prefix::

    $ git fetch upstream
    From git.km3net.de:km3py/km3pipe
     * [new branch]        develop                -> upstream/develop
     * [new branch]        feature/8.0-dev        -> upstream/feature/8.0-dev
     * [new branch]        gitlab_jenkins_ci_test -> upstream/gitlab_jenkins_ci_test
     * [new branch]        legacy                 -> upstream/legacy
     * [new branch]        master                 -> upstream/master


If you want to update for example your **own** ``develop`` branch, switch to it first with::

    git checkout develop

and then merge the ``upstream/develop`` into it::

    git merge upstream/develop

Make sure to regularly ``git fetch upstream`` and merge changes to your own branches.

Push your changes to Gitlab
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To push all your changes to your fork, run::

    git push


Install in Developer Mode
~~~~~~~~~~~~~~~~~~~~~~~~~

KM3Pipe can be installed in `dev-mode`, which means, it links itself to your
site-packages and you can edit the sources and test them without the need
to reinstall KM3Pipe all the time. Although you will need to restart any
``python``, ``ipython`` or ``jupyter``-notebook (only the kernel!) if you
imported km3pipe before  you made the changes.

Go to your own fork folder (as described above) and check out the branch you
want to work on::

    git checkout develop  # the main development branch (should always be stable)
    make install-dev


Running the Test Suite
~~~~~~~~~~~~~~~~~~~~~~

Make sure to run the test suite first to see if everything is working
correctly::

    $ make test

This should give you a green bar, with an output like this::

    $ make test
    py.test --junitxml=./reports/junit.xml km3pipe
    ================================== test session starts ===================================
    platform darwin -- Python 3.6.4, pytest-3.5.1, py-1.5.3, pluggy-0.6.0
    rootdir: ~/Dev/km3pipe, inifile: pytest.ini
    plugins: pylint-0.9.0, flake8-1.0.1, cov-2.5.1
    collected 309 items

    km3pipe/io/tests/test_aanet.py ....                         [  1%]
    km3pipe/io/tests/test_ch.py .                               [  1%]
    km3pipe/io/tests/test_clb.py ........                       [  4%]
    km3pipe/io/tests/test_daq.py ........                       [  6%]
    ...
    ...
    ...
    km3pipe/tests/test_style.py ........................        [ 87%]
    km3pipe/tests/test_testing.py ..                            [ 88%]
    km3pipe/tests/test_time.py ..................               [ 93%]
    km3pipe/tests/test_tools.py ...................             [100%]

    ----- generated xml file: ~/Dev/km3pipe/reports/junit.xml ------
    =================== 309 passed in 3.07 seconds ===================

Run the tests every time you make changes to see if you broke anything! It usually
takes just a few seconds and ensures that you don't break existing code. It's
also an easy way to spot syntax errors ;)
    
You can also start a script which will watch for file changes and retrigger
a test suite run every time for you. It's a nice practice to have a terminal
open running this script to check your test results continuously::

    make test-loop

Time to Code
~~~~~~~~~~~~

We develop new features and fix bugs on separate branches and merge them
back to ``develop`` when they are stable.

You can however stay on your develop branch if you want to, although we
recommend working on a separate branch.

We now assume that you thrust us and keep going on with creating a new branch.
**While on the ``develop`` branch**, create a feature branch::

    git checkout develop
    git checkout -b my_cool_new_class

Don't forget to push it to your fork regularly. Also keep in mind that the first
time you push a newly created branch, you will be prompted to set the target
branch on your fork. The command is then displayed, but for the sake of
completeness::

    git push --set-upstream origin my_cool_new_class

Once you set the upstream, you can push your latest commits any time you want with::

    git push


Create a Merge Request (aka Pull Request)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new issue on http://git.km3net.de/km3py/km3pipe/issues/new describing
what ou are up to (fixing a bug, adding a new feature etc.).
**Wait until** the issue is accepted by one of the main developers and **a separate
issue branch is created**, which will be used to test your code before it finally
gets merged into the ``develop`` branch.

Make sure the tests pass and that you have pushed the latest commits to your fork on the Git server by
executing::

    make test
    git push

If the issue branch is created, you can continue with submitting your merge request.

Let's say that the branch with the commits you want to merge into the original
KM3Pipe repository are on ``my_cool_new_class``. Go to the "New Merge
Request" page: http://git.km3net.de/tgal/km3pipe/merge_requests/new and select
your own ``my_cool_new_class`` branch as the "Source branch" and the
issue branch (which was created by one of the main developers earlier and looks
something like e.g. ``23-my-cool-now-class``, on ``km3py/km3pipe`` as the "Target branch".

Click on *"Compare branches and continue"*, change the *Title* and the *Description*, add
some *Labels* if you feel so and click on *"Submit merge request"*.

Your commits will be inspected and eventually merged to the issue branch. This will
trigger the Jenkins server, which will run the complete integration process, including 
installation tests, dependency check, building the documentation and of course running
the unit test suite (which you hopefully checked continuously ;)

After your merge request has been approved, check the issue page if the Jenkins server
is happy with your changes. If there are any changes needed, commit and push those to your own
fork, and create a new merge request to the same issue branch. Rinse and repeat...

That's it!
