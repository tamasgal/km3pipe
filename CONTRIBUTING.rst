Contact Us
----------
Join the KM3Pipe channel here: https://chat.km3net.de/channel/km3pipe


Filing Bugs or Feature Requests
-------------------------------

Please **always** create an issue when you encounter any bugs, problems or
need a new feature. Emails and private messages are not meant to communicate
such things!

Use the appropriate template and file a new issue here:
https://git.km3net.de/km3py/km3pipe/issues

You can browse all the issues here: https://git.km3net.de/km3py/km3pipe/issues

Please follow the instructions in the templates to provide all the 
necessary information which will help other people to understand the
situation.

Improve KM3Pipe
---------------

Check out our KanBan board http://git.km3net.de/km3py/km3pipe/boards,
which shows all the open issues in three columns:

- *discussion*: The issues which are yet to be discussed (e.g. not clear how to proceed)
- *todo*: Issues tagged with this label are ready to be tackled
- *doing*: These issues are currently "work in progress". They can however be
  put tossed back to *todo* column at any time if the development is suspended.

Here is the recommended workflow if you want to improve KM3Pipe. This is a
standard procedure for collaborative software development, nothing exotic!

Feel free to contribute ;)

Make a Fork of KM3Pipe
~~~~~~~~~~~~~~~~~~~~~~

You create a fork (your full own copy of the
repository), change the code and when you are happy with the changes, you create
a merge request, so we can review, discuss and add your contribution.
Merge requests are automatically tested on our GitLab CI server and you
don't have to do anything special.

Go to http://git.km3net.de/km3py/km3pipe and click on "Fork".

After that, you will have a full copy of KM3Pipe with write access under an URL
like this: ``http://git.km3net.de/your_git_username/km3pipe``

Clone your Fork to your PC
~~~~~~~~~~~~~~~~~~~~~~~~~~

Get a local copy to work on (use the SSH address `git@git...`, not the HTTP one)::

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
     * [new branch]        gitlab_jenkins_ci_test -> upstream/gitlab_jenkins_ci_test
     * [new branch]        legacy                 -> upstream/legacy
     * [new branch]        master                 -> upstream/master


If you want to update for example your **own** ``master`` branch
to contain all the changes on the official ``master`` branch of KM3Pipe,
switch to it first with::

    git checkout master

and then merge the ``upstream/master`` into it::

    git merge upstream/master

Make sure to regularly ``git fetch upstream`` and merge changes to your own branches.

Push your changes to Gitlab regularly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure to keep your fork up to date on the GitLab server by pushing
all your commits regularly using::

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

    git checkout master  # the main development branch (should always be stable)
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
    =================== 467 passed in 6.21 seconds ===================

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
back to ``master`` when they are stable. Merge requests (see below) are also
pointing towards this branch.

If you are working on your own fork, you can stay on your own ``master`` branch
and create merge requests from that.

Create a Merge Request (aka Pull Request)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Go to https://git.km3net.de/km3py/km3pipe/merge_requests/new and select
your source branch, which contains the changes you want to be included in KM3Pipe
and select the `develop` branch as target branch.

That's it, the merge will be accepted if everything is OK ;)

If you want to join the KM3Pipe dev-team, let us know!:)
