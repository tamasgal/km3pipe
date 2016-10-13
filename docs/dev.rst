Install in Developer Mode
-------------------------

Setup a Virtualenv for your New Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO.

Enter the virtualenv::

    workon km3pipe


Make a Fork of KM3Pipe
~~~~~~~~~~~~~~~~~~~~~~

Go to ``http://git.km3net.de/tgal/km3pipe`` and click on "Fork".


Install in Developer Mode
~~~~~~~~~~~~~~~~~~~~~~~~~

Fetch the develop branch from git repository (replace YOURUSER with you
git user name)::

    git clone git+http://git.km3net.de/YOURUSER/km3pipe.git@develop
    cd km3pipe

Install in editable mode::
    
    pip install -e ./ 

*While on the ``develop`` branch*, create a feature branch::

    git co -b feature/my_cool_new_class


When Editing ``.pyx`` files, recompile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make Sure all Tests pass
~~~~~~~~~~~~~~~~~~~~~~~~

Install & run the ``pytest`` suite::

    pip install pytest

    cd /path/to/km3pipe
    py.test


Create a Merge Request (aka Pull Request)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



