Contribute
==========

You want to hack new features into km3pipe or are just here to fix a
bug? Great! Let's get you started.

If you have any questions, join us in the `#km3pipe`_ Slack channel.

.. _#km3pipe: https://km3net.slack.com/messages/km3pipe

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


How to Write Documentation
--------------------------

Build docs locally::
  
  cd km3pipe/docs
  make html

How to Add Examples to the Gallery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
