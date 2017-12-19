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


Best Practices
--------------

Refrain from importing ROOT utils in an init file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Seriously, don't. ROOT is a very optional requirement, and even if you have 
it installed, you get very weird crashes in unexpected places.

Install in Developer Mode
-------------------------

Setup a Virtualenv for your New Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO.

Enter the virtualenv::

    workon km3pipe


Make a Fork of KM3Pipe
~~~~~~~~~~~~~~~~~~~~~~

Go to ``http://git.km3net.de/km3py/km3pipe`` and click on "Fork".


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


