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


Make a Fork of KM3Pipe
~~~~~~~~~~~~~~~~~~~~~~

Go to ``http://git.km3net.de/km3py/km3pipe`` and click on "Fork".


Install in Developer Mode
-------------------------

KM3Pipe can be install in `dev-mode`, which means, it links itself to your
site-packages and you can edit the sources and test them without the need
to reinstall KM3Pipe all the time.

Clone the git repository and run::

    $ git clone git+http://git.km3net.de/YOURUSER/km3pipe.git
    $ cd km3pipe
    $ git checkout develop  # our main development branch (should be always stable)
    $ make install-dev

Make sure to run the test suite after you made changes to see if you broke
anything::

    $ make test
    
You can also start a script which will watch for file changes and retrigger
a test suite run every time for you. It's a nice practice to have a terminal
open running this script to check your test results continuously::

    $ make test-loop



We develop new features and fix bugs on separate branches and merge them
back to ``develop`` when they are stable.

*While on the ``develop`` branch*, create a feature branch::

    $ git co -b feature/my_cool_new_class


Create a Merge Request (aka Pull Request)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TODO
