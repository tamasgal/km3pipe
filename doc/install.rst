Installation
============


.. contents:: :local:

Installing km3pipe is as easy as running::

    pip install km3pipe

To install also all the optional extras (Pandas, SciPy, ...)::

    pip install "km3pipe[extras]"

As of version 9, only Python 3.5+ is supported. version 8 is the last one which
supports Python 2.7.

To get the latest version from the ``master`` branch, use::

    pip install git+http://git.km3net.de/km3py/km3pipe.git

Once km3pipe is install, you can use the command line utility ``km3pipe`` to
install or update your copy::

    km3pipe update                  # install the latest master
    km3pipe update some-git-branch  # install the from "some-git-branch"
    km3pipe update v9.0.0-beta.2    # install a specific version

Using Virtual Environments
--------------------------

It is highly recommended to create an isolated Python environment for each of
your analyses and projects, so that you have full control over the installed
packages.

To create a virtualenv (Python 3.5+ required) run the command::

    python -m venv venv

which will create one in the ``venv`` folder. To activate it::

    . venv/bin/activate

After that you can ``pip install`` whatever you want.

To leave the environment, activate another one or just type::

    deactivate


Important Note for Users of the CC-IN3P3 in Lyon
------------------------------------------------

km3pipe is preinstalled on the Lyon computing centre. Put this into your
`~/.bashrc` or `~/.zshenv` (or whatever login script you prefer):::

    module load python/3.7.5

Alternatively, if you need the old Python 2 version, you can load it with::

    module load python/2.7.17

The Python environments also contain the latest versions of all important and
commonly used scientific packages like scipy, numpy, scikit-learn, pandas,
numba, astropy etc.

If you are missing any packages, contact us and we will install them.


Install from Source
-------------------

If you prefer to play around with the source code or contribute to the
development of km3pipe, you can also install it in an editable mode.
First, clone the repository with::

    git clone http://git.km3net.de/km3py/km3pipe.git

and run::

    make install-dev

This will install all the required development packages and create
a link to your Python site-packages folder, so that you can edit the
source and try it out immediately without having to reinstall it
every time.


Running the test suite
----------------------

To test if everything is working, make sure to install all the additional
requirements with::

    pip install "km3pipe[dev]"
    pip install "km3pipe[extras]"

and run the following command::

    km3pipe test

If you are a developer, you will like to explore the ``make test``
and ``make test-loop`` commands after installing km3pipe from source (see above).


Non-Python requirements
-----------------------

km3pipe uses HDF5 as a primary high-level format.
On Debian based Linux distributions (Ubuntu, ...), install the HDF5 libraries with::

    (sudo) apt-get install hdf5

On RedHat-based distributions (CentOS, Scientific Linux, ...)::

    (sudo) yum install hdf5

On ArchLinux, Manjaro and alike::

    (sudo) pacman -S hdf5

