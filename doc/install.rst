Installation
============


.. contents:: :local:


Important Note for Users of the CC-IN3P3 in Lyon
------------------------------------------------

KM3Pipe is preinstalled on the Lyon computing centre. Put this into your
`~/.bashrc` or `~/.zshenv` (or whatever login script you prefer):::

    source $KM3NET_THRONG_DIR/src/python/pyenv.sh

To test if everything is working, run the following command::

    km3pipe test

And you are ready to go! This will work on both Scientific Linux 6 and Cent OS
machines as the ``$KM3NET_THRONG_DIR``, set by your group environment, is
pointing to different directories.
The Python environment also contains the latest versions of all important and
commonly used scientific packages like scipy, numpy, scikit-learn, pandas etc.

If you are missing any packages, contact us and we will install them.

Requirements
------------

To install km3pipe, you need:

- Python >= 3.5

- pip (via ``$ easy_install pip``)

- C compiler, e.g. ``gcc``.

- HDF5 (the hdf5lib C library, e.g. `apt-get install hdf5` or `yum install hdf5`)

(Recommended) PyEnv or Virtual Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A very clean and simple way to install any version of Python (we recommend 3.6.1+) is PyEnv (https://github.com/pyenv/pyenv).
It is easily set up and gives you a fresh installation without messing around with your systems Python environment::

    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc

After that, log out and in (or close the terminal window and open a new one). To install and activate a Python version::

    pyenv install 3.6.1  # obviously installs Python 3.6.1
    pyenv global 3.6.1   # sets the global python version to 3.6.1

You can also use virtual environments (https://virtualenvwrapper.readthedocs.io) to isolate your Python projects.


Install
-------

To install the latest stable version fromm the PyPI repository:::

    $ pip install km3pipe
    
It might be advisable to use:::

    $ easy_install km3pipe

on Windows.

To get the latest version, use:::

    $ pip install git+http://git.km3net.de/km3py/km3pipe.git


Install from Source
-------------------

To install KM3Pipe from source, clone the git repository::

    $ git clone http://git.km3net.de/km3py/km3pipe.git

check out your desired branch::

    $ git checkout master  # or any other branch you are interested in

and run::

    $ make install

To install it in development-mode, which will just link the folder to your
Python site-packages, so you will be able to modify KM3Pipe and use it immediately
without the need to reinstall it::

    $ make install-dev


Run the Test Suite
------------------

To run the unit test suite, you can either run::

    $ km3pipe test

or if you have checked out the sources::

    $ make test


Updating
--------

KM3Pipe comes with a command line utility called `km3pipe`, which can
be used to update KM3Pipe itself::

    $ km3pipe update

Or you can of course use `pip`::

    $ pip install --upgrade km3pipe

To get the latest developer version::

    $ km3pipe update master

If you installed KM3Pipe from source via `make install-dev`,
you simply pull the changes from git and rebuild it::

    $ cd /path/to/km3pipe_repo
    $ git pull


Configuration
-------------

KM3Pipe can read frequently used information (like DB session cookies,
API tokens, etc.) from a configuration file, which is expected to
be `~/.km3net`.

Here is an example configuration::

    [General]
    check_for_updates=no

    [DB]
    cookie=sid_fooman_123.34.56.78_

