Installation
============


.. contents:: :local:


Important Note for Users of the CC-IN3P3 in Lyon
------------------------------------------------

KM3Pipe is preinstalled on the Lyon computing centre. Put this into your
`~/.bashrc` or `~/.zshenv` (or whatever login script you prefer):::

    source /afs/in2p3.fr/throng/km3net/src/python/pyenv.sh

And you are ready to go!
The Python environment also contains all important and commonly used scientific
packages like scipy, numpy, scikit-learn, pandas etc.

Requirements
------------

To install km3pipe, you need:

- Python >= 2.7 or >= 3.4

- pip (via ``$ easy_install pip``)

- C compiler, e.g. ``gcc``.

- HDF5 (the hdf5lib C library, e.g. `apt-get install hdf5`)

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


To install the latest stable version:::

    $ pip install km3pipe[full]

If you encounter any errors complaining about `pyx` files, install `Cython`
first. At this point you might also install `numpy`::

    $ pip install cython numpy

To get the development version, use:::

    $ pip install git+http://git.km3net.de/km3py/km3pipe.git@develop


Updating
--------

KM3Pipe comes with a command line utility called `km3pipe`, which can
be used to update KM3Pipe itself::

    $ km3pipe update

To get the latest developer version::

    $ km3pipe update develop

Or you can of course use `pip`::

    $ pip install --upgrade km3pipe


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

    [Slack]
    token=xoxp-2355837568-2397897846-8945924372-395f023485
