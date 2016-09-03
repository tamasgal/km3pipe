Quick Start
===========


Installation
------------

KM3Pipe is written in Python, so all you need is a working Python installation
with version 2.7.x or 3.4.x and above.

To install KM3Pipe, I recommend using ``pip`` (get it via ``easy_install pip``
if you don't have it already)::

    pip install numpy cython
    pip install km3pipe

This will pull the latest release from the PyPi repository.
To install the most recent development version, simply type::

    pip install numpy cython
    pip install git+http://git.km3net.de/tgal/km3pipe.git@develop


Configuration
-------------

KM3Pipe can read frequently used information (like DB credentials, API tokens,
etc.) from a configuration file, which is expected to be `~/.km3net`.

Here is an example configuration::

    [General]
    check_for_updates=no

    [DB]
    username=fooman
    password=god

    [Slack]
    token=xoxp-2355837568-2397897846-8945924372-395f023485


Updating
--------

KM3Pipe comes with a command line utility called `km3pipe`, which can
be used to update KM3Pipe itself::

    km3pipe update

To get the latest developer version::

    km3pipe update develop

Or you can of course use `pip`::

    pip install --upgrade km3pipe


Additional Software Recommendations
-----------------------------------

I highly recommend using ``Jupyter`` for prototyping and
playing around::

    pip install jupyter
