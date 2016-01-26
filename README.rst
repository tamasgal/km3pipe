KM3Pipe
=======

KM3Pipe is a framework for KM3NeT related stuff including MC, data files, live access to detectors and databases, parsers for different file formats and an easy to use framework for batch processing.

Read the docs at http://km3pipe.readthedocs.org

KM3NeT related documentation at http://wiki.km3net.de/index.php/KM3Pipe

Quick Install
=============
To install the latest stable version:

    pip install km3pipe
    
If you're not using a virtual environment (https://virtualenvwrapper.readthedocs.org), you can install it in your own home directory, however I recommend using virtual environments for any Python related stuff.

    pip install --user km3pipe

To install the latest developer version:

    git clone git@github.com:tamasgal/km3pipe.git
    cd km3pipe
    pip install -e .

.. image:: https://travis-ci.org/tamasgal/km3pipe.svg?branch=develop
    :target: https://travis-ci.org/tamasgal/km3pipe

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
    :target: http://km3pipe.readthedocs.org/en/latest/
