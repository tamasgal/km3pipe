KM3Pipe
=======

KM3Pipe is a framework for KM3NeT related stuff including MC, data files, live access to detectors and databases, parsers for different file formats and an easy to use framework for batch processing.

Read the docs at http://km3pipe.readthedocs.io

KM3NeT related documentation (internal) at http://wiki.km3net.de/index.php/KM3Pipe

KM3NeT public project homepage http://www.km3net.org

Quick Install
=============
To install the latest stable version:

.. code-block:: bash

    pip install km3pipe
    
If you're not using a virtual environment (https://virtualenvwrapper.readthedocs.io), you can install it in your own home directory, however I recommend using virtual environments for any Python related stuff.

.. code-block:: bash

    pip install --user km3pipe

To install the latest developer version:

.. code-block:: bash

    git clone git@github.com:tamasgal/km3pipe.git
    cd km3pipe
    pip install -e .
