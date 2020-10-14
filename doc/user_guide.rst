User Guide
==========

km3pipe is built upon `thepipe <https://github.com/tamasgal/thepipe>`_,
a Python package which once was the core of
km3pipe and has been split out as a separate package. It's a generic pipeline framework
which can be used in all kinds of modular workflow designs.

The km3pipe package is a collection of functions, classes and modules which are
more or less specific to KM3NeT. It provides native Python access to all kinds
of data formats like ROOT, EVT, binary DAQ, network protocols and more.

The preferred on-disk output file format is :doc:`hdf5`.

.. toctree:: 

    pipeline
    datastructures
    features
    hdf5
    cmd
    db
    local_db
    faq
