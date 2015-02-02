.. _pipeinspector:

PipeInspector
=============

PipeInspector is a tool to inspect different kinds of data formats used
within the KM3NeT collaboration. It utilises the KM3Pipe framework to
deal with data I/O and allows easy access to the stored information.

.. image:: http://tamasgal.com/km3net/PipeInspector_Screenshot.png
    :alt: PipeInspector
    :width: 700
    :align: center

It is currently in an early alpha status, but already able to handle the
DAQ binary data, ROOT and Aanet-ROOT format.

If you installed KM3Pipe via `pip`, you'll be able to launch `pipeinspector`
directly from the terminal::

    pipeinspector /path/to/data/file.ext

