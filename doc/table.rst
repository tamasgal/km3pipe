Storing Data
============

.. contents:: :local:


In science, people want to store results, either intermediate ones to have
a "checkpoint" in a potentially huge analysis chain or of course final results which are only
used for high level analysis, data visualisation and interpretation.
HDF5 is a dataformat which is open source, popular among (data) scientists
and flexible enough to store all kinds of data.

The ``Pipeline``, ``Table`` and ``HDF5Pump``/``HDF5Sink`` classes are very
good friends. In this document I'll demonstrate how to build a pipeline to
analyse a file, store intermediate results using the ``Table`` and ``HDF5Sink``
classes and then do some basic high level data analysis using the ``Pandas``
(https://pandas.pydata.org) framework.


Work in progress...

