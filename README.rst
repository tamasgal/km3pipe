KM3Pipe
=======

.. image:: https://travis-ci.org/tamasgal/km3pipe.svg?branch=develop
    :target: https://travis-ci.org/tamasgal/km3pipe
.. image:: https://zenodo.org/badge/24634697.svg
   :target: https://zenodo.org/badge/latestdoi/24634697

KM3Pipe is a framework for KM3NeT related stuff including MC, data files, live access to detectors and databases, parsers for different file formats and an easy to use framework for batch processing.


The framework tries to standardise the way the data is processed by providing a Pipeline-class, which can be used to put together different built-in or user made Pumps and Modules. Pumps act as data readers/parsers (from files, memory or even socket connections) and Modules take care of data processing and output. Such a Pipeline setup can then be used to iteratively process data in a file. In our case for example we store several thousands of neutrino interaction events in a bunch of files and KM3Pipe is used to put together an analysis chain which processes each event one-by-one.

Although it is mainly designed for the KM3NeT neutrino detectors, it can easily be extended to support any kind of data formats. Feel free to get in touch if you’re looking for a small, versatile framework which provides a quite straightforward module system to make code exchange between your project members as easy as possible. KM3Pipe already comes with several types of Pumps, so it should be easy to find an example to implement your owns. As of version 6.8.0 you find Pumps based on popular formats like HDF5 (https://www.hdfgroup.org), ROOT (https://root.cern.ch) but also some very specialised project internal binary data formats, which on the other hand can act as templates for your own ones. Just have a look at the io subpackage and of course the documention if you’re interested!

Read the docs at http://km3pipe.readthedocs.io (updated on each commit)

KM3NeT related documentation (internal) at http://wiki.km3net.de/index.php/KM3Pipe

KM3NeT public project homepage http://www.km3net.org


