KM3Pipe
=======

.. image:: https://travis-ci.org/tamasgal/km3pipe.svg?branch=develop
    :target: https://travis-ci.org/tamasgal/km3pipe

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
    :target: http://km3pipe.readthedocs.io/en/latest/

.. image:: https://zenodo.org/badge/24634697.svg
   :target: https://zenodo.org/badge/latestdoi/24634697


KM3Pipe is a framework for KM3NeT related stuff including MC, data files, live access to detectors and databases, parsers for different file formats and an easy to use framework for batch processing.

The main Git repository, where issues and merge requests are managed can be found at http://git.km3net.de

The framework tries to standardise the way the data is processed by providing a Pipeline-class, which can be used to put together different built-in or user made Pumps, Sinks and Modules. Pumps act as data readers/parsers (from files, memory or even socket connections), Sinks are responsible for writing data to disk and Modules take care of data processing, output and user interaction. Such a Pipeline setup can then be used to iteratively process data in a file or from a stream. In our case for example, we store several thousands of neutrino interaction events in a bunch of files and KM3Pipe is used to stitch together an analysis chain which processes each event one-by-one by passing them through a pipeline of modules.

Although it is mainly designed for the KM3NeT neutrino detectors, it can easily be extended to support any kind of data formats. The core functionality is written in a general way and is applicable to all kinds of data processing workflows.

Feel free to get in touch if you’re looking for a small, versatile framework which provides a quite straightforward module system to make code exchange between your project members as easily as possible. KM3Pipe already comes with several types of Pumps, so it should be easy to find an example to implement your owns. As of version 8.0.0 you find Pumps and Sinks based on popular formats like HDF5 (https://www.hdfgroup.org), ROOT (https://root.cern.ch) but also some very specialised project internal binary data formats, which on the other hand can act as templates for your own ones. Just have a look at the io subpackage and of course the documention if you’re interested!

Read the docs at http://km3pipe.readthedocs.io (updated on each commit)

KM3NeT related documentation (internal) at http://wiki.km3net.de/index.php/KM3Pipe

KM3NeT public project homepage http://www.km3net.org
