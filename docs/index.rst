.. KM3Pipe documentation master file, created by
   sphinx-quickstart on Sat Oct  4 19:16:43 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

KM3Pipe
=======

KM3Pipe is a framework for KM3NeT related stuff including MC, data files, live access to detectors and databases, parsers for different file formats and an easy to use framework for batch processing.

The framework tries to standardise the way the data is processed within our collaboration by providing a `Pipeline`-class, which can be used to put together different built-in or user made `Pumps` and `Modules`. `Pumps` act as data readers/parsers (from files, memory or even socket connections) and `Modules` take care of data processing and output. Such a `Pipeline` setup can then be used to iteratively process data in a file. In our case for example we store several thousands of neutrino interaction events in a bunch of files and KM3Pipe is used to put together an analysis chain which processes each event one-by-one.

Although it is mainly designed for the KM3NeT neutrino detectors, it can easily be extended to support any kind of data formats. Feel free to get in touch if you're looking for a small, versatile framework which provides a quite straightforward module system to make code exchange between your project members as easy as possible.
KM3Pipe already comes with several types of `Pumps` (the modules which act as a data-parser/reader) so it should be easy to find an example to implement your owns. As of version 1.2.3 you find `Pumps` based on popular formats like HDF5 (https://www.hdfgroup.org), ROOT (https://root.cern.ch) but also some very specialised project internal binary data formats, which on the other hand can act templates for your own ones. Just have a look at the `io` subpackage and of course the documention if you're interested!

Contents
========

.. toctree::
	 :maxdepth: 1 

   install
   user_guide
   cmd
   auto_examples/index
   api/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

