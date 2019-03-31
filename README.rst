KM3Pipe
=======

.. image:: https://git.km3net.de/km3py/km3pipe/badges/master/build.svg
    :target: https://git.km3net.de/km3py/km3pipe/pipelines

.. image:: https://git.km3net.de/km3py/km3pipe/badges/master/coverage.svg
    :target: https://km3py.pages.km3net.de/km3pipe/coverage

.. image:: https://api.codacy.com/project/badge/Grade/9df4849cb9f840289bf883de0dc8e28f
   :alt: Codacy Badge
   :target: https://app.codacy.com/app/tamasgal/km3pipe?utm_source=github.com&utm_medium=referral&utm_content=tamasgal/km3pipe&utm_campaign=Badge_Grade_Settings

.. image:: https://examples.pages.km3net.de/km3badges/docs-latest-brightgreen.svg
    :target: https://km3py.pages.km3net.de/km3pipe

.. image:: https://zenodo.org/badge/24634697.svg
   :target: https://doi.org/10.5281/zenodo.808829


KM3Pipe is a framework for KM3NeT related stuff including MC, data files, live
access to detectors and databases, parsers for different file formats and an
easy to use framework for batch processing.

The main Git repository, where issues and merge requests are managed can be
found at https://git.km3net.de/km3py/km3pipe.git

The framework tries to standardise the way the data is processed by providing
a Pipeline-class, which can be used to put together different built-in or user
made Pumps, Sinks and Modules. Pumps act as data readers/parsers (from files,
memory or even socket connections), Sinks are responsible for writing data to
disk and Modules take care of data processing, output and user interaction.
Such a Pipeline setup can then be used to iteratively process data in a file or
from a stream. In our case for example, we store several thousands of neutrino
interaction events in a bunch of files and KM3Pipe is used to stitch together
an analysis chain which processes each event one-by-one by passing them through
a pipeline of modules.

Although it is mainly designed for the KM3NeT neutrino detectors, it can easily
be extended to support any kind of data formats. The core functionality is
written in a general way and is applicable to all kinds of data processing
workflows.

To start off, run::

    pip install km3pipe

If you have Docker (https://www.docker.com) installed, you can start using
KM3Pipe immediately by typing::

    docker run -it docker.km3net.de/km3pipe

Feel free to get in touch if you’re looking for a small, versatile framework
which provides a quite straightforward module system to make code exchange
between your project members as easily as possible. KM3Pipe already comes with
several types of Pumps, so it should be easy to find an example to implement
your owns. As of version 8.0.0 you find Pumps and Sinks based on popular
formats like HDF5 (https://www.hdfgroup.org), ROOT (https://root.cern.ch) but
also some very specialised project internal binary data formats, which on the
other hand can act as templates for your own ones. Just have a look at the io
subpackage and of course the documentation if you’re interested!

Read the docs at https://km3py.pages.km3net.de/km3pipe or
(https://km3pipe.readthedocs.org), both updated on each push.

KM3NeT public project homepage http://www.km3net.org

