Command Line Tools
==================

.. contents:: :local:

If you've installed KM3Pipe via ``pip``, you have access to some useful
command line utilities out of the box.

KM3Pipe
-------

Most of the commands have to be prefixed with ``km3pipe`` to avoid possible
nameclashes and also for an improved overview.
You can for example simply run ``km3pipe -h`` in your shell to see all available
commands:

.. command-output:: km3pipe --help
   :shell:

``update``
~~~~~~~~~~

The command ``km3pipe update [GIT_BRANCH]`` should be used to (once installed)
get latest version of KM3Pipe. If no git branch is specified, it will pull
the master branch, which always holds the stable releases.

If you want to try the newest features, pull the develop branch via
``km3pipe update develop``. This is 99.9% stable, since we always do our
experiments in ``feature/x`` branches. However, we might break it sometimes.
Have a look at our git repository to see what we're working on if you're
interested.

``runtable``
~~~~~~~~~~~~

To get a list of runs taken with one of the KM3NeT detectors, you can use
the ``runtable`` command.

The following command pulls the last 10 runs which matches the regular
expression ``PHYS``. In other words, you'll get a list of physics runs::

    km3pipe runtable -n 10 -s PHYS 14

An example output is::

    RUN	UNIXSTARTTIME	STARTTIME_DEFINED	RUNSETUPID	RUNSETUPNAME	T0_CALIBSETID	DATETIME
    848	3611	1465506000553	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-09 21:00:00.553000+00:00
    849	3612	1465506060554	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-09 21:01:00.554000+00:00
    850	3613	1465509600606	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-09 22:00:00.606000+00:00
    851	3614	1465509660607	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-09 22:01:00.607000+00:00
    852	3615	1465520400799	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-10 01:00:00.799000+00:00
    853	3616	1465520460800	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-10 01:01:00.800000+00:00
    854	3617	1465531200966	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-10 04:00:00.966000+00:00
    855	3618	1465531260967	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-10 04:01:00.967000+00:00
    856	3619	1465542000119	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-10 07:00:00.119000+00:00
    857	3620	1465542060119	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-10 07:01:00.119000+00:00

``triggersetup``
~~~~~~~~~~~~~~~~
Get the trigger setup (description and optical/acoustic DataFilter settings)
for a given runsetup ID::

    $ triggersetup -h
    Prints the trigger information of a given run setup.

    Usage:
	triggersetup RUNSETUP_OID
	triggersetup (-h | --help)
	triggersetup --version

    Options:
	RUNSETUP_OID   The run setup identifier (e.g. A02004580)
	-h --help      Show this screen.

``triggermap``
~~~~~~~~~~~~~~
Shows a histogram (similar to the one on the online monitoring pages) of
the trigger contribution for events::

    $ triggermap -h
    This script creates histogram which shows the trigger contribution for events.

    Usage:
	triggermap [-d DET_ID -p PLOT_FILENAME -u DU] FILENAME
	triggermap --version

    Option:
	FILENAME          Name of the input file.
	-u DU             Only plot for the given DU.
	-d DET_ID         Detector ID [default: 29].
	-p PLOT_FILENAME  The filename of the plot [default: trigger_map.png].
	-h --help         Show this screen.


DataBase
--------

``streamds``
~~~~~~~~~~~~
The utility ``streamds`` can be used to
interact with the database directly from the shell::

    $ streamds --help
    Access the KM3NeT StreamDS DataBase service.

    Usage:
        streamds
        streamds list
        streamds upload [-q] CSV_FILE
        streamds info STREAM
        streamds get STREAM [PARAMETERS...]
        streamds (-h | --help)
        streamds --version

    Options:
        STREAM      Name of the stream.
        CSV_FILE    Tab separated data for the runsummary tables.
        PARAMETERS  List of parameters separated by space (e.g. detid=29).
        -q          Dryrun! This will upload the parameters with a TEST_ prefix.
        -h --help   Show this screen.

PipeInspector
-------------

PipeInspector is a tool to inspect different kinds of data formats used
within the KM3NeT collaboration. It utilises the KM3Pipe framework to
deal with data I/O and allows easy access to the stored information.

.. image:: _static/PipeInspector_Screenshot.png
    :alt: PipeInspector
    :width: 700
    :align: center

It is currently in an early alpha status, but already able to handle the
DAQ binary data, ROOT and Aanet-ROOT format.

If you installed KM3Pipe via `pip`, you'll be able to launch `pipeinspector`
directly from the terminal::

    pipeinspector /path/to/data/file.ext


.. _h5cli:

HDF5 CLI Utils
--------------

``h5info``
~~~~~~~~~~

Show some H5 metadata (KM3 H5 version, km3pipe version, etc).

Example::

    $ h5info km3net_jul13_90m_muatm50T655.km3_v5r1.JTE_r2356.root.0-499.h5
    format_version: b'4.1'
    km3pipe: b'7.1.2.dev'
    pytables: b'3.4.0'


    $ h5info --help

    Show the km3pipe etc. version used to write a H5 file.

    Usage:
      h5info FILE [-r]
      h5info (-h | --help)
      h5info --version

    Options:
      FILE        Input file.
      -r --raw    Dump raw metadata.
      -h --help   Show this screen.

``h5tree``
~~~~~~~~~~

Print header info (TODO)

``h5tree``
~~~~~~~~~~

Print the structure of a H5 file + minimal metadata.

For a less pretty, more verbose output, use the ``ptdump`` util instead.

Example::

  $ h5tree elec.h5
  KM3HDF5 v4.2
  Number of Events: 169163
  ├── hits
  │  ├── _indices
  │  ├── channel_id
  │  ├── dom_id
  │  ├── event_id
  │  ├── time
  │  ├── tot
  │  └── triggered
  ├── mc_hits
  │  ├── _indices
  │  ├── a
  │  ├── event_id
  │  ├── origin
  │  ├── pmt_id
  │  └── time
  ├── reco
  │  └── gandalf
  ├── talala


``ptdump`` (from PyTables)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Inspect the contents of a HDF5 file, walking through all the subgroups.

Read the `PyTables docs <http://www.pytables.org/usersguide/utilities.html#id1>`_ for more details.

Example output::

    ┌─[moritz@averroes ~/km3net/data ]
    └─╼ ptdump nueCC.h5
    / (RootGroup) ''
    /event_info (Table(121226,), shuffle, zlib(5)) ''
    /hits (Table(0,), shuffle, zlib(5)) ''
    /mc_hits (Table(0,), shuffle, zlib(5)) ''
    /mc_tracks (Table(242452,), shuffle, zlib(5)) ''
    /reco (Group) ''
    /reco/aa_shower_fit (Table(121226,), shuffle, zlib(5)) ''
    /reco/dusj (Table(121226,), shuffle, zlib(5)) ''
    /reco/j_gandalf (Table(121226,), shuffle, zlib(5)) ''
    /reco/q_strategy (Table(121226,), shuffle, zlib(5)) ''
    /reco/reco_lns (Table(121226,), shuffle, zlib(5)) ''
    /reco/thomas_features (Table(121226,), shuffle, zlib(5)) ''


``pttree`` (from PyTables)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Show the memory consumption of a HDF5 file. As you can see below, the 
overwhelming majority of space is used by the hits, as expected.

Example output::

    ┌─[moritz@ceres ~/pkg/km3pipe/examples/data ]
    └─╼ pttree km3net_jul13_90m_muatm50T655.km3_v5r1.JTE_r2356.root.0-499.h5

    ------------------------------------------------------------

    / (RootGroup)
    +--hits (Group)
    |     ... 7 leaves, mem=35.0MiB, disk=8.1MiB [66.3%]
    +--mc_hits (Group)
    |     ... 6 leaves, mem=15.2MiB, disk=3.8MiB [31.6%]
    +--mc_tracks (Table)
    |     mem=858.4KiB, disk=251.6KiB [ 2.0%]
    `--event_info (Table)
          mem=56.6KiB, disk=6.3KiB [ 0.1%]

    ------------------------------------------------------------
    Total branch leaves:    15
    Total branch size:      51.2MiB in memory, 12.2MiB on disk
    Mean compression ratio: 0.24
    HDF5 file size:         12.5MiB
    ------------------------------------------------------------
