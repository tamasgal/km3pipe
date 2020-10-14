Storing Data (HDF5)
===================

.. contents:: :local:

HDF5 files are the prefered input and primary output type in KM3Pipe.
It is a generic format for hierarchical storage of large amount of numerical
data. Unlike e.g. ROOT (the official KM3NeT data format),
HDF5 is a general-purpose dataformat and not
specifically designed for HEP experiments. The usual HDF5 structures are
either groups of 1D arrays or 2D tables in contrast to ROOT which is
optimised for nested jagged structures, specifically designed for HEP
experiments where the events are stored in irregular tree-like structures.
While ROOT files require a complete C++ framework, even for I/O
(nowadays there are luckily alternatives, see
`uproot <https://uproot.readthedocs.io/en/latest/>`_ which is also
used in `km3io <https://km3py.pages.km3net.de/km3io/>`_ to access
KM3NeT files without ROOT),
HDF5 only requires a tiny library (hdf5lib) and is accessible from almost
all popular programming languages (Python, Julia, C, C++, Java, Go, R,
Matlab, Rust...).

In KM3NeT, HDF5 is often used to store intermediate results of an analysis
or summary data, which is used to perform high-level analysis. HDF5 is also
our main open data format.

In the following, the usual (recommended) structure is described to store
low level data. Note that there are no strict requirements how you structure
your own data, but it is recommended to split compound structures into
single arrays for easy access.

The ``tohdf5`` command which was present in KM3Pipe until v8 is now removed
due to the growing complexity of our ROOT files. To read those files,
the `km3io <https://km3py.pages.km3net.de/km3io/>`_ package is recommended
as mentioned above. An alternative to the ``tohdf5`` converter is provided
by the ``h5extract`` command line utility which has a lot of options to extract
and combine different branches of the ROOT files to a single HDF5 file.


The following sections describe how to read and write HDF5 data with KM3Pipe
using the ``kp.io.HDF5Pump`` and ``kp.io.HDF5Sink`` modules.

Writing HDF5 Data
~~~~~~~~~~~~~~~~~

The ``Pipeline``, ``Table`` and ``HDF5Pump``/``HDF5Sink`` classes are very
good friends. In this document I'll demonstrate how to build a pipeline to
analyse a file, store intermediate results using the ``Table`` and ``HDF5Sink``
classes and then do some basic high level data analysis using the ``Pandas``
(https://pandas.pydata.org) framework.

(Work in progress)

Data Hierarchy
~~~~~~~~~~~~~~

HDF5 has an internal structure line a Unix file system: There are groups 
("folders") containing tables ("files") which hold the data. Every 
table/group is identified by a "filepath". This is the output of
``ptdump`` for a file which contains MUPAGE simulation data for the ORCA
detector and was converted with the now deprecated ``tohdf5`` command::

    / (RootGroup) 'KM3NeT'
    /event_info (Table(3476,), fletcher32, shuffle, zlib(5)) 'EventInfo'
    /mc_tracks (Table(25651,), fletcher32, shuffle, zlib(5)) 'McTracks'
    /hits (Group) 'RawHitSeries'
    /hits/_indices (Table(3476,), fletcher32, shuffle, zlib(5)) 'Indices'
    /hits/channel_id (EArray(7373599,), fletcher32, shuffle, zlib(5)) 'Channel_id'
    /hits/dom_id (EArray(7373599,), fletcher32, shuffle, zlib(5)) 'Dom_id'
    /hits/event_id (EArray(7373599,), fletcher32, shuffle, zlib(5)) 'Event_id'
    /hits/time (EArray(7373599,), fletcher32, shuffle, zlib(5)) 'Time'
    /hits/tot (EArray(7373599,), fletcher32, shuffle, zlib(5)) 'Tot'
    /hits/triggered (EArray(7373599,), fletcher32, shuffle, zlib(5)) 'Triggered'
    /mc_hits (Group) 'McHitSeries'
    /mc_hits/_indices (Table(3476,), fletcher32, shuffle, zlib(5)) 'Indices'
    /mc_hits/a (EArray(240744,), fletcher32, shuffle, zlib(5)) 'A'
    /mc_hits/event_id (EArray(240744,), fletcher32, shuffle, zlib(5)) 'Event_id'
    /mc_hits/origin (EArray(240744,), fletcher32, shuffle, zlib(5)) 'Origin'
    /mc_hits/pmt_id (EArray(240744,), fletcher32, shuffle, zlib(5)) 'Pmt_id'
    /mc_hits/time (EArray(240744,), fletcher32, shuffle, zlib(5)) 'Time'

All nodes with the type ``Table`` are 2D tables, which is a list of
``HDF5Compund`` data (similar to C-structs). This format is supported by all
HDF5 wrappers and can be read e.g. with the Pandas framework, which is
written in Python and designed for high level statistical analysis.
Other nodes are of the type ``EArray``, which is an "Extensible Arrays" in
the HDF5 context. It represents a 1D array. The ``hits`` and ``mc_hits`` are
stored this way to get the maximum performance when doing event-by-event
analysis. The ``_indices`` array holds the index information, which you
need to split up the huge 1D arrays into the corresponding events.

A typical km3net h5 file looks like this: The reconstruction tables, for 
example, have columns called "energy" or "zenith", and each row of the table
corresponds to a single event::

    ├── event_info        # 2D table
    ├── mc_tracks         # 2D table
    ├── hits              # group
    │   ├── _indices      # 2D table with the index information (index, n_items)
    │   ├── tot           # 1D int array
    │   ├── time          # 1D float array
    │   └── ...
    └── reco              # group
        ├── aashowerfit   # 2D table
        │   ├── E         # 1D float array
        │   ├── phi
        │   └── ...
        └── ...

(Experts) Data Substructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since HDF5 is a general-purpose table based format, there are some minor 
tweaks done by km3pipe to emulate an event-by-event based workflow:

For example, the event ``Hit`` storage: each hit in the whole file (!) is split up
into its "components" ``time``, ``tot`` etc. and stored under the ``/hits``
group. In order to get the hits for the event #23, you first have to read
in the ``/hits/_indices`` table (keep that in memory if you want to look at
multiple events, it's 1-2 MB or so!) and look at its entry at index 23.
You will see two numbers, the first is the index of the first hit (let's call 
it ``idx``) and the second is the number of hits (``n_items``).
Now you can read ``/hit/time[idx:idx+n_items]``, ``/hit/time[idx:idx+n_item``, 
etc. Of course KM3Pipe provides the ``km3pipe.io.hdf5.HDF5Pump(filename=...)``
instance which does this for you::

    p = km3pipe.io.hdf5.HDF5Pump(filename="path/to/file.h5")
    blob = p[23]
    hits = blob["Hits"]
