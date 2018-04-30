HDF5
====

.. contents:: :local:

HDF5 files are the prefered input and primary output type in KM3Pipe.
It is a general format for hierarchical storage of large amount of numerical
data. Unlike e.g. ROOT, HDF5 is a general-purpose dataformat and not 
specifically designed for HEP experiments. HDF5 also requires only a tiny
library (hdf5lib) and is accessible with almost all popular programming
languages (Python, C, C++, Java, Go, Julia, R, Matlab, Rust...).

In KM3NeT, it is used to store event information like PMT hits, 
reconstructed particles and all kind of other analysis results.


Data Hierarchy
--------------

HDF5 has an internal structure line a Unix file system: There are groups 
("folders") containing tables ("files") which hold the data. Every 
table/group is identified by a "filepath". This is the output of
``ptdump`` for a file which contains MUPAGE simulation data for the ORCA
detector::

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


Conversion Utils
----------------

To convert a ROOT/EVT file to KM3NeT HDF5 use ``tohdf5`` which comes with KM3Pipe::

    tohdf5 filename.root

See the :ref:`h5cli` on how to convert & inspect HDF5 files from the shell.
