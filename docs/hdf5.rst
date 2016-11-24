HDF5
====

.. contents:: :local:

HDF5 files are the prefered input and primary output type in KM3Pipe.
It is a general format for hierarchical storage of large amount of numerical
data. Unlike e.g. ROOT, HDF5 is a general-purpose dataformat and not 
specifically designed for HEP experiments.

In KM3NeT, It is used to store event information like PMT hits, 
reconstructed particles and all kind of other analysis results.

Data Hierarchy
--------------

HDF5 has an internal structure line a Unix file system: There are groups 
("folders") containing tables ("files") which hold the data. Every 
table/group is identified by a "filepath"::

  /
  /hits
  /reco/jgandalf
  /some/very/deep/group
  /l0_hits/0/0
  /l0_hits/0/1

The data itself is arranged in a 2D table format (rows + columns), where the
columns all can have different datatypes.

A typical km3net h5 file looks like this: The reconstruction tables, for 
example, have columns called "energy" or "zenith", and each row of the table
corresponds to a single event::

    ├── event_info
    │   ├── det_id
    │   └── ...
    ├── mc_tracks
    │   ├── dir
    │   └── ...
    ├── hits      # 2D Table
    │   ├── tot   # int
    │   ├── time  # float
    │   └── ...
    └── reco              # Group
        ├── aashowerfit   # 2D Table
        │   ├── E         # float
        │   ├── phi
        │   └── ...
        └── ...

(Experts) Data Substructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since HDF5 is a general-purpose table based format, there are some minor 
tweaks done by km3pipe to emulate an event-by-event based workflow:

For example, the ``Hit`` storage: each hit is represented by a row in the 
``/hits`` table. But since each event has multiple (a lot!) of hits,
these tables have an additional column ``'event_id'`` to identify the 
corresponding event. So, you can perform queries like 
``hits = hit_table.where('even_id == 42')`` to get all hits from event #42.

Reading/Writing
---------------

Table Workflow
~~~~~~~~~~~~~~

Pipeline Workflow
~~~~~~~~~~~~~~~~~


Conversion Utils
----------------

See the :ref:`h5cli` on how to convert & inspect HDF5 files from the shell.
