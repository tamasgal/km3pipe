HDF5
====

HDF5 files are the prefered input and primary output type in KM3Pipe.
It is used to store event information like PMT hits taken with KM3NeT
detectors, reconstructed particles and all kind of other analysis results.

Data Hierarchy
--------------

Three main groups are currently used to organise data in HDF5 files:
``event``, ``reco`` and ``analysis``, each holding 1D arrays of data.
A typical file looks like this::

    ├── hits                    # 2D Table
    │   ├── tot                 # int
    │   ├── time                # float
    │   └── ...
    ├── mc_tracks
    │   ├── dir
    │   └── ...
    ├── event_info
    │   ├── det_id
    │   └── ...
    └── reco                    # Group
        ├── aashowerfit         # 2D Table
        │   ├── E               # float
        │   ├── phi
        │   └── ...
        └── ...

Conversion Utils
----------------

See the :ref:`h5cli` on how to convert & inspect HDF5 files from the shell.
