File Formats
============

In the following you find a description of the file formats used to store
event and reconstruction information for KM3NeT simulations and real data.

HDF5
----

HDF5 files are the prefered input and primary output type in KM3Pipe.
It is used to store event information like PMT hits taken with KM3NeT
detectors, reconstructed particles and all kind of other analysis results.

Data Hierarchy
--------------

Three main groups are currently used to organise data in HDF5 files:
`event`, `reco` and `analysis`, each holding 1D arrays of data.
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

