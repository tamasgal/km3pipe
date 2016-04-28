.. _file_formats:

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

    ├── analysis
    │   ├── dusj_theta_times_n_hits
    │   ├── ...
    │   └── recolns_theta_times_n_hits_minus_reco_plus_seven
    ├── event
    │   ├── 1
    │   │   └── hits
    │   │       ├── channel_id
    │   │       ├── ...
    │   │       ├── time
    │   │       └── tot
    │   ├── 10
    │   ├── 11
    │   ├── 12
    │   ├── ...
    │   ├── 2
    │   ├── 3
    │   ├── 4
    │   ├── 5
    │   ├── 6
    │   ├── 7
    │   ├── 8
    │   ├── 9
    │   └── ...
    └── reco
        ├── aashowerfit
        │   ├── E
        │   ├── phi
        │   ├── ...
        │   └── theta
        ├── dusj
        ├── recolns
        ├── ...
        └── royfit

`event/`
~~~~~~~

`reco/`
~~~~~~

`analysis/`
~~~~~~~~~~

