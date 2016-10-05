HDF5
====

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

Command Line Utils
------------------

`h5tree`
~~~~~~~~

Shows the content of a HDF5 file::

    ┌─[moritz@averroes ~/km3net/data ]
    └─╼ h5tree nueCC.h5
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

`tohdf`
~~~~~~~
.. program-output:: tohdf5 --help

`hdf2root`
~~~~~~~~~~
.. program-output:: hdf2root --help
