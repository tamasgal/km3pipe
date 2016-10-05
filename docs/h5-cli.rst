``tohdf``
~~~~~~~~~
.. program-output:: tohdf5 --help

``hdf2root``
~~~~~~~~~~~~
.. program-output:: hdf2root --help

``h5info``
~~~~~~~~~~

.. program-output:: h5info --help

``h5tree``
~~~~~~~~~~

.. program-output:: h5tree --help

Example::

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

