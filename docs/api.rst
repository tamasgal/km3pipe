API Reference
=============

.. contents:: :local:

KM3Pipe: Main Framework
-----------------------

``km3pipe.io``: Data Input / Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.io
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.io

.. autosummary::
  :toctree: api

  AanetPump
  CHPump
  CLBPump
  DAQPump
  EvtPump
  GenericPump
  H5Chain
  HDF5Pump
  HDF5Sink
  JPPPump
  PicklePump
  df_to_h5
  read_hdf5
  write_table


``km3pipe.dataclasses``: Internal Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.dataclasses
  :no-members:
  :no-inherited-members:


.. currentmodule:: km3pipe.dataclasses

.. autosummary::
  :toctree: api

  EventInfo
  Hit
  HitSeries
  Track
  TrackSeries
  KM3Array
  KM3DataFrame


``km3pipe.core``
~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.core
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.core

.. autosummary::
  :toctree: api

  Pipeline
  Module
  Pump
  Blob
  Geometry
  AanetGeometry
  Run


``km3pipe.hardware``
~~~~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.hardware
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.hardware

.. autosummary::
  :toctree: api

  Detector
  PMT


``km3pipe.logger``
~~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.logger
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.logger

.. autosummary::
  :toctree: api

  LogIO


``km3pipe.srv``
~~~~~~~~~~~~~~~

.. automodule:: km3pipe.srv
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.srv

.. autosummary::
  :toctree: api

  ClientManager


``km3pipe.tools``
~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.tools
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.tools

.. autosummary::
  :toctree: api

  pdg2name
  geant2pdg
  decamelise
  camelise


``km3pipe.db``: Database Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.db
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.db

.. autosummary::
  :toctree: api

  DBManager

  
``km3pipe.cmd``: Command Line Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.cmd
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.cmd

.. autosummary::
  :toctree: api


  run_tests
  runtable
  runinfo
  update_km3pipe
  retrieve
  detx
  detectors
  rundetsn


KM3Modules: Pipeline Segments
-----------------------------

``km3modules``: Useful helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: km3modules
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3modules

.. autosummary::
  :toctree: api

  Wrap
  Dump
  Delete
  Keep
  HitCounter
  BlobIndexer
  StatusBar
  MemoryObserver
  GetAngle
  Cut

``km3modules.k40``: K40 calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: km3modules.k40
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3modules.k40

.. autosummary::
  :toctree: api

  IntraDOMCalibrator
  CoincidenceFinder
  calibrate_dom
  load_k40_coincidences_from_hdf5
  load_k40_coincidences_from_rootfile
  fit_delta_ts
  calculate_angles
  fit_angular_distribution
  minimize_t0s
  minimize_qes
  correct_means
  correct_rates
  calculate_rms_means
  calculate_rms_rates

``km3modules.hits``: Hit Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: km3modules.hits
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3modules.hits

.. autosummary::
  :toctree: api

  HitStatistics
  NDoms
  HitSelector
  FirstHits
  TrimmedHits

``km3modules.reco``: Simple Reconstructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: km3modules.reco
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3modules.reco

.. autosummary::
  :toctree: api
  
  Reconstruction
  SvdFit
  Trawler
  bimod
  uniform_chi2
  idr
  tensor_of_intertia

  
``km3modules.astro``: Astro Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: km3modules.astro
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3modules.astro

.. autosummary::
  :toctree: api

  to_frame


``km3modules.topology``
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: km3modules.topology
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3modules.topology

.. autosummary::
  :toctree: api
  
  TriggeredDUs

  
``km3modules.parser``
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: km3modules.parser
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3modules.parser

.. autosummary::
  :toctree: api
  
  CHParser
