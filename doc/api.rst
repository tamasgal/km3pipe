API Reference
=============

.. contents:: :local:

KM3Pipe: Main Framework
-----------------------

``km3pipe.core``
~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.core
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.core

.. autosummary::
  :toctree: api

  Blob
  Module
  Pipeline
  Pump
  Run


``km3pipe.calib``
~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.calib
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.calib

.. autosummary::
  :toctree: api

  Calibration




``km3pipe.cmd``: Command Line Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: km3pipe.cmd

.. autosummary::
  :toctree: api


  detectors
  detx
  retrieve
  run_tests
  rundetsn
  update_km3pipe


``km3pipe.dataclasses``: Internal Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.dataclasses
  :no-members:
  :no-inherited-members:


.. currentmodule:: km3pipe.dataclasses

.. autosummary::
  :toctree: api

  Table
  is_structured
  has_structured_dt
  inflate_dtype


``km3pipe.db``: Database Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.db
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.db

.. autosummary::
  :toctree: api

  DBManager
  StreamDS
  ParametersContainer
  DOMContainer
  DOM
  we_are_in_lyon
  clbupi2ahrsupi
  show_ahrs_calibration


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
  EventPump
  PicklePump
  df_to_h5
  read_hdf5
  write_table


``km3pipe.logger``
~~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.logger
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.logger

.. autosummary::
  :toctree: api

  LogIO
  get
  set_level


``km3pipe.mc``
~~~~~~~~~~~~~~

.. automodule:: km3pipe.mc
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.mc

.. autosummary::
  :toctree: api

  geant2pdg
  pdg2name
  name2pdg


``km3pipe.plot``: Plotting tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: km3pipe.plot

.. autosummary::
  :toctree: api

  hexbin


``km3pipe.srv``
~~~~~~~~~~~~~~~

.. automodule:: km3pipe.srv
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.srv

.. autosummary::
  :toctree: api

  ClientManager


``km3pipe.shell``
~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.shell
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.shell

.. autosummary::
  :toctree: api

  Script
  qsub
  gen_job
  hppsgrab
  get_jpp_env


``km3pipe.tools``
~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.tools
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.tools

.. autosummary::
  :toctree: api

  ifiles
  token_urlsafe
  prettyln
  irods_filepath
  unpack_nfirst
  split
  decamelise
  camelise
  deprecated


``km3pipe.time``
~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.time
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.time

.. autosummary::
  :toctree: api

  Timer
  Cuckoo
  total_seconds
  tai_timestamp


KM3Modules: Pipeline Segments
-----------------------------

``km3modules.common``: Useful helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: km3modules.common
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3modules.common

.. autosummary::
  :toctree: api

  BlobIndexer
  Cut
  Delete
  Dump
  GetAngle
  HitCounter
  Keep
  MemoryObserver
  StatusBar
  Wrap


``km3modules.k40``: K40 calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: km3modules.k40
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3modules.k40

.. autosummary::
  :toctree: api

  K40BackgroundSubtractor
  IntraDOMCalibrator
  TwofoldCounter
  HRVFIFOTimesliceFilter
  SummaryMedianPMTRateService
  MedianPMTRatesService
  ResetTwofoldCounts
  calculate_angles
  calculate_rms_means
  calculate_rms_rates
  calibrate_dom
  correct_means
  correct_rates
  fit_angular_distribution
  fit_delta_ts
  load_k40_coincidences_from_hdf5
  load_k40_coincidences_from_rootfile
  minimize_qes
  minimize_t0s


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


``km3modules.mc``
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: km3modules.mc
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3modules.mc

.. autosummary::
  :toctree: api
  
  MCTimeCorrector
