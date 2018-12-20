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


  createconf
  detectors
  detx
  retrieve
  run_tests
  rundetsn
  update_km3pipe


``km3pipe.controlhost``
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.controlhost
  :no-members:
  :no-inherited-members:


.. currentmodule:: km3pipe.controlhost

.. autosummary::
  :toctree: api

  Client
  Message
  Tag
  Prefix


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
  TriggerSetup
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
  UTMInfo


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
  HDF5Pump
  HDF5Sink
  EventPump
  PicklePump
  read_calibration


``km3pipe.logger``
~~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.logger
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.logger

.. autosummary::
  :toctree: api

  LogIO
  get_logger
  set_level
  get_printer
  hash_coloured
  hash_coloured_escapes


``km3pipe.math``
~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.math
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.math

.. autosummary::
  :toctree: api

  neutrino_to_source_direction
  source_to_neutrino_direction
  theta
  phi
  azimuth
  zenith
  angle_between
  innerprod_1d
  unit_vector
  pld3
  com
  circ_permutation
  hsin
  space_angle
  rotation_matrix
  Polygon
  IrregularPrism
  SparseCone
  inertia
  g_parameter
  gold_parameter
  log_b
  qrot
  qeuler
  qrot_yaw
  intersect_3d


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
  most_energetic
  leading_particle
  get_flavor
  is_neutrino
  is_muon


``km3pipe.plot``: Plotting tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: km3pipe.plot

.. autosummary::
  :toctree: api

  hexbin
  get_ax
  diag
  automeshgrid
  meshgrid
  prebinned_hist
  joint_hex
  plot_convexhull


``km3pipe.srv``
~~~~~~~~~~~~~~~

.. automodule:: km3pipe.srv
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.srv

.. autosummary::
  :toctree: api

  ClientManager


``km3pipe.testing``
~~~~~~~~~~~~~~~~~~~

.. automodule:: km3pipe.testing
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.testing

.. autosummary::
  :toctree: api

  surrogate


``km3pipe.sys``
~~~~~~~~~~~~~~~

.. automodule:: km3pipe.sys
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3pipe.sys

.. autosummary::
  :toctree: api

  ignored
  peak_memory_usage


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
  iexists
  token_urlsafe
  prettyln
  irods_filepath
  unpack_nfirst
  namedtuple_with_defaults
  split
  decamelise
  camelise
  colored
  supports_color
  cprint
  issorted
  lstrip
  chunks
  is_coherent
  zero_pad
  istype
  AnyBar



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
  np_to_datetime


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

  Dump
  Delete
  Keep
  HitCounter
  HitCalibrator
  BlobIndexer
  StatusBar
  TickTock
  MemoryObserver
  Siphon

``km3modules.ahrs``: AHRS calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: km3modules.ahrs
  :no-members:
  :no-inherited-members:

.. currentmodule:: km3modules.ahrs

.. autosummary::
  :toctree: api

  AHRSCalibrator
  fit_ahrs
  get_latest_ahrs_calibration


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
  calibrate_dom
  calculate_weights
  load_k40_coincidences_from_hdf5
  load_k40_coincidences_from_rootfile
  gaussian
  gaussian_wo_offset
  fit_delta_ts
  calculate_angles
  exponential_polinomial
  exponential
  fit_angular_distribution
  minimize_t0s
  minimize_sigmas
  minimize_qes
  correct_means
  correct_rates
  calculate_rms_means
  calculate_rms_rates
  get_comb_index
  add_to_twofold_matrix

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
  McTruth
  convert_mc_times_to_jte_times

