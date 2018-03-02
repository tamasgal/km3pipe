Unreleased changes
------------------
* Fixes ``KeyError`` when accessing McTracks via the aanet API

Version 7
---------

7.17.1 / 2018-02-28
~~~~~~~~~~~~~~~~~~~
* Fixed typo ``ligiermirro`` -> ``ligiermirror``


7.17.0 / 2018-02-27
~~~~~~~~~~~~~~~~~~~
* ``triggersetup`` command line utility added, which allows easy access to
  the trigger setup of a given run setup
* ``k40calib`` now accepts ``-s`` to select a ``JDAQTimeslice`` stream.
  an empty string will use the original stream and 'L1', 'L2' and 'SN' will 
  select the new streams introduced in Jpp v9
* ``kp.tools.AnyBar`` added to control the AnyBar macOS app, including a
  pipeline integration: ``kp.Pipeline(anybar=True)``
* ``km3pipe runtable`` is now a standalone command line tool: ``runtable``
* ``km3pipe runinfo`` is now a standalone command line tool: ``runinfo``
* ``UTMInfo`` added in ``kp.hardware`` to make access to UTM information easier
  in detector files ``Detector().utm_info``...
* ``ligiermirror`` command line utility added


7.16.0 / 2018-01-28
~~~~~~~~~~~~~~~~~~~

* ``km3pipe.plot``: Common plotting helpers
* A handful utility functions for ``km3pipe.shell.Script``, like ``cp``,
  ``iget``...
* ``kp.tools.bincenters`` now lives in ``kp.plot``. 
* ``kp.db.DBManager.trigger_setup`` can now retrieve trigger setups for a given
  OID
* Add ``n_digits`` option in ``kp.io.evt.EvtPump`` for file number index
  when iterating over multiple files.
* ``kp.math`` has some helpers for bootstrapping confidence intervals
  when fitting probability distributions via max LLH (in scipy.stats)
* Docs: move statistics examples to own section, show some distribution fits

7.15.0 / 2018-01-19
~~~~~~~~~~~~~~~~~~~
* ``TimeslicePump`` now supports the readout of any stream ("L0", "L1", "SN"...)
* Minor bugfixes (km3pipe has no attribute named hardware...)

7.14.3 / 2018-01-17
~~~~~~~~~~~~~~~~~~~
* add loguniform distribution (``kp.math``)
* add contextmanager for pumps (``with HDF5Pump(fname) as h5: print(h5[0])``)
* clean up makefile / installer docs
* debug compilation/makefile issues

7.14.1 / 2018-01-09
~~~~~~~~~~~~~~~~~~~
* Windows compatible version of `sys.peak_memory`. KM3Pipe should now compile
  and work under windows...
* fix issues with hit indexing when merging multiple h5 files

7.14.0 / 2017-12-22
~~~~~~~~~~~~~~~~~~~
* ``core.pyx`` and ``tools.pyx`` have been "depyxed"
* ``Calibration.apply**`` (should) always returns the hits
* ``Module.finish`` (and thus the pipeline!) actually return something now!
* ``Calibration`` shortcut removed from ``km3pipe``, so now  you have to use
  ``from km3pipe.calib import Calibration`` or just ``kp.calib.Calibration``
  if you ``importe km3pipe as kp``.
  This change was needed to be able to import __km3pipe__ in Julia.
* ``kp.io.hdf5.HDF5Pump`` now accepts the path of a boolean cut mask,
  e.g. ``cut_mask='/pid/survives_precut'``. If the bool mask is false, the 
  event is skipped.

7.13.2 / 2017-12-11
~~~~~~~~~~~~~~~~~~~
* makefile tuning
* linalg tuning (innerprod_1d etc)
* pandas mc utils simplification (`is_neutrino` takes Series, not DataFrame, etc)

7.13.2 / 2017-12-10
~~~~~~~~~~~~~~~~~~~
* add a makefile
* flake8 all the things
* make compatible for upcoming numpy 1.14
* add ``nb2shpx`` util for notebook -> sphinx gallery exampe
* some pandas bits and bobs


7.13.0 / 2017-12-07
~~~~~~~~~~~~~~~~~~~
* Improved CLB raw data readout
* Pipelines now return a ``finish blob`` which contains the return values
  of each modules finish method (this is for Tommaso)
* ``TimesliceParser`` now reads all timeslice streams (L0, L1, L2, SN)
* ``TimesliceParser`` now returns the blob even if it was not able to parse
  the data
* ``TMCHRepump`` now has an iterator interface
* Fixed bug in ``StreamDS`` where it tried to create a session in Lyon and
  failed. Now it uses the permanent session which was created by Cristiano
* Some smaller bugfixes and name-consistency-changes

7.12.1 / 2017-11-28
~~~~~~~~~~~~~~~~~~~
* ``kp.math``: ``zenith, azimuth, phi, theta`` now follow the correct 
  km3net definitions (finally)
* JFit pump now follows multipump paradigm
* improved logging in IO

7.12.0 / 2017-11-24
~~~~~~~~~~~~~~~~~~~
* Added preliminary ``kp.io.jpp.FitPump``, which reads ``JFit`` objects. 
  However, it does not yet read the ``fitinf`` vector, yet.
* ``Calibration`` moved to ``kp.calib``, since core.pyx was Cython and numba
  does not like cython files.
* ``streamds`` now requires the ``get`` command to retrieve info on command
  line
* ``streamds`` can now upload to runsummary tables
* remove obsolete ``kp.dev`` (now resides in ``kp.tools``
* fixes EOF hang in ``kp.io.daq.TMCHRepump``

7.11.0 / 2017-11-12
~~~~~~~~~~~~~~~~~~~
* Hotfix of the SummaryslicePump (rates/fifos/hrvs reference issue)
* ``Geometry`` has been renamed to ``Calibration``
* aanetpump now does not convert MC times by default

7.10.0 / 2017-11-07
~~~~~~~~~~~~~~~~~~~
* JPPPump removed
* New ``k40calib`` command line tool to calibrate DOMs using the K40
  method
* ``TimeslicePump`` and ``SummaryslicePump`` now add meta information about
  the slices to the blob: ``blob['TimesliceInfo']`` and 
  ``blob['SummarysliceInfo']``
* ``SummaryslicePump`` now reads out FIFO status and HRV for each PMT
* ``kp.shell.qsub()`` can be used to submit jobs to SGE clusters

7.9.1 / 2017-11-01
~~~~~~~~~~~~~~~~~~
* Massiv(!) speedup of the JPP timeslice pump (factor 3 to 4), now only about
  8% slower compared to raw JPP readout. We are at the I/O limit of ROOT ;)
* ``DTypeAttr`` now allows adding of additional fields to the numpy array
  using the ``.append_fields`` method.
* merge ``kp.dev`` into ``kp.tools``

7.9.0 / 2017-10-27
~~~~~~~~~~~~~~~~~~
* New command line utility to plot the trigger contributions: ``triggermap``
* fix wrong spaceangle computation (duh!)
* KM3HDF5 Version 4.4 (minimum 4.1): RawHit time is now int32 =
  instead of float32 and CRawHit*.time/CMcHit*.time is float64
  fixes bugs which occured due to precision loss for large hit times

7.8.1 / 2017-10-23
~~~~~~~~~~~~~~~~~~
* Fixes the ``io.jpp.EventPump`` to use ``RawHitSeries``

7.8.0 / 2017-10-23
~~~~~~~~~~~~~~~~~~
* A preliminary version of ``SummaryslicePump``
* A new pump for JPP events has been added: ``io.jpp.EventPump``. This will
  replace the ``JPPPump`` soon.
* several changes to km3modules.k40 to improve the calibration procedure


7.7.1 / 2017-10-12
~~~~~~~~~~~~~~~~~~
* (aanet/tohd5) run id is now read from header, per default; if that fails
  (or the flag ``--ignore-run-id-from-header`` is set, fall back to
  the ``event.run_id``

7.7.0 / 2017-10-11
~~~~~~~~~~~~~~~~~~
* (aanet/tohd5) new option to read run ID from header, not event.
  in old versions of JTE, the event.run_id is overwritten with the default, 1.
* there is now a new command line utility called ``streamds`` for non-pythonistas
* The new ``km3pipe.ahrs`` now contains AHRS calibration routines


7.6.1 / 2017-10-09
~~~~~~~~~~~~~~~~~~
* ``HDF5Sink`` now uses the new ``HDF5MetaData`` class two write more verbose
  metadata to the files (e.g. file conversion parameters)
  HDF5 metadata now contains much more information; e.g. if the mc hit time
  correction was applied, the aa-format, whether jppy was used etc
* introduce "services" to the pipeline model. these are addressed via the
  ``expose`` method
* aa/gand: fix up-vs-downgoing normalisation (now difference over sum)
* fix automatic JTE/MC time conversion
* fix the check if mc time correction needs to be applied
* ``h5tree`` CLI util, to print just the structure + nevents + nrows.
  less verbose than ``ptdump``
* KM3HDF5 4.3: introduce richer metadata

7.5.5 / 2017-09-27
~~~~~~~~~~~~~~~~~~
* Option to Ignore hits in pumps
* fix aanet fitinf enum

7.5.4 / 2017-09-25
~~~~~~~~~~~~~~~~~~
* fix aanet (optional) 4-element event.weight vector readout. the weights
  can now be read again :-)
* Use mc_t to detect if MC time conversion (from JTE to MC time) should be
  applied. Should be more reliable since some MC could use positive DET_ID
  which should only be used for real data

7.5.3 / 2017-09-23
~~~~~~~~~~~~~~~~~~
* Fixed bug which converted MC times in real data. Now it checks for a
  positive DET_ID and does not convert (even if told so...)
* Fixes zt-plot, which did not use the newly implemented datatypes

7.5.2 / 2017-09-22
~~~~~~~~~~~~~~~~~~
* fixed bug in math.spatial_angle (zenith vs latitude)
* (aanet) jgandalf_new now computes a ton of fit-spread-related metrics (updated in tohdf5 help string, too)
* added usage warning to math.azimuth. for rest-of-world compatible coordinates, use KM3Astro
* accept coords in polygon containment (contains_xy)

7.5.1 / 2017-09-19
~~~~~~~~~~~~~~~~~~
* The AANetPump now automatically converts hit times from JTE time to MC time.
  This should be now the default behaviour for all pumps.
* ``tohdf5`` now has the option to ``--do-not-correct-mc-times`` in case
  the automatic conversion from JTE to MC hit time is not wanted
* HDF5 version updated to 4.2 due to the new handling of JTE/MC times.
  It is however backwards compatible to 4.1.
* Freezes six-dependency to version 1.10 as the metaclass stuff for
  Python 2 is broken in 1.11

7.5.0 / 2017-09-14
~~~~~~~~~~~~~~~~~~
* Adds sorting for ``***Series``` and other `DTypeAttr` subclasses.

7.4.2 / 2017-09-11
~~~~~~~~~~~~~~~~~~
* Numpy style slicing for ``***Series``
* skip aanet header, optionally

7.4.1 / 2017-08-28
~~~~~~~~~~~~~~~~~~
* minor fixes for i3 files + old aanet
* Add arrival timestamp to controlhost Prefix

7.4.0 / 2017-08-18
~~~~~~~~~~~~~~~~~~
* Introduces ``StreamDS`` in ``km3pipe.db`` which allows easy access to all
  streamds tables

7.3.2 / 2017-08-08
~~~~~~~~~~~~~~~~~~
* add ``i3shower2hdf5`` CLI util for converting orcadusj files
* add ``kp.math.space_angle``

7.3.1 / 2017-08-02
~~~~~~~~~~~~~~~~~~
* add ``i3toroot`` and ``i3root2hdf5`` CLI utils for converting I3 files
* drop deprecated ``h5tree``, from ``km3pipe.utils`` (use ``ptdump`` instead)
* drop deprecated ``km3pipe.io.hdf5.H5Mono``
* read aanet ``mc_id = evt.frame_index - 1``

7.2.5 / 2017-07-20
~~~~~~~~~~~~~~~~~~
* drop ``read_hdf5`` and ``GenericPump`` from top level module import
  (would make pytables a hard requirement)

7.2.3 / 2017-07-19
~~~~~~~~~~~~~~~~~~
* No more error messages in ``Detector`` or ``Geometry`` (which uses
  ``Detector``) when reading in corrupt DETX with negative line ids.
* Fixes "TypeError: data type not understood" for Geometry.apply
* Various fixes to support the new HitSeries format (e.g. for RainbowaAlga2)
* New styles
* SciPy histogram showoff by Moritz
* Minor updates in docs
* Skeleton for future project bootstrap

7.2.2 / 2017-07-11
~~~~~~~~~~~~~~~~~~
* ``AANetPump`` now parses the full header and ``HDF5Pump`` writes it to
  /header as attributes

7.2.1 / 2017-07-11
~~~~~~~~~~~~~~~~~~
* Fixes ``HDF5Pump`` for Python3

7.2.0 / 2017-07-11
~~~~~~~~~~~~~~~~~~
* KM5HDF5 v4.1 now have DU and Floor information when calibrating
* Added 5 last lines in: daq.py - TMCHdata for reading the monitoring file

7.1.1 / 2017-07-11
~~~~~~~~~~~~~~~~~~
* Fixed bug with aanet pump

7.1.0 / 2017-07-11
~~~~~~~~~~~~~~~~~~
* Increased performance for Geometry.apply
* Changed type of time to float in ``RawHitSeries``
* Introducing ``CRawHitSeries`` and ``CMcHitSeries`` which represent calibrated
  hit series
* New command line argument to apply geometry/time calibration to an HDF5 file
  usage: ``calibrate DETXFILE HDF5FILE``

7.0.0 / 2017-07-09
~~~~~~~~~~~~~~~~~~
* New KM3HDF5 version 4.0
* HDF5Pump now creates ``RawHitSeries``. The other pumps will be updated too.
* ``Geometry.apply()`` will return ``HitSeries`` if a ``RawHitSeries`` instance
  is the input.
* Several bug fixes and speedups.

Version 6
---------

6.9.2 / 2017-07-06
~~~~~~~~~~~~~~~~~~
* Hotfix
* HDF5 version was accidentally set to 4.3 in km3pipe v6.9.1, now it is 3.4
* minor change in EvtPump

6.9.1 / 2017-07-04
~~~~~~~~~~~~~~~~~~
* Last version freeze before 7.0
* Fix event_id and run_id
* add ``MCHitSeries`` to represent Monte Carlo hitseries
* add ``MCTrackSeries`` to represent Monte Carlo trackseries
* add ``MCHit`` to represent Monte Carlo hits
* add ``MCTrack`` to represent Monte Carlo tracks
* add run id to event_info

6.9.0 / 2017-07-03
~~~~~~~~~~~~~~~~~~
* add ``TMCHRepump`` to replay IO_MONIT dumps
* add ``RawHitSeries`` to represent uncalibrated hitseries
* use ``RawHitSeries`` and nested structure in HDF5 files when converting
  from aanet
* HDF5 version changed from to 3.3. Only the hits-readout is affected though!
  DST, reco and track readout were not changed and should be compatible
  down to 3.0

6.8.2 / 2017-06-20
~~~~~~~~~~~~~~~~~~
* add option to create default config file
* fix wrong readout in `io.root.get_hist3d`

6.8.1 / 2017-06-15
~~~~~~~~~~~~~~~~~~
- DOI citation added
- tohdf5: aa pump: make zed correction (mc tracks) optional

6.8.0 / 2017-06-13
~~~~~~~~~~~~~~~~~~
* minor bugfixes
* git repository changed, ``km3pipe update develop`` is broken for
  all versions below 6.8.0

6.7.1 / 2017-06-08
~~~~~~~~~~~~~~~~~~
* ControlHost improvements
* Change ``every`` behavior in pipeline
* h5chain multifile fix

6.7.0 / 2017-05-08
~~~~~~~~~~~~~~~~~~
* ``totmonitor`` command line utility added
* bump library versions (scipy >=0.19)

6.6.6 / 2017-04-03
~~~~~~~~~~~~~~~~~~
* change blosc compression -> zlib compression
* add corsika evt tag reader (seamuon/seaneutrino)

6.5.5 / 2017-03-29
~~~~~~~~~~~~~~~~~~
* fix decoding issues in EvtPump

6.5.4 / 2017-03-21
~~~~~~~~~~~~~~~~~~
* fix aanet mc_tracks usr backwards compat

6.5.3 / 2017-03-21
~~~~~~~~~~~~~~~~~~
* Show initialisation time for pipeline and modules.
* Doc update / more examples
* aanet: fix ``mc_tracks.usr`` readout (use ``.getusr()``)

6.5.2 / 2017-03-12
~~~~~~~~~~~~~~~~~~
* Support for KM3PIPE_DEBUG env variable to enable line tracing (set it to 1)

6.5.1 / 2017-03-12
~~~~~~~~~~~~~~~~~~
* Fixed Cython/numpy dependency, now they should install automatically.

6.5.0 / 2017-03-11
~~~~~~~~~~~~~~~~~~
* remove astro stuff, move to git.km3net.de/moritz/km3astro
* fixed HDF5 version warning
* some cleanup in __init__.pys, so be prepared to change some import statements
  * split up tools into tools/math/sys/dev/time/mc
  * stuff under km3modules is now in km3modules.common


6.4.4 / 2017-02-27
~~~~~~~~~~~~~~~~~~
* h5concat (multi-h5-to-h5) deprecated because buggy. Going to drop all
  event_id for 7.0 (for now use ptconcat
* Clean up setup.py

6.4.3 / 2017-02-22
~~~~~~~~~~~~~~~~~~
* Fix pyroot segfault when reading aanet header

6.4.2 / 2017-02-21
~~~~~~~~~~~~~~~~~~
* Fix aanet header
* style update

6.4.1 / 2017-02-16
~~~~~~~~~~~~~~~~~~
* API doc fixes
* add missing requirements to setup.py
* minor py2/py3 compat fix

6.4.0 / 2017-02-08
~~~~~~~~~~~~~~~~~~
* K40 calibration module from Jonas!
* Pushover client! Push messages to your mobile phone or computer via
  ``pushover the message you want``.
* Minor bugfixes

6.3.0 / 2017-01-21
~~~~~~~~~~~~~~~~~~
* Introduces `BinaryStruct` which makes handling binary data much more easier.
* `Cuckoo` now allows args and kwargs to be passed to the callback function.
* km3modules.plot module added including a unified DOM plotter
* km3modules.fit module added including k40 coincidence fit

6.2.2 / 2017-01-19
~~~~~~~~~~~~~~~~~~
* add ``rundetsn`` cmd tool

6.2.1 / 2017-01-17
~~~~~~~~~~~~~~~~~~
* Use numpy-style imports
* AanetPump: Don't use `evt.id` for event_id by default, until we all agree on it

6.2.0 / 2017-01-16
~~~~~~~~~~~~~~~~~~
* The DB client now automatically uses the production cookie on Lyon.
  No need to deal with session requests anymore...
* New command line utility to download runs from iRODS: `km3pipe retrieve ...`
* Integrates the controlhost package

6.1.1 / 2017-01-12
~~~~~~~~~~~~~~~~~~
* H5Chain now is just a Multifile pd.HDFStore
* `prettyln` for nicely formatted headers
* Online DAQ readout is now Python3 proof

6.1.0 / 2017-01-02
~~~~~~~~~~~~~~~~~~
* H5Pump now supports multiple files
* h5concat util for concatenating multiple H5 files

6.0.4 / 2016-12-21
~~~~~~~~~~~~~~~~~~
* fix: H5Sink in py3 actually creates indextables + closes file now
* HDF5 3.1: Change compression to BLOSC, fallback to zlib
* MergeDF module
* Easier access to seconds in timer

6.0.3
~~~~~
* Fix Dataclass + IO conversion signatures towards consistency
* Ask for requesting new DB session when session expired.

6.0.2
~~~~~
* Make blob ordered by default + actually use it in the pumps.

6.0.1
~~~~~
* FIX freeze numpy version

6.0.0 2016-11-29
~~~~~~~~~~~~~~~~
* change all bool dataclasses to int
* add new fields to event_info: livetime_sec, n_evs_gen, n_files_gen
* update KM3HDF -> v3

Version 5
---------

5.5.3 / 2016/11/28
~~~~~~~~~~~~~~~~~~
* Add fix_event_id option to h5pump

5.5.2 / 2016-11-24
~~~~~~~~~~~~~~~~~~
* Updated docs

5.5.1 / 2016-11-24
~~~~~~~~~~~~~~~~~~
* Cuckoo now can be called directly
* CHPump uses Cuckoo for log.warn to avoid spamming in case of
  high network traffic
* DOM class to represent DOMs retrieved by the DBManager

5.5 / 2016-11-18
~~~~~~~~~~~~~~~~
* New ``KM3DataFrame + KM3Array`` dataclasses, np/pandas subclasses + metadata
* replaced ``ArrayTaco`` with ``KM3Array``
* ``H5Mono`` pump to read HDF5 with flat table structure

5.4 / 2016-11-08
~~~~~~~~~~~~~~~~
* Add a bunch of useful km3modules

5.3.3 / 2016-11-04
~~~~~~~~~~~~~~~~~~
* Fix time calib application

5.3.2 / 2016-11-03
~~~~~~~~~~~~~~~~~~
* add preliminary bootstrap script

5.3.0 / 2016-11-03
~~~~~~~~~~~~~~~~~~
* Detector.dom_positions now returns an OrderedDict instead of a list
* Cache DOM positions in Detector
* pld3 function in tools, to calculate point-line-distance in 3d

5.2.2 / 2016-10-26
~~~~~~~~~~~~~~~~~~
* Fixes Cython dependency
* ``kp.io.pandas.H5Chain`` now returns N _events_, not _rows_

5.2.0 / 2016-10-25
~~~~~~~~~~~~~~~~~~
* Introduce ``configure`` method in ``Module``, so you no longer need to
  override ``__init__`` and call ``super``. You can, though ;)

5.1.5 / 2016-10-24
~~~~~~~~~~~~~~~~~~
* DB/Dataclass bugfixes

5.1.2 / 2016-10-20
~~~~~~~~~~~~~~~~~~
* Unify Reco + Wrapper dataclass. Reco(map, dtype) -> ArrayTaco.from_dict()
* add ``to='pandas'`` option to ``Dataclass.serialise()``
* Tweak internal array/dataframe handling

5.1.0 / 2016-10-20
~~~~~~~~~~~~~~~~~~
* ...

5.0.0 / 2016-10-18
~~~~~~~~~~~~~~~~~~
* Major dataclass refactor:
  * hits now always have pos_x, .., dir_y, .., t0
  * completely flat hit datastructure

Version 4
---------

4.9.0 / 2016-10-14
~~~~~~~~~~~~~~~~~~
* New plot style handling and new styles: talk, poster, notebook
  (load them using `km3pipe.style.use(...)`)
  Just like in previous versions: `import km3pipe.style` will load
  the default style.

4.8.3 / 2016-10-13
~~~~~~~~~~~~~~~~~~
* Fixes t0 application in HitSeries

4.8.2 / 2016-10-13
~~~~~~~~~~~~~~~~~~
* Fixes geometry application in HitSeries

4.8.1 / 2016-10-12
~~~~~~~~~~~~~~~~~~
* Forcing matplotlib 2.0.0b4 as dependency. Don't blame us!
* New unified style for all plots, using `import km3pipe.style`
* aanet / jgandalf: write zeroed row if no track in event
* fix string handling in H5 attributes

4.8.0 / 2016-10-11
~~~~~~~~~~~~~~~~~~
* Group frames in summary slices under /timeslices/slice_id/frame_id
  when using ``tohdf5 -j -s FILE.root``
* ``hdf2root`` is now it's own command
* ``tohdf5`` and ``hdf2root`` no longer ``km3pipe`` CLI subcommands
* Use zlib instead of blosc for compatibility reasons
* add CLI option to make DB connection non-permanent
* ``tohdf5`` / ``GenericPump`` now supports multiple input files for aanet files

4.7.1 / 2016-09-29
~~~~~~~~~~~~~~~~~~
* Improved documentation
* Fixed event_id indexing for the /hits table in HDF5
* root sub-package added (via rootpy)
* Added arguments to allow optional parsing of L0 data and summaryslices
  when using the JPPPump
* New command line utility to convert to HDF5: ``tohdf5``

4.7.0 / 2016-09-25
~~~~~~~~~~~~~~~~~~
* Adds summary slice readout support via jppy
* Introducing astro package
* Use BLOSC compression library for HDF5

4.6.0
~~~~~
* ...

4.5.1
~~~~~
* Bugfixes

4.5.0
~~~~~
* Full L0 readout support via ``JPPPump``

4.4.1
~~~~~
* Bugfixes

4.4.0
~~~~~
* JEvt/JGandalf support
* Minor HDF5 Improvements

4.3.0
~~~~~
* Introduces HDF5 format versioning

4.2.2
~~~~~
* Bugfixes

4.2.1
~~~~~
* Bugfixes

4.2.0
~~~~~
* ...

4.1.2
~~~~~
* Bugfixes

4.1.1 / 2016-08-09
~~~~~~~~~~~~~~~~~~
* Bugfixes

4.1.0 / 2016-08-04
~~~~~~~~~~~~~~~~~~
* Ability to use simple functions as modules
