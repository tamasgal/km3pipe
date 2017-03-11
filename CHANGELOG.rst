Unreleased changes
------------------
* remove astro stuff, move to git.km3net.de/moritz/km3astro
* split up tools into tools/math/sys/dev/time/mc
* fixed HDF5 version warning

6.4.4 / 2017-02-27
------------------
* h5concat (multi-h5-to-h5) deprecated because buggy. Going to drop all 
  event_id for 7.0 (for now use ptconcat
* Clean up setup.py

6.4.3 / 2017-02-22
-----------------
* Fix pyroot segfault when reading aanet header

6.4.2 / 2017-02-21
------------------
* Fix aanet header
* style update

6.4.1 / 2017-02-16
------------------
* API doc fixes
* add missing requirements to setup.py
* minor py2/py3 compat fix

6.4.0 / 2017-02-08
------------------
* K40 calibration module from Jonas!
* Pushover client! Push messages to your mobile phone or computer via
  ``pushover the message you want``.
* Minor bugfixes

6.3.0 / 2017-01-21
------------------
* Introduces `BinaryStruct` which makes handling binary data much more easier.
* `Cuckoo` now allows args and kwargs to be passed to the callback function.
* km3modules.plot module added including a unified DOM plotter
* km3modules.fit module added including k40 coincidence fit

6.2.2 / 2017-01-19
------------------
* add ``rundetsn`` cmd tool

6.2.1 / 2017-01-17
------------------
* Use numpy-style imports
* AanetPump: Don't use `evt.id` for event_id by default, until we all agree on it

6.2.0 / 2017-01-16
------------------
* The DB client now automatically uses the production cookie on Lyon.
  No need to deal with session requests anymore...
* New command line utility to download runs from iRODS: `km3pipe retrieve ...`
* Integrates the controlhost package

6.1.1 / 2017-01-12
------------------
* H5Chain now is just a Multifile pd.HDFStore
* `prettyln` for nicely formatted headers
* Online DAQ readout is now Python3 proof

6.1.0 / 2017-01-02
------------------
* H5Pump now supports multiple files
* h5concat util for concatenating multiple H5 files

6.0.4 / 2016-12-21
------------------
* fix: H5Sink in py3 actually creates indextables + closes file now
* HDF5 3.1: Change compression to BLOSC, fallback to zlib
* MergeDF module
* Easier access to seconds in timer

6.0.3
-----
* Fix Dataclass + IO conversion signatures towards consistency
* Ask for requesting new DB session when session expired.

6.0.2
-----
* Make blob ordered by default + actually use it in the pumps.

6.0.1
-----
* FIX freeze numpy version

6.0.0 2016-11-29
------------------
* change all bool dataclasses to int
* add new fields to event_info: livetime_sec, n_evs_gen, n_files_gen
* update KM3HDF -> v3

5.5.3 / 2016/11/28
------------------
* Add fix_event_id option to h5pump

5.5.2 / 2016-11-24
------------------
* Updated docs

5.5.1 / 2016-11-24
------------------
* Cuckoo now can be called directly
* CHPump uses Cuckoo for log.warn to avoid spamming in case of
  high network traffic
* DOM class to represent DOMs retrieved by the DBManager

5.5 / 2016-11-18
----------------
* New ``KM3DataFrame + KM3Array`` dataclasses, np/pandas subclasses + metadata
* replaced ``ArrayTaco`` with ``KM3Array``
* ``H5Mono`` pump to read HDF5 with flat table structure

5.4 / 2016-11-08
----------------
* Add a bunch of useful km3modules

5.3.3 / 2016-11-04
------------------
* Fix time calib application

5.3.2 / 2016-11-03
------------------
* add preliminary bootstrap script

5.3.0 / 2016-11-03
------------------
* Detector.dom_positions now returns an OrderedDict instead of a list
* Cache DOM positions in Detector
* pld3 function in tools, to calculate point-line-distance in 3d

5.2.2 / 2016-10-26
------------------
* Fixes Cython dependency
* ``kp.io.pandas.H5Chain`` now returns N _events_, not _rows_

5.2.0 / 2016-10-25
------------------
* Introduce ``configure`` method in ``Module``, so you no longer need to
  override ``__init__`` and call ``super``. You can, though ;)

5.1.5 / 2016-10-24
------------------
* DB/Dataclass bugfixes

5.1.2 / 2016-10-20
------------------
* Unify Reco + Wrapper dataclass. Reco(map, dtype) -> ArrayTaco.from_dict()
* add ``to='pandas'`` option to ``Dataclass.serialise()``
* Tweak internal array/dataframe handling

5.1.0 / 2016-10-20
------------------
* ...

5.0.0 / 2016-10-18
------------------
* Major dataclass refactor:
  * hits now always have pos_x, .., dir_y, .., t0
  * completely flat hit datastructure

4.9.0 / 2016-10-14
------------------
* New plot style handling and new styles: talk, poster, notebook
  (load them using `km3pipe.style.use(...)`)
  Just like in previous versions: `import km3pipe.style` will load
  the default style.

4.8.3 / 2016-10-13
------------------
* Fixes t0 application in HitSeries

4.8.2 / 2016-10-13
------------------
* Fixes geometry application in HitSeries

4.8.1 / 2016-10-12
------------------
* Forcing matplotlib 2.0.0b4 as dependency. Don't blame us!
* New unified style for all plots, using `import km3pipe.style`
* aanet / jgandalf: write zeroed row if no track in event
* fix string handling in H5 attributes

4.8.0 / 2016-10-11
------------------
* Group frames in summary slices under /timeslices/slice_id/frame_id
  when using ``tohdf5 -j -s FILE.root``
* ``hdf2root`` is now it's own command
* ``tohdf5`` and ``hdf2root`` no longer ``km3pipe`` CLI subcommands
* Use zlib instead of blosc for compatibility reasons
* add CLI option to make DB connection non-permanent
* ``tohdf5`` / ``GenericPump`` now supports multiple input files for aanet files

4.7.1 / 2016-09-29
------------------
* Improved documentation
* Fixed event_id indexing for the /hits table in HDF5
* root sub-package added (via rootpy)
* Added arguments to allow optional parsing of L0 data and summaryslices
  when using the JPPPump
* New command line utility to convert to HDF5: ``tohdf5``

4.7.0 / 2016-09-25
------------------
* Adds summary slice readout support via jppy
* Introducing astro package
* Use BLOSC compression library for HDF5

4.6.0
-----
* ...

4.5.1
-----
* Bugfixes

4.5.0
-----
* Full L0 readout support via ``JPPPump``

4.4.1
-----
* Bugfixes

4.4.0
-----
* JEvt/JGandalf support
* Minor HDF5 Improvements

4.3.0
-----
* Introduces HDF5 format versioning

4.2.2
-----
* Bugfixes

4.2.1
-----
* Bugfixes

4.2.0
-----
* ...

4.1.2
-----
* Bugfixes

4.1.1 / 2016-08-09
------------------
* Bugfixes

4.1.0 / 2016-08-04
------------------
* Ability to use simple functions as modules
