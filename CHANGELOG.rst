Unreleased changes
------------------

5.5 / 2016-11-18
~~~~~~~~~~~~~~~~
* New ``KM3DataFrame + KM3Array`` dataclasses, np/pandas subclasses + metadata
* replaced ``ArrayTaco`` with ``KM3Array``

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
------------------
* Detector.dom_positions now returns an OrderedDict instead of a list
* Cache DOM positions in Detector
* pld3 function in tools, to calculate point-line-distance in 3d

5.2.2 / 2016-10-26
~~~~~~~~~~~~~~~~~~
* Fixes Cython dependency
* ``kp.io.pandas.H5Chain`` now returns N _events_, not _rows_

5.2.0 / 2016-10-25
------------------
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
------------------
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
------------------
* Adds summary slice readout support via jppy
* Introducing astro package
* Use BLOSC compression library for HDF5

4.6.0
-----
* ...

4.5.1
~~~~~
* Bugfixes

4.5.0
-----
* Full L0 readout support via ``JPPPump``

4.4.1
~~~~~
* Bugfixes

4.4.0
-----
* JEvt/JGandalf support
* Minor HDF5 Improvements

4.3.0
-----
* Introduces HDF5 format versioning

4.2.2
~~~~~
* Bugfixes

4.2.1
~~~~~
* Bugfixes

4.2.0
-----
* ...

4.1.2
~~~~~
* Bugfixes

4.1.1 / 2016-08-09
~~~~~~~~~~~~~~~~~~
* Bugfixes

4.1.0 / 2016-08-04
------------------
* Ability to use simple functions as modules
