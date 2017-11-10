Data Structures
===============

.. contents:: :local:

This section describes the basic data structures which a **pump** usually
provides via the **blob** dictionary. The pump is responsible to parse
the data and create a **blob** (a simple Python dictionary) for each
event in the file. When processing a data file with KM3Pipe, a module
chain is being utilised to cycle through the events. Each module within
the chain recieves the original, unaltered data from the pump and
further also additional information created by the preceeding modules.

Hits and McHits
---------------

If you want to analyse the hits or create your own recunstruction, these two
are the the most important ones.
The class used in KM3Pipe to represent a bunch of hits is called
``RawHitSeries`` and ``McHitSeries``.
The ``RawHitSeries`` comes with ``dom_id``, ``channel_id``, ``tot``, ``time``
and ``triggered`` and the ``McHitSeries`` has ``a``, ``origin``, ``time`` and
``pmt_id``.

+---------------+------------+---------------------------------+
| information   | dict key   | container type                  |
+===============+============+=================================+
| Raw Hits      | Hits       | RawHitSeries (np.ndarray-like)  |
+---------------+------------+---------------------------------+
| MC Hits       | McHits     | McHitSeries (np.ndarray-like)   |
+---------------+------------+---------------------------------+

The ``*Series`` classes are basically numpy ndarrays (you can access the
actual numpy array through the ``._arr`` attribute), but you can also iterate
through them (including slicing) and get instances of ``RawHit`` or ``McHit``
will.
Both the ``*Series`` and the elementary hit classes have attributes which can
be accessed through the following getters:

+---------------------+--------------+---------+-----------+----------+-----------+----------+
| information         | getter       | type    | RawHit    | McHit    | CRawHit   | CMcHit   |
+=====================+==============+=========+===========+==========+===========+==========+
| hit time            | .time        | float32 | X         | X        | X         | X        |
+---------------------+--------------+---------+-----------+----------+-----------+----------+
| time over threshold | .tot         | uint8   | X         |          | X         |          |
+---------------------+--------------+---------+-----------+----------+-----------+----------+
| a (number of p.e.)  | .a           | float32 |           | X        |           | X        |
+---------------------+--------------+---------+-----------+----------+-----------+----------+
| PMT ID              | .pmt_id      | uint32  |           | X        |           | X        |
+---------------------+--------------+---------+-----------+----------+-----------+----------+
| Channel ID          | .channel_id  | uint8   | X         |          | X         |          |
+---------------------+--------------+---------+-----------+----------+-----------+----------+
| DOM ID              | .dom_id      | uint32  | X         |          | X         |          |
+---------------------+--------------+---------+-----------+----------+-----------+----------+
| trigger information | .triggered   | bool    | X         |          | X         |          |
+---------------------+--------------+---------+-----------+----------+-----------+----------+
| origin (track ID)   | .origin      | uint32  |           | X        |           | X        |
+---------------------+--------------+---------+-----------+----------+-----------+----------+
| position            | .pos_[xzy]   | float32 |           |          | X         | X        |
+---------------------+--------------+---------+-----------+----------+-----------+----------+
| direction           | .dir_[xzy]   | float32 |           |          | X         | X        |
+---------------------+--------------+---------+-----------+----------+-----------+----------+
| t0                  | .t0          | float32 |           |          | X         |          |
+---------------------+--------------+---------+-----------+----------+-----------+----------+
| du                  | .du          | uint8   |           |          | X         | X        |
+---------------------+--------------+---------+-----------+----------+-----------+----------+
| floor               | .floor       | uint8   |           |          | X         | X        |
+---------------------+--------------+---------+-----------+----------+-----------+----------+

Note that if you access ``.tot`` of a ``RawHitSeries`` for example, you will
get a 1D numpy array containing all the ToTs of the hits (in the order of the
hits). So you can for example quickly have a look at the ToT distribution of
the full event.

Calibrating Hits and McHits
---------------------------

Both ``RawHitSeries`` and ``McHitSeries`` have a corresponding
``CRawHitSeries`` and ``CMcHitSeries``. The "C" stands for "Calibrated" and
those classes has additional attributes to access the position, direction and
calibrated hit times. They also provide access to the DU and floor which the
hit was registered.

In order to obtain the position, direction, the t0 correction and DU/floor, you
need to apply a calibration. KM3Pipe provides the ``Calibration`` class to do this
for you.

To create a calibration from a detector file::

    cal = kp.Calibration(filename="path/to/detector.detx")


To apply the calibration to a set of hits::

    calibrated_hits = cal.apply(hits)

That's it, you will get a ``CRawHitSeries`` or ``CMcHitSeries`` instance
respectively, with ``pos_x``, ``pos_y``, ... and also ``dir_x``, ``dir_y``...
and ``du``, ``floor``.


Another, even easier way is to calibrate your file beforehand, using the
``calibrate`` command line utility::

    calibrate DETXFILE HDF5FILE

If you read in the file with the ``km3pipe.io.hdf5.HDF5Pump``, it will 
automatically recognise the calibration and use the correct classes.

Sorting of Hits
---------------

All `HitSeries` classes derive from `DTypeAttr` which implements a very fast
sorting using the `numpy.argsort` method.

Here is an example showing how to sort a dummy hit series with 3 hits::

    In [1]: import km3pipe as kp

    In [2]: hits = kp.dataclasses.RawHitSeries.from_arrays(
       ...:     [13, 21, 12],
       ...:     [10, 11, 10],
       ...:     [3, 1, 2],
       ...:     [22, 23, 24],
       ...:     [False, True, True],
       ...:     23)
       ...: hits
       ...:
    Out[2]: RawHitSeries with 3 hits.

    In [3]: for h in hits:
       ...:     print(h)
       ...:
    RawHit: channel_id(13), dom_id(10), tot(22), time(3.0), triggered(0)
    RawHit: channel_id(21), dom_id(11), tot(23), time(1.0), triggered(1)
    RawHit: channel_id(12), dom_id(10), tot(24), time(2.0), triggered(1)

    In [4]: for h in hits.sorted():
       ...:     print(h)
       ...:
    RawHit: channel_id(21), dom_id(11), tot(23), time(1.0), triggered(1)
    RawHit: channel_id(12), dom_id(10), tot(24), time(2.0), triggered(1)
    RawHit: channel_id(13), dom_id(10), tot(22), time(3.0), triggered(0)

    In [5]: for h in hits.sorted('channel_id'):
       ...:     print(h)
       ...:
    RawHit: channel_id(12), dom_id(10), tot(24), time(2.0), triggered(1)
    RawHit: channel_id(13), dom_id(10), tot(22), time(3.0), triggered(0)
    RawHit: channel_id(21), dom_id(11), tot(23), time(1.0), triggered(1)

    In [6]: for h in hits.sorted('dom_id'):
       ...:     print(h)
       ...:
    RawHit: channel_id(13), dom_id(10), tot(22), time(3.0), triggered(0)
    RawHit: channel_id(12), dom_id(10), tot(24), time(2.0), triggered(1)
    RawHit: channel_id(21), dom_id(11), tot(23), time(1.0), triggered(1)
