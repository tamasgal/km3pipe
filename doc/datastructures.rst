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

The class used in KM3Pipe to represent almost any kinds of datastructures
which are written to or read from the disk or network connections is called
``Table``. It is a two-dimensional numpy ndarray (``np.recarray`` subclass),
where each "column" (1D array) is accessible using its specific attribute.

The ``Table`` is designed to work together with the ``HDF5Sink``, which
dumps the table data into a given location (``Table.h5loc`` attribute)
in an HDF5 file.

``Tables`` written using the ``HDF5Sink`` can be read bye the ``HDF5Pump``,
which will retrieve them in pieces (groups), just like they were written
during a pipeline. However, the HDF5 files containing ``Table`` data can also
be read by any other package since it uses only native HDF5 data structures
(``HDF5Compund``, for the experts).

Hits and McHits
---------------

If you want to analyse the hits or create your own reconstruction, 
the ``Hits`` and ``McHits`` datatypes are the most important ones. 

The ``Hits`` come with ``dom_id``, ``channel_id``, ``tot``, ``time``
and ``triggered`` and the ``McHits`` have ``a``, ``origin``, ``time`` and
``pmt_id``. Additional information about hit positions and directions etc
is available in the ``CalibHits`` datastructure.

All ``Table``-backed datastructures (hits, tracks, ...) have attributes which can
be accessed through the following getters:

+---------------------+--------------+---------+-----------+----------+-----------+------------+
| information         | getter       | type    | Hit       | McHit    | CalibHit  | CalibMcHit |
+=====================+==============+=========+===========+==========+===========+============+
| hit time            | .time        | float32 | X         | X        | X         | X          |
+---------------------+--------------+---------+-----------+----------+-----------+------------+
| time over threshold | .tot         | uint8   | X         |          | X         |            |
+---------------------+--------------+---------+-----------+----------+-----------+------------+
| a (number of p.e.)  | .a           | float32 |           | X        |           | X          |
+---------------------+--------------+---------+-----------+----------+-----------+------------+
| PMT ID              | .pmt_id      | uint32  |           | X        |           | X          |
+---------------------+--------------+---------+-----------+----------+-----------+------------+
| Channel ID          | .channel_id  | uint8   | X         |          | X         |            |
+---------------------+--------------+---------+-----------+----------+-----------+------------+
| DOM ID              | .dom_id      | uint32  | X         |          | X         |            |
+---------------------+--------------+---------+-----------+----------+-----------+------------+
| trigger information | .triggered   | bool    | X         |          | X         |            |
+---------------------+--------------+---------+-----------+----------+-----------+------------+
| origin (track ID)   | .origin      | uint32  |           | X        |           | X          |
+---------------------+--------------+---------+-----------+----------+-----------+------------+
| position            | .pos_[xzy]   | float32 |           |          | X         | X          |
+---------------------+--------------+---------+-----------+----------+-----------+------------+
| direction           | .dir_[xzy]   | float32 |           |          | X         | X          |
+---------------------+--------------+---------+-----------+----------+-----------+------------+
| t0                  | .t0          | float32 |           |          | X         |            |
+---------------------+--------------+---------+-----------+----------+-----------+------------+
| du                  | .du          | uint8   |           |          | X         | X          |
+---------------------+--------------+---------+-----------+----------+-----------+------------+
| floor               | .floor       | uint8   |           |          | X         | X          |
+---------------------+--------------+---------+-----------+----------+-----------+------------+

Note that if you access ``.tot`` of a ``Hits`` table for example, you will
get a 1D numpy array containing all the ToTs of the hits (in the order of the
hits). So you can for example quickly have a look at the ToT distribution of
the full event.

Calibrating Hits and McHits
---------------------------

Both ``Hits`` and ``McHits`` have corresponding
``CalibHits`` and ``CalibMcHits``. Those classes have additional attributes 
to access the position, direction and calibrated hit times. 
They also provide access to the DU and floor which the hit was registered.

In order to obtain the position, direction, the t0 correction and DU/floor, you
need to apply a calibration. KM3Pipe provides the ``Calibration`` class to do this
for you.

To create a calibration from a detector file::

    cal = kp.calib.Calibration(filename="path/to/detector.detx")


To apply the calibration to a set of hits::

    calibrated_hits = cal.apply(hits)

That's it, you will get a ``CalibHits`` or ``CalibMcHits`` table
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
    hi
    In [2]: hits = kp.dataclasses.Table.from_template([
       ...:    ...:     [13, 21, 12],
       ...:    ...:     [10, 11, 10],
       ...:    ...:     [3, 1, 2],
       ...:    ...:     [22, 23, 24],
       ...:    ...:     [False, True, True],
       ...:    ...:     23], 'Hits')
       ...:    ...:hits
       ...:
    Out[2]:
    Hits <class 'km3pipe.dataclasses.Table'>
    HDF5 location: /hits (split)
    channel_id (dtype: |u1) = [13 21 12]
    dom_id (dtype: <u4) = [10 11 10]
    time (dtype: <f8) = [3. 1. 2.]
    tot (dtype: |u1) = [22 23 24]
    triggered (dtype: |u1) = [0 1 1]
    group_id (dtype: <u4) = [23 23 23]
    
    In [3]: hits.sorted('time')
    Out[3]:
    Hits <class 'km3pipe.dataclasses.Table'>
    HDF5 location: /hits (split)
    channel_id (dtype: |u1) = [21 12 13]
    dom_id (dtype: <u4) = [11 10 10]
    time (dtype: <f8) = [1. 2. 3.]
    tot (dtype: |u1) = [23 24 22]
    triggered (dtype: |u1) = [1 1 0]
    group_id (dtype: <u4) = [23 23 23]


        In [4]: for h in hits:
       ...:     print(h.time)
       ...:     print(h['tot'])
       ...:
    3.0
    22
    1.0
    23
    2.0
    24

    In [5]: for k in hits.sorted('time'):
       ...:     print(h.time)
       ...:     print(h['tot'])
       ...:
    2.0
    24
    2.0
    24
    2.0
