Data Structures
===============

This section describes the basic data structures which a **pump**
provides via the **blob** dictionary. The pump is responsible to parse
the data and create a **blob** (a simple Python dictionary) for each
event in the file. When processing a data file with KM3Pipe, a module
chain is being utilised to cycle through the events. Each module within
the chain recieves the original, unaltered data from the pump and
further also additional information created by the preceeding modules.

Hits
----

There are two kinds of basic hit types: a **raw hit** representing either an
actual hit measured by the detector hardware or a calibrated MC hit which
does not contain MC information anyomre, and a **MC hit**, which
was created by a Monte Carlo simulation. The dictonary key naming
conventions for raw hits and MC hits are the following:

+---------------+------------+------------------------+
| information   | dict key   | container type         |
+===============+============+========================+
| Raw Hits      | Hits       | HitSeries (list-like)  |
+---------------+------------+------------------------+
| MC Hits       | MCHits     | HitSeries (list-like)  |
+---------------+------------+------------------------+

Both hit types have attributes which can be accessed through the
following getters:

+---------------------+--------------+-----------+-----------+----------+
| information         | getter       | type      | raw hit   | MC hit   |
+=====================+==============+===========+===========+==========+
| hit id              | .id          | numeric   | X         | X        |
+---------------------+--------------+-----------+-----------+----------+
| hit time            | .time        | numeric   | X         | X        |
+---------------------+--------------+-----------+-----------+----------+
| time over threshold | .tot         | numeric   | X         |          |
+---------------------+--------------+-----------+-----------+----------+
| PMT id              | .pmt_id      | numeric   | X         | X        |
+---------------------+--------------+-----------+-----------+----------+
| Channel id          | .channel_id  | numeric   | X         | X        |
+---------------------+--------------+-----------+-----------+----------+
| trigger information | ...          | ...       | X         |          |
+---------------------+--------------+-----------+-----------+----------+

to be continued...


Tracks
------

MC Tracks
~~~~~~~~~

Track Fits
~~~~~~~~~~
