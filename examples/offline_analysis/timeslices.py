#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et
"""
==================
Reading Timeslices
==================

This examples show how to access L0, L1, L2 and SN timeslice streams using
km3pipe. Note that you need an activated Jpp environment and jppy installed
(http://git.km3net.de/km3py/jppy.git).

Both are available in Lyon, wherebei ``jppy`` is included in the group
python installation.

A general note on timeslice streams: in older Jpp versions there was only
one timeslice stream present in the files. This was later separated into
L0, L1, L2 and SN streams. A convenient tool to check the availability of
streams is ``JPrintTree -f FILENAME``::

    > JPrintTree -f KM3NeT_00000014_00004451.root
    KM3NeT_00000014_00004451.root
    KM3NET_TIMESLICE     KM3NETDAQ::JDAQTimeslice     2808  8416 [MB]
    KM3NET_EVENT         KM3NETDAQ::JDAQEvent              70     0 [MB]
    KM3NET_SUMMARYSLICE  KM3NETDAQ::JDAQSummaryslice     2808     4 [MB]

Here is an example with split timeslice streams::

    > JPrintTree -f KM3NeT_00000029_00003242.root
    KM3NeT_00000029_00003242.root
    KM3NET_TIMESLICE     KM3NETDAQ::JDAQTimeslice          0      0 [MB]
    KM3NET_TIMESLICE_L0  KM3NETDAQ::JDAQTimesliceL0        0      0 [MB]
    KM3NET_TIMESLICE_L1  KM3NETDAQ::JDAQTimesliceL1     5390    319 [MB]
    KM3NET_TIMESLICE_L2  KM3NETDAQ::JDAQTimesliceL2        0      0 [MB]
    KM3NET_TIMESLICE_SN  KM3NETDAQ::JDAQTimesliceSN   107910    162 [MB]
    KM3NET_EVENT         KM3NETDAQ::JDAQEvent          21445     24 [MB]
    KM3NET_SUMMARYSLICE  KM3NETDAQ::JDAQSummaryslice  107910    109 [MB]


To access the Supernova Timeslice hits, you pass ``stream=SN`` to the
``TimeslicePump``, as seen here:

"""

ROOT_FILENAME = "KM3NeT_00000014_00004451.root"

import km3pipe as kp

pump = kp.io.jpp.TimeslicePump(filename=ROOT_FILENAME, stream='SN')
for blob in pump:
    hits = blob['TSHits']

####################################################################
# The timeslice pump is used to convert the timeslice objects in
# the ROOT file into numpy recarrays. We explicitly set the stream
# to an empty string, since we are opening an older file.
#
# Here is how to access the hits.

pump = kp.io.jpp.TimeslicePump(filename=ROOT_FILENAME, stream='')
for blob in pump:
    hits = blob['TSHits']
