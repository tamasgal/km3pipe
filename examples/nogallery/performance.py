#!/usr/bin/env python
from __future__ import division, print_function
import os
import km3pipe as kp
from km3pipe.dataclasses import HitSeriesA, CHitSeries, CPosition, Position
from km3modules.common import StatusBar

import ROOT

PATH='/sps/km3net/users/tgal/data/km3net/mu_oct14'
DATAFILE='km3net_jul13_90m_muatm10T23.km3_v5r1.JTE.root'

class HitStatisticsKm3pipe(kp.Module):
    def process(self, blob):
        hits = blob['Hits']
        triggered_hits = hits.triggered
        print("km3pipe: {0}".format(len(triggered_hits)))
        return blob

class HitStatistics2(kp.Module):
    def process(self, blob):
        hits = blob['Hits']
        triggered_hits = [hit for hit in hits if hit.triggered]
        print("2: {0}".format(len(triggered_hits)))
        return blob

class HitStatisticsAanet(kp.Module):
    def process(self, blob):
        hits = blob['Evt'].hits
        triggered_hits = [h for h in hits if h.trig]
        print("Aanet: {0}".format(len(triggered_hits)))
#        for hit in hits:
#            hit.pos = ROOT.Vec(1, 2, 3)
        return blob

class HitStatisticsA(kp.Module):
    def process(self, blob):
        hits = blob['Evt'].hits
        hits_a = HitSeriesA.from_aanet(hits)._data
        triggered_hits = [h for h in hits_a if h.triggered]
        print("A: {0}".format(len(triggered_hits)))
#        triggered_hits = [hit for hit in hits if hit.trig]
        return blob

class CHitStatistics(kp.Module):
    def process(self, blob):
        hits = blob['Evt'].hits
        chits = CHitSeries.from_aanet(hits)
        triggered_hits = chits.triggered
#        print(chits.time)
        print("CHits: {0}".format(len(triggered_hits)))
#        for hit in chits:
#            hit.pos = CPosition(1, 2, 3)
#        print(chits.time)
#        print(chits.tot)
#        print(chits.dom_id)
#        print(chits.channel_id)
#        print(chits.pmt_id)
        return blob

class CyHitStatistics(kp.Module):
    def process(self, blob):
        hits = blob['Evt'].hits
        chits = CHitSeries.from_aanet_as_cyhit(hits)
        triggered_hits = chits.triggered
#        print(chits.time)
        print("CHits: {0}".format(len(triggered_hits)))
#        for hit in chits:
#            hit.pos = CPosition(1, 2, 3)
#        print(chits.time)
#        print(chits.tot)
#        print(chits.dom_id)
#        print(chits.channel_id)
#        print(chits.pmt_id)
        return blob

pipe = kp.Pipeline(timeit=True)
pipe.attach(kp.io.AanetPump, filename=os.path.join(PATH, DATAFILE))
pipe.attach(StatusBar)
#pipe.attach(HitStatisticsKm3pipe)
#pipe.attach(HitStatistics2)
#pipe.attach(HitStatisticsA)
pipe.attach(CHitStatistics)
pipe.attach(CyHitStatistics)
pipe.attach(HitStatisticsAanet)
pipe.drain(100)

